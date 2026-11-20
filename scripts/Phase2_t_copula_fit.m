% PHASE2_T_COPULA_FIT Phase 2: fit the Student-t copula shape parameter to every channel set of a recording.
%
% Turns one recording's cleaned, band-limited X_filtered into the study's unit rows.
% Each channel set goes through the same four steps in this order: pseudo-observations, Kendall's tau-b, R by the sine relation with the eigenvalue-floor repair, and profile maximum likelihood of p = 1/nu at fixed R.
% tFitChannelSet carries all four steps; this script chooses the channel sets, schedules the work, checkpoints it and writes the tables.
%
% The units are every channel pair at d = 2 and groups_per_d seeded groups at each d of d_list.
% The group seed is channelGroupSeed(seed_base, d), tied to the recording's identity rather than to its place in a run list, and Phases 3 and 4 read these channel sets from fit_all.mat.
% p = 0 is the Gaussian copula and a valid estimate.
% Nothing is excluded on the value of p_hat: the heavy end of the domain, a near-tied remote basin and a gross correlation repair are recorded statuses, and a unit is dropped only on a numerical failure.
%
% Resuming: every chunk of chunk_size units is checkpointed under mat/checkpoints/, a finished dimension is merged into mat/fit_dXX.mat and drops its chunk files, and a recording whose mat/fit_all.mat exists is not refitted.
% A run cut off by a wall-clock limit therefore resumes at the first unfinished chunk, and a finished recording costs seconds.
% TC_OVERWRITE=1 recomputes everything and first deletes the recording's existing output (index, per-dimension files, chunks, tables and both stamps), so an interrupted overwrite can never be resumed from, or mistaken for, the run it overwrites.
%
% Every reuse is checked, and a recording is done only when it is published.
% Chunks, per-dimension files and the index all carry settings_identity, the hash of this phase's scientific configuration and of the recording.json it read (settingsIdentity), and a file whose identity is not this run's is refused, not merged.
% Publication order is chunks, per-dimension files, fit_all.mat, the CSVs, the figures, then run_config.json.
% A recording counts as finished only when the index, its identity, every table the index lists (localPublishedTables) and run_config.json are all present.
% An interruption after the fits, or a table lost later, therefore regenerates the tables from the index without refitting anything, instead of reporting a half-published recording as done.
%
% mat/fit_dXX.mat is the full record of one dimension: the unit rows with their coarse profiles, the packed tau and R upper triangles, the settings, the options and the timing.
% mat/fit_all.mat is a compact index over all of them: every unit row without its two profile vectors, plus the names of the per-dimension files that carry the profiles and the matrices.
% The index lets a later phase read every unit of a recording in one load, and it is not a second copy of the per-dimension files, which at production scale would double a multi-gigabyte tree for numbers only the profile figure reads.
%
% In both files `units` is the column struct of unitRowsToColumns: one array per field with one row per unit, not a struct array.
% A -v7.3 struct array stores one HDF5 dataset per field per element, so the roughly 180,000 rows of a production recording would cost gigabytes of headers for tens of megabytes of numbers.
% The chunk checkpoints under mat/checkpoints/ keep the plain struct array, because they are transient, deleted as soon as their dimension merges, and never larger than chunk_size units.
%
% Recordings are discovered, not configured: Phase 1 writes a recording.json beside every output tree, discoverPhase1Recordings finds them, and this phase carries no recording table.
%
% Environment: TC_RESULTS_ROOT, TC_RECORDINGS, TC_RECORDING_INDEX, TC_WORKERS, TC_OVERWRITE, TC_RENDER_FIGURES, plus the two pilot overrides TC_GROUPS_PER_D and TC_MAX_PAIRS.
%
% Run:
%   matlab -batch "cd('<repo>'); addpath('scripts'); Phase2_t_copula_fit"

% Find the repository root (the first folder above this script that holds src/ and scripts/) and put src/ on the path.
script_dir = fileparts(mfilename('fullpath'));
worktree_root = script_dir;
while ~(isfolder(fullfile(worktree_root, 'src')) && isfolder(fullfile(worktree_root, 'scripts')))
    parent = fileparts(worktree_root);
    if strcmp(parent, worktree_root)
        error('bootstrap:RepoRootNotFound', ...
              'repository root (a dir with src/ and scripts/) not found above %s', script_dir);
    end
    worktree_root = parent;
end
addpath(genpath(fullfile(worktree_root, 'src')));

%% ============================================================ settings
settings = struct( ...
                  'd_list',       3:20, ...   % d = 2 is always fitted, as every channel pair
                  'groups_per_d', 10000, ...  % seeded channel groups per dimension
                  'max_pairs',    [], ...     % [] = every pair; a count keeps the first that many
                  'chunk_size',   1000, ...   % units per checkpointed parfor call
                  'fit_opts',     struct());  % overrides of tFitDefaultOptions; see that file

steps = struct( ...
               'render_figures', getEnvFlag("TC_RENDER_FIGURES", true), ...
               'overwrite',      getEnvFlag("TC_OVERWRITE", false));

% The fit domain and the coarse profile grid are the fitter's, materialised here so that run_config.json records them and the quantile cache tabulates exactly the coarse-profile nodes fitTCopulaShape evaluates; the refinement steps between the nodes compute their quantiles without it.
% Deriving the grid from the options makes the cache a speed-up rather than a second, silently different grid.
fit_defaults = tFitDefaultOptions(settings.fit_opts);
settings.profile_grid = linspace(fit_defaults.p_bounds(1), fit_defaults.p_bounds(2), ...
                                 fit_defaults.profile_grid_size);

results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));
selected_keys = getEnvList("TC_RECORDINGS");
selected_index = getEnvInteger("TC_RECORDING_INDEX", [], [1, Inf]);
requested_workers = getEnvInteger("TC_WORKERS", [], [0, 1024]);
% Pilot overrides let a timing pilot run the whole phase end to end in minutes; a production run leaves both unset and gets the values above.
settings.groups_per_d = getEnvInteger("TC_GROUPS_PER_D", settings.groups_per_d, [1, Inf]);
settings.max_pairs = getEnvInteger("TC_MAX_PAIRS", settings.max_pairs, [1, Inf]);

% The settings identity is the scientific configuration of this phase, hashed and stamped into every chunk, every per-dimension file, fit_all.mat and run_config.json.
% Every reuse below compares it, so a checkpoint or a finished recording written under other settings is refused instead of merged under the same unit indices.
% chunk_size is left out because it only schedules the work and changes no number, while the pilot overrides are part of it because they change which channel sets are fitted.
% localIdentityFields adds the per-recording half from recording.json: the Phase 1 identity of the input, the recording's seed_base and retained channel count, which together decide the channel sets, and the Phase 1 schema version.
identity_fields = struct( ...
                         'phase',        'Phase2', ...
                         'd_list',       settings.d_list, ...
                         'groups_per_d', settings.groups_per_d, ...
                         'max_pairs',    settings.max_pairs, ...
                         'fit_opts',     tFitDefaultOptions(settings.fit_opts), ...
                         'profile_grid', settings.profile_grid);

%% ============================================================ run
recordings = localSelectRecordings( ...
                                   discoverPhase1Recordings(results_root, selected_keys), selected_index, selected_keys);
fprintf(['[Phase2] %d recording(s) | groups_per_d %d | max_pairs %s | chunk %d | ' ...
         'results %s\n'], numel(recordings), settings.groups_per_d, ...
        mat2str(settings.max_pairs), settings.chunk_size, results_root);

failures = strings(0, 1);
n_workers = [];
for i = 1:numel(recordings)
    rec = recordings(i);
    try
        identity = settingsIdentity(localIdentityFields(identity_fields, rec));
        probe = openPhaseOutput(rec.rel_base, 2, results_root, 'Diary', false);
        fit_all_path = fullfile(probe.mat_dir, 'fit_all.mat');
        if ~steps.overwrite && isfile(fit_all_path)
            % A recording is done when its index, its identity, every table the index lists and the provenance stamp are all present.
            % The stamp is written last, so a wall-clock cut between the index and the tables, or a table lost afterwards, sends the recording to the republish below, which rebuilds the tables from the index without recomputing any fit.
            localAssertIdentity(fit_all_path, identity);
            if localIsPublished(fit_all_path, probe.text_dir)
                fprintf('[Phase2] %s: already done, skipping (%s)\n', ...
                        rec.dataset_key, fit_all_path);
                continue
            end
            localRepublishRecording(rec, steps, results_root, fit_all_path, identity);
            continue
        end
        if isempty(n_workers)
            % The pool opens on the first recording that still has work, so a finished phase (the last link of a resume chain) exits in seconds instead of paying for a pool start it never uses.
            n_workers = localOpenPool(requested_workers);
            fprintf('[Phase2] parallel pool: %d worker(s)\n', n_workers);
        end
        localFitRecording(rec, settings, steps, results_root, n_workers, identity);
    catch recording_error
        failures(end + 1, 1) = string(rec.dataset_key) + " [" + ...
            string(recording_error.identifier) + "] " + ...
            string(recording_error.message); %#ok<SAGROW>
        fprintf(2, '\n[Phase2] %s failed (%s)\n%s\n', rec.dataset_key, ...
                recording_error.identifier, ...
                getReport(recording_error, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\n[Phase2] finished: %d of %d recording(s) failed\n', ...
        numel(failures), numel(recordings));
if ~isempty(failures)
    error('Phase2:RecordingFailures', ...
          'Phase 2 failed for %d recording(s):\n%s', numel(failures), ...
          strjoin(failures, newline));
end

% ===========================================================================
% Local functions
% ===========================================================================
function selected = localSelectRecordings(recordings, index, keys)
    % Apply TC_RECORDING_INDEX, which addresses the canonical order of discoverPhase1Recordings (ascending seed_base); TC_RECORDINGS has already filtered by key at discovery.
    if isempty(index)
        selected = recordings;
        return
    end
    assert(isempty(keys), 'Phase2:AmbiguousSelection', ...
           ['TC_RECORDINGS and TC_RECORDING_INDEX are mutually exclusive; got ' ...
            'keys "%s" and index %d.'], strjoin(keys, ', '), index);
    assert(index <= numel(recordings), 'Phase2:RecordingIndexOutOfRange', ...
           'TC_RECORDING_INDEX is %d but only %d recording(s) were discovered (%s).', ...
           index, numel(recordings), strjoin(string({recordings.dataset_key}), ', '));
    selected = recordings(index);
end

function fields = localIdentityFields(fields, rec)
    % Add the recording-dependent half of the identity from recording.json: the Phase 1 identity of the input (the table row and the preprocessing policy that produced the matrix), the seed_base that seeds every channel-group sample, the retained channel count every pair and group is drawn from, and the Phase 1 schema version.
    % A change to any of them means other channel sets or another input, so fits made under other values are never reused.
    fields.phase1_identity = localRecordText(rec, 'settings_identity');
    fields.phase1_schema_version = localRecordText(rec, 'phase1_schema_version');
    fields.seed_base = double(rec.seed_base);
    fields.n_channels = double(rec.n_channels_retained);
end

function text = localRecordText(rec, name)
    % One text field of recording.json, or '' when the sidecar does not carry it.
    text = '';
    if isfield(rec, name)
        text = char(string(rec.(name)));
    end
end

function done = localIsPublished(fit_all_path, text_dir)
    % True when the index lists the tables of the recording's publication, every one of them exists, and the provenance stamp, which is written last, exists.
    % An index that lists no table is not taken as published.
    tables = localStoredVariable(fit_all_path, 'published_tables', {});
    done = ~isempty(tables) && isfile(fullfile(text_dir, 'run_config.json')) ...
        && all(cellfun(@(name) isfile(fullfile(text_dir, name)), tables));
end

function tables = localPublishedTables(d_values)
    % The tables that make up one recording's publication, declared only here.
    % fit_all.mat records the list for its own units, localPublishRecording writes exactly those tables, and the recording counts as finished only when every one exists.
    % A unit kind the index does not hold has no table (there is no pair table without a pair), so d_values = [2 3] gives the full declared set.
    tables = {'pairs.csv', 'groups_long.csv', 'summary_by_d.csv', ...
              'status_counts_by_d.csv', 'timing_by_d.csv'};
    tables = tables([any(d_values == 2), any(d_values >= 3), true, true, true]);
end

function localWithdrawPublication(ctx, checkpoint_dir)
    % Under TC_OVERWRITE=1, delete the recording's existing output before anything is refitted: the index, every per-dimension file, every chunk, the tables and both stamps.
    % An interrupted overwrite then leaves nothing that looks finished, and the resume that follows can only continue this computation, never pick up a dimension or a chunk of the run it overwrites.
    stale = [ ...
             localListFiles(ctx.mat_dir, 'fit_*.mat'), ...
             localListFiles(checkpoint_dir, 'fit_d*_chunk*.mat'), ...
             fullfile(ctx.text_dir, [localPublishedTables([2, 3]), ...
                                {'run_config.json', 'figures_manifest.json'}]), ...
             {fullfile(ctx.images_dir, 'figures_manifest.json')}];
    localDeleteFiles(stale);
    fprintf('[Phase2] TC_OVERWRITE=1: the previous output of %s was withdrawn\n', ...
            ctx.base);
end

function paths = localListFiles(folder, pattern)
    % The files of one folder that match a pattern, as a row of full paths.
    listing = dir(fullfile(folder, pattern));
    paths = cell(1, numel(listing));
    for k = 1:numel(listing)
        paths{k} = fullfile(listing(k).folder, listing(k).name);
    end
end

function localDeleteFiles(paths)
    % Delete each listed file that exists.
    for k = 1:numel(paths)
        if isfile(paths{k})
            delete(paths{k});
        end
    end
end

function localAssertIdentity(path, identity)
    % Refuse an artifact written under another scientific configuration.
    % Artifacts are keyed by recording, dimension or chunk ordinal, so a file from other settings describes other numbers under the same key and is never merged.
    stored = localStoredIdentity(path);
    if strcmp(stored, identity)
        return
    end
    if isempty(stored)
        stored = '<none recorded>';
    end
    error('Phase2:SettingsIdentityMismatch', ...
          ['%s was written under the settings identity %s, but this run''s ' ...
           'settings are %s. Rerun with TC_OVERWRITE=1 to recompute it, or ' ...
           'restore the settings that produced it.'], path, stored, identity);
end

function identity = localStoredIdentity(path)
    % The settings identity a MAT file stores as a top-level variable, or '' when it stores none.
    identity = char(string(localStoredVariable(path, 'settings_identity', '')));
end

function value = localStoredVariable(path, name, default)
    % One top-level variable of a MAT file, or DEFAULT when the file does not contain it.
    value = default;
    if ismember(name, who('-file', path))
        loaded = load(path, name);
        value = loaded.(name);
    end
end

function n_workers = localOpenPool(requested_workers)
    % TC_WORKERS: 0 runs serially, and empty takes the worker count of MATLAB's local cluster profile.
    % Pool size is scheduling only: every unit is independent, so the worker count changes no reported result.
    if isequal(requested_workers, 0)
        n_workers = openParallelPool(false);
    else
        n_workers = openParallelPool(true, requested_workers);
    end
end

function localFitRecording(rec, settings, steps, results_root, n_workers, identity)
    % Fit every channel set of one recording, resuming from whatever is on disk; the caller has already found the recording unfinished.
    % ctx owns the diary through an onCleanup guard, so the diary flushes and closes when this function returns, including on an error.
    ctx = openPhaseOutput(rec.rel_base, 2, results_root);
    fit_all_path = fullfile(ctx.mat_dir, 'fit_all.mat');
    started_utc = datetime('now', 'TimeZone', 'UTC');
    checkpoint_dir = fullfile(ctx.mat_dir, 'checkpoints');
    if ~isfolder(checkpoint_dir)
        mkdir(checkpoint_dir);
    end
    if steps.overwrite
        localWithdrawPublication(ctx, checkpoint_dir);
    end

    [X, channel_ids, segment_boundaries] = localLoadPhase1Signal(rec);
    [n, n_channels] = size(X);
    fprintf('\n========================================================\n');
    fprintf('Phase 2 recording: %s\n', rec.dataset_key);
    fprintf('  input : %s\n', rec.phase1_mat_abs);
    fprintf('  signal: %d samples x %d channels in %d segment(s)\n', ...
            n, n_channels, numel(segment_boundaries));
    fprintf('  output: %s\n', ctx.base);
    fprintf('========================================================\n');

    fit_opts = tFitDefaultOptions(settings.fit_opts);
    cache_timer = tic;
    cache = tCopulaQuantileCache(n, settings.profile_grid);
    fprintf('[Phase2] quantile cache: %d distinct tail probabilities x %d nodes in %.1f s\n', ...
            cache.num_distinct, numel(settings.profile_grid), toc(cache_timer));

    dims = unique([2, settings.d_list]);
    per_dimension = cell(1, numel(dims));
    timing = nan(numel(dims), 3);   % [n_units, wall_seconds, median_fit_seconds]
    for k = 1:numel(dims)
        [per_dimension{k}, timing(k, :)] = localFitDimension(dims(k), X, ...
                                                             channel_ids, rec, settings, steps, fit_opts, cache, ctx, ...
                                                             checkpoint_dir, n_workers, identity);
    end

    kept = ~cellfun(@isempty, per_dimension);
    assert(any(kept), 'Phase2:NoUnits', ...
           ['No dimension of %s produced a channel set: the recording retained %d ' ...
            'channels and the requested dimensions were %s.'], ...
           rec.dataset_key, n_channels, mat2str(dims));
    units = [per_dimension{kept}];

    %% The compact index over the per-dimension files, then the tables
    % The profiles and the packed tau and R upper triangles left out here stay in the fit_dXX.mat files that per_dimension_files names.
    % Phases 3, 4 and 5 take their units from 'units' and read none of the fields left out, so this is the whole unit table they need in one load.
    payload = struct();
    payload.units = rmfield(unitRowsToColumns(units), ...
                            {'profile_p_grid', 'profile_loglik_grid'});
    payload.settings = settings;
    payload.settings_identity = identity;
    payload.fit_opts = fit_opts;
    payload.n = n;
    payload.n_channels = n_channels;
    payload.dataset_key = rec.dataset_key;
    payload.d_list = dims(kept);
    payload.per_dimension_files = arrayfun(@localDimensionFileName, dims(kept), ...
                                           'UniformOutput', false);
    % The timing table travels in the index because a republish regenerates the tables from this MAT alone and must be able to write timing_by_d.csv too.
    payload.timing_by_d = localTimingTable(dims(kept), timing(kept, :), n_workers);
    payload.published_tables = localPublishedTables([units.d]);
    localSaveMatAtomic(fit_all_path, payload);
    fprintf(['[Phase2] index over %d unit(s) and %d dimension(s), profiles and ' ...
             'packed matrices left in %s: %s\n'], numel(units), nnz(kept), ...
            strjoin(payload.per_dimension_files, ', '), fit_all_path);

    localPublishRecording(ctx, rec, payload, units, steps, identity, struct( ...
                                                                            'n_workers',    n_workers, ...
                                                                            'started_utc',  started_utc, ...
                                                                            'finished_utc', datetime('now', 'TimeZone', 'UTC'), ...
                                                                            'phase1_mat',   rec.phase1_mat_abs, ...
                                                                            'n',            n, ...
                                                                            'n_channels',   n_channels, ...
                                                                            'n_units',      numel(units)));
    % ctx goes out of scope here, and its onCleanup turns the diary off.
end

function localRepublishRecording(rec, steps, results_root, fit_all_path, identity)
    % Rebuild every table and figure from the index, without refitting any channel set, and stamp the recording so the next run reads it as finished.
    % This runs when the index is on disk with a matching identity but a table it lists, or run_config.json, is missing: the run that wrote the index stopped between the fits and the tables, or a table was lost later.
    ctx = openPhaseOutput(rec.rel_base, 2, results_root);
    started_utc = datetime('now', 'TimeZone', 'UTC');
    fprintf(['[Phase2] %s: the fits are complete but their publication is ' ...
             'incomplete; regenerating the tables and figures from %s WITHOUT ' ...
             'REFITTING\n'], rec.dataset_key, fit_all_path);
    payload = load(fit_all_path);
    units = unitColumnsToRows(payload.units);
    localPublishRecording(ctx, rec, payload, units, steps, identity, struct( ...
                                                                            'n_workers',        0, ...
                                                                            'started_utc',      started_utc, ...
                                                                            'finished_utc',     datetime('now', 'TimeZone', 'UTC'), ...
                                                                            'phase1_mat',       rec.phase1_mat_abs, ...
                                                                            'n',                payload.n, ...
                                                                            'n_channels',       payload.n_channels, ...
                                                                            'n_units',          numel(units), ...
                                                                            'regenerated_from', fit_all_path));
end

function localPublishRecording(ctx, rec, payload, units, steps, identity, provenance)
    % Write the tables, the figures and the provenance stamp of one recording in the order that makes run_config.json a completion marker: the MAT file (already written by the caller), the CSVs, the figures, then the stamp.
    % An interruption anywhere before the stamp leaves the recording unfinished.
    % payload is what fit_all.mat holds, whether just built or just read, so a fresh run and a republish write the same tables.
    localWriteUnitCsv(units([units.d] == 2), fullfile(ctx.text_dir, 'pairs.csv'));
    localWriteUnitCsv(units([units.d] >= 3), fullfile(ctx.text_dir, 'groups_long.csv'));
    [summary_by_d, status_counts_by_d] = summarizeUnitRowsByDimension(units);
    writeTableAtomic(summary_by_d, fullfile(ctx.text_dir, 'summary_by_d.csv'));
    writeTableAtomic(status_counts_by_d, ...
                     fullfile(ctx.text_dir, 'status_counts_by_d.csv'));
    writeTableAtomic(payload.timing_by_d, fullfile(ctx.text_dir, 'timing_by_d.csv'));
    disp(summary_by_d);

    if steps.render_figures
        try
            localReportFigureStatus(renderPhaseFigures('render_phase2_figures', ...
                                                       ctx.base, ctx.images_dir, ctx.text_dir), rec.dataset_key);
        catch figure_error
            fprintf(2, '[Phase2] %s figure layer failed (%s); the data are unaffected.\n', ...
                    rec.dataset_key, figure_error.message);
        end
    end

    run_settings = payload.settings;
    run_settings.steps = steps;
    run_settings.recording = rec;
    run_settings.settings_identity = identity;
    writeRunConfig(ctx, run_settings, provenance);
end

function [X, channel_ids, segment_boundaries] = localLoadPhase1Signal(rec)
    % Load the analysis input of one recording and check it against what its sidecar declares.
    % A transposed matrix does not error downstream but silently yields a wrong correlation structure, so the orientation is checked against the sample and channel counts Phase 1 recorded rather than guessed from the shape.
    stored = whos('-file', rec.phase1_mat_abs);
    names = {stored.name};
    wanted = {'X_filtered', 'channel_ids'};
    missing = wanted(~ismember(wanted, names));
    assert(isempty(missing), 'Phase2:MissingPhase1Variable', ...
           'Phase 1 output %s holds no %s.', rec.phase1_mat_abs, strjoin(missing, ', '));

    loaded = load(rec.phase1_mat_abs, 'X_filtered', 'channel_ids');
    X = double(loaded.X_filtered);
    channel_ids = double(loaded.channel_ids(:)).';
    assert(isnumeric(X) && ismatrix(X) && ...
           size(X, 1) == rec.n_samples && size(X, 2) == rec.n_channels_retained, ...
           'Phase2:UnexpectedSignalShape', ...
           ['X_filtered of %s must be [samples x channels] = [%d x %d], as its ' ...
            'recording.json declares; %s holds %d x %d.'], ...
           rec.dataset_key, rec.n_samples, rec.n_channels_retained, ...
           rec.phase1_mat_abs, size(X, 1), size(X, 2));
    assert(isreal(X) && all(isfinite(X), 'all'), 'Phase2:NonfiniteSignal', ...
           'X_filtered of %s holds non-finite or complex values: %s', ...
           rec.dataset_key, rec.phase1_mat_abs);
    assert(numel(channel_ids) == size(X, 2), 'Phase2:ChannelIdMismatch', ...
           'X_filtered of %s has %d columns but %d channel IDs: %s', ...
           rec.dataset_key, size(X, 2), numel(channel_ids), rec.phase1_mat_abs);

    if ismember('segment_boundaries', names)
        boundaries = load(rec.phase1_mat_abs, 'segment_boundaries');
        segment_boundaries = double(boundaries.segment_boundaries(:));
    else
        segment_boundaries = size(X, 1);
    end
end

function [units, timing] = localFitDimension(d, X, channel_ids, rec, settings, ...
                                             steps, fit_opts, cache, ctx, checkpoint_dir, n_workers, identity)
    % Fit every channel set of one dimension, checkpointed chunk by chunk.
    % timing is [n_units, wall_seconds, median_fit_seconds]; wall_seconds counts this run's work, so a resumed dimension reports what the resume cost and a reused one what its original run cost.
    units = [];
    timing = nan(1, 3);
    fit_path = fullfile(ctx.mat_dir, localDimensionFileName(d));
    if ~steps.overwrite && isfile(fit_path)
        localAssertIdentity(fit_path, identity);
        stored = load(fit_path, 'units', 'seconds');
        % The file holds the column struct, while the rest of this script works on the rows that the chunk loop produces and the CSV writers take.
        units = unitColumnsToRows(stored.units);
        timing = [numel(units), stored.seconds, ...
                  median([units.fit_seconds], 'omitnan')];
        fprintf('[Phase2] d = %2d: reusing %d unit(s) from %s\n', ...
                d, numel(units), fit_path);
        return
    end

    grid = localUnitsForDimension(size(X, 2), d, settings, rec.seed_base);
    if isempty(grid)
        fprintf(['[Phase2] d = %2d: skipped, the recording retains only %d ' ...
                 'channels\n'], d, size(X, 2));
        return
    end

    n_units = size(grid, 1);
    n_chunks = ceil(n_units / settings.chunk_size);
    chunks = cell(1, n_chunks);
    dimension_timer = tic;
    for c = 1:n_chunks
        first = (c - 1) * settings.chunk_size + 1;
        last = min(c * settings.chunk_size, n_units);
        chunk_path = fullfile(checkpoint_dir, sprintf('fit_d%02d_chunk%03d.mat', d, c));
        if ~steps.overwrite && isfile(chunk_path)
            % A chunk file is keyed by (d, ordinal) only, so a chunk written under different settings (groups per dimension, pair cap, fit options) would describe other channel sets, or other fits of them, under the same unit indices.
            % The identity check refuses every such file, and the grid check below also refuses one written with a different chunk size, which moves the indices without changing the science.
            localAssertIdentity(chunk_path, identity);
            stored = load(chunk_path, 'chunk');
            stored_channels = vertcat(stored.chunk.rows.channels);
            if ~isequal(stored.chunk.unit_indices, first:last) ...
                    || size(stored_channels, 1) ~= last - first + 1 ...
                    || ~isequaln(stored_channels(:, 1:d), grid(first:last, :))
                error('Phase2:CheckpointGridMismatch', ...
                      ['%s was written for a different unit grid than this run ' ...
                       '(d = %d, units %d to %d). Delete mat/checkpoints and rerun, ' ...
                       'or restore the settings that produced it.'], ...
                      chunk_path, d, first, last);
            end
            chunks{c} = stored.chunk;
            fprintf('[Phase2] d = %2d chunk %3d/%3d: reused %d unit(s) (%s)\n', ...
                    d, c, n_chunks, numel(stored.chunk.rows), chunk_path);
            continue
        end
        chunks{c} = localFitChunk(d, c, grid(first:last, :), first, X, rec, ...
                                  fit_opts, cache, n_workers);
        % The identity is a separate top-level variable, so the reuse check above reads it without loading the chunk's unit rows.
        localSaveMatAtomic(chunk_path, ...
                           struct('chunk', chunks{c}, 'settings_identity', identity));
        fprintf(['[Phase2] d = %2d chunk %3d/%3d: %d unit(s) in %.1f s ' ...
                 '(%.0f%% of the dimension, %.0f s elapsed)\n'], ...
                d, c, n_chunks, numel(chunks{c}.rows), chunks{c}.seconds, ...
                100 * last / n_units, toc(dimension_timer));
    end

    rows_by_chunk = cellfun(@(chunk) chunk.rows, chunks, 'UniformOutput', false);
    tau_by_chunk = cellfun(@(chunk) chunk.tau_packed, chunks, 'UniformOutput', false);
    r_by_chunk = cellfun(@(chunk) chunk.R_packed, chunks, 'UniformOutput', false);
    units = [rows_by_chunk{:}];

    payload = struct();
    payload.units = unitRowsToColumns(units);
    payload.tau_packed = vertcat(tau_by_chunk{:});
    payload.R_packed = vertcat(r_by_chunk{:});
    payload.settings = settings;
    payload.settings_identity = identity;
    payload.fit_opts = fit_opts;
    payload.seconds = toc(dimension_timer);
    payload.n = size(X, 1);
    payload.n_channels = size(X, 2);
    payload.channel_ids = channel_ids;
    payload.dataset_key = rec.dataset_key;
    payload.d = d;
    localSaveMatAtomic(fit_path, payload);
    for c = 1:n_chunks
        chunk_path = fullfile(checkpoint_dir, sprintf('fit_d%02d_chunk%03d.mat', d, c));
        if isfile(chunk_path)
            delete(chunk_path);
        end
    end
    timing = [numel(units), payload.seconds, median([units.fit_seconds], 'omitnan')];
    fprintf('[Phase2] d = %2d: %d unit(s) in %.1f s -> %s\n', ...
            d, numel(units), payload.seconds, fit_path);
end

function chunk = localFitChunk(d, c, groups, first_unit, X, rec, fit_opts, ...
                               cache, n_workers)
    % Fit one chunk of channel sets in one parfor call, returning their unit rows and their packed tau and R matrices.
    %
    % debt: X and the quantile cache are broadcast on every chunk (about 66 MB of cache at n = 4e5, plus the signal).
    % A parallel.pool.Constant hoisted to the recording would send them once per pool instead of once per chunk; the plain broadcast is kept while a recording has at most a few hundred chunks.
    m = size(groups, 1);
    rows = cell(1, m);
    taus = cell(1, m);
    correlations = cell(1, m);
    dataset_key = rec.dataset_key;
    chunk_timer = tic;
    parfor (u = 1:m, n_workers)
        [row, tau, R] = tFitChannelSet(X, groups(u, :), fit_opts, cache);
        row.dataset_key = dataset_key;
        row.unit_index = first_unit + u - 1;
        rows{u} = row;
        taus{u} = tau;
        correlations{u} = R;
    end

    % Packing happens outside the parfor so the loop body is the four study steps and nothing else; with matrices of at most 20 x 20 this loop costs nothing.
    tau_packed = nan(m, 190);
    r_packed = nan(m, 190);
    for u = 1:m
        tau_packed(u, :) = localPackUpperTriangle(taus{u});
        r_packed(u, :) = localPackUpperTriangle(correlations{u});
    end

    chunk = struct();
    chunk.d = d;
    chunk.chunk = c;
    chunk.unit_indices = first_unit + (0:m - 1);
    chunk.rows = [rows{:}];
    chunk.tau_packed = tau_packed;
    chunk.R_packed = r_packed;
    chunk.seconds = toc(chunk_timer);
end

function name = localDimensionFileName(d)
    % The per-dimension file name, spelled in one place so the merged index cannot name a file the dimension loop does not write.
    name = sprintf('fit_d%02d.mat', d);
end

function grid = localUnitsForDimension(n_channels, d, settings, seed_base)
    % The channel sets of one dimension: every pair at d = 2 (the first max_pairs in nchoosek order when capped), a seeded group sample above.
    % A dimension wider than the retained channel count has no channel set.
    if d > n_channels
        grid = zeros(0, d);
        return
    end
    if d == 2
        grid = selectAllPairs(n_channels);
        if ~isempty(settings.max_pairs)
            grid = grid(1:min(settings.max_pairs, size(grid, 1)), :);
        end
        return
    end
    grid = sampleChannelGroups(n_channels, d, settings.groups_per_d, ...
                               channelGroupSeed(seed_base, d));
end

function packed = localPackUpperTriangle(M)
    % The d(d-1)/2 strictly upper-triangular entries in column-major order, NaN-padded to the 190 a d = 20 channel set needs, so every dimension stacks into one [n_units x 190] array.
    values = M(triu(true(size(M, 1)), 1));
    packed = nan(1, 190);
    packed(1:numel(values)) = reshape(values, 1, []);
end

function localWriteUnitCsv(rows, path)
    % A unit CSV, or a printed note when the run selected no unit of that kind.
    if isempty(rows)
        fprintf('[Phase2] no unit for %s; not written\n', path);
        return
    end
    writeUnitRowsCsv(rows, path);
end

function T = localTimingTable(dims, timing, n_workers)
    % The cost of each dimension (units, wall time, median fit time and workers), from which run-time estimates for larger runs are built.
    T = table(dims(:), timing(:, 1), timing(:, 2), timing(:, 3), ...
              repmat(n_workers, numel(dims), 1), ...
              'VariableNames', {'d', 'n_units', 'wall_seconds', 'median_fit_seconds', ...
                          'workers'});
end

function localSaveMatAtomic(path, payload)
    % Save every field of payload as its own variable in a -v7.3 MAT file, through a temporary file and a rename, so an interrupted save cannot leave a partial file that a resume would trust.
    % The staging name carries a per-call token, as in saveStructAtomic, so two writers of one destination never delete or overwrite each other's half-written file.
    % saveStructAtomic itself is not used because it stores one variable, and these files are read field by field.
    [~, token] = fileparts(tempname);
    tmp = [path, '.', token, '.tmp.mat'];
    try
        save(tmp, '-struct', 'payload', '-v7.3');
    catch save_error
        localDeleteFiles({tmp});
        rethrow(save_error);
    end
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'Phase2:SaveFailed', ...
           'Could not move %s into place: %s', tmp, msg);
end

function localReportFigureStatus(status, dataset_key)
    % Report a figure-render failure; never fail the phase on one.
    if ~(isstruct(status) && isfield(status, 'ok') && status.ok)
        fprintf(2, ['[Phase2] %s figure renderer did not complete; the mats and ' ...
                    'the CSVs are unaffected.\n'], dataset_key);
        return
    end
    if ~(isfield(status, 'n_figures') && status.n_figures > 0)
        fprintf(2, ['[Phase2] %s renderer completed without producing a figure; ' ...
                    'the data are unaffected.\n'], dataset_key);
    end
end
