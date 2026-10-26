%PHASE3_GAUSSIAN_SURROGATES Phase 3: the Gaussian-copula surrogate reference, as a pooled comparison per dimension.
%
% The Gaussian copula is the p = 0 limit of the Student-t family, not an interior point, so a fitted p_hat > 0 is not by itself evidence of a non-Gaussian copula.
% About half the fits of Gaussian data land exactly on p = 0, and the rest return a finite value.
% This phase builds the reference the observed fits are read against.
% It draws a coherent multivariate phase-randomised surrogate of the recording's rank-normal scores: one shared random phase per frequency across all channels, applied independently within each declared trial segment.
% The surrogate keeps every auto- and cross-spectrum of the scores and has a Gaussian copula.
% It is re-ranked, and every Phase 2 channel set is refitted on it end to end, with its own Kendall matrix, eigenvalue-floor repair and profile fit, so the reference carries every bias of the estimator.
% The reference is therefore "a stationary process with a Gaussian copula and the same auto- and cross-spectra", and it cannot separate a non-Gaussian copula from non-stationarity.
%
% The comparison is descriptive: it has no p-value and no per-channel-set exceedance test.
% Realization b = 1 is the reported reference and realization b = 2 a yardstick.
% Per dimension, the pooled distribution of p_hat = 1/nu_hat over the channel sets is compared between the observed fits and realization 1 by the overlap coefficient OVL (distributionOverlap, mass outside = 1 - OVL), with p = 0 and p = 3 as atoms of their own and equal-count bins of the pooled values between them.
% Two yardsticks show how far OVL falls below 1 for a distribution that is a property of the recording rather than of the random draw: OVL between the two realizations, and OVL between two seeded disjoint halves of the observed channel sets (summarizeSurrogateComparison).
% The information differences the reference itself produces are the floors Phase 4 joins (realization 1, 95th percentile).
%
% Per realization b:
%   coherentPhaseSurrogate(Z, segment boundaries, its own RandStream)
%     -> tiedrank / (n+1)
%     -> one full-recording Kendall matrix (kendallTauMatrixParallel), checkpointed
%     -> per dimension, in Phase 2's unit order, chunks of chunk_size units:
%        unitCorrelationFromTau on the tau submatrix, then tFitUnit.
%
% Resume and sharding.
% Every chunk is checkpointed (mat/checkpoints/surrogate_bB_dDD_chunkCCC.mat).
% A finished dimension is merged into mat/surrogate_bB_dDD.mat, the full record with the profiles (41 nodes by default) like the Phase 2 fit_dDD.mat, and its chunk files are deleted.
% A realization whose every dimension is merged is skipped without generating its surrogate.
% A shard TC_SHARD=k/N owns the realizations with mod(b-1, N) == k-1.
% A single-shard run (N = 1) computes every realization and then publishes.
% A sharded run never publishes, because several shards finishing together would write the same files at the same moment; the TC_FINALIZE=1 job that scripts/hpc/submit.sh queues after the shard array owns that step.
% A TC_FINALIZE=1 run computes nothing: it needs every realization of every dimension on disk and refuses otherwise.
%
% Two identities, because the fits and their summary cost very different amounts of computation:
%   fit identity      - the hash of what defines the fits (the seed grid, the fit options and the Phase 2 identity of the channel sets).
%                       It is stamped into every Kendall checkpoint, chunk and per-dimension file, and a file carrying another one is refused with an error, because these files hold hours of computation and only TC_OVERWRITE=1 recomputes them.
%   settings identity - the fit identity plus the settings of the comparison (realizations, overlap bins, quantile levels, halves seed offset).
%                       It is stamped into mat/surrogates_all.mat and run_config.json, and Phase 4 folds it into its own identity.
%                       A publication under another settings identity is withdrawn by the next publishing run and merged again from the realization files in minutes, without refitting.
% Chunk and per-dimension files also record the seed of their realization and their unit indices, and both are checked on every reuse.
%
% Publication.
% The order is the realization files, surrogates_all.mat, the CSVs, the figures, then run_config.json, written by the publishing run alone.
% A recording counts as finished only when the merged MAT, its identity, every table it lists (localPublishedTables) and run_config.json are all present.
% A publishing run that finds the MAT but not the rest republishes the tables from it without refitting.
% TC_OVERWRITE=1 withdraws the existing publication in every run, the finalize job included, and in a computing run also the files of the realizations that run owns, never those of another shard, which may still be running.
%
% Reads : the Phase 1 mat (X_filtered, segment_boundaries) and the Phase 2 fit_all.mat of every recording discovered under the results root.
% Writes: <rel_base>/Phase3/{mat,text,images}:
%         mat/checkpoints/surrogate_bB_kendall.mat, surrogate_bB_dDD_chunkCCC.mat
%         mat/surrogate_bB_dDD.mat, mat/surrogates_all.mat
%         text/surrogate_comparison_by_d.csv, surrogate_floor_by_d.csv, surrogate_units.csv, timing_shard_K.csv, timing_realizations.csv, run_config.json, log.txt
%
% Environment: TC_RESULTS_ROOT, TC_RECORDINGS, TC_RECORDING_INDEX, TC_SHARD, TC_WORKERS, TC_OVERWRITE, TC_FINALIZE, TC_RENDER_FIGURES.
% There is no pilot override: the reference refits exactly the channel sets Phase 2 fitted, so a Phase 2 pilot (TC_GROUPS_PER_D, TC_MAX_PAIRS) is a Phase 3 pilot.
% Launch MATLAB with SHELL=/bin/bash, or the figure step cannot shell out.

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
% seed_grid is the declared size of the surrogate seed namespace, ten recordings by one thousand realizations.
% studySeed maps (seed_base, b) through these radices, so a change in the number of realizations never moves a seed.
%
% realizations is 2 by design: b = 1 is the reported reference and b = 2 the yardstick, and summarizeSurrogateComparison reads exactly those two.
%
% The halves yardstick splits each dimension's channel sets with RandStream('twister', 'Seed', channelGroupSeed(seed_base, d) + halves_seed_offset).
% The split therefore comes from the recording's own identity and stays apart from the Phase 2 channel-set sample, which draws on channelGroupSeed(seed_base, d) itself.
settings = struct( ...
    'realizations',       2, ...           % b = 1 reported, b = 2 the yardstick
    'seed_grid',          [10, 1000], ...  % declared (recordings, realizations) radices
    'fit_opts',           struct(), ...    % overrides of tFitDefaultOptions (profiles kept)
    'chunk_size',         1000, ...        % units per checkpointed parfor call
    'overlap_bins',       20, ...          % equal-count bins of the pooled non-atom p_hat
    'quantile_levels',    [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5], ... % per cent
    'halves_seed_offset', 2, ...           % halves stream: channelGroupSeed(seed_base, d) + 2
    'shard',              getEnvShard("TC_SHARD"), ...
    'workers',            getEnvInteger("TC_WORKERS", [], [0, Inf]));

steps = struct( ...
    'finalize_only',  getEnvFlag("TC_FINALIZE", false), ...
    'render_figures', getEnvFlag("TC_RENDER_FIGURES", true), ...
    'overwrite',      getEnvFlag("TC_OVERWRITE", false));

% The fit domain and the coarse profile grid are the fitter's, materialised here as in Phase 2, so the quantile cache tabulates exactly the coarse-profile nodes fitTCopulaShape evaluates.
% The domain endpoints are also the two atoms of the overlap, where fits land exactly: p = 0, the Gaussian copula, and p = 3, the heavy end.
fit_defaults = tFitDefaultOptions(settings.fit_opts);
settings.profile_grid = linspace(fit_defaults.p_bounds(1), fit_defaults.p_bounds(2), ...
    fit_defaults.profile_grid_size);
settings.overlap_atoms = fit_defaults.p_bounds;

results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));
selected_keys = getEnvList("TC_RECORDINGS");
selected_index = getEnvInteger("TC_RECORDING_INDEX", [], [1, Inf]);

% The two identities described in the header.
% The shard, the worker count, chunk_size and the step switches are in neither, because they schedule the work and change no number.
% localIdentity adds the per-recording part of the fit identity, the identity of the Phase 2 fits whose channel sets are refitted.
fit_identity_fields = struct( ...
    'phase',     'Phase3', ...
    'seed_grid', settings.seed_grid, ...
    'fit_opts',  tFitDefaultOptions(settings.fit_opts));
comparison_fields = struct( ...
    'realizations',       settings.realizations, ...
    'overlap_bins',       settings.overlap_bins, ...
    'quantile_levels',    settings.quantile_levels, ...
    'halves_seed_offset', settings.halves_seed_offset);

%% ============================================================ run
assert(isempty(selected_keys) || isempty(selected_index), 'Phase3:AmbiguousSelection', ...
    ['TC_RECORDINGS and TC_RECORDING_INDEX are mutually exclusive; got keys ' ...
    '"%s" and index %d.'], strjoin(selected_keys, ', '), selected_index);
recordings = discoverPhase1Recordings(results_root, selected_keys);
if ~isempty(selected_index)
    assert(selected_index <= numel(recordings), 'Phase3:RecordingIndexOutOfRange', ...
        'TC_RECORDING_INDEX is %d but only %d recording(s) were discovered under %s.', ...
        selected_index, numel(recordings), results_root);
    recordings = recordings(selected_index);
end
run_mode = 'compute';
if steps.finalize_only
    run_mode = 'finalize only';
end
fprintf('[Phase3] %d recording(s) | shard %d/%d | %s | %d realization(s) | results %s\n', ...
    numel(recordings), settings.shard(1), settings.shard(2), run_mode, ...
    settings.realizations, results_root);

% The first realization that still has work opens the pool, so a finished phase, such as the last link of a resume chain or a rerun, exits in seconds.
n_workers = [];
failures = strings(0, 1);
for i = 1:numel(recordings)
    rec = recordings(i);
    try
        n_workers = localRunRecording(rec, settings, steps, results_root, n_workers, ...
            fit_identity_fields, comparison_fields);
    catch recording_error
        failures(end + 1, 1) = string(rec.dataset_key) + " [" + ...
            string(recording_error.identifier) + "] " + ...
            string(recording_error.message); %#ok<SAGROW>
        fprintf(2, '\n[Phase3] %s failed (%s)\n%s\n', rec.dataset_key, ...
            recording_error.identifier, ...
            getReport(recording_error, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\n[Phase3] finished: %d of %d recording(s) failed\n', ...
    numel(failures), numel(recordings));
if ~isempty(failures)
    error('Phase3:RecordingFailures', ...
        'Phase 3 failed for %d recording(s):\n%s', numel(failures), ...
        strjoin(failures, newline));
end


% ===========================================================================
% Local functions
% ===========================================================================
function n_workers = localRunRecording(rec, settings, steps, results_root, n_workers, ...
        fit_identity_fields, comparison_fields)
    % One recording: skip it when published, compute the realizations this run owns, and publish when every realization is on disk and this run owns the publication.
    %
    % Only a publishing run touches the publication, except that TC_OVERWRITE=1 withdraws it in every run.
    % A run publishes when it is the single shard of its run (N = 1) or the TC_FINALIZE=1 job; a shard of a sharded run computes its realizations and never writes the merged MAT, a table or a stamp.
    probe = openPhaseOutput(rec.rel_base, 3, results_root, 'Diary', false);
    merged_path = fullfile(probe.mat_dir, 'surrogates_all.mat');
    fit_all_path = fullfile(fileparts(probe.base), 'Phase2', 'mat', 'fit_all.mat');
    [fit_identity, identity, phase2_identity] = localIdentity(fit_identity_fields, ...
        comparison_fields, fit_all_path, rec);
    publishing = steps.finalize_only || settings.shard(2) == 1;

    stale_publication = false;
    if ~steps.overwrite && isfile(merged_path)
        if strcmp(localStoredIdentity(merged_path), identity)
            if localIsPublished(merged_path, probe.text_dir)
                fprintf('[Phase3] %s: already done, skipping (%s)\n', ...
                    rec.dataset_key, merged_path);
                return;
            end
            if publishing
                % Merged but not wholly published, after a run cut off between the merged MAT and its tables, or a table lost afterwards.
                % The tables are regenerated from the merged MAT, and nothing is refitted.
                ctx = openPhaseOutput(rec.rel_base, 3, results_root);
                started_utc = datetime('now', 'TimeZone', 'UTC');
                fprintf(['[Phase3] %s: the reference is merged but its publication ' ...
                    'is incomplete; regenerating the tables and figures from ' ...
                    '%s WITHOUT REFITTING\n'], rec.dataset_key, merged_path);
                localPublish(ctx, load(merged_path), settings, steps, identity, ...
                    localWorkerCount(n_workers), started_utc);
                return;
            end
        elseif publishing
            stale_publication = true;
        end
    end

    ctx = openPhaseOutput(rec.rel_base, 3, results_root);
    started_utc = datetime('now', 'TimeZone', 'UTC');
    owned = [];
    if ~steps.finalize_only
        owned = find(mod((1:settings.realizations) - 1, settings.shard(2)) ...
            == settings.shard(1) - 1);
    end
    fprintf('\n========================================================\n');
    fprintf('Phase 3 recording: %s (seed_base %d)\n', rec.dataset_key, rec.seed_base);
    fprintf('  output: %s\n', ctx.base);
    fprintf('  fit identity: %s\n', fit_identity);
    fprintf('  identity: %s\n', identity);
    fprintf('  realizations computed here: %s\n', mat2str(owned));
    fprintf('========================================================\n');
    if steps.overwrite
        localWithdrawPublication(ctx);
        localWithdrawRealizations(ctx, owned);
    elseif stale_publication
        fprintf(['[Phase3] %s was published under the identity %s, but the settings ' ...
            'now give %s; it is withdrawn and merged again from the ' ...
            'realization files\n'], merged_path, localStoredIdentity(merged_path), ...
            identity);
        localWithdrawPublication(ctx);
    end

    phase2 = load(fit_all_path, 'units', 'fit_opts');
    units = phase2.units;
    % The surrogate distribution is a reference for the observed one only if both come from the same estimator, so the materialised Phase 2 fit options must equal the ones every surrogate unit is refitted with.
    assert(isfield(phase2, 'fit_opts') && ...
        isequaln(phase2.fit_opts, tFitDefaultOptions(settings.fit_opts)), ...
        'Phase3:FitOptionsMismatch', ...
        ['%s was fitted with other fit options than Phase 3 refits its surrogates ' ...
        'with (settings.fit_opts); the observed and surrogate distributions would ' ...
        'not come from the same estimator: %s'], rec.dataset_key, fit_all_path);
    dims = reshape(unique(units.d), 1, []);
    signal = [];
    for b = owned
        [n_workers, signal] = localRunRealization(ctx, rec, b, units, dims, ...
            settings, steps, fit_identity, n_workers, signal);
    end

    present = localRealizationFilesPresent(ctx, settings.realizations, dims);
    if steps.finalize_only
        [b_missing, d_missing] = find(~present, 1);
        assert(all(present, 'all'), 'Phase3:IncompleteReference', ...
            ['TC_FINALIZE=1 merges existing realizations, but %d of %d ' ...
            'realization-dimension files are missing under %s (first missing: ' ...
            'b = %d, d = %d).'], nnz(~present), numel(present), ctx.mat_dir, ...
            b_missing, dims(d_missing));
    end
    if all(present, 'all') && publishing
        localFinalize(ctx, rec, units, dims, settings, steps, merged_path, ...
            fit_identity, identity, phase2_identity, localWorkerCount(n_workers), ...
            started_utc);
    elseif all(present, 'all')
        fprintf(['[Phase3] %s: every realization is on disk; the merge belongs to ' ...
            'the TC_FINALIZE=1 job, not to shard %d of %d\n'], ...
            rec.dataset_key, settings.shard(1), settings.shard(2));
    else
        fprintf(['[Phase3] %s: %d of %d realization-dimension files on disk, ' ...
            'finalize deferred\n'], rec.dataset_key, nnz(present), numel(present));
    end
end


function [fit_identity, identity, phase2_identity] = localIdentity(fit_fields, ...
        comparison_fields, fit_all_path, rec)
    % The two identities of one recording's reference.
    % The fit identity carries the identity of the Phase 2 fits whose channel sets are refitted, so refitting the observed units under other options or other channel sets invalidates every realization drawn against them.
    % The Phase 2 identity is returned too, because the merged MAT records it in the clear, and that is how Phase 4 checks that a reference belongs to the fits it joins it to (a hash cannot be read back).
    assert(isfile(fit_all_path), 'Phase3:MissingPhase2Fit', ...
        ['Phase 3 needs the Phase 2 unit rows of %s, which are not at %s. ' ...
        'Run Phase 2 for this recording first.'], rec.dataset_key, fit_all_path);
    phase2_identity = localStoredIdentity(fit_all_path);
    fit_fields.phase2_identity = phase2_identity;
    fit_identity = settingsIdentity(fit_fields);
    comparison_fields.fit_identity = fit_identity;
    identity = settingsIdentity(comparison_fields);
end


function n = localWorkerCount(n_workers)
    % The worker count for the provenance stamp: 0 when this run opened no pool.
    n = 0;
    if ~isempty(n_workers)
        n = n_workers;
    end
end


function n_workers = localOpenPool(requested_workers)
    % TC_WORKERS = 0 runs serially, and an empty value takes the default size of MATLAB's local pool.
    % The pool size only schedules the work: every unit is independent, and no result depends on the worker count.
    if isequal(requested_workers, 0)
        n_workers = openParallelPool(false);
    else
        n_workers = openParallelPool(true, requested_workers);
    end
    fprintf('[Phase3] parallel pool: %d worker(s)\n', n_workers);
end


function done = localIsPublished(merged_path, text_dir)
    % A reference is published when its merged MAT lists the tables its publication consists of, every one of them exists, and the provenance stamp, written last, exists.
    % A merged MAT that lists no table is not taken as published.
    tables = localStoredVariable(merged_path, 'published_tables', {});
    done = ~isempty(tables) && isfile(fullfile(text_dir, 'run_config.json')) ...
        && all(cellfun(@(name) isfile(fullfile(text_dir, name)), tables));
end


function tables = localPublishedTables()
    % The tables one recording's reference publishes, declared here and nowhere else: surrogates_all.mat records the list, localPublish writes exactly those, and the recording counts as finished only when every one exists.
    tables = {'surrogate_comparison_by_d.csv', 'surrogate_floor_by_d.csv', ...
        'surrogate_units.csv', 'timing_realizations.csv'};
end


function localWithdrawPublication(ctx)
    % Withdraw the recording's publication: the merged reference, every table and both stamps.
    % It also deletes files this phase does not write but that may sit in the same folders (the four tables listed next, subset.mat and surrogate_long_d*.csv), so none of them can be read as part of this publication.
    % The realization files stay, because the fit identity governs them (localWithdrawRealizations).
    retired_tables = {'surrogate_unit_summary.csv', 'surrogate_summary_by_d.csv', ...
        'subset_units.csv', 'timing_replicates.csv'};
    stale = [ ...
        fullfile(ctx.mat_dir, {'surrogates_all.mat', 'subset.mat'}), ...
        fullfile(ctx.text_dir, [localPublishedTables(), retired_tables, ...
        {'run_config.json', 'figures_manifest.json'}]), ...
        localListFiles(ctx.text_dir, 'surrogate_long_d*.csv'), ...
        {fullfile(ctx.images_dir, 'figures_manifest.json')}];
    localDeleteFiles(stale);
    fprintf('[Phase3] the previous publication of %s was withdrawn\n', ctx.base);
end


function localWithdrawRealizations(ctx, owned)
    % TC_OVERWRITE=1 recomputes the realizations this run owns from scratch, so their Kendall checkpoint, chunks and per-dimension files are deleted first, and an interrupted overwrite can only be resumed as this computation.
    % Another shard's realizations are never touched, because that shard may be computing them at the same time.
    for b = reshape(owned, 1, [])
        localDeleteFiles([ ...
            localListFiles(ctx.mat_dir, sprintf('surrogate_b%d_d*.mat', b)), ...
            localListFiles(fullfile(ctx.mat_dir, 'checkpoints'), ...
            sprintf('surrogate_b%d_*.mat', b))]);
        fprintf('[Phase3] TC_OVERWRITE=1: the files of realization %d were withdrawn\n', b);
    end
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
    % Refuse an artifact written under another configuration.
    % A realization file is keyed by (b, d, chunk) alone, so a file from other settings describes other fits under the same key and must never be merged.
    stored = localStoredIdentity(path);
    if strcmp(stored, identity)
        return;
    end
    if isempty(stored)
        stored = '<none recorded>';
    end
    error('Phase3:SettingsIdentityMismatch', ...
        ['%s was written under the identity %s, but this run''s fit identity is ' ...
        '%s. Rerun with TC_OVERWRITE=1 to recompute it, or restore the ' ...
        'settings that produced it.'], path, stored, identity);
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


function seed = localSeed(rec, b, settings)
    % The seed of realization b of this recording, from the declared seed grid.
    assert(rec.seed_base >= 1 && rec.seed_base <= settings.seed_grid(1), ...
        'Phase3:SeedBaseOutsideGrid', ...
        ['Recording %s has seed_base %d, outside the declared seed grid of %d ' ...
        'recordings (settings.seed_grid(1)); raise the declared size before ' ...
        'adding recordings.'], rec.dataset_key, rec.seed_base, settings.seed_grid(1));
    seed = studySeed('surrogate', [rec.seed_base, b], settings.seed_grid);
end


function path = localDimensionPath(ctx, b, d)
    % The per-dimension file of one realization, named in one place.
    path = fullfile(ctx.mat_dir, sprintf('surrogate_b%d_d%02d.mat', b, d));
end


function present = localRealizationFilesPresent(ctx, realizations, dims)
    % [realizations x numel(dims)]: which per-dimension files are on disk.
    present = false(realizations, numel(dims));
    for b = 1:realizations
        for k = 1:numel(dims)
            present(b, k) = isfile(localDimensionPath(ctx, b, dims(k)));
        end
    end
end


function [n_workers, signal] = localRunRealization(ctx, rec, b, units, dims, settings, ...
        steps, fit_identity, n_workers, signal)
    % Realization b: every dimension whose file is not yet on disk, chunk by chunk.
    % The surrogate, its pseudo-observations and its Kendall matrix are built only when a dimension is left to fit, and the recording's rank-normal scores and quantile cache (signal) are built once for all its realizations.
    seed = localSeed(rec, b, settings);
    todo = false(1, numel(dims));
    for k = 1:numel(dims)
        path = localDimensionPath(ctx, b, dims(k));
        if isfile(path)
            localAssertDimensionFile(path, fit_identity, seed, units, dims(k));
        else
            todo(k) = true;
        end
    end
    if ~any(todo)
        fprintf('[Phase3] b = %d: every dimension is merged, skipping\n', b);
        return;
    end
    if ~all(todo)
        fprintf('[Phase3] b = %d: reusing the merged dimension(s) d = %s\n', b, ...
            mat2str(dims(~todo)));
    end

    if isempty(signal)
        [signal.Z, signal.boundaries] = localRankNormalScores(rec);
        cache_timer = tic;
        signal.cache = tCopulaQuantileCache(size(signal.Z, 1), settings.profile_grid);
        fprintf(['[Phase3] %s: %d samples x %d channels, %d segment(s); quantile ' ...
            'cache of %d tail probabilities x %d nodes in %.1f s\n'], ...
            rec.dataset_key, size(signal.Z, 1), size(signal.Z, 2), ...
            numel(signal.boundaries), signal.cache.num_distinct, ...
            numel(settings.profile_grid), toc(cache_timer));
    end
    if isempty(n_workers)
        n_workers = localOpenPool(settings.workers);
    end

    realization_timer = tic;
    Xs = coherentPhaseSurrogate(signal.Z, signal.boundaries, ...
        RandStream('twister', 'Seed', seed));
    n = size(Xs, 1);
    Us = tiedrank(Xs) ./ (n + 1);
    tau_full = localKendall(ctx, b, seed, Xs, fit_identity, n_workers);
    clear Xs
    fprintf('[Phase3] b = %d (seed %d): surrogate, ranks and Kendall matrix in %.1f s\n', ...
        b, seed, toc(realization_timer));

    for k = find(todo)
        localFitDimension(ctx, rec, b, dims(k), seed, units, Us, tau_full, settings, ...
            steps, fit_identity, signal.cache, n_workers);
    end
    fprintf('[Phase3] b = %d: done in %.1f s\n', b, toc(realization_timer));
end


function tau_full = localKendall(ctx, b, seed, Xs, fit_identity, n_workers)
    % The realization's full Kendall matrix, checkpointed, because at production size it takes minutes of pool time and every resume of the realization needs it again.
    checkpoint_dir = fullfile(ctx.mat_dir, 'checkpoints');
    if ~isfolder(checkpoint_dir)
        mkdir(checkpoint_dir);
    end
    path = fullfile(checkpoint_dir, sprintf('surrogate_b%d_kendall.mat', b));
    if isfile(path)
        localAssertIdentity(path, fit_identity);
        stored = load(path, 'tau_full', 'seed');
        assert(isequal(stored.seed, seed) && isequal(size(stored.tau_full), ...
            [size(Xs, 2), size(Xs, 2)]), 'Phase3:CheckpointSeedMismatch', ...
            ['%s holds the %d x %d Kendall matrix of seed %d, but realization %d ' ...
            'of this recording has seed %d and %d channels. Delete it and rerun.'], ...
            path, size(stored.tau_full, 1), size(stored.tau_full, 2), stored.seed, ...
            b, seed, size(Xs, 2));
        tau_full = stored.tau_full;
        fprintf('[Phase3] b = %d: Kendall matrix reused (%s)\n', b, path);
        return;
    end
    kendall_timer = tic;
    tau_full = kendallTauMatrixParallel(Xs, n_workers);
    kendall_seconds = toc(kendall_timer);
    localSaveFields(path, struct('tau_full', tau_full, 'seed', seed, 'b', b, ...
        'kendall_seconds', kendall_seconds, 'n_workers', n_workers, ...
        'settings_identity', fit_identity));
    fprintf('[Phase3] b = %d: Kendall matrix of %d channels in %.1f s\n', ...
        b, size(tau_full, 1), kendall_seconds);
end


function localAssertDimensionFile(path, fit_identity, seed, units, d)
    % A per-dimension file is reused only when it is this recording's realization of this dimension: same fit identity, same seed, and the Phase 2 channel sets of d in Phase 2's order.
    localAssertIdentity(path, fit_identity);
    stored = load(path, 'seed', 'unit_indices');
    at_d = units.d == d;
    assert(isequal(stored.seed, seed) && ...
        isequal(stored.unit_indices(:), units.unit_index(at_d)), ...
        'Phase3:RealizationUnitMismatch', ...
        ['%s holds seed %d and %d unit(s), but this realization has seed %d and ' ...
        'd = %d has %d Phase 2 unit(s) with other indices. Delete it and rerun, ' ...
        'or rerun with TC_OVERWRITE=1.'], path, stored.seed, ...
        numel(stored.unit_indices), seed, d, nnz(at_d));
end


function localFitDimension(ctx, rec, b, d, seed, units, Us, tau_full, settings, ...
        steps, fit_identity, cache, n_workers)
    % Every channel set of one dimension on realization b, checkpointed chunk by chunk and merged into the per-dimension file, whose chunk files are then deleted.
    at_d = find(units.d == d);
    unit_indices = units.unit_index(at_d);
    channels = units.channels(at_d, :);
    n_units = numel(at_d);
    n_chunks = ceil(n_units / settings.chunk_size);
    fit_opts = tFitDefaultOptions(settings.fit_opts);
    checkpoint_dir = fullfile(ctx.mat_dir, 'checkpoints');
    timing_path = fullfile(ctx.text_dir, sprintf('timing_shard_%d.csv', settings.shard(1)));

    chunks = cell(1, n_chunks);
    dimension_timer = tic;
    for c = 1:n_chunks
        first = (c - 1) * settings.chunk_size + 1;
        last = min(c * settings.chunk_size, n_units);
        chunk_path = localChunkPath(checkpoint_dir, b, d, c);
        if ~steps.overwrite && isfile(chunk_path)
            % A chunk is keyed by (b, d, ordinal) only, so its seed and unit indices are checked as well as its identity: a chunk_size change moves the indices without changing the science.
            localAssertIdentity(chunk_path, fit_identity);
            stored = load(chunk_path, 'chunk');
            if ~isequal(stored.chunk.seed, seed) || ...
                    ~isequal(stored.chunk.unit_indices(:), unit_indices(first:last))
                error('Phase3:CheckpointGridMismatch', ...
                    ['%s was written for another seed or unit grid than this run ' ...
                    '(b = %d, seed %d, d = %d, units %d to %d). Delete ' ...
                    'mat/checkpoints and rerun, or restore the settings that ' ...
                    'produced it.'], chunk_path, b, seed, d, first, last);
            end
            chunks{c} = stored.chunk;
            fprintf('[Phase3] b = %d d = %2d chunk %3d/%3d: reused %d unit(s)\n', ...
                b, d, c, n_chunks, numel(stored.chunk.unit_indices));
            continue;
        end
        chunks{c} = localFitChunk(b, d, c, seed, channels(first:last, :), ...
            unit_indices(first:last), Us, tau_full, rec, fit_opts, cache, n_workers);
        % The identity is a separate top-level variable, so the reuse check above reads it without loading the chunk's unit rows.
        localSaveFields(chunk_path, struct('chunk', chunks{c}, ...
            'settings_identity', fit_identity));
        localAppendTiming(timing_path, chunks{c});
        fprintf(['[Phase3] b = %d d = %2d chunk %3d/%3d: %d unit(s) in %.1f s ' ...
            '(%.0f%% of the dimension, %.0f s elapsed)\n'], b, d, c, n_chunks, ...
            last - first + 1, chunks{c}.seconds, 100 * last / n_units, ...
            toc(dimension_timer));
    end

    % Each chunk codes its text fields against its own label lists, so the merge goes through the rows before it becomes one column struct again.
    rows = cellfun(@(chunk) unitColumnsToRows(chunk.rows), chunks, 'UniformOutput', false);
    rows = [rows{:}];
    payload = struct();
    payload.units = unitRowsToColumns(rows);
    payload.settings = settings;
    payload.settings_identity = fit_identity;
    payload.fit_opts = fit_opts;
    payload.seed = seed;
    payload.b = b;
    payload.d = d;
    payload.unit_indices = unit_indices;   % top level, so a reuse check skips the rows
    payload.seconds = toc(dimension_timer);
    payload.n = size(Us, 1);
    payload.n_channels = size(Us, 2);
    payload.dataset_key = rec.dataset_key;
    localSaveFields(localDimensionPath(ctx, b, d), payload);
    for c = 1:n_chunks
        localDeleteFiles({localChunkPath(checkpoint_dir, b, d, c)});
    end
    fprintf('[Phase3] b = %d d = %2d: %d unit(s) in %.1f s, median p_hat %.4f\n', ...
        b, d, n_units, payload.seconds, median(payload.units.p_hat, 'omitnan'));
end


function chunk = localFitChunk(b, d, c, seed, channel_block, unit_indices, Us, ...
        tau_full, rec, fit_opts, cache, n_workers)
    % One parfor call: the unit rows of one chunk of channel sets on one realization.
    %
    % debt: Us and the quantile cache are broadcast to the workers on every chunk, a transfer that grows with the recording's length and channel count and is accepted, as in Phase 2, while a realization has at most a few hundred chunks.
    % The upgrade path is a parallel.pool.Constant hoisted to the realization, which sends them once per pool instead of once per chunk.
    m = size(channel_block, 1);
    rows = cell(1, m);
    dataset_key = char(rec.dataset_key);
    chunk_timer = tic;
    parfor (u = 1:m, n_workers)
        % Every unit reads the realization's pseudo-observations and Kendall matrix, and there is nothing to slice, because a unit's channel set is an arbitrary column subset.
        padded = channel_block(u, :);
        grp = padded(isfinite(padded));
        [R, corr] = unitCorrelationFromTau(tau_full(grp, grp), fit_opts); %#ok<PFBNS>
        row = tFitUnit(Us(:, grp), R, corr, fit_opts, cache); %#ok<PFBNS>
        row.dataset_key = dataset_key;
        row.unit_index = unit_indices(u);
        row.channels = padded;
        rows{u} = row;
    end

    chunk = struct();
    chunk.b = b;
    chunk.d = d;
    chunk.chunk = c;
    chunk.seed = seed;
    chunk.unit_indices = reshape(unit_indices, 1, []);
    chunk.rows = unitRowsToColumns([rows{:}]);
    chunk.seconds = toc(chunk_timer);
    chunk.n_workers = n_workers;
end


function path = localChunkPath(checkpoint_dir, b, d, c)
    % The checkpoint file of chunk c of dimension d on realization b.
    path = fullfile(checkpoint_dir, sprintf('surrogate_b%d_d%02d_chunk%03d.mat', b, d, c));
end


function [Z, boundaries] = localRankNormalScores(rec)
    % The surrogate's input: the per-column rank-normal scores of the cleaned matrix, with the trial segmentation Phase 1 recorded (a continuous recording is one segment ending at the last sample).
    variables = who('-file', rec.phase1_mat_abs);
    wanted = {'X_filtered'};
    if ismember('segment_boundaries', variables)
        wanted{end + 1} = 'segment_boundaries';
    end
    phase1 = load(rec.phase1_mat_abs, wanted{:});
    X = double(phase1.X_filtered);

    required = {'n_samples', 'n_channels_retained'};
    missing = required(~isfield(rec, required));
    assert(isempty(missing), 'Phase3:MissingRecordingMetadata', ...
        'recording.json of %s carries no %s field.', rec.dataset_key, ...
        strjoin(missing, ', '));
    assert(size(X, 1) == rec.n_samples && size(X, 2) == rec.n_channels_retained, ...
        'Phase3:SignalMatrixOrientation', ...
        ['X_filtered of %s must be [samples x channels] = [%d x %d] as ' ...
        'recording.json declares; got %d x %d. A transposed matrix does not ' ...
        'error, it silently yields a different correlation structure.'], ...
        rec.dataset_key, rec.n_samples, rec.n_channels_retained, ...
        size(X, 1), size(X, 2));
    assert(all(isfinite(X), 'all'), 'Phase3:NonFiniteSignal', ...
        ['X_filtered of %s must be finite: the Fourier transform of a ' ...
        'non-finite row contaminates every frequency of that column ' ...
        '(%d of %d entries are not finite).'], rec.dataset_key, ...
        nnz(~isfinite(X)), numel(X));

    Z = norminv(tiedrank(X) ./ (size(X, 1) + 1));
    if isfield(phase1, 'segment_boundaries')
        boundaries = reshape(double(phase1.segment_boundaries), 1, []);
    else
        boundaries = size(X, 1);
    end
    % Without this check, a trial recording whose boundaries are missing would be read as one continuous segment, and its phases would silently be randomised across trial ends.
    if isfield(rec, 'n_segments')
        assert(numel(boundaries) == rec.n_segments, 'Phase3:SegmentCountMismatch', ...
            ['%s declares %d segments in recording.json but %s yields %d segment ' ...
            'end(s).'], rec.dataset_key, rec.n_segments, rec.phase1_mat_abs, ...
            numel(boundaries));
    end
end


function localFinalize(ctx, rec, units, dims, settings, steps, merged_path, ...
        fit_identity, identity, phase2_identity, n_workers, started_utc)
    % Merge every realization's per-dimension files into [units x realizations] matrices aligned to the Phase 2 units, compare them with the observed fits, save the merged MAT, then publish the tables, the figures and the provenance stamp.
    %
    % Beside the matrices, the merged MAT records in the clear the Phase 2 identity the reference was built on (Phase 4 checks it before joining), the fit identity of the realization files, the seeds, the comparison and floor tables, and the tables its publication consists of.
    status_labels = {'ok', 'heavy_boundary', 'ambiguous_profile', ...
        'numerical_failure', 'degenerate_pair'};
    numeric_fields = {'p_hat', 'nu_hat', 'loglik_t', 'loglik_gaussian', 'logdet_R', ...
        'repair_shift_rms', 'fit_seconds'};
    numeric_names = {'p_hat_s', 'nu_hat_s', 'loglik_t_s', 'loglik_gaussian_s', ...
        'logdet_R_s', 'repair_shift_s', 'fit_seconds_s'};
    flag_fields = {'numerical_ok', 'gaussian_snapped', 'heavy_boundary', ...
        'ambiguous_profile'};
    R = settings.realizations;
    m = numel(units.d);

    merged = struct();
    for f = 1:numel(numeric_names)
        merged.(numeric_names{f}) = nan(m, R);
    end
    for f = 1:numel(flag_fields)
        merged.([flag_fields{f} '_s']) = false(m, R);
    end
    merged.status_s = zeros(m, R, 'int8');
    merged.status_labels = status_labels;
    seeds = zeros(1, R);

    merge_timer = tic;
    for b = 1:R
        seeds(b) = localSeed(rec, b, settings);
        for d = dims
            path = localDimensionPath(ctx, b, d);
            localAssertIdentity(path, fit_identity);
            stored = load(path, 'units', 'seed');
            rows_at = find(units.d == d);
            assert(isequal(stored.seed, seeds(b)), 'Phase3:CheckpointSeedMismatch', ...
                ['%s was fitted on the surrogate of seed %d, but realization %d of ' ...
                '%s has seed %d (seed_base %d, seed_grid %s). Rerun with ' ...
                'TC_OVERWRITE=1, or restore the seed settings that produced it.'], ...
                path, stored.seed, b, rec.dataset_key, seeds(b), rec.seed_base, ...
                mat2str(settings.seed_grid));
            assert(isequal(stored.units.unit_index(:), units.unit_index(rows_at)) && ...
                isequaln(stored.units.channels, units.channels(rows_at, :)), ...
                'Phase3:RealizationUnitMismatch', ...
                ['%s holds %d channel set(s) that are not the %d Phase 2 channel ' ...
                'sets of d = %d in Phase 2''s order. Rerun with TC_OVERWRITE=1.'], ...
                path, numel(stored.units.unit_index), numel(rows_at), d);
            for f = 1:numel(numeric_fields)
                merged.(numeric_names{f})(rows_at, b) = stored.units.(numeric_fields{f});
            end
            for f = 1:numel(flag_fields)
                merged.([flag_fields{f} '_s'])(rows_at, b) = ...
                    logical(stored.units.(flag_fields{f}));
            end
            % The file codes its statuses against its own sorted label list, so they are decoded before being recoded against the canonical one.
            decoded = string(stored.units.status_labels(double(stored.units.status)));
            [known, code] = ismember(decoded, string(status_labels));
            assert(all(known), 'Phase3:UnknownStatus', ...
                '%s carries the unknown unit status "%s".', path, ...
                strjoin(unique(reshape(decoded(~known), 1, [])), ', '));
            merged.status_s(rows_at, b) = int8(code);
        end
    end
    fprintf('[Phase3] merged %d realization(s) x %d dimension(s) in %.1f s\n', ...
        R, numel(dims), toc(merge_timer));

    [comparison, floors, half] = summarizeSurrogateComparison( ...
        struct('d', units.d, 'unit_index', units.unit_index, 'p_hat', units.p_hat, ...
        'numerical_ok', units.numerical_ok, 'heavy_boundary', units.heavy_boundary), ...
        struct('p_hat', merged.p_hat_s, 'numerical_ok', merged.numerical_ok_s, ...
        'heavy_boundary', merged.heavy_boundary_s, 'status', merged.status_s, ...
        'status_labels', {status_labels}), ...
        rec.seed_base, struct('overlap_bins', settings.overlap_bins, ...
        'atoms', settings.overlap_atoms, 'quantile_levels', settings.quantile_levels, ...
        'halves_seed_offset', settings.halves_seed_offset));

    % The observed side, in the column layout of unitRowsToColumns, aligned row for row with the realization matrices.
    merged.units = struct( ...
        'd', units.d, ...
        'unit_index', units.unit_index, ...
        'channels', units.channels, ...
        'p_hat_obs', units.p_hat, ...
        'nu_hat_obs', units.nu_hat, ...
        'numerical_ok_obs', double(logical(units.numerical_ok)), ...
        'heavy_boundary_obs', double(logical(units.heavy_boundary)), ...
        'status_obs', units.status, ...
        'status_obs_labels', {units.status_labels}, ...
        'half', half);
    merged.comparison = comparison;
    merged.floors = floors;
    merged.seeds = seeds;
    merged.settings = settings;
    merged.settings_identity = identity;
    merged.fit_identity = fit_identity;
    merged.phase2_identity = phase2_identity;
    merged.published_tables = localPublishedTables();
    localSaveFields(merged_path, merged);
    localPublish(ctx, merged, settings, steps, identity, n_workers, started_utc);
end


function localPublish(ctx, merged, settings, steps, identity, n_workers, started_utc)
    % Everything derived from the merged reference, in the one order that makes run_config.json a completion marker: the CSVs (localPublishedTables), then the figures, then the stamp.
    % Nothing here refits a unit, so a run that lost only its tables regenerates them from surrogates_all.mat at the cost of reading it.
    writeTableAtomic(merged.comparison, ...
        fullfile(ctx.text_dir, 'surrogate_comparison_by_d.csv'));
    writeTableAtomic(merged.floors, fullfile(ctx.text_dir, 'surrogate_floor_by_d.csv'));
    writeTableAtomic(localUnitTable(merged), fullfile(ctx.text_dir, 'surrogate_units.csv'));
    localMergeTiming(ctx);
    disp(merged.comparison(:, {'d', 'n_units', 'n_obs_ok', 'n_sur1_ok', 'n_sur2_ok', ...
        'ovl_obs_sur', 'mass_outside', 'ovl_sur_sur2', 'ovl_obs_halves', ...
        'obs_fraction_p0', 'sur_fraction_p0'}));

    if steps.render_figures
        renderPhaseFigures('render_phase3_figures', ctx.base, ctx.images_dir, ctx.text_dir);
    end

    run_settings = settings;
    run_settings.steps = steps;
    run_settings.settings_identity = identity;
    run_settings.fit_identity = merged.fit_identity;
    run_settings.phase2_identity = merged.phase2_identity;
    writeRunConfig(ctx, run_settings, struct('n_workers', n_workers, ...
        'started_utc', started_utc, ...
        'finished_utc', datetime('now', 'TimeZone', 'UTC')));
end


function T = localUnitTable(merged)
    % One row per channel set: the observed fit, its half, and its fit on every realization, which is what the distribution figures are drawn from.
    units = merged.units;
    m = numel(units.d);
    channels_str = cell(m, 1);
    for k = 1:m
        present = units.channels(k, isfinite(units.channels(k, :)));
        channels_str{k} = strtrim(sprintf('%d ', present));
    end
    T = table(units.d, units.unit_index, channels_str, units.half, units.p_hat_obs, ...
        units.numerical_ok_obs, 'VariableNames', {'d', 'unit_index', 'channels_str', ...
        'half', 'p_hat_obs', 'numerical_ok_obs'});
    for b = 1:size(merged.p_hat_s, 2)
        T.(sprintf('p_hat_s%d', b)) = merged.p_hat_s(:, b);
        T.(sprintf('numerical_ok_s%d', b)) = double(merged.numerical_ok_s(:, b));
        T.(sprintf('status_s%d', b)) = reshape( ...
            merged.status_labels(double(merged.status_s(:, b))), [], 1);
    end
end


function localAppendTiming(path, chunk)
    % One line per chunk, appended as the chunk finishes, so a job stopped at its time limit still leaves its timings.
    % There is one file per shard, because the shards of one recording run as separate jobs and must not interleave in one file.
    new_file = ~isfile(path);
    fid = fopen(path, 'a');
    assert(fid ~= -1, 'Phase3:TimingOpenFailed', 'Could not open %s for appending.', path);
    closer = onCleanup(@() fclose(fid));
    if new_file
        fprintf(fid, 'b,d,chunk,units,seconds,workers\n');
    end
    fprintf(fid, '%d,%d,%d,%d,%.6f,%d\n', chunk.b, chunk.d, chunk.chunk, ...
        numel(chunk.unit_indices), chunk.seconds, chunk.n_workers);
end


function localMergeTiming(ctx)
    % Every shard's timing file as one table, ascending in (b, d, chunk), with the most recent measurement per chunk.
    % The files are read oldest first, so the last row for a chunk is the newest one a resumed or overwritten run wrote.
    % The table is part of the publication, so it is written even when no shard log survives, empty with its header, rather than left missing.
    % A timing file with other columns is not a chunk log and is not merged.
    names = {'b', 'd', 'chunk', 'units', 'seconds', 'workers'};
    timing = array2table(zeros(0, numel(names)), 'VariableNames', names);
    shards = dir(fullfile(ctx.text_dir, 'timing_shard_*.csv'));
    [~, oldest_first] = sort([shards.datenum]);
    parts = {timing};
    for k = reshape(oldest_first, 1, [])
        part = readtable(fullfile(shards(k).folder, shards(k).name));
        if isequal(part.Properties.VariableNames, names)
            parts{end + 1} = part; %#ok<AGROW>
        else
            fprintf('[Phase3] %s is not a chunk timing log; not merged\n', shards(k).name);
        end
    end
    timing = vertcat(parts{:});
    [~, keep] = unique([timing.b, timing.d, timing.chunk], 'rows', 'last');
    writeTableAtomic(sortrows(timing(keep, :), {'b', 'd', 'chunk'}), ...
        fullfile(ctx.text_dir, 'timing_realizations.csv'));
end


function localSaveFields(path, payload)
    % Save every field of payload as its own top-level variable, through a temporary file and a rename.
    % A half-written file then cannot look like a finished one to the next run's skip check, and a top-level settings_identity can be read back without loading the rows beside it.
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
    [ok, message] = movefile(tmp, path, 'f');
    assert(ok, 'Phase3:SaveMoveFailed', 'Could not move %s into place: %s', tmp, message);
end
