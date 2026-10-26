%PHASE4_INFORMATION_MEASURES Phase 4: information measures of the fitted t copula and their differences from the Gaussian, in nats.
%
% With p_hat in hand, this phase computes how far the information-theoretic measures move away from their Gaussian closed form.
% For an elliptical copula the answer is exact.
% The Gaussian level is a function of log det R and of the summed log determinants of its leave-one-out minors, the Student-t level adds a term that depends on (d, nu) only, and all of these inputs are already in the Phase 2 unit row, so this phase is arithmetic over stored numbers:
%
%   TC_G = -1/2 logdet_R,  DTC_G = 1/2 sum_logdet_minors - (d-1)/2 logdet_R,
%   O_G  = TC_G - DTC_G,   MI_G = TC_G at d = 2,
%   X_t  = X_G + Delta(X)(d, p_hat)     (tInformationDifferences, closed form)
%
% Each difference is reported in nats and as a percentage of both the Gaussian value and the fitted value, under the suppression rules of informationMeasuresT: an absolute floor of 1e-3 nats on each denominator, and the O-information cancellation ratio per denominator.
% A unit whose fit failed numerically carries NaN measures and status code 0.
%
% Three things are joined to that arithmetic:
%   - the model-free KSG anchor at the dimensions where it is affordable and meaningful (mutual information at d = 2, total correlation at d = 3);
%   - the surrogate floor of Phase 3, by dimension: the 95th percentile of each difference over the surrogate fits of realization 1 at that dimension;
%   - the (d, p) theory grid, which is the same for every recording and is written beside each one so a figure never has to recompute it.
%
% Phase 4 is fast and has no checkpoints, because nothing in it runs long enough to need a resume.
% Unless TC_OVERWRITE=1, a recording is skipped when its mat/information_measures.mat carries this run's settings identity and every table the MAT lists, as well as run_config.json, exists.
% The publication order is the MAT, the CSVs, the figures, then run_config.json.
% After a run cut off past the MAT, or a table lost later, the next run therefore regenerates the tables and figures from the MAT without recomputing a measure, instead of reporting a half-published recording as done.
%
% The identity follows the inputs, and a stale MAT is recomputed.
% The identity covers this phase's settings, the Phase 2 fits and the joined Phase 3 reference ('no_surrogates' when none is published).
% Phase 4 merges nothing, because every number is rebuilt from those inputs in seconds, so a MAT under another identity is withdrawn and recomputed rather than refused, and an ordinary resume after Phase 3 completes picks up the reference.
% A published Phase 3 reference built on other Phase 2 fits is refused with an error, never joined, because it was fitted on other channel sets.
%
% Reads : Phase 2 fit_all.mat, the Phase 1 mat (for KSG), and, when Phase 3 has published it, Phase 3 text/surrogate_floor_by_d.csv with the identities recorded in mat/surrogates_all.mat.
% Writes: <rel_base>/Phase4/{mat,text,images}:
%         mat/information_measures.mat
%         text/im_pairs.csv, im_long.csv, im_summary_by_d.csv, im_by_gaussian_decile.csv, o_sign_flips_by_d.csv, theory_grid.csv, ksg_low_d.csv, run_config.json, log.txt
%
% Environment: TC_RESULTS_ROOT, TC_RECORDINGS, TC_RECORDING_INDEX, TC_WORKERS, TC_OVERWRITE, TC_RENDER_FIGURES.
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
% The two percentage thresholds repeat the prespecified defaults of informationMeasuresT, because a phase carries its own configuration.
% 1e-3 nats is the millinat reporting unit, below which a relative percentage means nothing.
% 0.1 is the O-information cancellation ratio that catches a small difference of two much larger quantities that still clears that floor.
settings = struct( ...
    'ksg_k',                        6, ...
    'ksg_dimensions',               [2, 3], ...
    'pct_min_abs_denominator',      1e-3, ...
    'o_pct_min_cancellation_ratio', 0.1, ...
    'theory_p',                     0:0.005:3, ...
    'theory_d',                     2:20, ...
    'workers',                      getEnvInteger("TC_WORKERS", [], [0, Inf]));

steps = struct( ...
    'ksg',             true, ...
    'theory_grid',     true, ...
    'join_surrogates', true, ...
    'render_figures',  getEnvFlag("TC_RENDER_FIGURES", true), ...
    'overwrite',       getEnvFlag("TC_OVERWRITE", false));

results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));
selected_keys = getEnvList("TC_RECORDINGS");
selected_index = getEnvInteger("TC_RECORDING_INDEX", [], [1, Inf]);

% The settings identity is the scientific configuration of this phase, hashed and stamped into information_measures.mat and run_config.json.
% The worker count and the step switches are not part of it: the worker count only schedules the work, and the switches decide which anchors, joins, tables and figures a run produces, never the value of a measure.
% localIdentity adds the per-recording part: the identity of the Phase 2 fits every measure is an arithmetic function of, and that of the Phase 3 reference joined to them.
identity_fields = struct( ...
    'phase',                        'Phase4', ...
    'ksg_k',                        settings.ksg_k, ...
    'ksg_dimensions',               settings.ksg_dimensions, ...
    'pct_min_abs_denominator',      settings.pct_min_abs_denominator, ...
    'o_pct_min_cancellation_ratio', settings.o_pct_min_cancellation_ratio, ...
    'theory_p',                     settings.theory_p, ...
    'theory_d',                     settings.theory_d);

%% ============================================================ run
assert(isempty(selected_keys) || isempty(selected_index), 'Phase4:AmbiguousSelection', ...
    ['TC_RECORDINGS and TC_RECORDING_INDEX are mutually exclusive; got keys ' ...
    '"%s" and index %d.'], strjoin(selected_keys, ', '), selected_index);
recordings = discoverPhase1Recordings(results_root, selected_keys);
if ~isempty(selected_index)
    assert(selected_index <= numel(recordings), 'Phase4:RecordingIndexOutOfRange', ...
        'TC_RECORDING_INDEX is %d but only %d recording(s) were discovered under %s.', ...
        selected_index, numel(recordings), results_root);
    recordings = recordings(selected_index);
end

n_workers = openParallelPool(~isequal(settings.workers, 0) && steps.ksg, settings.workers);
fprintf('[Phase4] %d recording(s) | %d worker(s) | results %s\n', ...
    numel(recordings), n_workers, results_root);

failures = strings(0, 1);
for i = 1:numel(recordings)
    rec = recordings(i);
    try
        localRunRecording(rec, settings, steps, results_root, n_workers, ...
            identity_fields);
    catch recording_error
        failures(end + 1, 1) = string(rec.dataset_key) + " [" + ...
            string(recording_error.identifier) + "] " + ...
            string(recording_error.message); %#ok<SAGROW>
        fprintf(2, '\n[Phase4] %s failed (%s)\n%s\n', rec.dataset_key, ...
            recording_error.identifier, ...
            getReport(recording_error, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\n[Phase4] finished: %d of %d recording(s) failed\n', ...
    numel(failures), numel(recordings));
if ~isempty(failures)
    error('Phase4:RecordingFailures', ...
        'Phase 4 failed for %d recording(s):\n%s', numel(failures), ...
        strjoin(failures, newline));
end


% ===========================================================================
% Local functions
% ===========================================================================
function localRunRecording(rec, settings, steps, results_root, n_workers, ...
        identity_fields)
    % One recording end to end: measures, KSG anchors, joins, tables and figures.
    probe = openPhaseOutput(rec.rel_base, 4, results_root, 'Diary', false);
    measures_path = fullfile(probe.mat_dir, 'information_measures.mat');
    [identity, reference] = localIdentity(identity_fields, fileparts(probe.base), ...
        rec, steps);

    stale_identity = '';
    if ~steps.overwrite && isfile(measures_path)
        stored = localStoredIdentity(measures_path);
        if strcmp(stored, identity)
            if localIsPublished(measures_path, probe.text_dir)
                fprintf('[Phase4] %s: already done, skipping (%s)\n', ...
                    rec.dataset_key, measures_path);
                return;
            end
            ctx = openPhaseOutput(rec.rel_base, 4, results_root);
            started_utc = datetime('now', 'TimeZone', 'UTC');
            fprintf(['[Phase4] %s: the measures are saved but their publication ' ...
                'is incomplete; regenerating the tables and figures from %s ' ...
                'WITHOUT RECOMPUTING THEM\n'], rec.dataset_key, measures_path);
            payload = load(measures_path);
            localPublishRecording(ctx, payload, settings, steps, identity, ...
                0, started_utc);
            return;
        end
        stale_identity = stored;
        if isempty(stale_identity)
            stale_identity = '<none recorded>';
        end
    end

    ctx = openPhaseOutput(rec.rel_base, 4, results_root);
    started_utc = datetime('now', 'TimeZone', 'UTC');
    fprintf('\n========================================================\n');
    fprintf('Phase 4 recording: %s\n', rec.dataset_key);
    fprintf('  output: %s\n', ctx.base);
    fprintf('  identity: %s\n', identity);
    fprintf('========================================================\n');
    if ~isempty(stale_identity)
        fprintf(['[Phase4] %s was computed under the identity %s, but the settings ' ...
            'and inputs now give %s (the Phase 2 fits or the Phase 3 reference ' ...
            'changed, or a setting did); Phase 4 merges nothing, so it is ' ...
            'recomputed\n'], measures_path, stale_identity, identity);
    end
    if steps.overwrite || ~isempty(stale_identity)
        localWithdrawPublication(ctx);
    end

    % Phase 2 stores its unit rows column-wise (one array per field, one row per unit), so this phase indexes the columns rather than a struct array.
    fit = load(fullfile(fileparts(ctx.base), 'Phase2', 'mat', 'fit_all.mat'), 'units');
    units = fit.units;
    fprintf('[Phase4] %d unit(s) over d = %s\n', numel(units.d), ...
        mat2str(reshape(unique(units.d), 1, [])));

    im = localUnitTable(units, settings);
    im = localAppendKSG(im, rec, units, settings, steps, n_workers);
    im = localAppendSurrogates(im, reference);

    payload = struct();
    payload.measures = im;
    payload.settings = settings;
    payload.settings_identity = identity;
    if steps.theory_grid
        payload.theory_grid = localTheoryGrid(settings);
    else
        fprintf('[Phase4] theory grid step off: theory_grid.csv not written\n');
    end
    payload.published_tables = localPublishedTables(steps.ksg, steps.theory_grid);
    localSaveFields(measures_path, payload);

    localPublishRecording(ctx, payload, settings, steps, identity, n_workers, ...
        started_utc);
end


function localPublishRecording(ctx, payload, settings, steps, identity, ...
        n_workers, started_utc)
    % The tables, the figures and the provenance stamp, in the one order that makes run_config.json a completion marker: the MAT (already written by the caller), then the CSVs, then the figures, then the stamp.
    % Everything here is a projection of payload, and the two optional tables are written exactly when payload lists them, so a fresh run and a republish write the same files.
    im = payload.measures;
    writeTableAtomic(im(im.d == 2, :), fullfile(ctx.text_dir, 'im_pairs.csv'));
    writeTableAtomic(im(im.d >= 3, :), fullfile(ctx.text_dir, 'im_long.csv'));
    writeTableAtomic(localSummaryByDimension(im), ...
        fullfile(ctx.text_dir, 'im_summary_by_d.csv'));
    writeTableAtomic(localDecileTable(im), ...
        fullfile(ctx.text_dir, 'im_by_gaussian_decile.csv'));
    writeTableAtomic(localSignFlipTable(im), ...
        fullfile(ctx.text_dir, 'o_sign_flips_by_d.csv'));
    if ismember('ksg_low_d.csv', payload.published_tables)
        writeTableAtomic(localKsgTable(im, settings), ...
            fullfile(ctx.text_dir, 'ksg_low_d.csv'));
    end
    if ismember('theory_grid.csv', payload.published_tables)
        writeTableAtomic(payload.theory_grid, fullfile(ctx.text_dir, 'theory_grid.csv'));
    end

    if steps.render_figures
        renderPhaseFigures('render_phase4_figures', ctx.base, ctx.images_dir, ctx.text_dir);
    end

    run_settings = settings;
    run_settings.settings_identity = identity;
    writeRunConfig(ctx, run_settings, struct('n_workers', n_workers, ...
        'started_utc', started_utc, ...
        'finished_utc', datetime('now', 'TimeZone', 'UTC')));
end


function tables = localPublishedTables(with_ksg, with_theory)
    % The tables one recording's measures publish, declared here and nowhere else: information_measures.mat records the list, localPublishRecording writes exactly those, and the recording counts as finished only when every one exists.
    % The KSG and theory tables exist only when their steps ran.
    tables = {'im_pairs.csv', 'im_long.csv', 'im_summary_by_d.csv', ...
        'im_by_gaussian_decile.csv', 'o_sign_flips_by_d.csv', 'ksg_low_d.csv', ...
        'theory_grid.csv'};
    tables = tables([true(1, 5), logical(with_ksg), logical(with_theory)]);
end


function done = localIsPublished(measures_path, text_dir)
    % The measures are published when their MAT lists the tables its publication consists of, every one of them exists, and the provenance stamp, written last, exists.
    % A MAT that lists no table is not taken as published.
    tables = localStoredVariable(measures_path, 'published_tables', {});
    done = ~isempty(tables) && isfile(fullfile(text_dir, 'run_config.json')) ...
        && all(cellfun(@(name) isfile(fullfile(text_dir, name)), tables));
end


function localWithdrawPublication(ctx)
    % TC_OVERWRITE=1, or a MAT computed under another identity, withdraws the recording's existing publication (the MAT, every table and both stamps) before anything is recomputed, so an interrupted recompute can never be read as the finished one.
    localDeleteFiles([ ...
        {fullfile(ctx.mat_dir, 'information_measures.mat')}, ...
        fullfile(ctx.text_dir, [localPublishedTables(true, true), ...
        {'run_config.json', 'figures_manifest.json'}]), ...
        {fullfile(ctx.images_dir, 'figures_manifest.json')}]);
end


function localDeleteFiles(paths)
    % Delete each listed file that exists.
    for k = 1:numel(paths)
        if isfile(paths{k})
            delete(paths{k});
        end
    end
end


function [identity, reference] = localIdentity(fields, recording_base, rec, steps)
    % The identity of one recording's measures: this phase's settings, the identity of the Phase 2 fits every measure is an arithmetic function of, and the identity of the Phase 3 reference joined to them ('no_surrogates' when none is).
    % The measures are therefore recomputed whenever the fits change and whenever a reference appears, disappears or is rebuilt.
    fit_all_path = fullfile(recording_base, 'Phase2', 'mat', 'fit_all.mat');
    assert(isfile(fit_all_path), 'Phase4:MissingPhase2Fit', ...
        ['Phase 4 needs the Phase 2 unit rows of %s, which are not at %s. ' ...
        'Run Phase 2 for this recording first.'], rec.dataset_key, fit_all_path);
    fields.phase2_identity = localStoredIdentity(fit_all_path);
    reference = localSurrogateReference(recording_base, fields.phase2_identity, steps);
    fields.phase3_identity = reference.identity;
    identity = settingsIdentity(fields);
end


function reference = localSurrogateReference(recording_base, phase2_identity, steps)
    % The Phase 3 reference this run joins, if any.
    % A reference is joined only when it is published (merged MAT, floor table and stamp all present) and only when it was built on the Phase 2 fits this run reads, which the merged MAT records in the clear.
    % A reference fitted on the channel sets of other Phase 2 fits is not the reference of these ones, so it is refused with an error rather than joined.
    phase3 = fullfile(recording_base, 'Phase3');
    merged_path = fullfile(phase3, 'mat', 'surrogates_all.mat');
    floor_path = fullfile(phase3, 'text', 'surrogate_floor_by_d.csv');
    reference = struct('identity', 'no_surrogates', 'floor_path', '', ...
        'merged_path', merged_path);
    published = isfile(merged_path) && isfile(floor_path) && ...
        isfile(fullfile(phase3, 'text', 'run_config.json'));
    if ~steps.join_surrogates || ~published
        return;
    end
    built_on = char(string(localStoredVariable(merged_path, 'phase2_identity', '')));
    phase3_identity = localStoredIdentity(merged_path);
    if ~strcmp(built_on, phase2_identity) || isempty(phase3_identity)
        if isempty(built_on)
            built_on = '<none recorded>';
        end
        error('Phase4:StaleSurrogateReference', ...
            ['The Phase 3 reference %s was built on the Phase 2 fits %s, but the ' ...
            'Phase 2 fits are now %s, so it was fitted on other channel sets. ' ...
            'Rerun Phase 3 with TC_OVERWRITE=1 before Phase 4, or remove that ' ...
            'Phase 3 output to run Phase 4 without a reference.'], ...
            merged_path, built_on, phase2_identity);
    end
    reference.identity = phase3_identity;
    reference.floor_path = floor_path;
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


function im = localUnitTable(units, settings)
    % The per-unit measures beside the unit's key columns.
    %
    % units is the Phase 2 column struct: one array per field, with one row per unit.
    %
    % The Gaussian reference is rebuilt from the two log-determinants the unit row already carries rather than from R, the same closed form without a Cholesky factorisation per unit.
    % A unit with no usable fit is passed in as p = NaN, the fail-closed path of the measure function: NaN measures, unreportable percentages, status code 0.
    opts = struct( ...
        'pct_min_abs_denominator', settings.pct_min_abs_denominator, ...
        'o_pct_min_cancellation_ratio', settings.o_pct_min_cancellation_ratio);
    measure_timer = tic;
    n_units = numel(units.d);
    rows = cell(1, n_units);
    for i = 1:n_units
        if units.numerical_ok(i)
            rows{i} = tInformationMeasuresFromLogdets(units.d(i), units.p_hat(i), ...
                units.logdet_R(i), units.sum_logdet_minors(i), opts);
        else
            rows{i} = tInformationMeasuresFromLogdets(units.d(i), NaN, NaN, NaN, opts);
        end
    end
    measures = struct2table([rows{:}]);
    % d, p and nu repeat the key columns d, p_hat and nu_hat, and a table cannot carry a name twice.
    measures = removevars(measures, {'d', 'p', 'nu'});
    % The measure struct carries the field names of informationMeasuresT, so the two entry points can be compared field for field.
    % The CSV uses the column names the renderers read: <M>_G for the Gaussian value and <M>_pct_status for each denominator's status code.
    % The renaming happens here, at the table edge, and nowhere else.
    for name = {'MI', 'TC', 'DTC', 'O'}
        measures = renamevars(measures, ...
            {[name{1} '_gauss'], [name{1} '_pct_status_code'], ...
            [name{1} '_pct_status_code_t']}, ...
            {[name{1} '_G'], [name{1} '_pct_status'], [name{1} '_pct_status_t']});
    end

    % The key columns carry the unit's identity and the three numbers every measure is a function of (p_hat and the two log-determinants), so a reader can rederive the whole row from the CSV alone.
    keys = unitRowsToTable(units);
    keys = keys(:, {'dataset_key', 'd', 'unit_index', 'channels_str', 'n', ...
        'p_hat', 'nu_hat', 'rho', 'logdet_R', 'sum_logdet_minors', ...
        'numerical_ok', 'status'});
    im = [keys, measures];
    fprintf('[Phase4] measures: %d unit(s), %d reportable, %.1f s\n', ...
        height(im), nnz(im.valid), toc(measure_timer));
end


function im = localAppendKSG(im, rec, units, settings, steps, n_workers)
    % The model-free anchor at the dimensions where it is affordable, settings.ksg_dimensions: mutual information at d = 2 and total correlation above it, on the same cleaned matrix the fit used.
    % units is the Phase 2 column struct, whose row order is the row order of im, so a position in one addresses the same channel set in the other.
    % KSG is a reference beside the closed forms, not the result, so a channel set it cannot handle (tied normal scores, too few complete rows) gets NaN with a counted, reported reason instead of ending the recording.
    im.ksg_MI = nan(height(im), 1);
    im.ksg_TC = nan(height(im), 1);
    if ~steps.ksg
        fprintf('[Phase4] KSG step off: ksg_MI and ksg_TC stay NaN\n');
        return;
    end

    targets = find(ismember(im.d, settings.ksg_dimensions));
    if isempty(targets)
        fprintf('[Phase4] KSG: no unit at d = %s\n', mat2str(settings.ksg_dimensions));
        return;
    end
    X = localLoadSignal(rec);
    groups = cell(1, numel(targets));
    for t = 1:numel(targets)
        channels = units.channels(targets(t), :);
        groups{t} = channels(isfinite(channels));
    end

    ksg_timer = tic;
    values = nan(numel(targets), 1);
    reasons = strings(numel(targets), 1);
    k = settings.ksg_k;
    parfor (t = 1:numel(targets), n_workers)
        try
            % X is a broadcast variable: every channel set reads its own columns of the one cleaned matrix, so there is nothing to slice.
            values(t) = estimateMIKSG(X(:, groups{t}), k); %#ok<PFBNS>
        catch anchor_error
            reasons(t) = string(anchor_error.identifier);
        end
    end

    pairs = im.d(targets) == 2;
    im.ksg_MI(targets(pairs)) = values(pairs);
    im.ksg_TC(targets(~pairs)) = values(~pairs);
    failed = strlength(reasons) > 0;
    fprintf('[Phase4] KSG: %d anchor(s) at d = %s in %.1f s, %d unavailable%s\n', ...
        numel(targets), mat2str(settings.ksg_dimensions), toc(ksg_timer), ...
        nnz(failed), localFirstReason(reasons(failed)));
end


function im = localAppendSurrogates(im, reference)
    % The Phase 3 floor, joined by dimension.
    % floor95_<M> of a unit is the 95th percentile of the difference <M> over the successful surrogate fits of its dimension on realization 1, the reported reference; realization 2 is the Phase 3 yardstick and is not joined.
    % Every channel set of a dimension carries the same floor, because the reference is a pooled distribution per dimension, not a per-set one.
    % Mutual information has a floor at d = 2 only, so floor95_MI stays NaN above it, as does any column at a dimension the reference has no row for.
    %
    % reference comes from localSurrogateReference, which has already refused a reference built on other Phase 2 fits.
    measures = {'MI', 'TC', 'DTC', 'O'};
    for m = 1:numel(measures)
        im.(['floor95_' measures{m}]) = nan(height(im), 1);
    end
    if isempty(reference.floor_path)
        fprintf(['[Phase4] no published Phase 3 reference to join (%s), or the ' ...
            'join step is off: floor95_* stay NaN\n'], reference.merged_path);
        return;
    end

    floors = readtable(reference.floor_path, 'TextType', 'char');
    required = {'d', 'realization', 'measure', 'q95'};
    missing = required(~ismember(required, floors.Properties.VariableNames));
    assert(isempty(missing), 'Phase4:SurrogateFloorColumns', ...
        ['%s carries no %s column; it was written by an earlier Phase 3. Rerun ' ...
        'Phase 3 with TC_OVERWRITE=1.'], reference.floor_path, strjoin(missing, ', '));
    floors = floors(floors.realization == 1, :);
    for m = 1:numel(measures)
        rows = floors(strcmp(floors.measure, measures{m}), :);
        [covered, where] = ismember(im.d, rows.d);
        im.(['floor95_' measures{m}])(covered) = rows.q95(where(covered));
    end
    fprintf('[Phase4] surrogate floor join: %d of %d unit(s) at a dimension it covers\n', ...
        nnz(ismember(im.d, floors.d)), height(im));
end


function X = localLoadSignal(rec)
    % The cleaned analysis matrix, with the declared orientation asserted: a transposed matrix does not error, it silently yields a different KSG estimate.
    phase1 = load(rec.phase1_mat_abs, 'X_filtered');
    X = double(phase1.X_filtered);
    required = {'n_samples', 'n_channels_retained'};
    missing = required(~isfield(rec, required));
    assert(isempty(missing), 'Phase4:MissingRecordingMetadata', ...
        'recording.json of %s carries no %s field.', rec.dataset_key, ...
        strjoin(missing, ', '));
    assert(size(X, 1) == rec.n_samples && size(X, 2) == rec.n_channels_retained, ...
        'Phase4:SignalMatrixOrientation', ...
        ['X_filtered of %s must be [samples x channels] = [%d x %d] as ' ...
        'recording.json declares; got %d x %d.'], rec.dataset_key, ...
        rec.n_samples, rec.n_channels_retained, size(X, 1), size(X, 2));
end


function suffix = localFirstReason(reasons)
    % The distinct error identifiers behind the unavailable anchors, for the diary, or '' when there are none.
    if isempty(reasons)
        suffix = '';
    else
        suffix = sprintf(' (%s)', strjoin(unique(reasons).', ', '));
    end
end


function summary = localSummaryByDimension(im)
    % Location and spread of every difference at every dimension, in nats and against both denominators, plus the status-code counts that say how much of each percentage column is suppressed and why.
    measures = {'MI', 'TC', 'DTC', 'O'};
    quantities = {'delta_nats', 'delta_pct', 'delta_pct_t'};
    dims = reshape(unique(im.d), [], 1);
    summary = table(dims, 'VariableNames', {'d'});
    summary.n_units = zeros(numel(dims), 1);
    summary.n_ok = zeros(numel(dims), 1);
    for m = 1:numel(measures)
        for q = 1:numel(quantities)
            for statistic = {'median', 'q25', 'q75', 'mean'}
                summary.([measures{m} '_' quantities{q} '_' statistic{1}]) = ...
                    nan(numel(dims), 1);
            end
        end
        for code = 0:3
            summary.(sprintf('%s_pct_status_%d', measures{m}, code)) = zeros(numel(dims), 1);
            summary.(sprintf('%s_pct_status_t_%d', measures{m}, code)) = zeros(numel(dims), 1);
        end
    end

    for k = 1:numel(dims)
        at_d = im.d == dims(k);
        summary.n_units(k) = nnz(at_d);
        summary.n_ok(k) = nnz(at_d & im.valid);
        for m = 1:numel(measures)
            for q = 1:numel(quantities)
                statistics = localLocationSpread(im.([measures{m} '_' quantities{q}])(at_d));
                names = {'median', 'q25', 'q75', 'mean'};
                for s = 1:numel(names)
                    summary.([measures{m} '_' quantities{q} '_' names{s}])(k) = statistics(s);
                end
            end
            for code = 0:3
                summary.(sprintf('%s_pct_status_%d', measures{m}, code))(k) = ...
                    nnz(im.([measures{m} '_pct_status'])(at_d) == code);
                summary.(sprintf('%s_pct_status_t_%d', measures{m}, code))(k) = ...
                    nnz(im.([measures{m} '_pct_status_t'])(at_d) == code);
            end
        end
    end
end


function statistics = localLocationSpread(values)
    % [median q25 q75 mean] over the finite entries; all NaN when there are none.
    values = values(isfinite(values));
    if isempty(values)
        statistics = nan(1, 4);
        return;
    end
    quartiles = quantile(values, [0.25, 0.75]);
    statistics = [median(values), quartiles(1), quartiles(2), mean(values)];
end


function deciles = localDecileTable(im)
    % Is a large percentage only an artefact of a small Gaussian denominator?
    % Each (dimension, measure) is split into ten equal groups by the rank of its Gaussian value, rank deciles rather than value quantiles, so tied denominators cannot collapse the bins.
    % Each group reports the median difference, the median percentage, and the mean-before-ratio percentage, the ratio of the means, which no single small denominator can inflate.
    measures = {'MI', 'TC', 'DTC', 'O'};
    n_bins = 10;
    d_column = []; measure_column = {}; decile_column = []; n_column = [];
    median_nats = []; median_pct = []; mean_before_ratio = [];
    for d = reshape(unique(im.d), 1, [])
        at_d = im.d == d;
        for m = 1:numel(measures)
            gauss = im.([measures{m} '_G'])(at_d);
            delta = im.([measures{m} '_delta_nats'])(at_d);
            pct = im.([measures{m} '_delta_pct'])(at_d);
            keep = isfinite(gauss) & isfinite(delta);
            if ~any(keep)
                continue;      % MI above d = 2, or a dimension with no fit
            end
            gauss = gauss(keep); delta = delta(keep); pct = pct(keep);
            bin = min(n_bins, ...
                floor(n_bins * (tiedrank(gauss) - 0.5) / numel(gauss)) + 1);
            for q = 1:n_bins
                in_bin = bin == q;
                if ~any(in_bin)
                    continue;
                end
                d_column(end + 1, 1) = d; %#ok<AGROW>
                measure_column{end + 1, 1} = measures{m}; %#ok<AGROW>
                decile_column(end + 1, 1) = q; %#ok<AGROW>
                n_column(end + 1, 1) = nnz(in_bin); %#ok<AGROW>
                median_nats(end + 1, 1) = median(delta(in_bin)); %#ok<AGROW>
                median_pct(end + 1, 1) = median(pct(in_bin), 'omitnan'); %#ok<AGROW>
                mean_before_ratio(end + 1, 1) = ...
                    100 * mean(delta(in_bin)) / mean(gauss(in_bin)); %#ok<AGROW>
            end
        end
    end
    deciles = table(d_column, measure_column, decile_column, n_column, ...
        median_nats, median_pct, mean_before_ratio, 'VariableNames', ...
        {'d', 'measure', 'decile', 'n', 'median_delta_nats', 'median_pct', ...
        'mean_before_ratio_pct'});
end


function flips = localSignFlipTable(im)
    % How often the O-information changes sign between the Gaussian and the fitted value.
    % Only channel sets where both values are finite and non-zero can flip, so they form the denominator.
    dims = reshape(unique(im.d), [], 1);
    n = zeros(numel(dims), 1);
    n_flips = zeros(numel(dims), 1);
    for k = 1:numel(dims)
        at_d = im.d == dims(k);
        gauss = im.O_G(at_d);
        fitted = im.O_t(at_d);
        comparable = isfinite(gauss) & isfinite(fitted) & gauss ~= 0 & fitted ~= 0;
        n(k) = nnz(comparable);
        n_flips(k) = nnz(sign(fitted(comparable)) ~= sign(gauss(comparable)));
    end
    flips = table(dims, n, n_flips, n_flips ./ n, ...
        'VariableNames', {'d', 'n', 'n_flips', 'rate'});
end


function anchors = localKsgTable(im, settings)
    % The model-free anchor beside the two model values it is compared with: mutual information at d = 2, total correlation above it.
    rows = ismember(im.d, settings.ksg_dimensions);
    block = im(rows, :);
    pairs = block.d == 2;
    ksg = block.ksg_TC;
    ksg(pairs) = block.ksg_MI(pairs);
    gauss = block.TC_G;
    gauss(pairs) = block.MI_G(pairs);
    fitted = block.TC_t;
    fitted(pairs) = block.MI_t(pairs);
    measure = repmat({'TC'}, height(block), 1);
    measure(pairs) = {'MI'};
    anchors = table(block.d, block.unit_index, block.channels_str, block.n, ...
        block.p_hat, block.nu_hat, measure, ksg, gauss, fitted, ...
        'VariableNames', {'d', 'unit_index', 'channels_str', 'n', 'p_hat', ...
        'nu_hat', 'measure', 'ksg_nats', 'gauss_nats', 't_nats'});
end


function grid = localTheoryGrid(settings)
    % The recording-independent part of the result: kappa_t and the four differences as functions of (d, p).
    % It is written beside every recording so a figure never recomputes it, and it is identical in every results tree.
    p = reshape(settings.theory_p, 1, []);
    parts = cell(1, numel(settings.theory_d));
    for k = 1:numel(settings.theory_d)
        d = settings.theory_d(k);
        differences = tInformationDifferences(d, p);
        parts{k} = table(p.', differences.nu.', repmat(d, numel(p), 1), ...
            differences.kappa_d.', differences.kappa_dm1.', differences.MI.', ...
            differences.TC.', differences.DTC.', differences.O.', ...
            'VariableNames', {'p', 'nu', 'd', 'kappa_d', 'kappa_dm1', ...
            'MI_delta', 'TC_delta', 'DTC_delta', 'O_delta'});
    end
    grid = vertcat(parts{:});
end


function localSaveFields(path, payload)
    % Save every field of payload as its own top-level variable, through a temporary file and a rename, so a half-written MAT cannot satisfy the next run's skip check.
    % The staging name carries a per-call token, as in saveStructAtomic, so two writers of one destination never delete or overwrite each other's half-written file.
    % saveStructAtomic itself is not used because it stores one variable, and this file is read field by field.
    [~, token] = fileparts(tempname);
    tmp = [path, '.', token, '.tmp.mat'];
    try
        save(tmp, '-struct', 'payload', '-v7.3');
    catch save_error
        localDeleteFiles({tmp});
        rethrow(save_error);
    end
    [ok, message] = movefile(tmp, path, 'f');
    assert(ok, 'Phase4:SaveMoveFailed', 'Could not move %s into place: %s', tmp, message);
end
