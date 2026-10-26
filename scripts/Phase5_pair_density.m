%PHASE5_PAIR_DENSITY Phase 5: d = 2 copula density panels for a handful of representative pairs.
%
% It shows, for a few pairs of each recording, what the fitted Student-t copula and the Gaussian copula look like beside the data, and how far each is from it.
% Phase 4 reports the departure as a number; this phase is the picture behind that number, summarized per panel by its RMS residual.
%
% Per recording:
%   Phase 2's d = 2 unit rows -> two pairs per |tau| tertile (selectTertilePairs, no random numbers, nearest the band median) -> for each pair the empirical, Gaussian and Student-t cell-averaged densities on one 20 x 20 grid, the three signed residual maps and their RMS (pairDensityPanel).
%
% Nothing is fitted here: rho and p_hat are read from the Phase 2 row, so a panel can never disagree with the fit it illustrates.
% Only pairs with numerical_ok are candidates; the heavy end, an ambiguous profile and a gross repair are recorded conditions, not exclusions, and their pairs stay eligible.
%
% Every density panel is a cell average (cell probability / cell area), the quantity a histogram of pseudo-observations estimates, so each of the three integrates to 1 over the unit square and the three are comparable cell by cell.
%
% Output (<results_root>/<rel_base>/Phase5/):
%   mat/pair_density.mat           top-level pairs (1 x K struct array: unit_index, band, band_index, rank_in_band, channels, channels_str, tau, rho, p_hat, nu_hat, n, n_panel, u_edges, v_edges, empirical, gaussian, t, the three residual grids, their RMS and ratio_gauss_over_t), plus settings, settings_identity, published_tables, dataset_key, rel_base, u_edges, v_edges, tertile_edges and n_pairs_available
%   text/pair_selection.csv        band, band_index, rank_in_band, channels_str, tau, rho, p_hat, nu_hat, n
%   text/pair_density_summary.csv  channels_str, band, band_index, rms_emp_gauss, rms_emp_t, rms_t_gauss, ratio_gauss_over_t
%   text/run_config.json, text/log.txt, images/
%
% The tertile band is stored twice: band is its name ('low', 'middle', 'high'), which figure titles and file names read, and band_index is 1, 2 or 3, which tables sort on.
%
% Resumability: a recording whose pair_density.mat exists and carries this run's settings identity, and whose two tables and run_config.json exist, is skipped unless TC_OVERWRITE=1.
% The phase takes seconds per recording, so it has no checkpoint below that granularity.
% Publication order is the mat, then the CSVs, then the figures, then run_config.json, so a run cut off after the mat, or a table lost later, regenerates the tables and figures from the mat instead of reporting a half-published recording as done.
% An existing mat whose identity is not this run's is refused rather than reused.
% TC_OVERWRITE=1 first withdraws the previous mat, tables and stamps.
%
% Environment: TC_RESULTS_ROOT, TC_RECORDINGS, TC_RECORDING_INDEX, TC_OVERWRITE, TC_RENDER_FIGURES, TC_WORKERS (recorded, never used: see settings.workers).
% MATLAB must start with SHELL set to an executable path such as /bin/bash, or the Python renderers cannot start.

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
% The grid is 20 x 20 on [0, 1]^2.
% Twenty cells per margin put about n/400 points in a cell, a few hundred even for the shortest recording: fine enough to show the corner structure the t copula adds, coarse enough that a cell's count is not itself noise.
settings = struct( ...
    'u_edges',    (0:0.05:1)', ...
    'v_edges',    (0:0.05:1)', ...
    'n_per_band', 2, ...          % pairs per |tau| tertile; 6 panels per recording
    'workers',    getEnvInteger("TC_WORKERS", 0, [0, 1024]));
% workers is read and recorded but never used: the six 20 x 20 panels take a few seconds per recording, so Phase 5 opens no pool and the provenance stamp says n_workers = 0.

steps = struct( ...
    'render_figures', getEnvFlag("TC_RENDER_FIGURES", true), ...
    'overwrite',      getEnvFlag("TC_OVERWRITE", false));

results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));
selected_keys = getEnvList("TC_RECORDINGS");

% The settings identity: the scientific configuration of this phase, hashed and stamped into pair_density.mat and run_config.json.
% workers is absent because it is recorded provenance and never used.
% localIdentity adds the per-recording half, the identity of the Phase 2 fits the panels illustrate.
identity_fields = struct( ...
    'phase',      'Phase5', ...
    'u_edges',    reshape(settings.u_edges, 1, []), ...
    'v_edges',    reshape(settings.v_edges, 1, []), ...
    'n_per_band', settings.n_per_band);

%% ============================================================ run
% The canonical order is the discovery order (recordings sorted by seed_base), so TC_RECORDING_INDEX means the same row here as in every other phase.
all_recordings = discoverPhase1Recordings(results_root, {});
selected_index = getEnvInteger("TC_RECORDING_INDEX", [], [1, numel(all_recordings)]);
assert(isempty(selected_keys) || isempty(selected_index), ...
    'Phase5:AmbiguousSelection', ...
    ['TC_RECORDINGS and TC_RECORDING_INDEX are mutually exclusive; got keys ' ...
    '"%s" and index %d.'], strjoin(selected_keys, ', '), selected_index);
if ~isempty(selected_index)
    recordings = all_recordings(selected_index);
elseif isempty(selected_keys)
    recordings = all_recordings;
else
    recordings = discoverPhase1Recordings(results_root, selected_keys);
end

fprintf('[Phase5] %d of %d recording(s) | results %s | overwrite %d\n', ...
    numel(recordings), numel(all_recordings), results_root, steps.overwrite);

failures = strings(0, 1);
for i = 1:numel(recordings)
    rec = recordings(i);
    try
        localProcessRecording(rec, settings, steps, results_root, identity_fields);
    catch recording_error
        failures(end + 1, 1) = string(rec.dataset_key) + " [" + ...
            string(recording_error.identifier) + "] " + ...
            string(recording_error.message); %#ok<SAGROW>
        fprintf(2, '\n[Phase5] %s failed (%s)\n%s\n', rec.dataset_key, ...
            recording_error.identifier, ...
            getReport(recording_error, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\n[Phase5] finished: %d of %d recording(s) failed\n', ...
    numel(failures), numel(recordings));
if ~isempty(failures)
    error('Phase5:RecordingFailures', ...
        'Phase 5 failed for %d recording(s):\n%s', numel(failures), ...
        strjoin(failures, newline));
end


% ===========================================================================
% Local functions
% ===========================================================================
function localProcessRecording(rec, settings, steps, results_root, identity_fields)
    % One recording: select the pairs, build the panels, write the mat and the CSVs.
    %
    % ctx owns the diary through an onCleanup guard, so it flushes and closes when this function returns, including on an error.
    rel_base = char(rec.rel_base);
    probe = openPhaseOutput(rel_base, 5, results_root, 'Diary', false);
    mat_path = fullfile(probe.mat_dir, 'pair_density.mat');
    identity = localIdentity(identity_fields, rel_base, results_root, rec);

    if ~steps.overwrite && isfile(mat_path)
        localAssertIdentity(mat_path, identity);
        if localIsPublished(mat_path, probe.text_dir)
            fprintf('[Phase5] %s: already done, skipping (%s)\n', ...
                rec.dataset_key, mat_path);
            return;
        end
        ctx = openPhaseOutput(rel_base, 5, results_root);
        started_utc = datetime('now', 'TimeZone', 'UTC');
        fprintf(['[Phase5] %s: the panels are saved but their publication is ' ...
            'incomplete; regenerating the tables and figures from %s ' ...
            'WITHOUT REBUILDING THEM\n'], rec.dataset_key, mat_path);
        stored = load(mat_path, 'pairs');
        localPublishRecording(ctx, rec, stored.pairs, settings, steps, identity, ...
            rel_base, started_utc);
        return;
    end

    ctx = openPhaseOutput(rel_base, 5, results_root);
    started_utc = datetime('now', 'TimeZone', 'UTC');
    fprintf('\n========================================================\n');
    fprintf('Phase 5 recording: %s\n', rec.dataset_key);
    fprintf('  output: %s\n', ctx.base);
    fprintf('  identity: %s\n', identity);
    fprintf('========================================================\n');
    if steps.overwrite
        localWithdrawPublication(ctx);
    end

    u_edges = reshape(settings.u_edges, 1, []);
    v_edges = reshape(settings.v_edges, 1, []);

    X = loadProcessedData(rec.phase1_mat_abs, 'X_filtered');
    pair_units = localReadPhase2Pairs(rel_base, results_root);
    [selected, tertile_edges] = localSelectPairs(pair_units, settings.n_per_band);

    fprintf('[Phase5] %d numerically ok pair(s); |tau| tertile edges [%.4f %.4f]; %d panel(s)\n', ...
        numel(pair_units), tertile_edges(1), tertile_edges(2), numel(selected));

    pairs = localBuildPanels(X, selected, u_edges, v_edges, rec.dataset_key);

    %% 5.1 Save -- explicit list, every grid kept, the edges both at the top level and in every pair
    out = struct( ...
        'dataset_key',       char(rec.dataset_key), ...
        'rel_base',          rel_base, ...
        'u_edges',           u_edges, ...
        'v_edges',           v_edges, ...
        'tertile_edges',     tertile_edges, ...
        'n_pairs_available', numel(pair_units));
    out.settings = settings;
    out.settings_identity = identity;
    out.published_tables = localPublishedTables();
    out.pairs = pairs;
    localSaveAtomic(mat_path, out);
    fprintf('[Phase5] saved %s\n', mat_path);

    %% 5.2 CSVs, figures and the provenance stamp
    localPublishRecording(ctx, rec, pairs, settings, steps, identity, rel_base, ...
        started_utc);
end


function localPublishRecording(ctx, rec, pairs, settings, steps, identity, ...
        rel_base, started_utc)
    % The two CSVs, the figures and the provenance stamp, in the one order that makes run_config.json a completion marker: the mat (already written by the caller), then the CSVs, then the figures, then the stamp.
    % Both tables are projections of pairs, so a fresh run and a republish write the same files.
    writeTableAtomic(localSelectionTable(pairs), ...
        fullfile(ctx.text_dir, 'pair_selection.csv'));
    writeTableAtomic(localSummaryTable(pairs), ...
        fullfile(ctx.text_dir, 'pair_density_summary.csv'));

    % A figure failure never fails the phase.
    if steps.render_figures
        status = renderPhaseFigures('render_phase5_figures', ctx.base, ...
            ctx.images_dir, ctx.text_dir);
        if isstruct(status) && isfield(status, 'ok') && status.ok
            fprintf('[Phase5] %d figure file(s)\n', status.n_figures);
        else
            fprintf(2, ['[Phase5] figures did not complete; the mat and the ' ...
                'CSVs of %s are unaffected.\n'], rec.dataset_key);
        end
    end

    run_settings = settings;
    run_settings.steps = steps;
    run_settings.rel_base = rel_base;
    run_settings.dataset_key = rec.dataset_key;
    run_settings.settings_identity = identity;
    writeRunConfig(ctx, run_settings, struct( ...
        'n_workers',    0, ...
        'started_utc',  started_utc, ...
        'finished_utc', datetime('now', 'TimeZone', 'UTC'), ...
        'phase1_mat',   rec.phase1_mat_abs));
end


function identity = localIdentity(fields, rel_base, results_root, rec)
    % The identity of one recording's panels: this phase's settings plus the identity of the Phase 2 fits the panels draw.
    % rho and p_hat are read from the unit row, never refitted, so a panel belongs to exactly one set of fits.
    fit_all_path = localFitAllPath(rel_base, results_root);
    assert(isfile(fit_all_path), 'Phase5:MissingPhase2Fit', ...
        ['Phase 5 needs the Phase 2 unit rows of %s, which are not at %s. ' ...
        'Run Phase 2 for this recording first.'], rec.dataset_key, fit_all_path);
    fields.phase2_identity = localStoredIdentity(fit_all_path);
    identity = settingsIdentity(fields);
end


function localAssertIdentity(path, identity)
    % Refuse an artifact written under another scientific configuration.
    stored = localStoredIdentity(path);
    if strcmp(stored, identity)
        return;
    end
    if isempty(stored)
        stored = '<none recorded>';
    end
    error('Phase5:SettingsIdentityMismatch', ...
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


function tables = localPublishedTables()
    % The tables one recording's panels publish, declared here and nowhere else: pair_density.mat records the list, localPublishRecording writes both, and the recording counts as finished only when both exist.
    tables = {'pair_selection.csv', 'pair_density_summary.csv'};
end


function done = localIsPublished(mat_path, text_dir)
    % The panels are published when their mat lists the tables its publication consists of, every one of them exists, and the provenance stamp, written last, exists.
    % A mat that lists no tables is not taken as published.
    tables = localStoredVariable(mat_path, 'published_tables', {});
    done = ~isempty(tables) && isfile(fullfile(text_dir, 'run_config.json')) ...
        && all(cellfun(@(name) isfile(fullfile(text_dir, name)), tables));
end


function localWithdrawPublication(ctx)
    % TC_OVERWRITE=1 withdraws the recording's previous publication (the mat, both tables, run_config.json and the figure manifests) before anything is rebuilt, so an interrupted overwrite can never be read as the finished one.
    localDeleteFiles([ ...
        {fullfile(ctx.mat_dir, 'pair_density.mat')}, ...
        fullfile(ctx.text_dir, [localPublishedTables(), ...
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


function fit_path = localFitAllPath(rel_base, results_root)
    % The path of one recording's Phase 2 fit_all.mat.
    % It is built rather than opened through openPhaseOutput, which would create an empty Phase2 tree for a recording Phase 2 never ran.
    parts = split(string(rel_base), "/");
    fit_path = fullfile(results_root, char(parts(1)), char(parts(2)), ...
        char(parts(3)), 'Phase2', 'mat', 'fit_all.mat');
end


function pair_units = localReadPhase2Pairs(rel_base, results_root)
    % The d = 2 unit rows Phase 2 fitted for this recording, with numerical failures dropped.
    fit_path = localFitAllPath(rel_base, results_root);
    assert(isfile(fit_path), 'Phase5:MissingPhase2Fit', ...
        ['Phase 5 needs the Phase 2 unit rows of %s, which are not at %s. ' ...
        'Run Phase 2 for this recording first.'], rel_base, fit_path);

    loaded = load(fit_path, 'units');
    assert(isfield(loaded, 'units') && isstruct(loaded.units), ...
        'Phase5:EmptyPhase2Fit', ...
        '%s carries no units variable, or it is not a struct.', fit_path);
    units = loaded.units;

    required = {'d', 'unit_index', 'numerical_ok', 'channels', 'tau', 'rho', ...
        'p_hat', 'nu_hat', 'n'};
    missing = required(~isfield(units, required));
    assert(isempty(missing), 'Phase5:Phase2SchemaMismatch', ...
        'The unit rows in %s have no %s field.', fit_path, strjoin(missing, ', '));

    % Phase 2 stores its unit rows column-wise, so the selection is a mask over the columns.
    % The pairs that survive it are expanded into a struct array, because localSelectPairs attaches a band, a band index and a rank to each one.
    pairs = find(units.d == 2 & units.numerical_ok ~= 0);
    pair_units = unitColumnsToRows(units, pairs);
    assert(~isempty(pair_units), 'Phase5:NoUsablePairs', ...
        ['%s holds %d unit row(s) but none is a numerically ok d = 2 pair, ' ...
        'so there is nothing to draw.'], fit_path, numel(units.d));
end


function [selected, tertile_edges] = localSelectPairs(pair_units, n_per_band)
    % n_per_band pairs per |tau| tertile, nearest that band's median |tau|, with the band label and the rank inside the band attached to each chosen unit.
    abs_tau = abs([pair_units.tau]);
    [sel_idx, band_of] = selectTertilePairs(abs_tau, n_per_band);
    tertile_edges = quantile(abs_tau, [1/3, 2/3]);

    % The band is stored as its name as well as its index: figure titles and file names read the name, and tables sort on 1/2/3.
    band_names = {'low', 'middle', 'high'};
    selected = pair_units(sel_idx);
    seen = zeros(1, 3);
    for k = 1:numel(sel_idx)
        band = band_of(sel_idx(k));
        seen(band) = seen(band) + 1;
        selected(k).band_index = band;
        selected(k).band = band_names{band};
        selected(k).rank_in_band = seen(band);
    end
end


function pairs = localBuildPanels(X, selected, u_edges, v_edges, dataset_key)
    % One panel per selected pair, in the selection's order (low |tau| band first).
    panels = cell(1, numel(selected));
    for k = 1:numel(selected)
        unit = selected(k);
        channels = unit.channels(isfinite(unit.channels));
        assert(numel(channels) == 2, 'Phase5:NotAPair', ...
            ['A d = 2 unit must name exactly two channels; unit %d of %s names ' ...
            '%d.'], k, dataset_key, numel(channels));
        assert(max(channels) <= size(X, 2), 'Phase5:ChannelOutOfRange', ...
            ['X_filtered is [samples x channels] and is %d x %d, so channel %d ' ...
            'of %s does not exist.'], size(X, 1), size(X, 2), max(channels), ...
            dataset_key);

        panel = pairDensityPanel(X(:, channels(1)), X(:, channels(2)), ...
            unit.rho, unit.p_hat, u_edges, v_edges);
        panels{k} = localPanelRow(unit, channels, panel);
        fprintf(['[Phase5] band %-6s rank %d | channels %s | tau %+.4f rho %+.4f ' ...
            'p_hat %.4f | rms(emp-G) %.4f rms(emp-t) %.4f ratio %.3f\n'], ...
            unit.band, unit.rank_in_band, panels{k}.channels_str, unit.tau, ...
            unit.rho, unit.p_hat, panel.rms_emp_gauss, panel.rms_emp_t, ...
            panel.ratio_gauss_over_t);
    end
    pairs = [panels{:}];
end


function row = localPanelRow(unit, channels, panel)
    % One element of the saved struct array; the field order is the saved order.
    row = struct( ...
        'unit_index',         unit.unit_index, ...
        'band',               unit.band, ...
        'band_index',         unit.band_index, ...
        'rank_in_band',       unit.rank_in_band, ...
        'channels',           reshape(channels, 1, []), ...
        'channels_str',       strtrim(sprintf('%d ', channels)), ...
        'tau',                unit.tau, ...
        'rho',                unit.rho, ...
        'p_hat',              unit.p_hat, ...
        'nu_hat',             unit.nu_hat, ...
        'n',                  unit.n, ...
        'n_panel',            panel.n, ...
        'u_edges',            panel.u_edges, ...
        'v_edges',            panel.v_edges, ...
        'empirical',          panel.empirical, ...
        'gaussian',           panel.gaussian, ...
        't',                  panel.t, ...
        'resid_emp_gauss',    panel.resid_emp_gauss, ...
        'resid_emp_t',        panel.resid_emp_t, ...
        'resid_t_gauss',      panel.resid_t_gauss, ...
        'rms_emp_gauss',      panel.rms_emp_gauss, ...
        'rms_emp_t',          panel.rms_emp_t, ...
        'rms_t_gauss',        panel.rms_t_gauss, ...
        'ratio_gauss_over_t', panel.ratio_gauss_over_t);
end


function T = localSelectionTable(pairs)
    % The pair_selection.csv table: the drawn pairs and their Phase 2 fit.
    T = table(localText(pairs, 'band'), [pairs.band_index].', ...
        [pairs.rank_in_band].', localText(pairs, 'channels_str'), ...
        [pairs.tau].', [pairs.rho].', [pairs.p_hat].', [pairs.nu_hat].', ...
        [pairs.n].', ...
        'VariableNames', {'band', 'band_index', 'rank_in_band', ...
        'channels_str', 'tau', 'rho', 'p_hat', 'nu_hat', 'n'});
end


function T = localSummaryTable(pairs)
    % The pair_density_summary.csv table: the residual RMS values of each panel.
    T = table(localText(pairs, 'channels_str'), localText(pairs, 'band'), ...
        [pairs.band_index].', [pairs.rms_emp_gauss].', [pairs.rms_emp_t].', ...
        [pairs.rms_t_gauss].', [pairs.ratio_gauss_over_t].', ...
        'VariableNames', {'channels_str', 'band', 'band_index', ...
        'rms_emp_gauss', 'rms_emp_t', 'rms_t_gauss', 'ratio_gauss_over_t'});
end


function column = localText(pairs, name)
    % One text field of every pair as a cellstr column.
    column = cellstr(reshape(string({pairs.(name)}), [], 1));
end


function localSaveAtomic(path, payload)
    % Save every field of PAYLOAD as its own top-level variable, through a temporary file and a rename, so an interrupted save cannot leave a half-written mat that the skip rule would read as a finished recording.
    % The staging name carries a per-call token, as saveStructAtomic's does, so two writers of one destination can never delete or overwrite each other's half-written file.
    % saveStructAtomic itself is not used because it stores one variable, and this mat is read field by field.
    [~, token] = fileparts(tempname);
    tmp = [path, '.', token, '.tmp.mat'];
    try
        save(tmp, '-struct', 'payload', '-v7.3');
    catch save_error
        localDeleteFiles({tmp});
        rethrow(save_error);
    end
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'Phase5:MoveFailed', 'Could not move %s into place: %s', tmp, msg);
end
