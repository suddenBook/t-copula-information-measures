%PHASE6_AUTOCORRELATION_TIMESCALE Phase 6: autocorrelation time scales of each recording.
%
% For each recording, the phase states how long its signal remembers itself (tau_env) and how long its amplitude does (tau_amp), in samples and in seconds.
% The simulation study (Phase 7) runs in units of its process time constant, on a grid of T/tau and samples per tau declared independently of the data.
% tau_env gives each recording's T/tau and samples per tau, by which the cross-recording renderer places it on that grid.
% The same functions measure the simulated paths, so data and model are compared on one transform and one estimator.
%
% Per recording:
%   1. Phase 1's X_filtered is mapped to rank-normal scores Z, channel by channel over the whole recording.
%   2. The autocorrelations of Z ('score') and of Z.^2 ('square') at lags 0..max_lag come from lagged products taken within trial segments only, averaged over channels (averagedRankAutocorrelation).
%   3. The envelope of each averaged autocorrelation is fitted with exp(-lag / tau) (fitEnvelopeTimescale).
% How the envelope is read is decided once, from the score (envelopeReading).
% If the score dips below -envelope_floor it oscillates, and every curve is read at lag 0 and at the lags of the score's peaks, up to the first point below envelope_floor.
% Otherwise every curve is read along itself, up to where it first falls below envelope_floor.
% The score gives the signal-memory time tau_env and the square the amplitude-memory time tau_amp.
% The square's Gaussian reference, the channel mean of the squared score autocorrelations, is what the square's autocorrelation would be if every channel were Gaussian with its own autocorrelation, and its time tau_amp_ref is what tau_amp is read against.
% The amplitude excess amp_excess = tau_amp / tau_amp_ref is 1 for a Gaussian signal in any band, and above 1 when the amplitude remembers longer than the signal's own autocorrelation explains.
% The fit is anchored at lag 0 and dominated by the first few points it reads, so amp_excess weighs a slow amplitude component by its height more than by its duration, and cannot see it beyond the window.
% Beside it, the integrated amplitude excess amp_excess_int = tau_amp_int / tau_amp_ref_int counts height and duration together, and is also 1 for a Gaussian signal.
% Its two times are the integrated autocorrelation times dt (1 + 2 sum_{k=1..max_lag} rho(k)) of the square and of its reference over the whole window (integratedAutocorrelationTime).
% Beside each time, the Ornstein-Uhlenbeck-equivalent time, twice the integral of the squared autocorrelation (ouEquivalentTimescale), is kept as the memory that sets the effective sample size of second-order statistics; the simulation's T/tau axis uses the envelope times.
% Every channel's own autocorrelations are fitted the same way, each read as its own score says, as the spread behind the channel average.
%
% Two centrings are published for every recording.
% The headline centres every transformed series by its mean over the whole recording.
% The within-trial set, every curve, fit and table column again with the suffix _trialc, centres each series by its mean within each declared segment before the within-segment lagged products.
% On a trial recording the headline reads the between-trial variance of the signal power as amplitude memory at every lag inside a trial, and the within-trial set removes it, at the cost of pulling every lag down by about 1 / n_eff of a segment (averagedRankAutocorrelation).
% Its Gaussian reference therefore carries the same pull.
% On a multi-segment recording it is, channel by channel, the autocorrelation the within-segment-centred square would have in expectation if the channel were Gaussian with the autocorrelation its headline score shows out to the longest segment (segmentCentredGaussianReference).
% On the stationary Gaussian trials of autocorrelationTimescaleTest (about 800 samples each), the square of the likewise-centred score autocorrelation, which is not pulled down, reads the fitted excess as 0.885, and the calibrated reference reads it as 1.00 (sd 0.01 over 24 seeds).
% The within-trial integrated times and excess are withheld (NaN) on a multi-segment recording.
% Centring within a segment of L samples makes its autocovariances sum to about zero and pulls every lag down by about tau_int / L.
% Over a window of W lags that takes about 2 W tau_int / L from the integrated time, which is all of it once the window reaches its cap of half the shortest segment, so no integrated time is defined within segments.
% The within-trial Ornstein-Uhlenbeck-equivalent times are published but carry the same pull, several per cent of tau_amp_ou on sub-second trials.
% On a one-segment recording the two centrings coincide, and every _trialc column equals its headline column.
% The reading mode and the peak lags are decided once, from the headline score, and every curve of both sets is read in them.
%
% The lag window is min(max_lag_seconds * fs, half the shortest segment), so every lag in the window draws pairs from every segment.
%
% A fit the estimator refuses (fitEnvelopeTimescale:BoundaryFit, an envelope that does not decay like an exponential over its range) is recorded as NaN with its error identifier and message, for the channel average as for a single channel, and the rest of the recording is still published.
% Any other error of the envelope fit stops the recording.
% A constant channel has no autocorrelation and is recorded with the status constant_series.
% No channel is dropped silently.
%
% Output (<results_root>/<rel_base>/Phase6/):
%   mat/timescale.mat - every field a top-level variable:
%     dataset_key, rel_base, modality, fs, n_samples, duration_seconds, n_channels, n_segments, shortest_segment_samples, channel_ids
%     max_lag_samples, lags_samples, lags_seconds, pair_counts (the within-segment sample pairs at each lag)
%     reading - the headline score's reading: mode, peak_index, channel_modes
%     score, square, square_gaussian - one struct per curve: transform, acf_mean, acf_by_channel, acf_quantiles, n_valid_channels, fit (the channel average's fit with its envelope, fitted curve, OU-equivalent and integrated times) and channels (every channel's fit as columns)
%     score_trialc, square_trialc, square_gaussian_trialc - the same structs under within-trial centring, with the calibrated reference and NaN integrated times on a multi-segment recording
%     settings, settings_identity, phase1_identity, published_tables
%   text/timescale_summary.csv - one row:
%     dataset_key, modality, fs, n_samples, duration_s, n_channels, n_segments, max_lag_samples, max_lag_s, envelope_floor
%     tau_env_samples, tau_env_s, tau_amp_samples, tau_amp_s, ratio_amp_env, samples_per_tau (fs * tau_env_s), T_over_tau (duration_s / tau_env_s)
%     env_fit_last_lag_s, env_fit_rmse, env_floor_reached, amp_fit_last_lag_s, amp_fit_rmse, amp_floor_reached
%     n_valid_channels, env_fit_error, amp_fit_error, env_fit_mode, amp_fit_mode (peaks or monotone, both the score's reading)
%     tau_env_ou_samples, tau_env_ou_s, tau_amp_ou_samples, tau_amp_ou_s, ratio_amp_env_ou
%     tau_amp_ref_samples, tau_amp_ref_s, amp_excess (tau_amp_s / tau_amp_ref_s), amp_ref_fit_last_lag_s, amp_ref_fit_rmse, amp_ref_floor_reached, amp_ref_fit_error
%     tau_amp_int_samples, tau_amp_int_s, tau_amp_ref_int_samples, tau_amp_ref_int_s, amp_excess_int (tau_amp_int_s / tau_amp_ref_int_s)
%     env_fit_n_points, amp_fit_n_points, amp_ref_fit_n_points
%     <column>_trialc - the within-trial twin of every column from tau_env_samples on, except n_valid_channels, env_fit_mode and amp_fit_mode
%     The twins are NaN, with the errors not_computed, when settings.centrings does not name 'segment', and the five integrated twins are NaN on a multi-segment recording.
%   text/acf_curves.csv - one row per lag:
%     lag_samples, lag_s, pair_count
%     per curve <t> (score, square, square_gaussian): acf_<t>, envelope_<t>, fitted_<t> and the across-channel quantiles acf_<t>_q10, acf_<t>_q50, acf_<t>_q90
%     <column>_trialc - the same columns of the within-trial curves
%   text/timescale_by_channel.csv - one row per channel:
%     channel_id
%     per time scale <r> (env, amp, amp_ref): tau_<r>_samples, tau_<r>_s, <r>_fit_last_lag_s, <r>_fit_rmse, <r>_floor_reached, <r>_status, <r>_error, <r>_fit_mode, tau_<r>_ou_s
%     amp_excess, tau_amp_int_samples, tau_amp_int_s, tau_amp_ref_int_samples, tau_amp_ref_int_s, amp_excess_int
%     <r>_fit_n_points per time scale
%     <column>_trialc - the within-trial twin of every column after channel_id except the <r>_fit_mode columns, the five integrated ones NaN on a multi-segment recording
%   text/run_config.json, text/log.txt, images/
% duration_s is the analysed time, n_samples / fs, which on a trial recording is the summed trial time.
% A floor_reached of 0 means no peak (or, for a monotone autocorrelation, no value) of the envelope fell below envelope_floor inside the lag window, so the fit read to the end of the window; NaN means the fit did not run or errored.
% envelope_<t> in acf_curves.csv is defined over the fit range only and is NaN beyond it.
%
% Resuming: unless TC_OVERWRITE=1, a recording is skipped when its timescale.mat exists and carries this run's settings identity, and its three tables and its run_config.json exist.
% The phase takes seconds per recording, so there is no checkpoint below that granularity.
% Publication order is the mat, then the CSVs, then the figures, then run_config.json.
% A run cut off after the mat, or a table lost later, therefore regenerates the tables and figures from the mat instead of reporting a half-published recording as done.
% An existing mat whose identity is not this run's is refused rather than reused.
% TC_OVERWRITE=1 first withdraws the previous mat, tables and stamps.
%
% Environment: TC_RESULTS_ROOT (default <repository>/results), TC_RECORDINGS (comma-separated dataset keys) or TC_RECORDING_INDEX (one position in the discovery order), TC_OVERWRITE (default 0), TC_RENDER_FIGURES (default 1).
% The phase is serial: it opens no pool and reads no TC_WORKERS.
% Launch MATLAB with SHELL=/bin/bash, or the Python renderers cannot start.

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
% max_lag_seconds: the lag window on a continuous recording, in s; half the shortest segment caps it on a trial recording.
% envelope_floor: the level that ends the fit range, so the fit follows the envelope from 1 down to 0.05, three time constants of a pure exponential (ln 20 = 3.0).
% envelope_method: the envelope fitEnvelopeTimescale implements; it enters the settings identity, so time scales written with another envelope are refused rather than reused.
% transforms: the series, 'score' (Z itself, giving tau_env) and 'square' (Z.^2, giving tau_amp); the score is always computed, because it decides how every curve is read.
% centrings: how each transformed series is centred before its lagged products, 'recording' by its mean over the whole recording (the headline, whose score decides the reading) and 'segment' by its mean within each declared segment (the within-trial set, published as the _trialc curves and columns).
% Both centrings are published for every recording, and on a one-segment recording they coincide.
% trial_reference: the within-trial Gaussian reference of the square, the expectation of the within-trial estimator for a Gaussian series (segmentCentredGaussianReference); it enters the settings identity, so a within-trial set read against another reference is refused rather than reused.
% channel_quantiles: the quantile levels across channels of each lag's autocorrelation, written beside the channel average as the spread band of the figure.
settings = struct( ...
    'max_lag_seconds',   2, ...
    'envelope_floor',    0.05, ...
    'envelope_method',   'score_mode_peak_envelope', ...
    'transforms',        {{'score', 'square'}}, ...
    'centrings',         {{'recording', 'segment'}}, ...
    'trial_reference',   'segment_centred_gaussian', ...
    'channel_quantiles', [0.10, 0.50, 0.90]);
assert(iscellstr(settings.transforms) && ismember('score', settings.transforms) && ...
    all(ismember(settings.transforms, {'score', 'square'})), ...
    'Phase6:BadTransforms', ...
    ['settings.transforms must name ''score'', which decides how every curve is ' ...
    'read, and may add ''square''.']);
assert(iscellstr(settings.centrings) && ismember('recording', settings.centrings) && ...
    all(ismember(settings.centrings, {'recording', 'segment'})), ...
    'Phase6:BadCentrings', ...
    ['settings.centrings must name ''recording'', the headline whose score decides ' ...
    'how every curve is read, and may add ''segment''.']);

steps = struct( ...
    'render_figures', getEnvFlag("TC_RENDER_FIGURES", true), ...
    'overwrite',      getEnvFlag("TC_OVERWRITE", false));

results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));
selected_keys = getEnvList("TC_RECORDINGS");

% The settings identity: the scientific configuration of this phase, hashed and stamped into timescale.mat and run_config.json.
% localIdentity adds the per-recording half, the identity of the Phase 1 matrix the time scales are computed from.
identity_fields = struct( ...
    'phase',             'Phase6', ...
    'max_lag_seconds',   settings.max_lag_seconds, ...
    'envelope_floor',    settings.envelope_floor, ...
    'envelope_method',   settings.envelope_method, ...
    'transforms',        {settings.transforms}, ...
    'centrings',         {settings.centrings}, ...
    'trial_reference',   settings.trial_reference, ...
    'channel_quantiles', settings.channel_quantiles);

%% ============================================================ run
% The canonical order is the discovery order (recordings sorted by seed_base), so TC_RECORDING_INDEX means the same row here as in every other phase.
all_recordings = discoverPhase1Recordings(results_root, {});
selected_index = getEnvInteger("TC_RECORDING_INDEX", [], [1, numel(all_recordings)]);
assert(isempty(selected_keys) || isempty(selected_index), ...
    'Phase6:AmbiguousSelection', ...
    ['TC_RECORDINGS and TC_RECORDING_INDEX are mutually exclusive; got keys ' ...
    '"%s" and index %d.'], strjoin(selected_keys, ', '), selected_index);
if ~isempty(selected_index)
    recordings = all_recordings(selected_index);
elseif isempty(selected_keys)
    recordings = all_recordings;
else
    recordings = discoverPhase1Recordings(results_root, selected_keys);
end

fprintf('[Phase6] %d of %d recording(s) | results %s | overwrite %d\n', ...
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
        fprintf(2, '\n[Phase6] %s failed (%s)\n%s\n', rec.dataset_key, ...
            recording_error.identifier, ...
            getReport(recording_error, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\n[Phase6] finished: %d of %d recording(s) failed\n', ...
    numel(failures), numel(recordings));
if ~isempty(failures)
    error('Phase6:RecordingFailures', ...
        'Phase 6 failed for %d recording(s):\n%s', numel(failures), ...
        strjoin(failures, newline));
end


% ===========================================================================
% Local functions
% ===========================================================================
function localProcessRecording(rec, settings, steps, results_root, identity_fields)
    % One recording: the autocorrelations, their fits, the mat and the CSVs.
    % ctx owns the diary through an onCleanup guard, so the diary flushes and closes when this function returns, including on an error.
    rel_base = char(rec.rel_base);
    probe = openPhaseOutput(rel_base, 6, results_root, 'Diary', false);
    mat_path = fullfile(probe.mat_dir, 'timescale.mat');
    [identity, phase1_identity] = localIdentity(identity_fields, rec);

    if ~steps.overwrite && isfile(mat_path)
        localAssertIdentity(mat_path, identity);
        if localIsPublished(mat_path, probe.text_dir)
            fprintf('[Phase6] %s: already done, skipping (%s)\n', ...
                rec.dataset_key, mat_path);
            return;
        end
        ctx = openPhaseOutput(rel_base, 6, results_root);
        started_utc = datetime('now', 'TimeZone', 'UTC');
        fprintf(['[Phase6] %s: the time scales are saved but their publication ' ...
            'is incomplete; regenerating the tables and figures from %s ' ...
            'WITHOUT RECOMPUTING THEM\n'], rec.dataset_key, mat_path);
        localPublishRecording(ctx, rec, load(mat_path), settings, steps, ...
            identity, rel_base, started_utc);
        return;
    end

    ctx = openPhaseOutput(rel_base, 6, results_root);
    started_utc = datetime('now', 'TimeZone', 'UTC');
    fprintf('\n========================================================\n');
    fprintf('Phase 6 recording: %s\n', rec.dataset_key);
    fprintf('  input : %s\n', rec.phase1_mat_abs);
    fprintf('  output: %s\n', ctx.base);
    fprintf('  identity: %s\n', identity);
    fprintf('========================================================\n');
    if steps.overwrite
        localWithdrawPublication(ctx);
    end

    [X, fs, channel_ids, segment_ends] = localLoadPhase1Signal(rec);
    [n, n_channels] = size(X);
    segment_lengths = diff([0; segment_ends]);
    max_lag = min(round(settings.max_lag_seconds * fs), floor(min(segment_lengths) / 2));
    assert(max_lag >= 2, 'Phase6:LagWindowTooShort', ...
        ['The lag window of %s is %d sample(s) (max_lag_seconds %g at fs %g Hz, ' ...
        'shortest of %d segment(s) %d samples); an envelope fit needs lags 0 to 2 ' ...
        'at least.'], rec.dataset_key, max_lag, settings.max_lag_seconds, fs, ...
        numel(segment_lengths), min(segment_lengths));
    fprintf(['[Phase6] %d samples x %d channels at %g Hz in %d segment(s), shortest ' ...
        '%d samples | lags 0..%d (%.3f s)\n'], n, n_channels, fs, ...
        numel(segment_ends), min(segment_lengths), max_lag, max_lag / fs);

    %% 6.1 The autocorrelation of each transform and its fits
    lags = (0:max_lag)';
    out = struct( ...
        'dataset_key',              char(rec.dataset_key), ...
        'rel_base',                 rel_base, ...
        'modality',                 localModality(rec), ...
        'fs',                       fs, ...
        'n_samples',                n, ...
        'duration_seconds',         n / fs, ...
        'n_channels',               n_channels, ...
        'n_segments',               numel(segment_ends), ...
        'shortest_segment_samples', min(segment_lengths), ...
        'channel_ids',              channel_ids, ...
        'max_lag_samples',          max_lag, ...
        'lags_samples',             lags, ...
        'lags_seconds',             lags / fs, ...
        'pair_counts',              []);
    % The headline score first: its reading (the channel average's, and every channel's own) decides how every curve of this recording is read, under both centrings.
    score = averagedRankAutocorrelation(X, segment_ends, max_lag, 'score');
    readings = localReadings(score, settings.envelope_floor);
    out.pair_counts = score.pair_counts;   % the same lagged pairs for every transform
    out.reading = struct('mode', readings.mean.mode, ...
        'peak_index', readings.mean.peak_index, ...
        'channel_modes', {cellfun(@localModeOf, readings.channels, 'UniformOutput', false)});
    out = localAddCurves(out, score, X, segment_ends, 'recording', settings, readings);
    if ismember('segment', settings.centrings)
        out = localAddCurves(out, ...
            averagedRankAutocorrelation(X, segment_ends, max_lag, 'score', 'segment'), ...
            X, segment_ends, 'segment', settings, readings);
    end

    %% 6.2 Save -- explicit list, nothing duplicated from Phase 1
    out.settings = settings;
    out.settings_identity = identity;
    out.phase1_identity = phase1_identity;
    out.published_tables = localPublishedTables();
    localSaveAtomic(mat_path, out);
    fprintf('[Phase6] saved %s\n', mat_path);

    %% 6.3 CSVs, figures and the provenance stamp
    localPublishRecording(ctx, rec, out, settings, steps, identity, rel_base, ...
        started_utc);
end


function localPublishRecording(ctx, rec, out, settings, steps, identity, ...
        rel_base, started_utc)
    % The three CSVs, the figures and the provenance stamp, in the one order that makes run_config.json a completion marker: the mat (already written by the caller), then the CSVs, then the figures, then the stamp.
    % Every table is a projection of OUT, so a fresh run and a republish write the same files.
    summary = localSummaryTable(out);
    writeTableAtomic(summary, fullfile(ctx.text_dir, 'timescale_summary.csv'));
    writeTableAtomic(localCurvesTable(out), fullfile(ctx.text_dir, 'acf_curves.csv'));
    writeTableAtomic(localChannelTable(out), ...
        fullfile(ctx.text_dir, 'timescale_by_channel.csv'));
    fprintf(['[Phase6] %s: tau_env %.4g s (%.4g samples), tau_amp %.4g s ' ...
        '(%.4g samples), ratio %.3g, Gaussian reference %.4g s, excess %.3g ' ...
        '(integrated %.3g) | samples per tau %.4g | T/tau %.4g\n'], ...
        rec.dataset_key, summary.tau_env_s, summary.tau_env_samples, ...
        summary.tau_amp_s, summary.tau_amp_samples, summary.ratio_amp_env, ...
        summary.tau_amp_ref_s, summary.amp_excess, summary.amp_excess_int, ...
        summary.samples_per_tau, summary.T_over_tau);
    if out.n_segments > 1
        fprintf(['[Phase6] %s: centred within each of its %d segments, excess %.3g ' ...
            'against the calibrated Gaussian reference (integrated excess ' ...
            'withheld: no integrated time is defined within segments)\n'], ...
            rec.dataset_key, out.n_segments, summary.amp_excess_trialc);
    end

    % The figures never fail the phase.
    if steps.render_figures
        status = renderPhaseFigures('render_phase6_figures', ctx.base, ...
            ctx.images_dir, ctx.text_dir);
        if isstruct(status) && isfield(status, 'ok') && status.ok
            fprintf('[Phase6] %d figure file(s)\n', status.n_figures);
        else
            fprintf(2, ['[Phase6] figures did not complete; the mat and the ' ...
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


function [X, fs, channel_ids, segment_ends] = localLoadPhase1Signal(rec)
    % The analysis input of one recording, checked against what its sidecar declares.
    % A transposed matrix does not error downstream but silently yields the autocorrelation along the wrong axis, so the orientation is checked here against the sample and channel counts Phase 1 recorded rather than guessed from the shape.
    % segment_ends are the cumulative trial ends Phase 1 persisted, or the sample count for a continuous recording.
    stored = whos('-file', rec.phase1_mat_abs);
    names = {stored.name};
    wanted = {'X_filtered', 'fs', 'channel_ids'};
    missing = wanted(~ismember(wanted, names));
    assert(isempty(missing), 'Phase6:MissingPhase1Variable', ...
        'Phase 1 output %s holds no %s.', rec.phase1_mat_abs, strjoin(missing, ', '));
    if ismember('segment_boundaries', names)
        wanted{end + 1} = 'segment_boundaries';
    end

    loaded = load(rec.phase1_mat_abs, wanted{:});
    X = double(loaded.X_filtered);
    fs = double(loaded.fs);
    channel_ids = double(loaded.channel_ids(:)).';
    assert(ismatrix(X) && size(X, 1) == rec.n_samples && ...
        size(X, 2) == rec.n_channels_retained, 'Phase6:UnexpectedSignalShape', ...
        ['X_filtered of %s must be [samples x channels] = [%d x %d], as its ' ...
        'recording.json declares; %s holds %d x %d.'], ...
        rec.dataset_key, rec.n_samples, rec.n_channels_retained, ...
        rec.phase1_mat_abs, size(X, 1), size(X, 2));
    assert(isreal(X) && all(isfinite(X), 'all'), 'Phase6:NonfiniteSignal', ...
        'X_filtered of %s holds non-finite or complex values: %s', ...
        rec.dataset_key, rec.phase1_mat_abs);
    assert(numel(channel_ids) == size(X, 2), 'Phase6:ChannelIdMismatch', ...
        'X_filtered of %s has %d columns but %d channel IDs: %s', ...
        rec.dataset_key, size(X, 2), numel(channel_ids), rec.phase1_mat_abs);
    assert(isscalar(fs) && isfinite(fs) && fs > 0, 'Phase6:BadSampleRate', ...
        'The sample rate fs of %s must be a positive finite scalar in Hz: %s', ...
        rec.dataset_key, rec.phase1_mat_abs);

    if isfield(loaded, 'segment_boundaries')
        segment_ends = double(loaded.segment_boundaries(:));
    else
        segment_ends = size(X, 1);
    end
    % A trial recording without its boundaries would otherwise be read as one continuous segment, and lagged pairs would silently cross trial ends.
    if isfield(rec, 'n_segments')
        assert(numel(segment_ends) == rec.n_segments, 'Phase6:SegmentCountMismatch', ...
            ['%s declares %d segments in recording.json but %s yields %d segment ' ...
            'end(s).'], rec.dataset_key, rec.n_segments, rec.phase1_mat_abs, ...
            numel(segment_ends));
    end
    if isfield(rec, 'fs')
        assert(abs(fs - rec.fs) <= 1e-9 * max(1, rec.fs), 'Phase6:SampleRateMismatch', ...
            '%s declares fs = %g Hz in recording.json but %s holds fs = %g Hz.', ...
            rec.dataset_key, rec.fs, rec.phase1_mat_abs, fs);
    end
end


function readings = localReadings(score, floor_level)
    % How every curve is read: the channel average's reading from the channel-mean score autocorrelation, and each channel's from its own ([] for a constant one).
    n_channels = size(score.acf_by_channel, 2);
    readings = struct('mean', envelopeReading(score.acf_mean, floor_level), ...
        'channels', {cell(n_channels, 1)});
    for c = 1:n_channels
        if ~isnan(score.acf_by_channel(1, c))
            readings.channels{c} = envelopeReading(score.acf_by_channel(:, c), floor_level);
        end
    end
end


function mode = localModeOf(reading)
    % A channel's reading mode, '' for a constant channel.
    mode = '';
    if ~isempty(reading)
        mode = reading.mode;
    end
end


function reference = localGaussianSquare(score)
    % The square transform's Gaussian reference in the layout of an averagedRankAutocorrelation output: each channel's squared score autocorrelation, and their channel mean.
    reference = struct( ...
        'lags',             score.lags, ...
        'acf_mean',         score.acf_square_gaussian, ...
        'acf_by_channel',   score.acf_by_channel .^ 2, ...
        'pair_counts',      score.pair_counts, ...
        'n_valid_channels', score.n_valid_channels, ...
        'transform',        'square_gaussian');
end


function out = localAddCurves(out, score, X, segment_ends, centring, settings, readings)
    % The curves of one centring, each read as READINGS (the headline score's) says and stored under its name plus the centring's suffix ('' for 'recording', '_trialc' for 'segment').
    % They are SCORE, the score under that centring, and, when the square is computed, the square under the same centring and its Gaussian reference.
    % The reference comes from SCORE's squared channel autocorrelations (localGaussianSquare), except in the within-trial set of a multi-segment recording, where it is calibrated to carry the pull of centring within each segment (localCalibratedGaussianSquare) and where the integrated times are withheld (localWithholdIntegrated).
    % The diary gets one line per curve and the amplitude excesses, except for the within-trial set of a one-segment recording, which is the headline set again.
    suffix = '';
    if strcmp(centring, 'segment')
        suffix = '_trialc';
    end
    within_trials = ~isempty(suffix) && out.n_segments > 1;
    names = {'score'};
    out.(['score', suffix]) = localTransformResult(score, out.fs, settings, readings);
    if ismember('square', settings.transforms)
        names = {'score', 'square', 'square_gaussian'};
        out.(['square', suffix]) = localTransformResult( ...
            averagedRankAutocorrelation(X, segment_ends, score.lags(end), 'square', ...
            centring), out.fs, settings, readings);
        if within_trials
            reference = localCalibratedGaussianSquare(X, segment_ends, score);
        else
            reference = localGaussianSquare(score);
        end
        out.(['square_gaussian', suffix]) = localTransformResult(reference, out.fs, ...
            settings, readings);
    end
    if within_trials
        for name = strcat(names, suffix)
            out.(name{1}) = localWithholdIntegrated(out.(name{1}));
        end
    end

    if ~isempty(suffix) && out.n_segments == 1
        fprintf(['[Phase6] one segment: centring within it is the headline centring, ' ...
            'so the %s curves are the headline curves\n'], suffix);
        return;
    end
    for name = strcat(names, suffix)
        localReportTransform(out.(name{1}), name{1});
    end
    if ismember('square', settings.transforms)
        amp = out.(['square', suffix]).fit;
        ref = out.(['square_gaussian', suffix]).fit;
        if within_trials
            fprintf(['[Phase6] amplitude excess%s tau_amp / tau_amp_ref = %.4g against ' ...
                'the calibrated Gaussian reference (%s reading); the integrated ' ...
                'times and excess are withheld: centring within each segment makes ' ...
                'the within-trial autocovariances sum to about zero, so no ' ...
                'integrated time is defined\n'], suffix, ...
                amp.tau_samples / ref.tau_samples, out.reading.mode);
        else
            fprintf(['[Phase6] amplitude excess%s tau_amp / tau_amp_ref = %.4g, ' ...
                'integrated %.4g (%s reading)\n'], suffix, ...
                amp.tau_samples / ref.tau_samples, ...
                amp.tau_int_samples / ref.tau_int_samples, out.reading.mode);
        end
    end
end


function reference = localCalibratedGaussianSquare(X, segment_ends, score)
    % The within-trial Gaussian reference of a multi-segment recording, in the layout of an averagedRankAutocorrelation output.
    % For each channel it is the autocorrelation its square, centred within each segment, would have in expectation if the channel were Gaussian with the autocorrelation its headline (whole-recording-centred) score shows out to the longest segment (segmentCentredGaussianReference); acf_mean is their mean over the channels whose score is not constant.
    % SCORE, the within-trial score, gives the lags and the pair counts.
    longest = max(diff([0; segment_ends]));
    headline = averagedRankAutocorrelation(X, segment_ends, longest - 1, 'score');
    by_channel = segmentCentredGaussianReference(headline.acf_by_channel, segment_ends, ...
        score.lags(end));
    valid = ~isnan(by_channel(1, :));
    reference = struct( ...
        'lags',             score.lags, ...
        'acf_mean',         mean(by_channel(:, valid), 2), ...
        'acf_by_channel',   by_channel, ...
        'pair_counts',      score.pair_counts, ...
        'n_valid_channels', nnz(valid), ...
        'transform',        'square_gaussian');
end


function result = localWithholdIntegrated(result)
    % RESULT with the integrated autocorrelation times of its channel average and of every channel withheld as NaN.
    % Centring within each segment makes a curve's within-trial autocovariances sum to about zero, so it defines no integrated time.
    result.fit.tau_int_samples = NaN;
    result.fit.tau_int_seconds = NaN;
    result.channels.tau_int_samples(:) = NaN;
    result.channels.tau_int_seconds(:) = NaN;
end


function result = localTransformResult(acf, fs, settings, readings)
    % One curve: the channel-averaged autocorrelation and its fit, the across-channel quantiles of each lag, and every channel's own fit as columns, each read as its score says (READINGS).
    % A constant channel carries a NaN autocorrelation and is not fitted.
    lags = acf.lags;
    n_channels = size(acf.acf_by_channel, 2);
    valid = ~isnan(acf.acf_by_channel(1, :));

    fits = cell(n_channels, 1);
    for c = 1:n_channels
        if valid(c)
            fits{c} = localFit(acf.acf_by_channel(:, c), lags, fs, settings, ...
                readings.channels{c});
        else
            fits{c} = localEmptyFit(numel(lags), 'constant_series');
        end
    end
    per_channel = [fits{:}];

    result = struct( ...
        'transform',        acf.transform, ...
        'acf_mean',         acf.acf_mean, ...
        'acf_by_channel',   acf.acf_by_channel, ...
        'acf_quantiles',    quantile(acf.acf_by_channel(:, valid), ...
        settings.channel_quantiles, 2), ...
        'n_valid_channels', acf.n_valid_channels, ...
        'fit',              localFit(acf.acf_mean, lags, fs, settings, readings.mean));
    % Every channel's fit as columns [n_channels x 1], in channel order; the per-channel envelopes and fitted curves are not kept, because the channel's autocorrelation and its tau fix both.
    result.channels = struct( ...
        'tau_samples',      [per_channel.tau_samples]', ...
        'tau_seconds',      [per_channel.tau_seconds]', ...
        'last_lag_samples', [per_channel.last_lag_samples]', ...
        'last_lag_seconds', [per_channel.last_lag_seconds]', ...
        'n_points',         [per_channel.n_points]', ...
        'rmse',             [per_channel.rmse]', ...
        'floor_reached',    [per_channel.floor_reached]', ...
        'mode',             {{per_channel.mode}'}, ...
        'tau_ou_samples',   [per_channel.tau_ou_samples]', ...
        'tau_ou_seconds',   [per_channel.tau_ou_seconds]', ...
        'tau_int_samples',  [per_channel.tau_int_samples]', ...
        'tau_int_seconds',  [per_channel.tau_int_seconds]', ...
        'status',           {{per_channel.status}'}, ...
        'error_id',         {{per_channel.error_id}'}, ...
        'error_message',    {{per_channel.error_message}'});
end


function fit = localFit(acf, lags, fs, settings, reading)
    % fitEnvelopeTimescale on lags in samples, read as READING says, with tau and the fit range also in seconds.
    % The Ornstein-Uhlenbeck-equivalent time and the integrated autocorrelation time over the whole window stand beside it; they need no envelope, so they are kept even when the envelope fit errors.
    % A fit that fitEnvelopeTimescale refuses is recorded (NaN, the status fit_error, its identifier and message, and the reading's mode) instead of stopping the recording.
    fit = localEmptyFit(numel(lags), 'ok');
    fit.mode = reading.mode;
    fit.tau_ou_samples = ouEquivalentTimescale(acf, lags);
    fit.tau_ou_seconds = fit.tau_ou_samples / fs;
    fit.tau_int_samples = integratedAutocorrelationTime(acf, lags);
    fit.tau_int_seconds = fit.tau_int_samples / fs;
    try
        envelope_fit = fitEnvelopeTimescale(acf, lags, settings.envelope_floor, reading);
    catch fit_error
        % Only the estimator's own refusal (an envelope that does not decay like an exponential over its range) is a recorded outcome; any other error is a bug and stops the recording.
        if ~strcmp(fit_error.identifier, 'fitEnvelopeTimescale:BoundaryFit')
            rethrow(fit_error);
        end
        fit.status = 'fit_error';
        fit.error_id = fit_error.identifier;
        fit.error_message = fit_error.message;
        return;
    end
    assert(strcmp(envelope_fit.method, settings.envelope_method), ...
        'Phase6:EnvelopeMethodMismatch', ...
        ['fitEnvelopeTimescale implements the %s envelope but the settings declare ' ...
        '%s; update settings.envelope_method so the identity names the code.'], ...
        envelope_fit.method, settings.envelope_method);
    fit.mode = envelope_fit.mode;
    fit.tau_samples = envelope_fit.tau;
    fit.tau_seconds = envelope_fit.tau / fs;
    fit.envelope = envelope_fit.envelope;
    fit.fitted = envelope_fit.fitted;
    fit.last_lag_samples = envelope_fit.last_lag;
    fit.last_lag_seconds = envelope_fit.last_lag / fs;
    fit.n_points = envelope_fit.n_points;
    fit.rmse = envelope_fit.rmse;
    fit.floor_reached = double(envelope_fit.floor_reached);
    fit.method = envelope_fit.method;
end


function fit = localEmptyFit(n_lags, status)
    % The record of a fit that did not produce a number: NaN everywhere, the floor_reached flag included (0 or 1 when a fit ran, NaN when none did).
    fit = struct( ...
        'tau_samples',      NaN, ...
        'tau_seconds',      NaN, ...
        'envelope',         nan(n_lags, 1), ...
        'fitted',           nan(n_lags, 1), ...
        'last_lag_samples', NaN, ...
        'last_lag_seconds', NaN, ...
        'n_points',         NaN, ...
        'rmse',             NaN, ...
        'floor_reached',    NaN, ...
        'method',           '', ...
        'mode',             '', ...
        'tau_ou_samples',   NaN, ...
        'tau_ou_seconds',   NaN, ...
        'tau_int_samples',  NaN, ...
        'tau_int_seconds',  NaN, ...
        'status',           status, ...
        'error_id',         '', ...
        'error_message',    '');
end


function localReportTransform(result, label)
    % One diary line per curve, named LABEL, plus one per distinct per-channel failure.
    fit = result.fit;
    if strcmp(fit.status, 'ok')
        fprintf(['[Phase6] %-6s | tau %.4g samples = %.4g s (%s) | fit to lag %d ' ...
            '(%.3f s), rmse %.3g, floor reached %d | OU-equivalent %.4g s\n'], ...
            label, fit.tau_samples, fit.tau_seconds, fit.mode, ...
            fit.last_lag_samples, fit.last_lag_seconds, fit.rmse, fit.floor_reached, ...
            fit.tau_ou_seconds);
    else
        fprintf(2, '[Phase6] %-6s | the channel-average fit FAILED: %s (%s)\n', ...
            label, fit.error_id, fit.error_message);
    end
    status = result.channels.status;
    fitted = strcmp(status, 'ok');
    tau = [result.channels.tau_seconds(fitted); NaN];   % NaN keeps an empty set printable
    fprintf(['[Phase6] %-6s | channels fitted %d of %d | per-channel tau median ' ...
        '%.4g s, range [%.4g, %.4g] s\n'], label, nnz(fitted), ...
        numel(status), median(tau, 'omitnan'), min(tau), max(tau));
    if any(~fitted)
        reasons = strtrim(strcat(status(~fitted), {' '}, result.channels.error_id(~fitted)));
        [labels, ~, index] = unique(reasons);
        for k = 1:numel(labels)
            fprintf('[Phase6] %-6s | %d channel(s) not fitted: %s\n', ...
                label, nnz(index == k), labels{k});
        end
    end
end


function T = localSummaryTable(out)
    % One row per recording: the headline columns, then the within-trial twin of every time-scale column, suffixed _trialc.
    % The recording's metadata, the channel count and the reading modes are shared by the two centrings and are not repeated.
    T = localSummaryColumns(out, '');
    shared = {'dataset_key', 'modality', 'fs', 'n_samples', 'duration_s', ...
        'n_channels', 'n_segments', 'max_lag_samples', 'max_lag_s', ...
        'envelope_floor', 'n_valid_channels', 'env_fit_mode', 'amp_fit_mode'};
    trial = removevars(localSummaryColumns(out, '_trialc'), shared);
    trial.Properties.VariableNames = strcat(trial.Properties.VariableNames, '_trialc');
    T = [T, trial];
end


function T = localSummaryColumns(out, suffix)
    % The summary row of the curves of one centring (SUFFIX '' for the headline, '_trialc' for the within-trial set), under the headline's column names.
    % tau_env is the score transform's time scale and tau_amp the square transform's.
    % A curve that was not computed, or whose channel-average fit errored, leaves its columns NaN and names the error.
    env = localRoleFit(out, ['score', suffix]);
    amp = localRoleFit(out, ['square', suffix]);
    ref = localRoleFit(out, ['square_gaussian', suffix]);
    % Channels valid under every transform computed (a constant channel is constant under both), counted on the headline curves.
    n_valid = min(cellfun(@(name) out.(name).n_valid_channels, out.settings.transforms));
    T = table( ...
        string(out.dataset_key), string(out.modality), out.fs, out.n_samples, ...
        out.duration_seconds, out.n_channels, out.n_segments, ...
        out.max_lag_samples, out.max_lag_samples / out.fs, ...
        out.settings.envelope_floor, ...
        env.tau_samples, env.tau_seconds, amp.tau_samples, amp.tau_seconds, ...
        amp.tau_seconds / env.tau_seconds, out.fs * env.tau_seconds, ...
        out.duration_seconds / env.tau_seconds, ...
        env.last_lag_seconds, env.rmse, env.floor_reached, ...
        amp.last_lag_seconds, amp.rmse, amp.floor_reached, n_valid, ...
        string(env.error_id), string(amp.error_id), ...
        string(env.mode), string(amp.mode), ...
        env.tau_ou_samples, env.tau_ou_seconds, amp.tau_ou_samples, amp.tau_ou_seconds, ...
        amp.tau_ou_seconds / env.tau_ou_seconds, ...
        ref.tau_samples, ref.tau_seconds, amp.tau_seconds / ref.tau_seconds, ...
        ref.last_lag_seconds, ref.rmse, ref.floor_reached, string(ref.error_id), ...
        amp.tau_int_samples, amp.tau_int_seconds, ref.tau_int_samples, ...
        ref.tau_int_seconds, amp.tau_int_seconds / ref.tau_int_seconds, ...
        env.n_points, amp.n_points, ref.n_points, ...
        'VariableNames', {'dataset_key', 'modality', 'fs', 'n_samples', ...
        'duration_s', 'n_channels', 'n_segments', 'max_lag_samples', ...
        'max_lag_s', 'envelope_floor', 'tau_env_samples', 'tau_env_s', ...
        'tau_amp_samples', 'tau_amp_s', 'ratio_amp_env', 'samples_per_tau', ...
        'T_over_tau', ...
        'env_fit_last_lag_s', 'env_fit_rmse', 'env_floor_reached', ...
        'amp_fit_last_lag_s', 'amp_fit_rmse', 'amp_floor_reached', ...
        'n_valid_channels', 'env_fit_error', 'amp_fit_error', ...
        'env_fit_mode', 'amp_fit_mode', ...
        'tau_env_ou_samples', 'tau_env_ou_s', 'tau_amp_ou_samples', ...
        'tau_amp_ou_s', 'ratio_amp_env_ou', ...
        'tau_amp_ref_samples', 'tau_amp_ref_s', 'amp_excess', ...
        'amp_ref_fit_last_lag_s', 'amp_ref_fit_rmse', 'amp_ref_floor_reached', ...
        'amp_ref_fit_error', ...
        'tau_amp_int_samples', 'tau_amp_int_s', 'tau_amp_ref_int_samples', ...
        'tau_amp_ref_int_s', 'amp_excess_int', ...
        'env_fit_n_points', 'amp_fit_n_points', 'amp_ref_fit_n_points'});
end


function names = localCurveNames(out)
    % The curves this recording carries, in table order: the transforms, then the square's Gaussian reference when the square was computed.
    names = reshape(out.settings.transforms, 1, []);
    if isfield(out, 'square_gaussian')
        names{end + 1} = 'square_gaussian';
    end
end


function fit = localRoleFit(out, transform)
    % The channel-average fit of one curve, or the empty record with the error not_computed when this run's settings did not compute the curve.
    if isfield(out, transform)
        fit = out.(transform).fit;
    else
        fit = localEmptyFit(numel(out.lags_samples), 'not_computed');
        fit.error_id = 'not_computed';
    end
end


function T = localCurvesTable(out)
    % One row per lag: the channel-average autocorrelation of each curve, its envelope and fitted exponential, and its across-channel quantiles, then the same columns of the within-trial curves, suffixed _trialc.
    T = table(out.lags_samples, out.lags_seconds, out.pair_counts, ...
        'VariableNames', {'lag_samples', 'lag_s', 'pair_count'});
    levels = out.settings.channel_quantiles;
    suffixes = {''};
    if isfield(out, 'score_trialc')
        suffixes{end + 1} = '_trialc';
    end
    for suffix = suffixes
        for transform = localCurveNames(out)
            name = transform{1};
            result = out.([name, suffix{1}]);
            T.(['acf_', name, suffix{1}]) = result.acf_mean;
            T.(['envelope_', name, suffix{1}]) = result.fit.envelope;
            T.(['fitted_', name, suffix{1}]) = result.fit.fitted;
            for q = 1:numel(levels)
                T.(sprintf('acf_%s_q%02d%s', name, round(100 * levels(q)), suffix{1})) = ...
                    result.acf_quantiles(:, q);
            end
        end
    end
end


function T = localChannelTable(out)
    % One row per channel: the headline columns, then the within-trial twin of every column but the reading modes, which the two centrings share, suffixed _trialc.
    T = localChannelColumns(out, '');
    if isfield(out, 'score_trialc')
        trial = localChannelColumns(out, '_trialc');
        names = trial.Properties.VariableNames;
        trial = trial(:, ~(strcmp(names, 'channel_id') | endsWith(names, '_fit_mode')));
        trial.Properties.VariableNames = strcat(trial.Properties.VariableNames, '_trialc');
        T = [T, trial];
    end
end


function T = localChannelColumns(out, suffix)
    % The per-channel columns of the curves of one centring (SUFFIX '' for the headline, '_trialc' for the within-trial set), under the headline's column names.
    % They hold each channel's own tau of each curve, with the fit status and the error identifier of a fit that did not produce one, the fitted and integrated amplitude excesses, and the number of lags each fit read.
    roles = struct('score', 'env', 'square', 'amp', 'square_gaussian', 'amp_ref');
    T = table(reshape(out.channel_ids, [], 1), 'VariableNames', {'channel_id'});
    for transform = localCurveNames(out)
        role = roles.(transform{1});
        channels = out.([transform{1}, suffix]).channels;
        T.(['tau_', role, '_samples']) = channels.tau_samples;
        T.(['tau_', role, '_s']) = channels.tau_seconds;
        T.([role, '_fit_last_lag_s']) = channels.last_lag_seconds;
        T.([role, '_fit_rmse']) = channels.rmse;
        T.([role, '_floor_reached']) = channels.floor_reached;
        T.([role, '_status']) = channels.status;
        T.([role, '_error']) = channels.error_id;
        T.([role, '_fit_mode']) = channels.mode;
        T.(['tau_', role, '_ou_s']) = channels.tau_ou_seconds;
    end
    if isfield(out, 'square_gaussian')
        T.amp_excess = T.tau_amp_s ./ T.tau_amp_ref_s;
        T.tau_amp_int_samples = out.(['square', suffix]).channels.tau_int_samples;
        T.tau_amp_int_s = out.(['square', suffix]).channels.tau_int_seconds;
        T.tau_amp_ref_int_samples = out.(['square_gaussian', suffix]).channels.tau_int_samples;
        T.tau_amp_ref_int_s = out.(['square_gaussian', suffix]).channels.tau_int_seconds;
        T.amp_excess_int = T.tau_amp_int_s ./ T.tau_amp_ref_int_s;
    end
    for transform = localCurveNames(out)
        T.([roles.(transform{1}), '_fit_n_points']) = ...
            out.([transform{1}, suffix]).channels.n_points;
    end
end


function modality = localModality(rec)
    % The modality recording.json declares, or '' for a sidecar without one.
    modality = '';
    if isfield(rec, 'modality')
        modality = char(string(rec.modality));
    end
end


function [identity, phase1_identity] = localIdentity(fields, rec)
    % The identity of one recording's time scales: this phase's settings plus the identity and the schema of the Phase 1 matrix they are computed from.
    required = {'settings_identity', 'phase1_schema_version'};
    missing = required(~isfield(rec, required));
    assert(isempty(missing), 'Phase6:MissingPhase1Identity', ...
        ['The recording.json of %s carries no %s; rerun Phase 1 for it, whose ' ...
        'sidecar records both.'], rec.dataset_key, strjoin(missing, ', '));
    phase1_identity = char(string(rec.settings_identity));
    fields.phase1_identity = phase1_identity;
    fields.phase1_schema_version = char(string(rec.phase1_schema_version));
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
    error('Phase6:SettingsIdentityMismatch', ...
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
    % The tables one recording's time scales publish, declared here and nowhere else.
    % timescale.mat records the list, localPublishRecording writes the tables, and the recording counts as finished only when all of them exist.
    tables = {'timescale_summary.csv', 'acf_curves.csv', 'timescale_by_channel.csv'};
end


function done = localIsPublished(mat_path, text_dir)
    % The time scales are published when their mat lists the tables of its publication, every one of them exists, and the provenance stamp, written last, exists.
    tables = localStoredVariable(mat_path, 'published_tables', {});
    done = ~isempty(tables) && isfile(fullfile(text_dir, 'run_config.json')) ...
        && all(cellfun(@(name) isfile(fullfile(text_dir, name)), tables));
end


function localWithdrawPublication(ctx)
    % Under TC_OVERWRITE=1, withdraw the recording's previous publication (the mat, the tables and both stamps) before anything is recomputed, so an interrupted overwrite can never be read as the finished one.
    localDeleteFiles([ ...
        {fullfile(ctx.mat_dir, 'timescale.mat')}, ...
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


function localSaveAtomic(path, payload)
    % Save every field of PAYLOAD as its own top-level variable, through a temporary file and a rename, so an interrupted save cannot leave a half-written mat that the skip rule would read as a finished recording.
    % The staging name carries a per-call token, so two writers of one destination can never delete or overwrite each other's half-written file.
    [~, token] = fileparts(tempname);
    tmp = [path, '.', token, '.tmp.mat'];
    try
        save(tmp, '-struct', 'payload', '-v7.3');
    catch save_error
        localDeleteFiles({tmp});
        rethrow(save_error);
    end
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'Phase6:MoveFailed', 'Could not move %s into place: %s', tmp, msg);
end
