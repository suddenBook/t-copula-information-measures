% PREPARE_LFP Monkey LFP preparation: trial-segmented, band-passed LFP of monkey L or N as Phase 1 input.
%
% By default the script runs the fixed, non-interactive production preparation: monkey L or N, [12, 25] Hz, TS-ON -> CUE-ON, with the band-pass applied separately inside every retained trial.
% Set TC_LFP_MONKEY=L|N to select the recording (default L) and TC_LFP_OUTPUT_ROOT to choose a scratch output root.
% TC_LFP_INPUT_ROOT may point at another directory that holds the two extracted *_lfp.mat files.
% The default output root is data/monkey_LFP/reproduced/, so the script never overwrites the Phase 1 inputs by accident.
% Optional TC_LFP_FILTER_RANGE, TC_LFP_START_TAG and TC_LFP_END_TAG values create an explicitly labelled custom run; the canonical reproduction is the default.
% The canonical run reproduces the Phase 1 inputs data/monkey_LFP/preprocessed/monkey_{L,N}/baseline_correct_epoch.mat, and verify_lfp_reproduction.m compares a reproduced file with them.
%
% The script loads the extracted LFP and events from data/monkey_LFP/raw/lfp_extracted/<session_id>_lfp.mat, written by extract_lfp_from_nix.py from the dataset's NIX files.
% It removes structurally invalid trials, applies the dataset's signal-quality rejections, keeps behaviourally correct and error trials apart, cuts each trial between two configured event tags, band-passes each trial segment on its own and saves tag-named outputs for downstream analyses.
%
% Key design choices:
%   * trial-wise products are the primary outputs
%   * pooled epoch files are convenience derivatives for state-level analyses
%   * signal-quality rejection is filter-aware: only the odML/NIX quality bands that overlap the analysis band are applied
%   * output lives under TC_LFP_OUTPUT_ROOT (default data/monkey_LFP/reproduced/) and never under results/
%   * other bands and tags are set only through explicit environment variables
%   * the chosen interval is exported separately for behaviourally correct and behaviourally error trials
%
% Dataset: the Brochier et al. reach-to-grasp dataset, recorded with a 96-channel Utah array in macaque motor cortex during an instructed delayed reach-to-grasp task with four trial types.
%
% Outputs (under <TC_LFP_OUTPUT_ROOT>/monkey_N/ or monkey_L/):
%   whole_session.mat                    - raw LFP after the applicable channel rejection
%   trial_table.mat                      - decoded per-trial event table and bookkeeping, including correct_mask and error_mask
%   <start>_to_<end>_correct_trials.mat  - correct-trial segments (raw and filtered)
%   <start>_to_<end>_correct_epoch.mat   - pooled filtered correct-trial product
%   <start>_to_<end>_error_trials.mat    - error-trial segments (raw and filtered)
%   <start>_to_<end>_error_epoch.mat     - pooled filtered error-trial product
%   ..._segment_zscored variants         - the same products z-scored per segment and channel (segment_standardization_mode)
%   log.txt                              - console log of the run
%
% The canonical run also copies its TS-ON -> CUE-ON correct-trial pooled product to baseline_correct_epoch.mat, the Phase 1 input name.
% Phase 1 does not filter this recording again, because the band-pass is applied here within each trial segment.

clearvars;
close all;
clc;

%% Step 0: Configuration

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

% ---- Monkey selection (TC_LFP_MONKEY, default L) ----
monkey = upper(strtrim(getenv('TC_LFP_MONKEY')));
if isempty(monkey)
    monkey = 'L';
end

switch monkey
    case 'N'
        session_id = 'i140703-001';
        monkey_name = 'N (Nikos)';
    case 'L'
        session_id = 'l101210-001';
        monkey_name = 'L (Lilou)';
    otherwise
        error('Unknown monkey: %s. Use ''N'' or ''L''.', monkey);
end

input_root = strtrim(getenv('TC_LFP_INPUT_ROOT'));
if isempty(input_root)
    input_root = fullfile(worktree_root, 'data', 'monkey_LFP', 'raw', ...
                          'lfp_extracted');
end
output_root = strtrim(getenv('TC_LFP_OUTPUT_ROOT'));
if isempty(output_root)
    output_root = fullfile(worktree_root, 'data', 'monkey_LFP', 'reproduced');
end
data_file = fullfile(input_root, [session_id '_lfp.mat']);
output_dir = fullfile(output_root, sprintf('monkey_%s', monkey));

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

diary(fullfile(output_dir, 'log.txt'));

% Frozen production values; explicit environment overrides create a custom run.
filter_range = [12, 25];  % Hz
filter_text = strtrim(getenv('TC_LFP_FILTER_RANGE'));
if ~isempty(filter_text)
    filter_values = sscanf(regexprep(filter_text, '[,;]', ' '), '%f').';
    assert(numel(filter_values) == 2 && all(isfinite(filter_values)) && ...
           filter_values(1) > 0 && filter_values(2) > filter_values(1), ...
           'prepare_lfp:InvalidFilterRange', ...
           'TC_LFP_FILTER_RANGE must contain two increasing positive numbers.');
    filter_range = filter_values;
end
filter_order = 4;         % Butterworth order (doubled by filtfilt)
segment_standardization_mode = 'zscore_per_segment_channel';  % or 'none'; this mode also writes the _segment_zscored products
allowed_segment_standardization_modes = {'none', 'zscore_per_segment_channel'};
beta_band = [15, 30];  % Hz; used only by the PSD diagnostic
configured_start_tag = strtrim(getenv('TC_LFP_START_TAG'));
configured_end_tag = strtrim(getenv('TC_LFP_END_TAG'));
if isempty(configured_start_tag)
    configured_start_tag = 'TS-ON';
end
if isempty(configured_end_tag)
    configured_end_tag = 'CUE-ON';
end
canonical_mode = isequal(filter_range, [12, 25]) && ...
    strcmp(configured_start_tag, 'TS-ON') && ...
    strcmp(configured_end_tag, 'CUE-ON');

if ~ismember(segment_standardization_mode, allowed_segment_standardization_modes)
    error('Unknown segment_standardization_mode: %s', segment_standardization_mode);
end

% Frequency bands (Hz) of the dataset's electrode and trial rejection flags: low (LFC), intermediate (IFC) and high (HFC).
band_defs = struct( ...
                   'LFC', [3, 10], ...
                   'IFC', [12, 40], ...
                   'HFC', [60, 250]);
band_order = {'LFC', 'IFC', 'HFC'};

event_catalog = buildEventCatalog();

fprintf('=== Monkey %s LFP Data Preparation ===\n', monkey_name);
fprintf('Source: %s\n', data_file);
fprintf('Filter: [%d, %d] Hz, order %d\n', filter_range(1), filter_range(2), filter_order);
fprintf('Mode: %s\n', ternary(canonical_mode, ...
                              'canonical non-interactive', 'custom non-interactive'));
fprintf('Structural validity gate: ON (frozen)\n');
fprintf('Output directory: %s\n', output_dir);
fprintf('Results/ is not touched by this script.\n\n');

%% Step 1: Load flat-schema extracted LFP + events

fprintf('Loading extracted LFP + events (flat schema)...\n');
if ~exist(data_file, 'file')
    error(['Extracted file not found: %s\n', ...
           'Run extract_lfp_from_nix.py first.'], data_file);
end
d = load(data_file);

% --- Neural LFP ---
X = d.X;                                % [samples x channels], single, uV
fs = double(d.fs);                      % Hz
assert(filter_range(2) < fs / 2, 'prepare_lfp:FilterAboveNyquist', ...
       'Filter upper edge %.6g Hz must be below Nyquist %.6g Hz.', ...
       filter_range(2), fs / 2);
channel_ids_all = double(d.channel_ids(:)).';    % row vector, sorted 1..96
t_start = double(d.t_start);            % seconds (explicit)
signal_units = char(d.signal_units);    % 'uV'
connector_aligned_ids_all = double(d.connector_aligned_ids(:)).';
electrode_coords_mm_all = double(d.electrode_coords_mm);
electrode_reject_HFC = logical(d.electrode_reject_HFC(:)).';
electrode_reject_LFC = logical(d.electrode_reject_LFC(:)).';
electrode_reject_IFC = logical(d.electrode_reject_IFC(:)).';

% --- Behavioural analog signals (only their size, rate and unit are logged) ---
X_behav = d.X_behav;
behav_fs = double(d.behav_fs);
behav_channel_ids = double(d.behav_channel_ids(:)).';
behav_channel_names = d.behav_channel_names;
behav_signal_units = char(d.behav_signal_units);
behav_t_start = double(d.behav_t_start);

% --- Unpack trial_events into vectors ---
assert(strcmp(char(d.trial_events_units), 's'), ...
       'Expected trial_events_units == ''s'' in extracted file, got %s', ...
       char(d.trial_events_units));
te = d.trial_events;
ev_time_sec = double(te.time_sec(:));
ev_label = cellstr(te.label(:));
ev_trial_id = double(te.trial_id(:));
ev_trial_type = cellstr(te.trial_type(:));
ev_performance_str = cellstr(te.performance_str(:));
ev_performance_code = double(te.performance_code(:));
ev_reject_HFC = logical(te.reject_HFC(:));
ev_reject_LFC = logical(te.reject_LFC(:));
ev_reject_IFC = logical(te.reject_IFC(:));
N_events = length(ev_time_sec);

source_metadata = d.metadata;
clear d te;

n_samples = size(X, 1);
n_channels_all = size(X, 2);

fprintf('LFP: [%d samples x %d channels], fs = %g Hz, duration = %.1f s\n', ...
        n_samples, n_channels_all, fs, n_samples / fs);
fprintf('Units: %s, t_start = %.6f s\n', signal_units, t_start);
fprintf('Channel IDs: %d to %d\n', channel_ids_all(1), channel_ids_all(end));
fprintf('Behavioural analog: [%d samples x %d channels], fs = %g Hz, units = %s\n', ...
        size(X_behav, 1), size(X_behav, 2), behav_fs, behav_signal_units);
fprintf('TrialEvents: %d events\n\n', N_events);

%% Step 1b: Build raw band-specific quality info

quality_info = struct();
quality_info.band_ranges_hz = band_defs;
quality_info.electrodes.HFC = channel_ids_all(electrode_reject_HFC);
quality_info.electrodes.LFC = channel_ids_all(electrode_reject_LFC);
quality_info.electrodes.IFC = channel_ids_all(electrode_reject_IFC);
quality_info.electrodes.union_all_bands = channel_ids_all( ...
                                                          electrode_reject_HFC | electrode_reject_LFC | electrode_reject_IFC);

mask_valid_ev = ev_trial_id > 0;
quality_info.trials.HFC = reshape(unique(ev_trial_id(ev_reject_HFC & mask_valid_ev)), 1, []);
quality_info.trials.LFC = reshape(unique(ev_trial_id(ev_reject_LFC & mask_valid_ev)), 1, []);
quality_info.trials.IFC = reshape(unique(ev_trial_id(ev_reject_IFC & mask_valid_ev)), 1, []);
quality_info.trials.union_all_bands = reshape(unique([ ...
                                                      quality_info.trials.HFC, ...
                                                      quality_info.trials.LFC, ...
                                                      quality_info.trials.IFC]), 1, []);

%% Step 2: Convert event times to LFP sample indices

% Event times (s) become 1-based sample indices of X, counted from the LFP start time t_start.
event_samples = round((ev_time_sec - t_start) * fs) + 1;

unique_labels = unique(ev_label);
fprintf('Unique event labels (%d types):\n', length(unique_labels));
for i = 1:length(unique_labels)
    n = sum(strcmp(ev_label, unique_labels{i}));
    fprintf('  %-20s  %d\n', unique_labels{i}, n);
end
fprintf('\n');

%% Step 2b: Structural-validity rejection (before the correct/error split)

% A trial with a TS-ON-ERROR event or the trial type NONE is malformed and removed.
unique_trial_ids_all = reshape(unique(ev_trial_id(ev_trial_id > 0)), 1, []);
n_total_pre_structural = length(unique_trial_ids_all);

structurally_rejected = [];
structurally_rejected_reasons = {};
for i = 1:length(unique_trial_ids_all)
    tid = unique_trial_ids_all(i);
    mask_i = (ev_trial_id == tid);
    labels_i = ev_label(mask_i);
    ttypes_i = ev_trial_type(mask_i);
    has_ts_on_error = any(strcmp(labels_i, 'TS-ON-ERROR'));
    is_none_type = any(strcmp(ttypes_i, 'NONE'));
    if has_ts_on_error || is_none_type
        structurally_rejected(end + 1) = tid; %#ok<SAGROW>
        if has_ts_on_error
            reason = 'TS-ON-ERROR present';
        else
            reason = 'trial_type==NONE';
        end
        structurally_rejected_reasons{end + 1} = reason; %#ok<SAGROW>
    end
end
unique_trial_ids = setdiff(unique_trial_ids_all, structurally_rejected, 'stable');

fprintf('Structural-validity rejection: removed %d trial(s)\n', ...
        length(structurally_rejected));
if ~isempty(structurally_rejected)
    if length(structurally_rejected) <= 20
        fprintf('  trial ids: %s\n', mat2str(structurally_rejected));
    else
        fprintf('  first 20 trial ids: %s ... (total %d)\n', ...
                mat2str(structurally_rejected(1:20)), length(structurally_rejected));
    end
    reason_counts = containers.Map();
    for r = 1:length(structurally_rejected_reasons)
        key = structurally_rejected_reasons{r};
        if isKey(reason_counts, key)
            reason_counts(key) = reason_counts(key) + 1;
        else
            reason_counts(key) = 1;
        end
    end
    keys_r = reason_counts.keys;
    for r = 1:length(keys_r)
        fprintf('  reason %-22s : %d\n', keys_r{r}, reason_counts(keys_r{r}));
    end
end
fprintf('  n_total: %d -> %d\n\n', n_total_pre_structural, length(unique_trial_ids));

%% Step 3: Build full trial table

n_trials = length(unique_trial_ids);
fprintf('Building trial table for %d trials...\n', n_trials);

% Tags whose first sample and time (s) become trial-table fields, NaN when the trial lacks the tag.
quick_tags = {'TS-ON', 'WS-ON', 'CUE-ON', 'CUE-OFF', 'GO-ON', 'SR', 'RW-ON'};
quick_fields = {'ts_on', 'ws_on', 'cue_on', 'cue_off', 'go_on', 'sr', 'rw_on'};

trials = struct([]);
for t = 1:n_trials
    tid = unique_trial_ids(t);
    mask = (ev_trial_id == tid);

    labels_this = ev_label(mask);
    samples_this = event_samples(mask);
    times_sec_this = ev_time_sec(mask);
    [times_sec_this, sort_idx] = sort(times_sec_this, 'ascend');
    labels_this = labels_this(sort_idx);
    samples_this = samples_this(sort_idx);

    trial_type_candidates = ev_trial_type(mask);
    trial_type_candidates = trial_type_candidates(sort_idx);
    perf_this = ev_performance_str(mask);
    perf_this = perf_this(sort_idx);
    code_this = ev_performance_code(mask);
    code_this = code_this(sort_idx);

    trials(t).trial_id = tid;
    non_empty = trial_type_candidates(~cellfun(@isempty, trial_type_candidates));
    if ~isempty(non_empty)
        trials(t).trial_type = non_empty{1};
    else
        trials(t).trial_type = '';
    end
    trials(t).correct = any(strcmp(perf_this, 'correct_trial'));
    if ~isempty(code_this)
        trials(t).performance_code = code_this(1);
    else
        trials(t).performance_code = NaN;
    end
    trials(t).event_labels = labels_this(:).';
    trials(t).event_samples = samples_this(:).';
    trials(t).event_times_sec = times_sec_this(:).';

    for e = 1:length(quick_tags)
        idx = find(strcmp(labels_this, quick_tags{e}), 1, 'first');
        if ~isempty(idx)
            trials(t).(quick_fields{e}) = samples_this(idx);
            trials(t).([quick_fields{e} '_sec']) = times_sec_this(idx);
        else
            trials(t).(quick_fields{e}) = NaN;
            trials(t).([quick_fields{e} '_sec']) = NaN;
        end
    end
end

correct_mask = [trials.correct];
error_mask = ~correct_mask;
n_behaviorally_correct = sum(correct_mask);
n_behaviorally_error = sum(error_mask);
correct_trials = trials(correct_mask);
error_trials = trials(error_mask);

fprintf('\nTrial summary:\n');
fprintf('  Total trials (post-structural): %d\n', n_trials);
fprintf('  Correct trials:                 %d\n', n_behaviorally_correct);
fprintf('  Error trials:                   %d\n', n_behaviorally_error);
printTrialTypeSummary(correct_trials, 'correct');
printTrialTypeSummary(error_trials, 'error');
fprintf('\n');

coverage_rows = computeTagCoverage(event_catalog, trials, correct_trials, error_trials);
printTagCatalog(coverage_rows);

%% Step 4: PSD inspection before filter-aware quality rejection

fprintf('Selecting representative segments for PSD inspection...\n');
[psd_reference_trials, psd_reference_label] = choosePsdReferenceTrials(correct_trials, trials);
[psd_start_tag, psd_end_tag, psd_segments_raw, valid_psd_ids] = ...
    buildReferencePSDSegments(psd_reference_trials, X, n_samples);

fprintf('PSD reference subset: %s\n', psd_reference_label);
fprintf('PSD reference interval: %s -> %s (%d valid trials)\n', ...
        psd_start_tag, psd_end_tag, length(valid_psd_ids));
fprintf('Computing PSD from representative raw segments for filter range inspection...\n');

n_psd = min(5, length(psd_segments_raw));
psd_segments_raw = psd_segments_raw(1:n_psd);
psd_diag = buildReferencePsdDiagnostic(psd_segments_raw, fs, channel_ids_all, filter_range, beta_band);

fprintf('Filter range set to [%d, %d] Hz (shown in the PSD figure).\n', filter_range(1), filter_range(2));
fprintf('Use the summary curve and top beta channels to choose between a broad band and beta-focused band.\n');
fprintf('Top beta-prominent channels from the reference segments:\n');
for i = 1:numel(psd_diag.top_channel_ids)
    fprintf('  %d. Ch%d  peak=%.1f Hz  peak PSD=%.2f dB  prominence=%.2f dB\n', ...
            i, psd_diag.top_channel_ids(i), psd_diag.top_peak_freqs_hz(i), ...
            psd_diag.top_peak_db(i), psd_diag.top_prominence_db(i));
end
if canonical_mode
    fprintf('Canonical non-interactive run: locked filter_range = [12,25] Hz.\n');
else
    fprintf('Custom non-interactive run: explicit filter range accepted.\n');
end
fprintf('Proceeding with filter_range = [%d, %d] Hz.\n\n', filter_range(1), filter_range(2));

%% Step 5: Filter-aware signal-quality rejection

[applicable_bands, rejected_el, rejected_tr] = computeApplicableQualityRejection( ...
                                                                                 quality_info, filter_range, band_defs, band_order);

quality_info.applied = struct( ...
                              'filter_range', filter_range, ...
                              'applicable_quality_bands', {applicable_bands}, ...
                              'electrode_union', rejected_el, ...
                              'trial_union', rejected_tr);

fprintf('Filter-aware signal-quality rejection:\n');
fprintf('  Applicable quality bands: %s\n', fmt_cellstr(applicable_bands));
fprintf('  Electrode union for this run: %s\n', fmt_int_set(rejected_el));
fprintf('  Trial union for this run:     %s\n\n', fmt_int_set(rejected_tr));

channel_ids = channel_ids_all;
connector_aligned_ids = connector_aligned_ids_all;
electrode_coords_mm = electrode_coords_mm_all;

if ~isempty(rejected_el)
    keep_mask = ~ismember(channel_ids, rejected_el);
    removed_ids = channel_ids(~keep_mask);
    X = X(:, keep_mask);
    channel_ids = channel_ids(keep_mask);
    connector_aligned_ids = connector_aligned_ids(keep_mask);
    electrode_coords_mm = electrode_coords_mm(keep_mask, :);
    fprintf('Signal-quality electrode rejection applied:\n');
    fprintf('  Removed %d electrode(s): %s\n', length(removed_ids), mat2str(removed_ids));
    fprintf('  Remaining: %d channels\n\n', size(X, 2));
else
    fprintf('Signal-quality electrode rejection applied: none.\n\n');
end

n_quality_rejected_correct = 0;
n_quality_rejected_error = 0;
rejected_correct_ids_actual = [];
rejected_error_ids_actual = [];
retained_correct_trials = correct_trials;
retained_error_trials = error_trials;
if ~isempty(rejected_tr)
    correct_trial_ids = [correct_trials.trial_id];
    reject_mask_correct = ismember(correct_trial_ids, rejected_tr);
    n_quality_rejected_correct = sum(reject_mask_correct);
    rejected_correct_ids_actual = correct_trial_ids(reject_mask_correct);
    retained_correct_trials = correct_trials(~reject_mask_correct);

    % Error trials go through the same rejection as correct trials.
    % An empty error_trials leaves the rejected-error bookkeeping empty.
    error_trial_ids = [error_trials.trial_id];
    reject_mask_error = ismember(error_trial_ids, rejected_tr);
    n_quality_rejected_error = sum(reject_mask_error);
    rejected_error_ids_actual = error_trial_ids(reject_mask_error);
    retained_error_trials = error_trials(~reject_mask_error);

    fprintf('Signal-quality trial rejection applied:\n');
    fprintf('  Removed %d trial(s) from correct set: %s\n', ...
            n_quality_rejected_correct, fmt_int_set(sort(rejected_correct_ids_actual)));
    fprintf('  Removed %d trial(s) from error set:   %s\n', ...
            n_quality_rejected_error, fmt_int_set(sort(rejected_error_ids_actual)));
    fprintf('  Remaining correct trials: %d\n', length(retained_correct_trials));
    fprintf('  Remaining error trials:   %d\n\n', length(retained_error_trials));
else
    fprintf('Signal-quality trial rejection applied: none.\n\n');
end

if isempty(retained_correct_trials) && isempty(retained_error_trials)
    error('No retained correct or error trials remain after filter-aware quality rejection.');
end

%% Step 6: Configured non-interactive segmentation

available_tags = {coverage_rows([coverage_rows.present_in_session]).tag};
start_tag = resolveTagChoice(configured_start_tag, available_tags);
end_tag = resolveTagChoice(configured_end_tag, available_tags);
assert(~isempty(start_tag) && ~isempty(end_tag), ...
       'prepare_lfp:ConfiguredTagsMissing', ...
       'Configured %s -> %s tags are unavailable for %s.', ...
       configured_start_tag, configured_end_tag, session_id);

fprintf('Selected interval: %s -> %s\n\n', start_tag, end_tag);

subset_defs = struct( ...
                     'subset_name', {'correct', 'error'}, ...
                     'display_name', {'Correct', 'Error'}, ...
                     'all_trials', {correct_trials, error_trials}, ...
                     'retained_trials', {retained_correct_trials, retained_error_trials}, ...
                     'quality_rejected_trial_ids', {rejected_correct_ids_actual, rejected_error_ids_actual}, ...
                     'n_quality_rejected', {n_quality_rejected_correct, n_quality_rejected_error});

subset_results = cell(numel(subset_defs), 1);
for s = 1:numel(subset_defs)
    subset_results{s} = processTrialSubsetIntervals( ...
                                                    subset_defs(s).retained_trials, X, n_samples, start_tag, end_tag, ...
                                                    fs, filter_range, filter_order, segment_standardization_mode, subset_defs(s).display_name);
end

if ~any(cellfun(@(res) res.has_valid_interval, subset_results))
    error('No retained correct or error trials contain a valid %s -> %s interval.', start_tag, end_tag);
end

%% Step 7: Save outputs

fprintf('Saving outputs to %s\n', output_dir);

filter_params = struct( ...
                       'filter_range', filter_range, ...
                       'filter_order', filter_order, ...
                       'method', 'Butterworth bandpass via filterData.m, applied per trial segment');

common_metadata = struct( ...
                         'monkey', monkey_name, ...
                         'session', session_id, ...
                         'source_file', data_file, ...
                         'signal_units', signal_units, ...
                         't_start', t_start, ...
                         't_start_units', 's', ...
                         'source_extracted_file', source_metadata, ...
                         'quality_rejection', quality_info, ...
                         'n_electrodes_removed', length(rejected_el), ...
                         'n_trials_removed_from_correct', n_quality_rejected_correct, ...
                         'n_trials_removed_from_error', n_quality_rejected_error, ...
                         'n_trials_removed_structural', length(structurally_rejected), ...
                         'structurally_rejected_trial_ids', structurally_rejected, ...
                         'segment_standardization_mode', segment_standardization_mode, ...
                         'preparation_mode', ternary(canonical_mode, ...
                                'canonical_noninteractive', 'custom_noninteractive'), ...
                         'date_created', char(datetime('now')));

metadata = common_metadata;
metadata.description = 'Full session raw LFP after applicable channel rejection. Benchmark/sanity-check path.';
save(fullfile(output_dir, 'whole_session.mat'), ...
     'X', 'fs', 'channel_ids', 'metadata', '-v7.3');
fprintf('  whole_session.mat  [%d x %d]\n', size(X, 1), size(X, 2));

event_info = struct( ...
                    't_start', t_start, ...
                    't_start_units', 's', ...
                    'decoding_method', ['Flat schema from extract_lfp_from_nix.py ', ...
                        '(nix-derived, trial_events scalar struct)'], ...
                    'n_total_pre_structural', n_total_pre_structural, ...
                    'n_total_post_structural', n_trials, ...
                    'n_behaviorally_correct', n_behaviorally_correct, ...
                    'n_behaviorally_error', n_behaviorally_error, ...
                    'n_quality_rejected_from_correct', n_quality_rejected_correct, ...
                    'n_quality_rejected_from_error', n_quality_rejected_error, ...
                    'n_retained_correct_after_quality', length(retained_correct_trials), ...
                    'n_retained_error_after_quality', length(retained_error_trials), ...
                    'structurally_rejected_trial_ids', structurally_rejected, ...
                    'quality_rejection_union_trial_ids', rejected_tr, ...
                    'quality_rejected_from_correct_trial_ids', rejected_correct_ids_actual, ...
                    'quality_rejected_from_error_trial_ids', rejected_error_ids_actual, ...
                    'available_tags_post_structural', {available_tags});
save(fullfile(output_dir, 'trial_table.mat'), ...
     'trials', 'event_info', 'correct_mask', 'error_mask', 'quality_info', 'event_catalog', ...
     'coverage_rows', '-v7.3');
fprintf('  trial_table.mat    %d trials (%d correct, %d error)\n', ...
        n_trials, n_behaviorally_correct, n_behaviorally_error);

slug = intervalSlug(start_tag, end_tag);

for s = 1:numel(subset_defs)
    interval_metadata = common_metadata;
    interval_metadata.segment_start_tag = start_tag;
    interval_metadata.segment_end_tag = end_tag;
    interval_metadata.segment_slug = slug;
    interval_metadata.filter_params = filter_params;
    interval_metadata.trial_subset = subset_defs(s).subset_name;
    interval_metadata.trial_subset_display = subset_defs(s).display_name;
    interval_metadata.n_trials_in_subset_pre_quality = length(subset_defs(s).all_trials);
    interval_metadata.n_trials_removed_from_subset_by_quality = subset_defs(s).n_quality_rejected;
    interval_metadata.quality_rejected_trial_ids = subset_defs(s).quality_rejected_trial_ids;
    interval_metadata.n_input_trials = subset_results{s}.n_input_trials;
    interval_metadata.n_valid = subset_results{s}.n_valid_interval;
    interval_metadata.n_invalid = subset_results{s}.n_invalid_interval;
    interval_metadata.trial_type_counts_pre_quality = buildTrialTypeCountStruct(subset_defs(s).all_trials);
    interval_metadata.trial_type_counts_retained = buildTrialTypeCountStruct(subset_defs(s).retained_trials);
    interval_metadata.trial_type_counts_valid = buildTrialTypeCountStruct(subset_results{s}.valid_trial_types);

    saveIntervalProducts(output_dir, slug, subset_defs(s).subset_name, ...
                         subset_results{s}.valid_segments_raw, subset_results{s}.segments_filtered, ...
                         subset_results{s}.valid_trial_ids, subset_results{s}.valid_trial_types, ...
                         subset_results{s}.valid_start_samples, subset_results{s}.valid_end_samples, ...
                         subset_results{s}.invalid_trial_ids, subset_results{s}.invalid_reasons, ...
                         subset_results{s}.X_pooled, subset_results{s}.segment_boundaries, ...
                         fs, channel_ids, filter_params, interval_metadata);

    fprintf('  %s_%s_trials.mat / %s_%s_epoch.mat\n', ...
            slug, subset_defs(s).subset_name, slug, subset_defs(s).subset_name);

    if strcmp(segment_standardization_mode, 'zscore_per_segment_channel')
        interval_metadata_z = interval_metadata;
        interval_metadata_z.standardization = struct( ...
                                                     'method', 'zscore_per_segment_channel', ...
                                                     'applied_stage', 'after segment filtering before pooling', ...
                                                     'constant_or_near_constant_channels_by_trial', {subset_results{s}.segment_standardization_info});

        saveIntervalProducts(output_dir, slug, subset_defs(s).subset_name, ...
                             subset_results{s}.valid_segments_raw, subset_results{s}.segments_filtered_segment_zscored, ...
                             subset_results{s}.valid_trial_ids, subset_results{s}.valid_trial_types, ...
                             subset_results{s}.valid_start_samples, subset_results{s}.valid_end_samples, ...
                             subset_results{s}.invalid_trial_ids, subset_results{s}.invalid_reasons, ...
                             subset_results{s}.X_pooled_segment_zscored, subset_results{s}.segment_boundaries, ...
                             fs, channel_ids, filter_params, interval_metadata_z, ...
                             'FileSuffix', '_segment_zscored');

        fprintf('  %s_%s_trials_segment_zscored.mat / %s_%s_epoch_segment_zscored.mat\n', ...
                slug, subset_defs(s).subset_name, slug, subset_defs(s).subset_name);
    end
end

if canonical_mode
    canonical_source = fullfile(output_dir, ...
                                [intervalSlug('TS-ON', 'CUE-ON') '_correct_epoch.mat']);
    canonical_target = fullfile(output_dir, 'baseline_correct_epoch.mat');
    assert(isfile(canonical_source), 'prepare_lfp:MissingCanonicalProduct', ...
           'Canonical pooled product was not generated: %s', canonical_source);
    [copy_ok, copy_message] = copyfile(canonical_source, canonical_target, 'f');
    assert(copy_ok, 'prepare_lfp:CanonicalCopyFailed', ...
           'Could not create %s: %s', canonical_target, copy_message);
    fprintf('  baseline_correct_epoch.mat (canonical Phase-1 input alias)\n');
end

%% Step 8: Summary

fprintf('\n=== Monkey %s Preparation Complete ===\n', monkey_name);
fprintf('LFP: [%d x %d], fs = %g Hz, duration = %.1f s, units = %s\n', ...
        n_samples, size(X, 2), fs, n_samples / fs, signal_units);
fprintf('Trials: %d pre-structural, %d post-structural, %d correct, %d error\n', ...
        n_total_pre_structural, n_trials, n_behaviorally_correct, n_behaviorally_error);
fprintf('Retained after quality: %d correct, %d error\n', ...
        length(retained_correct_trials), length(retained_error_trials));
fprintf('Filter: [%d, %d] Hz, order %d (per-segment)\n', ...
        filter_range(1), filter_range(2), filter_order);
fprintf('Applicable quality bands: %s\n', fmt_cellstr(applicable_bands));
fprintf('Selected interval: %s -> %s\n', start_tag, end_tag);
for s = 1:numel(subset_defs)
    fprintf('%s subset valid interval trials: %d / %d\n', ...
            subset_defs(s).display_name, subset_results{s}.n_valid_interval, subset_results{s}.n_input_trials);
end
fprintf('Segment standardization mode: %s\n', segment_standardization_mode);
fprintf('Output files in: %s\n', output_dir);

diary off;

% ---------------------------------------------------------------------
% Local helpers
% ---------------------------------------------------------------------

function catalog = buildEventCatalog()
    % The dataset's event tags with their class (digital, analog-derived or auxiliary), meaning and whether they are recommended as segment boundaries.
    catalog = struct( ...
                     'tag', { ...
                'TS-ON', 'WS-ON', 'CUE-ON', 'CUE-OFF', 'GO-ON', 'SR', 'RW-ON', 'STOP', ...
                'OT', 'OR', 'DO', 'OBB', 'FSRplat-ON', 'FSRplat-OFF', 'HEplat-ON', 'HEplat-OFF', ...
                'TS-OFF', 'TS-ON-ERROR', 'GO/RW-OFF', 'RW-OFF', 'SR-REP', 'RW-ON-REP', ...
                'ERROR-FLASH-ON', 'ERROR-FLASH-OFF', 'NONE'}, ...
                     'class', { ...
                  'digital', 'digital', 'digital', 'digital', 'digital', 'digital', 'digital', 'digital', ...
                  'analog-derived', 'analog-derived', 'analog-derived', 'analog-derived', ...
                  'analog-derived', 'analog-derived', 'analog-derived', 'analog-derived', ...
                  'auxiliary', 'auxiliary', 'auxiliary', 'auxiliary', 'auxiliary', 'auxiliary', ...
                  'auxiliary', 'auxiliary', 'auxiliary'}, ...
                     'meaning', { ...
                    'Trial start generated by LabView.', ...
                    'Warning-signal onset (central LED on).', ...
                    'First cue onset; reveals grip instruction.', ...
                    'First cue offset; starts preparatory delay.', ...
                    'Second cue onset; reveals force and acts as GO signal.', ...
                    'Switch release after GO.', ...
                    'Reward pump on.', ...
                    'Trial stop / end marker.', ...
                    'Object touch.', ...
                    'Object release.', ...
                    'Displacement onset.', ...
                    'Object back to baseline position.', ...
                    'Grip-force plateau onset.', ...
                    'Grip-force plateau offset.', ...
                    'Displacement plateau onset / offline hold-like anchor.', ...
                    'Displacement plateau offset.', ...
                    'Trial-start bit off; bookkeeping event.', ...
                    'Malformed / duplicate TS-ON marker.', ...
                    'Combined GO-off / reward-off digital transition.', ...
                    'Reward off event inferred from digital transitions.', ...
                    'Repeated switch-release-like event.', ...
                    'Repeated reward-on event.', ...
                    'Negative-feedback flash onset on error trials.', ...
                    'Negative-feedback flash offset on error trials.', ...
                    'Placeholder label for malformed / pre-cue aborted trials.'}, ...
                     'recommended', num2cell([ ...
                                 true, true, true, true, true, true, true, true, ...
                                 true, true, true, true, true, true, true, true, ...
                                 false, false, false, false, false, false, false, false, false]));
end

function value = ternary(condition, true_value, false_value)
    if condition
        value = true_value;
    else
        value = false_value;
    end
end

function coverage_rows = computeTagCoverage(catalog, trials, correct_trials, error_trials)
    % Add to each catalog row how many trials (all, correct, error) contain the tag, as counts and percentages.
    coverage_rows = catalog;
    n_trials_total = length(trials);
    n_trials_correct = length(correct_trials);
    n_trials_error = length(error_trials);

    for i = 1:length(catalog)
        tag = catalog(i).tag;
        post_n = sum(arrayfun(@(tr) any(strcmp(tr.event_labels, tag)), trials));
        corr_n = sum(arrayfun(@(tr) any(strcmp(tr.event_labels, tag)), correct_trials));
        err_n = sum(arrayfun(@(tr) any(strcmp(tr.event_labels, tag)), error_trials));
        coverage_rows(i).post_structural_n = post_n;
        coverage_rows(i).correct_n = corr_n;
        coverage_rows(i).error_n = err_n;
        coverage_rows(i).post_structural_pct = pct(post_n, n_trials_total);
        coverage_rows(i).correct_pct = pct(corr_n, n_trials_correct);
        coverage_rows(i).error_pct = pct(err_n, n_trials_error);
        coverage_rows(i).present_in_session = post_n > 0;
    end
end

function printTagCatalog(coverage_rows)
    % Print the catalog rows of the tags present in this session.
    fprintf('Available event tags in this session:\n');
    fprintf('%-18s %-16s %-4s %-18s %-18s %-18s %s\n', ...
            'tag', 'class', 'rec?', 'post-structural', 'correct', 'error', 'meaning');
    for i = 1:length(coverage_rows)
        if ~coverage_rows(i).present_in_session
            continue
        end
        rec = 'no';
        if coverage_rows(i).recommended
            rec = 'yes';
        end
        fprintf('%-18s %-16s %-4s %6d (%5.1f%%) %6d (%5.1f%%) %6d (%5.1f%%) %s\n', ...
                coverage_rows(i).tag, ...
                coverage_rows(i).class, ...
                rec, ...
                coverage_rows(i).post_structural_n, coverage_rows(i).post_structural_pct, ...
                coverage_rows(i).correct_n, coverage_rows(i).correct_pct, ...
                coverage_rows(i).error_n, coverage_rows(i).error_pct, ...
                coverage_rows(i).meaning);
    end
    fprintf('\n');
end

function printTrialTypeSummary(trials_subset, subset_label)
    % Print the number of trials per trial type in one subset.
    if isempty(trials_subset)
        fprintf('  %-10s 0 %s\n', '(none)', subset_label);
        return
    end

    type_info = buildTrialTypeCountStruct(trials_subset);
    if isempty(type_info.labels)
        fprintf('  %-10s %d %s\n', '(untyped)', length(trials_subset), subset_label);
        return
    end

    for i = 1:numel(type_info.labels)
        fprintf('  %-10s %d %s\n', type_info.labels{i}, type_info.counts(i), subset_label);
    end
end

function resolved = resolveTagChoice(choice, available_tags)
    % The available tag equal to choice, else its unique case-insensitive match, else ''.
    resolved = '';
    if isempty(choice)
        return
    end
    exact_match = find(strcmp(available_tags, choice), 1, 'first');
    if ~isempty(exact_match)
        resolved = available_tags{exact_match};
        return
    end
    ci_match = find(strcmpi(available_tags, choice));
    if isscalar(ci_match)
        resolved = available_tags{ci_match};
    end
end

function [segments_raw_all, start_samples_all, end_samples_all, valid_mask, ...
        invalid_trial_ids, invalid_reasons] = extractIntervalSegments(trials_in, X_in, n_samples_in, start_tag, end_tag)
    % Cut each trial from the sample of its first start tag to the sample before its first end tag, clipped to the recording.
    % A trial without a valid interval gets an empty segment, and its ID and reason are listed in invalid_trial_ids and invalid_reasons.

    n_in = length(trials_in);
    segments_raw_all = cell(n_in, 1);
    start_samples_all = nan(n_in, 1);
    end_samples_all = nan(n_in, 1);
    valid_mask = false(n_in, 1);
    invalid_trial_ids = [];
    invalid_reasons = {};

    for k = 1:n_in
        tr = trials_in(k);
        [s1, ok1] = firstOccurrence(tr, start_tag);
        [s2, ok2] = firstOccurrence(tr, end_tag);

        if ~ok1
            invalid_trial_ids(end + 1) = tr.trial_id; %#ok<AGROW>
            invalid_reasons{end + 1} = sprintf('missing start tag %s', start_tag); %#ok<AGROW>
            segments_raw_all{k} = zeros(0, size(X_in, 2), 'single');
            continue
        end
        if ~ok2
            invalid_trial_ids(end + 1) = tr.trial_id; %#ok<AGROW>
            invalid_reasons{end + 1} = sprintf('missing end tag %s', end_tag); %#ok<AGROW>
            segments_raw_all{k} = zeros(0, size(X_in, 2), 'single');
            continue
        end
        if s2 <= s1
            invalid_trial_ids(end + 1) = tr.trial_id; %#ok<AGROW>
            invalid_reasons{end + 1} = sprintf('end tag %s not after start tag %s', end_tag, start_tag); %#ok<AGROW>
            segments_raw_all{k} = zeros(0, size(X_in, 2), 'single');
            continue
        end

        s1 = max(s1, 1);
        s2 = min(s2 - 1, n_samples_in);
        if s2 < s1
            invalid_trial_ids(end + 1) = tr.trial_id; %#ok<AGROW>
            invalid_reasons{end + 1} = 'empty interval after boundary clipping'; %#ok<AGROW>
            segments_raw_all{k} = zeros(0, size(X_in, 2), 'single');
            continue
        end

        segments_raw_all{k} = X_in(s1:s2, :);
        start_samples_all(k) = s1;
        end_samples_all(k) = s2;
        valid_mask(k) = true;
    end
end

function [sample, ok] = firstOccurrence(tr, tag)
    % Sample of the first event with this tag in trial tr, and whether the trial has one.
    idx = find(strcmp(tr.event_labels, tag), 1, 'first');
    if isempty(idx)
        sample = NaN;
        ok = false;
    else
        sample = tr.event_samples(idx);
        ok = true;
    end
end

function result = processTrialSubsetIntervals(trials_subset, X_in, n_samples_in, start_tag, end_tag, ...
                                              fs, filter_range, filter_order, segment_standardization_mode, subset_label)
    % Segment, band-pass and pool one trial subset.
    % Each segment is filtered on its own; a segment too short for filtfilt, or whose filtering fails, stays unfiltered with a warning.
    % The filtered segments, and optionally their per-segment z-scores, are stacked, and segment_boundaries holds the cumulative segment ends.

    n_channels = size(X_in, 2);
    empty_matrix = zeros(0, n_channels, 'single');

    result = struct();
    result.n_input_trials = length(trials_subset);
    result.n_valid_interval = 0;
    result.n_invalid_interval = 0;
    result.has_valid_interval = false;
    result.invalid_trial_ids = [];
    result.invalid_reasons = {};
    result.valid_trial_ids = zeros(0, 1);
    result.valid_trial_types = cell(0, 1);
    result.valid_segments_raw = cell(0, 1);
    result.valid_start_samples = zeros(0, 1);
    result.valid_end_samples = zeros(0, 1);
    result.segment_lengths = zeros(0, 1);
    result.segments_filtered = cell(0, 1);
    result.segments_filtered_segment_zscored = cell(0, 1);
    result.segment_standardization_info = struct('trial_id', {}, 'constant_channels', {});
    result.X_pooled = empty_matrix;
    result.X_pooled_segment_zscored = empty_matrix;
    result.segment_boundaries = zeros(0, 1);

    fprintf('%s subset interval extraction summary:\n', subset_label);
    fprintf('  Input retained trials: %d\n', result.n_input_trials);

    if isempty(trials_subset)
        fprintf('  No trials available in this subset after quality rejection.\n\n');
        return
    end

    [segments_raw_all, start_samples_all, end_samples_all, valid_mask_interval, ...
        invalid_trial_ids, invalid_reasons] = extractIntervalSegments( ...
                                                                      trials_subset, X_in, n_samples_in, start_tag, end_tag);

    trial_ids_retained = [trials_subset.trial_id];
    trial_types_retained = {trials_subset.trial_type}.';

    result.invalid_trial_ids = invalid_trial_ids;
    result.invalid_reasons = invalid_reasons;
    result.n_invalid_interval = length(invalid_trial_ids);
    result.n_valid_interval = sum(valid_mask_interval);

    result.valid_trial_ids = trial_ids_retained(valid_mask_interval).';
    result.valid_trial_types = trial_types_retained(valid_mask_interval);
    result.valid_segments_raw = segments_raw_all(valid_mask_interval);
    result.valid_start_samples = start_samples_all(valid_mask_interval);
    result.valid_end_samples = end_samples_all(valid_mask_interval);

    fprintf('  Valid interval trials: %d\n', result.n_valid_interval);
    fprintf('  Invalid interval trials: %d\n', result.n_invalid_interval);

    if result.n_valid_interval == 0
        fprintf('  No valid %s -> %s intervals in this subset. Empty products will be saved.\n\n', ...
                start_tag, end_tag);
        return
    end

    result.has_valid_interval = true;
    result.segment_lengths = cellfun(@(s) size(s, 1), result.valid_segments_raw);
    fprintf('  Samples/trial: min=%d, max=%d, mean=%.0f\n', ...
            min(result.segment_lengths), max(result.segment_lengths), mean(result.segment_lengths));

    fprintf('Filtering each %s trial segment independently ([%d, %d] Hz, order %d)...\n', ...
            lower(subset_label), filter_range(1), filter_range(2), filter_order);

    min_samples_for_filter = 3 * (2 * filter_order) + 1;
    result.segments_filtered = cell(result.n_valid_interval, 1);

    for k = 1:result.n_valid_interval
        if result.segment_lengths(k) >= min_samples_for_filter
            try
                result.segments_filtered{k} = single(filterData( ...
                                                                double(result.valid_segments_raw{k}), fs, filter_range, filter_order));
            catch ME
                result.segments_filtered{k} = result.valid_segments_raw{k};
                warning(['%s interval trial %d filtering failed (%s). ', ...
                         'Keeping raw segment.'], ...
                        subset_label, result.valid_trial_ids(k), ME.message);
            end
        else
            result.segments_filtered{k} = result.valid_segments_raw{k};
            fprintf('  Warning: %s trial %d too short for filtering (%d < %d samples), kept raw.\n', ...
                    lower(subset_label), result.valid_trial_ids(k), result.segment_lengths(k), min_samples_for_filter);
        end
    end

    fprintf('Filtering complete for %s subset.\n', lower(subset_label));

    if strcmp(segment_standardization_mode, 'zscore_per_segment_channel')
        fprintf('Applying per-segment channel-wise z-score standardization in %s subset...\n', ...
                lower(subset_label));
        result.segments_filtered_segment_zscored = cell(result.n_valid_interval, 1);
        result.segment_standardization_info = repmat(struct( ...
                                                            'trial_id', NaN, ...
                                                            'constant_channels', []), result.n_valid_interval, 1);

        for k = 1:result.n_valid_interval
            [seg_z, seg_info] = zscoreColumnsSafe(result.segments_filtered{k});
            result.segments_filtered_segment_zscored{k} = seg_z;
            result.segment_standardization_info(k).trial_id = result.valid_trial_ids(k);
            result.segment_standardization_info(k).constant_channels = seg_info.constant_channels;
        end

        result.X_pooled_segment_zscored = vertcat(result.segments_filtered_segment_zscored{:});
        fprintf('Segment-wise standardization complete for %s subset.\n', lower(subset_label));
    end

    result.X_pooled = vertcat(result.segments_filtered{:});
    result.segment_boundaries = cumsum(result.segment_lengths);
    fprintf('  Pooled %s interval: [%d x %d]\n\n', ...
            lower(subset_label), size(result.X_pooled, 1), size(result.X_pooled, 2));
end

function [start_tag, end_tag, segments_raw, valid_trial_ids] = buildReferencePSDSegments(correct_trials, X_in, n_samples_in)
    % Raw segments of the first preferred interval with at least one valid trial: CUE-OFF -> GO-ON, then TS-ON -> CUE-ON, then TS-ON -> STOP.
    preferred_intervals = {
                           'CUE-OFF', 'GO-ON'
                           'TS-ON', 'CUE-ON'
                           'TS-ON', 'STOP'
                          };

    for i = 1:size(preferred_intervals, 1)
        a = preferred_intervals{i, 1};
        b = preferred_intervals{i, 2};
        [segments_raw_all, ~, ~, valid_mask, ~, ~] = extractIntervalSegments( ...
                                                                             correct_trials, X_in, n_samples_in, a, b);
        if any(valid_mask)
            start_tag = a;
            end_tag = b;
            segments_raw = segments_raw_all(valid_mask);
            valid_trial_ids = [correct_trials(valid_mask).trial_id];
            return
        end
    end

    error('No valid reference interval found for PSD inspection.');
end

function [reference_trials, reference_label] = choosePsdReferenceTrials(correct_trials, all_trials)
    % The correct trials, or all structurally valid trials when there are none.
    if ~isempty(correct_trials)
        reference_trials = correct_trials;
        reference_label = 'correct trials';
    else
        reference_trials = all_trials;
        reference_label = 'all structurally valid trials';
    end
end

function psd_diag = buildReferencePsdDiagnostic(psd_segments_raw, fs, channel_ids, filter_range, beta_band)
    % Welch PSDs of the reference segments, drawn as a cross-channel summary beside the (at most five) channels with the most prominent beta peak.
    % Beta prominence is the channel's peak in beta_band minus its median level over 5-45 Hz outside beta, in dB.
    % Diagnostic only; filtering and all saved products use filter_range.

    min_seg_len = min(cellfun(@(s) size(s, 1), psd_segments_raw));
    window_length = min(round(2 * fs), min_seg_len);
    nfft = 2^nextpow2(window_length);
    window = hamming(window_length);
    overlap = round(window_length / 2);
    n_segments = numel(psd_segments_raw);
    n_channels = size(psd_segments_raw{1}, 2);

    psd_linear_all = zeros(nfft / 2 + 1, n_channels, n_segments);
    freq = [];
    for i = 1:n_segments
        seg_data = double(psd_segments_raw{i});
        [pxx_ch, freq] = pwelch(seg_data, window, overlap, nfft, fs);
        psd_linear_all(:, :, i) = pxx_ch;
    end

    psd_db_all = 10 * log10(max(psd_linear_all, realmin('double')));
    psd_db_stack = reshape(psd_db_all, size(psd_db_all, 1), []);
    summary_median_db = median(psd_db_stack, 2);
    summary_p10_db = prctile(psd_db_stack, 10, 2);
    summary_p90_db = prctile(psd_db_stack, 90, 2);

    channel_mean_linear = mean(psd_linear_all, 3);
    channel_mean_db = 10 * log10(max(channel_mean_linear, realmin('double')));

    beta_mask = freq >= beta_band(1) & freq <= beta_band(2);
    control_mask = freq >= 5 & freq <= min(45, freq(end)) & ~beta_mask;
    if ~any(control_mask)
        control_mask = ~beta_mask;
    end

    peak_beta_db = nan(1, n_channels);
    peak_beta_freq_hz = nan(1, n_channels);
    beta_prominence_db = nan(1, n_channels);

    beta_freq = freq(beta_mask);
    for ch = 1:n_channels
        beta_curve = channel_mean_db(beta_mask, ch);
        if isempty(beta_curve)
            continue
        end
        [peak_beta_db(ch), peak_idx] = max(beta_curve);
        peak_beta_freq_hz(ch) = beta_freq(peak_idx);

        baseline_db = median(channel_mean_db(control_mask, ch));
        if isfinite(baseline_db)
            beta_prominence_db(ch) = peak_beta_db(ch) - baseline_db;
        else
            beta_prominence_db(ch) = peak_beta_db(ch);
        end
    end

    [~, sort_idx] = sort(beta_prominence_db, 'descend');
    top_n = min(5, n_channels);
    top_idx = sort_idx(1:top_n);

    figure('Position', [100 100 1200 520], 'Name', 'Reference PSD Diagnostic');

    subplot(1, 2, 1);
    hold on;
    fill([freq; flipud(freq)], [summary_p10_db; flipud(summary_p90_db)], ...
         [0.75 0.82 0.92], 'FaceAlpha', 0.35, 'EdgeColor', 'none', ...
         'DisplayName', '10-90% channel/segment envelope');
    plot(freq, summary_median_db, 'k-', 'LineWidth', 1.8, ...
         'DisplayName', 'Median PSD across channel-segment combinations');
    shadePsdBand(gca, filter_range, [1.00 0.90 0.82], 0.35);
    xline(50, ':', '50 Hz line noise', 'Color', [0.85 0.33 0.10], ...
          'LineWidth', 1.2, 'Alpha', 0.8);
    xline(filter_range(1), '--r', sprintf('%.0f Hz', filter_range(1)), 'LineWidth', 1.1);
    xline(filter_range(2), '--r', sprintf('%.0f Hz', filter_range(2)), 'LineWidth', 1.1);
    xlim([0, min(100, freq(end))]);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB/Hz)');
    title('Cross-channel/segment PSD summary');
    legend('Location', 'best');
    box off;
    set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
    hold off;

    subplot(1, 2, 2);
    hold on;
    colors = lines(top_n);
    for i = 1:top_n
        ch_idx = top_idx(i);
        plot(freq, channel_mean_db(:, ch_idx), 'LineWidth', 1.5, ...
             'Color', colors(i, :), ...
             'DisplayName', sprintf('Ch%d', channel_ids(ch_idx)));
        text(peak_beta_freq_hz(ch_idx), peak_beta_db(ch_idx), ...
             sprintf('Ch%d %.1f Hz', channel_ids(ch_idx), peak_beta_freq_hz(ch_idx)), ...
             'Color', colors(i, :), 'FontSize', 8, ...
             'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
    end
    shadePsdBand(gca, beta_band, [0.84 0.92 1.00], 0.30);
    xline(50, ':', '50 Hz', 'Color', [0.85 0.33 0.10], ...
          'LineWidth', 1.2, 'Alpha', 0.8, 'HandleVisibility', 'off');
    xlim([0, min(100, freq(end))]);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB/Hz)');
    title(sprintf('Top %d channels by beta prominence', top_n));
    legend('Location', 'best');
    box off;
    set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
    hold off;

    psd_diag = struct();
    psd_diag.freq = freq;
    psd_diag.summary_median_db = summary_median_db;
    psd_diag.summary_p10_db = summary_p10_db;
    psd_diag.summary_p90_db = summary_p90_db;
    psd_diag.channel_mean_db = channel_mean_db;
    psd_diag.beta_prominence_db = beta_prominence_db;
    psd_diag.peak_beta_freq_hz = peak_beta_freq_hz;
    psd_diag.peak_beta_db = peak_beta_db;
    psd_diag.top_channel_indices = top_idx;
    psd_diag.top_channel_ids = channel_ids(top_idx);
    psd_diag.top_peak_freqs_hz = peak_beta_freq_hz(top_idx);
    psd_diag.top_peak_db = peak_beta_db(top_idx);
    psd_diag.top_prominence_db = beta_prominence_db(top_idx);

end

function shadePsdBand(ax, band_range, color_rgb, alpha_val)
    % Shade a frequency band over the current y-range of ax, behind the curves.
    y_limits = ylim(ax);
    patch(ax, ...
          [band_range(1), band_range(2), band_range(2), band_range(1)], ...
          [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
          color_rgb, 'FaceAlpha', alpha_val, 'EdgeColor', 'none', ...
          'HandleVisibility', 'off');
    uistack(findobj(ax, 'Type', 'patch'), 'bottom');
end

function [applicable_bands, rejected_el, rejected_tr] = computeApplicableQualityRejection(quality_info, filter_range_in, band_defs, band_order)
    % The quality bands that overlap the filter band, and the union of their rejected electrodes and trials.
    applicable_bands = {};
    for i = 1:length(band_order)
        band = band_order{i};
        band_range = band_defs.(band);
        overlap_lo = max(filter_range_in(1), band_range(1));
        overlap_hi = min(filter_range_in(2), band_range(2));
        if overlap_hi > overlap_lo
            applicable_bands{end + 1} = band; %#ok<AGROW>
        end
    end

    rejected_el = [];
    rejected_tr = [];
    for i = 1:length(applicable_bands)
        band = applicable_bands{i};
        rejected_el = unique([rejected_el, quality_info.electrodes.(band)]);
        rejected_tr = unique([rejected_tr, quality_info.trials.(band)]);
    end
    rejected_el = reshape(rejected_el, 1, []);
    rejected_tr = reshape(rejected_tr, 1, []);
end

function slug = intervalSlug(start_tag, end_tag)
    % File-name stem <start>_to_<end> of an interval.
    slug = sprintf('%s_to_%s', slugifyTag(start_tag), slugifyTag(end_tag));
end

function slug = slugifyTag(tag)
    % Lower-case tag in which each run of non-alphanumeric characters becomes one underscore, with no leading or trailing underscore.
    slug = lower(regexprep(tag, '[^A-Za-z0-9]+', '_'));
    slug = regexprep(slug, '_+', '_');
    slug = regexprep(slug, '^_|_$', '');
end

function saveIntervalProducts(output_dir, slug, subset_name, ...
                              segments_raw, segments_filtered, trial_ids, trial_types, ...
                              start_samples, end_samples, invalid_trial_ids, invalid_reasons, ...
                              X_pooled, segment_boundaries, fs, channel_ids, filter_params, metadata, varargin)
    % Save the trial-wise file <slug>_<subset>_trials<suffix>.mat and the pooled file <slug>_<subset>_epoch<suffix>.mat (MAT -v7.3).
    % The optional name-value pair FileSuffix sets <suffix> (default '').

    p = inputParser;
    addParameter(p, 'FileSuffix', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    file_suffix = char(p.Results.FileSuffix);

    segment_start_tag = metadata.segment_start_tag;
    segment_end_tag = metadata.segment_end_tag;

    metadata_trials = metadata;
    metadata_trials.product_type = 'trial-wise';
    metadata_trials.pooling_note = '';
    metadata_trials.invalid_trial_ids = invalid_trial_ids;
    metadata_trials.invalid_reasons = invalid_reasons;

    metadata_epoch = metadata;
    metadata_epoch.product_type = 'pooled-epoch';
    metadata_epoch.pooling_note = ['Pooled repeated-state samples across trials. ', ...
                                   'Filtering was done per trial segment before pooling (not across joins). ', ...
                                   'Artificial discontinuities remain at segment_boundaries.'];

    trial_file = fullfile(output_dir, sprintf('%s_%s_trials%s.mat', slug, subset_name, file_suffix));
    epoch_file = fullfile(output_dir, sprintf('%s_%s_epoch%s.mat', slug, subset_name, file_suffix));
    metadata = metadata_trials;
    save(trial_file, ...
         'segments_raw', 'segments_filtered', 'trial_ids', 'trial_types', ...
         'segment_start_tag', 'segment_end_tag', ...
         'start_samples', 'end_samples', 'invalid_trial_ids', 'invalid_reasons', ...
         'fs', 'channel_ids', 'filter_params', 'metadata', '-v7.3');
    X = X_pooled;
    metadata = metadata_epoch;
    save(epoch_file, ...
         'X', 'segment_boundaries', 'trial_ids', ...
         'segment_start_tag', 'segment_end_tag', ...
         'fs', 'channel_ids', 'filter_params', 'metadata', '-v7.3');
end

function info = buildTrialTypeCountStruct(trials_or_types)
    % Trial-type labels and their counts, from a trial struct array or a cell array of types; empty types are ignored.
    info = struct('labels', {{}}, 'counts', []);

    if isempty(trials_or_types)
        return
    end

    if iscell(trials_or_types)
        trial_types = trials_or_types(:);
    else
        trial_types = {trials_or_types.trial_type}.';
    end

    keep_mask = ~cellfun(@isempty, trial_types);
    trial_types = trial_types(keep_mask);
    if isempty(trial_types)
        return
    end

    [labels, ~, label_idx] = unique(trial_types);
    counts = accumarray(label_idx, 1);
    info.labels = reshape(labels, 1, []);
    info.counts = reshape(counts, 1, []);
end

function s = fmt_int_set(v)
    if isempty(v)
        s = '[]';
    else
        s = mat2str(v);
    end
end

function s = fmt_cellstr(v)
    if isempty(v)
        s = '[]';
    else
        s = ['{' strjoin(v, ', ') '}'];
    end
end

function value = pct(numer, denom)
    if denom == 0
        value = 0;
    else
        value = 100 * numer / denom;
    end
end
