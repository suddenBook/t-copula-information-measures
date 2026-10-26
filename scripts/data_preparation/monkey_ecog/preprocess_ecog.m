%PREPROCESS_ECOG Monkey ECoG preparation: turn the Neurotycho recordings of monkey George into Phase 1 input files.
%
% Stage 1 loads each session, band-passes it at 1-60 Hz, extracts the condition segment, z-scores each channel and, where configured, trims the segment to an artifact-free range.
% The eyes-closed recordings then keep only the channels that pass the bad-channel test in both of them.
% Stage 2 band-passes at 1-8 Hz (delta/theta), downsamples to 100 Hz and z-scores each channel again.
% Both band-passes are zero-phase Butterworth filters (filtfilt).
%
% Inputs: the Neurotycho session folders under data/monkey_ECoG/raw/, one <folder>/<session>/ per row of the rec struct below.
%
% Outputs: one .mat file per recording (eight by default) in data/monkey_ECoG/preprocessed/, each containing:
%   X        - [samples x channels] preprocessed signal matrix
%   fs       - sampling rate in Hz (100 after downsampling)
%   metadata - struct with recording info, filter parameters and channel indices
%
% Both roots are resolved from the repository root, so the script runs from any working directory.
% Set TC_DATA_ROOT to point at a data tree held outside the repository.
%
% Set MONKEY_ECOG_LABELS to a comma-separated subset of labels (for example MONKEY_ECOG_LABELS=MD1,MD2) to preprocess only those recordings.
% A subset must select both eyes-closed recordings or neither, because the bad-channel test needs both; one alone raises an error.

clear all; close all; clc;

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
data_root = resolveRootFromEnv("TC_DATA_ROOT", string(fullfile(worktree_root, 'data')));

%% Configuration

raw_dir = fullfile(data_root, 'monkey_ECoG', 'raw');
fs_raw = 1000;
fs_target = 100;
n_ch = 128;
filter_order = 4;
wideband = [1 60];
narrowband = [1 8];
bad_ch_threshold = 8;  % |z| threshold of the eyes-closed bad-channel test

% Butterworth band-pass designs; butter takes cutoffs normalized to the Nyquist frequency.
[b_wide, a_wide] = butter(filter_order, 2 * wideband / fs_raw);
% The narrowband filter uses second-order sections, because its normalized cutoffs (0.002 and 0.016) make the [b, a] form ill-conditioned.
[z, p, k] = butter(filter_order, 2 * narrowband / fs_raw);
[sos_narrow, g_narrow] = zp2sos(z, p, k);

downsample_factor = fs_raw / fs_target;  % = 10

output_dir = fullfile(data_root, 'monkey_ECoG', 'preprocessed');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% Dataset definitions
%
% Each entry encodes the extraction choices of one recording:
%   folder      - Neurotycho experiment folder
%   session     - Session1 (baseline) or Session2 (anesthesia)
%   cond_range  - [i, j] indices into ConditionIndex; the segment runs from sample ConditionIndex(i) to ConditionIndex(j)
%   manual_trim - [first, last] sample (at 1000 Hz, counted from the segment start) of the artifact-free range to keep, or [] for the whole segment
%   label, condition, agent, date - recording label, condition, anesthetic agent ('None' for the eyes-closed baseline) and recording date, stored in metadata
%
% The cond_range differs between recordings because the number of events in ConditionIndex varies.
% For example, 20120724 KT and 20120726 MD have two injection events, which shifts the anesthetized segment to indices 3-4 instead of 2-3.

rec = struct( ...
    'folder', { ...
    '20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120724KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120724KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120731PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120726MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', ...
    '20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128'  ...
    }, ...
    'session',    {'Session1','Session1','Session2','Session2','Session2','Session2','Session2','Session2'}, ...
    'cond_range', {[3 4],     [3 4],     [2 3],     [3 4],     [2 3],     [2 3],     [3 4],     [2 3]}, ...
    'manual_trim',{[],        [],        [1 300000],[100000 260000],[1 250000],[1 500000],[],        []}, ...
    'label',      {'EC1',     'EC2',     'KT1',     'KT2',     'PF1',     'PF2',     'MD1',     'MD2'}, ...
    'condition',  {'EyesClosed','EyesClosed','Anesthetized','Anesthetized','Anesthetized','Anesthetized','Anesthetized','Anesthetized'}, ...
    'agent',      {'None',    'None',    'Ketamine','Ketamine','Propofol','Propofol','Medetomidine','Medetomidine'}, ...
    'date',       {'20120810','20120724','20120810','20120724','20120731','20120803','20120726','20120814'} ...
    );

label_filter = strtrim(getenv('MONKEY_ECOG_LABELS'));
if ~isempty(label_filter)
    requested_labels = strtrim(strsplit(label_filter, ','));
    labels = {rec.label};
    keep = ismember(labels, requested_labels);
    missing_labels = setdiff(requested_labels, labels, 'stable');
    assert(isempty(missing_labels), 'Unknown MONKEY_ECOG_LABELS entry: %s', ...
        strjoin(missing_labels, ', '));
    rec = rec(keep);
    fprintf('[MONKEY_ECOG_LABELS] running labels: %s\n', strjoin({rec.label}, ', '));
end

%% Stage 1: Load, wideband filter, condition extraction, z-score, artifact-free trim

fprintf('=== Stage 1: Load, wideband filter (1-60 Hz), segment extraction ===\n\n');

all_data = cell(1, numel(rec));

for d = 1:length(rec)
    session_path = fullfile(raw_dir, rec(d).folder, rec(d).session);
    fprintf('[%s] %s / %s\n', rec(d).label, rec(d).folder, rec(d).session);

    % Condition markers: ConditionIndex holds the sample indices (1000 Hz) of the session's events.
    cond = load(fullfile(session_path, 'Condition.mat'));
    ConditionIndex = cond.ConditionIndex;

    % Load 128-channel ECoG (first channel to get sample count, then preallocate)
    x = load(fullfile(session_path, 'ECoG_ch1.mat'));
    ch_data = x.ECoGData_ch1;
    n_samples = numel(ch_data);
    X = zeros(n_samples, n_ch);
    X(:, 1) = ch_data(:);
    for k = 2:n_ch
        x = load(fullfile(session_path, sprintf('ECoG_ch%d.mat', k)));
        ch_data = x.(sprintf('ECoGData_ch%d', k));
        X(:, k) = ch_data(:);
    end
    fprintf('  Loaded: %d samples x %d channels\n', size(X, 1), size(X, 2));

    % Zero-phase wideband band-pass over the whole session
    X = filtfilt(b_wide, a_wide, X);

    % Extract experimental condition segment
    seg_start = ConditionIndex(rec(d).cond_range(1));
    seg_end   = ConditionIndex(rec(d).cond_range(2));
    X = X(seg_start:seg_end, :);
    fprintf('  Segment [%d:%d]: %d samples (%.1f s)\n', ...
        seg_start, seg_end, size(X, 1), size(X, 1) / fs_raw);

    % Z-score per channel
    for k = 1:size(X, 2)
        X(:, k) = (X(:, k) - mean(X(:, k))) / std(X(:, k));
    end

    % Manual artifact-free trimming when a recording-specific range is configured
    if ~isempty(rec(d).manual_trim)
        trim = rec(d).manual_trim;
        X = X(trim(1):trim(2), :);
        fprintf('  Trimmed to [%d:%d]: %d samples (%.1f s)\n', ...
            trim(1), trim(2), size(X, 1), size(X, 1) / fs_raw);
        % Re-z-score after trimming
        for k = 1:size(X, 2)
            X(:, k) = (X(:, k) - mean(X(:, k))) / std(X(:, k));
        end
    end

    all_data{d} = X;
end

%% Bad channel detection (EC recordings only)
%
% A channel is bad when any of its z-scored samples exceeds bad_ch_threshold (8) in absolute value.
% The eyes-closed recordings keep only the channels that are good in both of them.

fprintf('\n=== Bad channel detection (Eyes-Closed, threshold = %d sigma) ===\n', bad_ch_threshold);

ec_idx = find(strcmp({rec.agent}, 'None') & strcmp({rec.condition}, 'EyesClosed'));
good_EC = [];
if isempty(ec_idx)
    fprintf('  No eyes-closed baseline selected; skipping EC bad-channel mask.\n');
else
    assert(numel(ec_idx) == 2, 'Expected either zero or two eyes-closed baseline recordings.');

    bad_count_EC1 = zeros(1, n_ch);
    bad_count_EC2 = zeros(1, n_ch);
    for k = 1:n_ch
        bad_count_EC1(k) = sum(abs(all_data{ec_idx(1)}(:, k)) > bad_ch_threshold);
        bad_count_EC2(k) = sum(abs(all_data{ec_idx(2)}(:, k)) > bad_ch_threshold);
    end

    good_EC1 = find(bad_count_EC1 == 0);
    good_EC2 = find(bad_count_EC2 == 0);
    good_EC  = intersect(good_EC1, good_EC2);

    fprintf('  %s: %d/%d good | %s: %d/%d good | Intersection: %d retained\n', ...
        rec(ec_idx(1)).label, length(good_EC1), n_ch, ...
        rec(ec_idx(2)).label, length(good_EC2), n_ch, length(good_EC));

    for idx = ec_idx
        all_data{idx} = all_data{idx}(:, good_EC);
    end
end
% The anesthetized recordings keep all 128 channels.

%% Stage 2: Narrowband filter (delta/theta), downsample, final z-score

fprintf('\n=== Stage 2: Narrowband filter (1-8 Hz), downsample to %d Hz ===\n\n', fs_target);

for d = 1:length(rec)
    X = all_data{d};

    % Narrowband bandpass (delta/theta, 1-8 Hz) — SOS form for numerical stability
    X = filtfilt(sos_narrow, g_narrow, X);

    % Downsample 1000 Hz -> 100 Hz in one step by keeping every tenth sample.
    % The 1-8 Hz band-pass above is the anti-aliasing filter, so the band stays far below the 50 Hz Nyquist frequency.
    X = X(1:downsample_factor:end, :);

    % Final z-score
    for k = 1:size(X, 2)
        X(:, k) = (X(:, k) - mean(X(:, k))) / std(X(:, k));
    end

    all_data{d} = X;
    fprintf('[%s] %d samples x %d ch  (%.1f s at %d Hz)\n', ...
        rec(d).label, size(X, 1), size(X, 2), size(X, 1) / fs_target, fs_target);
end

%% Save each recording as a separate .mat file
%
% Each file (MAT -v7.3) contains:
%   X        - [samples x channels], the Phase 1 input signal
%   fs       - sampling rate, 100 Hz
%   metadata - provenance and parameters of the preparation

fprintf('\n=== Saving ===\n');

fs = fs_target;

for d = 1:length(rec)
    X = all_data{d};

    metadata = struct();
    metadata.label         = rec(d).label;
    metadata.condition     = rec(d).condition;
    metadata.agent         = rec(d).agent;
    metadata.date          = rec(d).date;
    metadata.monkey        = 'George';
    metadata.modality      = 'ECoG128';
    metadata.n_ch_original = n_ch;
    metadata.n_ch          = size(X, 2);
    metadata.fs_original   = fs_raw;
    metadata.fs            = fs_target;
    metadata.filter_wide   = wideband;
    metadata.filter_narrow = narrowband;
    metadata.filter_order  = filter_order;
    metadata.manual_trim   = rec(d).manual_trim;
    metadata.source_folder = rec(d).folder;
    metadata.source_session = rec(d).session;
    metadata.cond_range    = rec(d).cond_range;

    if strcmp(rec(d).agent, 'None')
        metadata.good_channels = good_EC;
        metadata.bad_channels  = setdiff(1:n_ch, good_EC);
    else
        metadata.good_channels = 1:n_ch;
        metadata.bad_channels  = [];
    end

    outfile = fullfile(output_dir, sprintf('%s.mat', rec(d).label));
    save(outfile, 'X', 'fs', 'metadata', '-v7.3');
    fprintf('  %s  ->  %s\n', rec(d).label, outfile);
end

fprintf('\nDone. Output directory: %s\n', output_dir);
fprintf('Usage: load(''%s'')  %%  gives X, fs, metadata\n', fullfile(output_dir, 'EC1.mat'));
