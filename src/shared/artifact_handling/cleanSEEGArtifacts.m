%CLEANSEEGARTIFACTS Remove artifact channels from a recording using the detector output.
%
%   [X_clean, clean_info] = cleanSEEGArtifacts(X, fs, artifact_regions, artifact_info, params)
%
% In automatic mode, the study's fixed policy, only the channels flagged for abnormal kurtosis are removed; the other four reasons (low power, isolation, redundancy, high temporal artifact rate) are reported and the channels kept.
% In interactive mode an operator chooses which detector candidates to remove.
% The function can also notch-filter powerline interference on the channels where the detector found it and, with cleaning_strategy 'channel_and_time', cut artifact time windows.
% It applies to any electrophysiology modality despite its name.
%
% Inputs:
%   X                - [samples x channels] signal.
%   fs               - sampling rate in Hz.
%   artifact_regions - region struct from detectSEEGArtifacts ([start end] times in s); read only by 'channel_and_time'.
%   artifact_info    - detector struct from detectSEEGArtifacts.
%   params           - optional struct overriding the defaults below.
%
% Outputs:
%   X_clean    - cleaned signal [samples x retained channels]; samples are cut only under 'channel_and_time'.
%   clean_info - struct with the removed, candidate and retained channels, any cut segments, retention percentages and the parameters used.

function [X_clean, clean_info] = cleanSEEGArtifacts(X, fs, artifact_regions, artifact_info, params)

    if nargin < 5
        params = struct();
    end

    % Default parameters
    default_params = struct( ...
        'severe_threshold', 10, ...  % percent; a detector score above it makes a channel a candidate
        'moderate_threshold', 5, ...  % percent; 'channel_and_time' cuts the regions of channels scoring from here to severe_threshold
        'merge_gap', 0.1, ...  % s; cut segments closer than this are merged
        'apply_powerline_filter', false, ...  % notch-filter the channels with detected line noise
        'powerline_freqs', [50, 60], ...  % Hz, notch centres
        'powerline_bandwidth', 2, ...  % Hz, notch width
        'cleaning_strategy', 'channel_only', ...  % 'channel_only' or 'channel_and_time'
        'artifact_selection_mode', 'auto', ...  % 'auto' (no prompt; abnormal kurtosis only) or 'interactive'
        'interactive_channel_selection', false, ...  % true forces 'interactive'; set to the resolved mode below
        'channel_labels', {{}}, ...  % optional per-channel labels for the log and the sidecar
        'decision_sidecar_path', '', ...  % path of artifact_clean_decision.json; '' writes none
        'visualize', false ...  % plot the candidates in interactive mode
        );

    % Merge with user parameters
    fields = fieldnames(params);
    for i = 1:length(fields)
        default_params.(fields{i}) = params.(fields{i});
    end
    params = default_params;

    % Validate cleaning_strategy
    assert(ismember(params.cleaning_strategy, {'channel_only', 'channel_and_time'}), ...
        'cleaning_strategy must be ''channel_only'' or ''channel_and_time''.');

    % artifact_selection_mode chooses the mode ('auto' by default); the boolean interactive_channel_selection, when true, forces 'interactive'.
    if ~isfield(params, 'artifact_selection_mode') || isempty(params.artifact_selection_mode)
        params.artifact_selection_mode = 'auto';
    end
    assert(ismember(params.artifact_selection_mode, {'auto', 'interactive'}), ...
        'artifact_selection_mode must be ''auto'' or ''interactive''.');
    if params.interactive_channel_selection
        params.artifact_selection_mode = 'interactive';
    end
    % Set the boolean to the resolved mode, so clean_info.interactive_channel_selection records the path actually taken.
    params.interactive_channel_selection = strcmp(params.artifact_selection_mode, 'interactive');

    [n_samples, n_channels] = size(X);
    original_duration = n_samples / fs;

    fprintf('\n=== Electrophysiology Artifact Cleaning ===\n');
    fprintf('Original data: %d samples x %d channels (%.2f seconds)\n', ...
        n_samples, n_channels, original_duration);
    %% Step 1: Apply powerline interference filtering
    X_clean = X;
    powerline_filtered_channels = [];

    if params.apply_powerline_filter && ~isempty(artifact_info.powerline_channels)
        fprintf('\nApplying powerline notch filter...\n');

        for ch = artifact_info.powerline_channels
            for freq = params.powerline_freqs
                % Design notch filter
                nyquist = fs / 2;
                low_freq = (freq - params.powerline_bandwidth / 2) / nyquist;
                high_freq = (freq + params.powerline_bandwidth / 2) / nyquist;

                % Ensure frequencies are within valid range
                low_freq = max(low_freq, 0.001);
                high_freq = min(high_freq, 0.999);

                if low_freq < high_freq
                    [b, a] = butter(2, [low_freq, high_freq], 'stop');
                    X_clean(:, ch) = filtfilt(b, a, X_clean(:, ch));
                    powerline_filtered_channels = [powerline_filtered_channels, ch];
                end
            end
        end

        fprintf('Notch filter applied to %d channels: %s\n', ...
            length(unique(powerline_filtered_channels)), ...
            mat2str(unique(powerline_filtered_channels)));
    end
    %% Step 2: Candidate list from the detector score
    channel_artifact_pct = artifact_info.channel_artifact_pct;

    % channel_artifact_pct is the detector score: the percentage of samples the temporal detector marks, set to 100 for a channel with a channel-level flag.
    % The score bands below are therefore not temporal-load categories.
    % The candidate list (score above severe_threshold) feeds the interactive review and the sidecar; automatic removal is decided by reason in Step 3.
    auto_severe_channels = find(channel_artifact_pct > params.severe_threshold);
    moderate_channels = find(channel_artifact_pct >= params.moderate_threshold & ...
        channel_artifact_pct <= params.severe_threshold);
    clean_channels = find(channel_artifact_pct < params.moderate_threshold);

    fprintf(['\nLegacy sentinel-inclusive detector score categories ' ...
        '(not temporal-load or removal verdicts):\n']);
    fprintf('Candidate score >%.1f: %d channels - %s\n', ...
        params.severe_threshold, length(auto_severe_channels), mat2str(auto_severe_channels));
    fprintf('Intermediate score %.1f-%.1f: %d channels - %s\n', ...
        params.moderate_threshold, params.severe_threshold, ...
        length(moderate_channels), mat2str(moderate_channels));
    fprintf('Below-candidate score <%.1f: %d channels\n', ...
        params.moderate_threshold, length(clean_channels));
    %% Step 3: Apply the declared reason-specific channel decision
    fprintf('\nPreparing reason-specific channel decision...\n');

    exclusion_reasons = buildExclusionReasons(auto_severe_channels, artifact_info);

    % In interactive mode this draws an interactive figure used as a human input surface: the operator reads the candidate traces before answering the promptArtifactChannelSelection prompt.
    % In automatic mode no figure is drawn here; render_phase1_excluded_channels.py renders the same traces from the saved X_raw, clean_info and artifact_clean_decision.json.
    if strcmp(params.artifact_selection_mode, 'interactive') && params.visualize && ~isempty(auto_severe_channels)
        % Only plot channels that exist in the current X
        valid_mask = auto_severe_channels <= n_channels;
        visualizeExcludedChannels(X, auto_severe_channels(valid_mask), ...
            channel_artifact_pct, fs, exclusion_reasons(valid_mask));

        if any(~valid_mask)
            fprintf('Note: %d detector candidates (indices > %d) are not in current data — skipped in plot.\n', ...
                sum(~valid_mask), n_channels);
            for k = find(~valid_mask)'
                fprintf('  Channel %d (legacy detector score %.1f) — %s\n', ...
                    auto_severe_channels(k), channel_artifact_pct(auto_severe_channels(k)), exclusion_reasons{k});
            end
        end
    end

    % The five per-reason flags of detectSEEGArtifacts as 1 x n_channels logical rows.
    % A flag missing from artifact_info (a hand-built test struct, for example) becomes an all-false row; resolveReasonFlags describes the temporal-rate fallback.
    [flag_low_power, flag_abnormal_kurt, flag_isolated, flag_redundant, ...
        flag_temporal_rate] = resolveReasonFlags(artifact_info, n_channels);

    if strcmp(params.artifact_selection_mode, 'interactive')
        selected_channels = promptArtifactChannelSelection( ...
            auto_severe_channels, exclusion_reasons, params.severe_threshold);
    else
        % Automatic mode removes only the channels flagged for abnormal kurtosis.
        % Low-power, isolated, redundant and high-temporal-rate channels are reported below and kept; removing any of them takes artifact_selection_mode = 'interactive'.
        selected_channels = find(flag_abnormal_kurt);
    end

    % severe_channels is the removed set.
    severe_channels = selected_channels(:)';

    fprintf('\nApplying selected channel removals...\n');
    keep_channels = setdiff(1:n_channels, severe_channels);
    X_clean = X_clean(:, keep_channels);

    fprintf('Retained %d channels (excluded %d channels)\n', ...
        length(keep_channels), length(severe_channels));

    % Detected-but-kept channels per reason, for the log and the sidecar: channels flagged for one of the four non-kurtosis reasons that are not removed.
    % In automatic mode these are all such channels without an abnormal-kurtosis flag; in interactive mode, all such channels the operator kept.
    removed_set = false(1, n_channels);
    removed_set(severe_channels) = true;
    detected_low_power     = find(flag_low_power & ~removed_set);
    detected_isolated      = find(flag_isolated & ~removed_set);
    detected_redundant     = find(flag_redundant & ~removed_set);
    detected_temporal_rate = find(flag_temporal_rate & ~removed_set);
    detected_other_diagnostics = unique([detected_low_power, detected_isolated, ...
        detected_redundant, detected_temporal_rate]);

    % One log line with all five reason categories, so a batch run records the full classification.
    flagged_label_str = formatChannelLabelList(severe_channels, params.channel_labels);
    fprintf(['Phase 1 artifact-clean: mode=%s, abnormal_kurt removed=%d, ', ...
        'diagnostic candidates retained=%d (low_power=%d, isolated=%d, ', ...
        'redundant=%d, temporal_rate=%d). Removed channels: %s\n'], ...
        params.artifact_selection_mode, ...
        sum(flag_abnormal_kurt & removed_set), ...
        numel(detected_other_diagnostics), ...
        numel(detected_low_power), ...
        numel(detected_isolated), ...
        numel(detected_redundant), ...
        numel(detected_temporal_rate), ...
        flagged_label_str);
    if strcmp(params.artifact_selection_mode, 'interactive')
        fprintf('Channels selected for exclusion: %s\n', mat2str(severe_channels));
        fprintf('Legacy detector-score candidates retained by operator: %s\n', ...
            mat2str(setdiff(auto_severe_channels, severe_channels)));
    end

    % Optional sidecar: the decision with all five reason flags as JSON, so the run is documented without parsing the log.
    % Field names containing "severe" refer to the candidate list and its threshold, not to a severity verdict.
    % It is written only when decision_sidecar_path is set, as Phase 1 does.
    if ~isempty(params.decision_sidecar_path)
        % Reason strings indexed by the removed set, so reasons{k} in the sidecar describes flagged_channels(k).
        % exclusion_reasons stays indexed by the candidate list, the order promptArtifactChannelSelection and the candidate plot use; the two sets differ in length, so one array cannot describe both.
        % buildExclusionReasons depends only on the indices and artifact_info, so it can describe the removed set as well.
        removed_channel_reasons = buildExclusionReasons(severe_channels, artifact_info);

        flag_payload = struct( ...
            'flag_low_power',     double(flag_low_power), ...
            'flag_abnormal_kurt', double(flag_abnormal_kurt), ...
            'flag_isolated',      double(flag_isolated), ...
            'flag_redundant',     double(flag_redundant), ...
            'flag_temporal_rate', double(flag_temporal_rate), ...
            'detected_low_power',     detected_low_power, ...
            'detected_isolated',      detected_isolated, ...
            'detected_redundant',     detected_redundant, ...
            'detected_temporal_rate', detected_temporal_rate);

        % Assigned field by field, because struct() reads a cell-valued argument as a struct-array specification.
        decision = struct();
        decision.mode = params.artifact_selection_mode;
        decision.severe_threshold_pct = params.severe_threshold;
        decision.removed_channels = severe_channels;
        decision.removed_reasons = removed_channel_reasons(:)';
        decision.candidate_channels = auto_severe_channels;
        decision.candidate_reasons = exclusion_reasons(:)';
        decision.channel_labels = params.channel_labels;
        decision.flag_payload = flag_payload;
        peWriteArtifactCleanDecision(params.decision_sidecar_path, decision);
    end

    if strcmp(params.cleaning_strategy, 'channel_and_time')
        %% Step 4: Collect the time segments to cut, in original time stamps
        fprintf('\nCollecting time segments to cut...\n');

        cut_segments = [];

        % Add global artifact regions
        for i = 1:length(artifact_regions.global)
            cut_segments = [cut_segments; artifact_regions.global{i}];
        end
        fprintf('Added %d global artifact regions\n', length(artifact_regions.global));

        % Add artifact regions from moderate channels (use original channel indices)
        moderate_regions_count = 0;
        for i = 1:length(moderate_channels)
            original_ch = moderate_channels(i);
            % Check if channel index is valid and artifact_regions.channel exists
            if isfield(artifact_regions, 'channel') && ...
                    iscell(artifact_regions.channel) && ...
                    original_ch <= length(artifact_regions.channel) && ...
                    ~isempty(artifact_regions.channel{original_ch})
                ch_regions = artifact_regions.channel{original_ch};
                cut_segments = [cut_segments; ch_regions];
                moderate_regions_count = moderate_regions_count + size(ch_regions, 1);
            end
        end
        fprintf('Added %d artifact regions from moderate channels\n', moderate_regions_count);
        %% Step 5: Merge overlapping segments and convert to sample indices
        fprintf('\nProcessing cut segments...\n');

        if ~isempty(cut_segments)
            % Sort segments by start time
            cut_segments = sortrows(cut_segments, 1);

            % Merge overlapping segments
            merged_segments = [];
            current_start = cut_segments(1, 1);
            current_end = cut_segments(1, 2);

            for i = 2:size(cut_segments, 1)
                segment_start = cut_segments(i, 1);
                segment_end = cut_segments(i, 2);

                % Check if segments overlap or are close enough to merge
                if segment_start <= current_end + params.merge_gap
                    % Merge segments
                    current_end = max(current_end, segment_end);
                else
                    % Save current segment and start new one
                    merged_segments = [merged_segments; current_start, current_end];
                    current_start = segment_start;
                    current_end = segment_end;
                end
            end

            % Add the last segment
            merged_segments = [merged_segments; current_start, current_end];

            fprintf('Merged %d segments into %d segments\n', ...
                size(cut_segments, 1), size(merged_segments, 1));

            % Convert time segments to sample indices
            cut_indices = [];
            for i = 1:size(merged_segments, 1)
                start_sample = max(1, round(merged_segments(i, 1)*fs));
                end_sample = min(n_samples, round(merged_segments(i, 2)*fs));
                cut_indices = [cut_indices; start_sample, end_sample];
            end

        else
            merged_segments = [];
            cut_indices = [];
            fprintf('No segments to cut\n');
        end
        %% Step 6: Mask of the samples to keep
        fprintf('\nCreating data mask...\n');

        keep_mask = true(n_samples, 1);
        total_cut_samples = 0;

        if ~isempty(cut_indices)
            for i = 1:size(cut_indices, 1)
                start_idx = cut_indices(i, 1);
                end_idx = cut_indices(i, 2);
                keep_mask(start_idx:end_idx) = false;
                total_cut_samples = total_cut_samples + (end_idx - start_idx + 1);

                fprintf('  Cutting samples %d-%d (%.3f-%.3f s, %.3f s)\n', ...
                    start_idx, end_idx, (start_idx - 1)/fs, (end_idx - 1)/fs, (end_idx - start_idx + 1)/fs);
            end
        end
        %% Step 7: Apply the mask in a single cut
        fprintf('\nApplying cuts...\n');

        X_clean = X_clean(keep_mask, :);
        total_cut_time = total_cut_samples / fs;

        fprintf('Cut %d samples (%.3f seconds) in one operation\n', ...
            total_cut_samples, total_cut_time);

    else
        % channel_only: skip time cutting entirely
        fprintf('\nCleaning strategy: channel_only — skipping time cuts.\n');
        merged_segments = [];
        cut_indices = [];
        total_cut_samples = 0;
        total_cut_time = 0;
    end
    %% Step 8: Generate summary
    final_duration = size(X_clean, 1) / fs;
    data_retention_pct = (final_duration / original_duration) * 100;
    channel_retention_pct = (size(X_clean, 2) / n_channels) * 100;

    fprintf('\n=== Cleaning Summary ===\n');
    fprintf('Original: %d channels, %.2f seconds\n', n_channels, original_duration);
    fprintf('Final: %d channels, %.2f seconds\n', size(X_clean, 2), final_duration);
    fprintf('Channel retention: %.1f%% (%d/%d)\n', ...
        channel_retention_pct, size(X_clean, 2), n_channels);
    fprintf('Time retention: %.1f%% (%.2f/%.2f seconds)\n', ...
        data_retention_pct, final_duration, original_duration);
    fprintf('Total cut time: %.2f seconds (%.1f%%)\n', ...
        total_cut_time, (total_cut_time / original_duration)*100);
    %% Step 9: Store cleaning information
    clean_info = struct();
    clean_info.original_size = [n_samples, n_channels];
    clean_info.final_size = size(X_clean);
    clean_info.original_duration = original_duration;
    clean_info.final_duration = final_duration;
    clean_info.excluded_channels = severe_channels;
    clean_info.auto_detected_severe_channels = auto_severe_channels;
    clean_info.moderate_channels = moderate_channels;
    clean_info.retained_channels = keep_channels;
    clean_info.powerline_filtered_channels = unique(powerline_filtered_channels);
    clean_info.cut_segments_time = merged_segments;
    clean_info.cut_segments_samples = cut_indices;
    clean_info.total_cut_time = total_cut_time;
    clean_info.total_cut_samples = total_cut_samples;
    clean_info.channel_retention_pct = channel_retention_pct;
    clean_info.time_retention_pct = data_retention_pct;
    clean_info.interactive_channel_selection = params.interactive_channel_selection;
    clean_info.artifact_selection_mode = params.artifact_selection_mode;
    clean_info.params = params;
    %% Step 10: Cleaning-summary figure
    % render_phase1_cleaning.py draws the six-panel cleaning summary from X_raw, X and clean_info in preprocessed_data.mat; no MATLAB figure is drawn here.
end


function s = formatChannelLabelList(channel_indices, channel_labels)
    % Comma-separated channel list for the log: the channel label where one exists, the numeric index otherwise, or '(none)'.
    if isempty(channel_indices)
        s = '(none)';
        return;
    end
    parts = cell(1, numel(channel_indices));
    for k = 1:numel(channel_indices)
        ch = channel_indices(k);
        if iscell(channel_labels) && ch >= 1 && ch <= numel(channel_labels) && ...
                ~isempty(channel_labels{ch})
            parts{k} = char(string(channel_labels{ch}));
        else
            parts{k} = sprintf('%d', ch);
        end
    end
    s = strjoin(parts, ', ');
end


function [low_power, abnormal_kurt, isolated, redundant, temporal_rate] = ...
        resolveReasonFlags(artifact_info, n_channels)
    % The five per-channel reason flags of artifact_info as 1 x n_channels logical rows.
    % A missing or malformed flag becomes all false, except that a missing flag_temporal_rate is rebuilt from channel_artifact_pct when possible.
    low_power = falseRow(artifact_info, 'flag_low_power', n_channels);
    abnormal_kurt = falseRow(artifact_info, 'flag_abnormal_kurt', n_channels);
    isolated = falseRow(artifact_info, 'flag_isolated', n_channels);
    redundant = falseRow(artifact_info, 'flag_redundant', n_channels);
    temporal_rate = falseRow(artifact_info, 'flag_temporal_rate', n_channels);

    if isfield(artifact_info, 'flag_temporal_rate')
        return;
    end
    % Without flag_temporal_rate, apply the 10 percent default of peArtifactDetectionDefaults to channel_artifact_pct, leaving out the score of 100 that marks a channel-level flag.
    if isfield(artifact_info, 'channel_artifact_pct')
        temporal_rate = (artifact_info.channel_artifact_pct(:)' > 10) & ...
            (artifact_info.channel_artifact_pct(:)' < 100);
        if numel(temporal_rate) ~= n_channels
            temporal_rate = false(1, n_channels);
        end
    end
end


function row = falseRow(s, field, n)
    % s.(field) as a 1 x n logical row, or all false when it is absent, not logical or of another length.
    if isfield(s, field) && islogical(s.(field)) && numel(s.(field)) == n
        row = logical(s.(field)(:)).';
    else
        row = false(1, n);
    end
end
