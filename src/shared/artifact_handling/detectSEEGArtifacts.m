% DETECTSEEGARTIFACTS Mark artifact samples and flag atypical channels in a multichannel recording.
%
%   [artifact_regions, artifact_info] = detectSEEGArtifacts(X, fs, params)
%
% The temporal detector scores every sample for extreme values, rapid changes, flatlines, sustained runs and multi-channel coherent events, and a separate pass looks for powerline interference.
% Channel-level diagnostics then flag low power, abnormal kurtosis, isolation (low correlation with the other channels) and redundancy (near-duplicate signals).
% It applies to any electrophysiology modality despite its name.
% A raw-voltage input (SEEG, LFP) uses the absolute flatline threshold and a standardized input (ECoG) a scale-relative one, chosen by the artifact profile of its row in the recording table of scripts/Phase1_preprocessing.m.
%
% Inputs:
%   X      - [samples x channels] signal.
%   fs     - sampling rate in Hz.
%   params - optional struct overriding peArtifactDetectionDefaults.
%
% Outputs:
%   artifact_regions - struct with global, a cell of [start end] times in s where many channels are marked, and channel, an n_channels x 1 cell of per-channel [start end] rows in s.
%   artifact_info    - struct with the per-sample mask, scores and type tags, channel statistics, the detector score channel_artifact_pct, the powerline channels and their SNR, the quality flags and metrics, and the five per-channel reason flags.

function [artifact_regions, artifact_info] = detectSEEGArtifacts(X, fs, params)

    % Set default parameters if not provided
    if nargin < 3
        params = struct();
    end

    % The complete defaults also enter the Phase 1 settings identity, so changing one changes that identity.
    defaultParams = peArtifactDetectionDefaults();

    % Merge with user parameters
    fields = fieldnames(params);
    for i = 1:length(fields)
        defaultParams.(fields{i}) = params.(fields{i});
    end
    params = defaultParams;

    if isempty(params.random_seed)
        detector_stream = [];
    else
        validateattributes(params.random_seed, {'numeric'}, ...
                           {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'random_seed');
        detector_stream = RandStream('twister', 'Seed', double(params.random_seed));
    end

    [n_samples, n_channels] = size(X);
    time_vector = (0:n_samples - 1) / fs;

    fprintf('=== Electrophysiology Artifact Detection ===\n');
    fprintf('Data: %d samples x %d channels\n', n_samples, n_channels);
    fprintf('Duration: %.2f seconds\n', n_samples / fs);
    fprintf('Sampling rate: %d Hz\n', fs);
    %% Calculate robust statistics for the data
    fprintf('\nComputing channel statistics...\n');
    channel_stats = struct();
    validateattributes(params.gradient_percentile, {'numeric'}, ...
                       {'scalar', 'finite', '>', 0, '<', 100}, mfilename, ...
                       'gradient_percentile');

    for ch = 1:n_channels
        ch_data = X(:, ch);

        % Use median and MAD (Median Absolute Deviation)
        channel_stats(ch).median = median(ch_data);
        channel_stats(ch).mad = median(abs(ch_data - channel_stats(ch).median));
        channel_stats(ch).robust_std = 1.4826 * channel_stats(ch).mad; % Convert MAD to std equivalent

        % Calculate percentiles
        channel_stats(ch).percentiles = prctile(ch_data, [0.1, 1, 5, 95, 99, 99.9]);

        % Calculate gradient statistics
        ch_gradient = abs([0; diff(ch_data)]);
        channel_stats(ch).gradient_percentiles = prctile(ch_gradient, [95, 99, 99.5, 99.9]);

        % Adaptive amplitude threshold on the median-centred signal: the larger of the percentile_threshold percentile of |x - median| and mad_factor robust standard deviations.
        % Samples and threshold share the centred coordinate, so a constant baseline offset does not change the result.
        centered_data = ch_data - channel_stats(ch).median;
        channel_stats(ch).amp_threshold = max( ...
                                              prctile(abs(centered_data), params.percentile_threshold), ...
                                              params.mad_factor * channel_stats(ch).robust_std);

        channel_stats(ch).grad_threshold = ...
            prctile(ch_gradient, params.gradient_percentile);
    end

    % Global statistics
    all_data = X(:);
    global_percentiles = prctile(all_data, [0.01, 0.1, 1, 99, 99.9, 99.99]);
    fprintf(['Global percentiles (input amplitude units): 0.01%%=%.1f, ' ...
             '0.1%%=%.1f, 99.9%%=%.1f, 99.99%%=%.1f\n'], ...
            global_percentiles(1), global_percentiles(2), global_percentiles(5), global_percentiles(6));
    %% Initialize detection results
    artifact_mask = false(n_samples, n_channels);
    artifact_scores = zeros(n_samples, n_channels);
    artifact_types = cell(n_samples, n_channels);
    %% 1. Extreme value detection
    fprintf('\nDetecting extreme values...\n');
    extreme_count = 0;

    for ch = 1:n_channels
        % Use adaptive threshold
        centered_data = X(:, ch) - channel_stats(ch).median;
        ch_extreme = abs(centered_data) > channel_stats(ch).amp_threshold;

        artifact_mask(:, ch) = artifact_mask(:, ch) | ch_extreme;
        artifact_scores(ch_extreme, ch) = artifact_scores(ch_extreme, ch) + 1.0;

        % Mark type
        extreme_idx = find(ch_extreme);
        for idx = extreme_idx'
            artifact_types{idx, ch} = [artifact_types{idx, ch}, 'EXT;'];
        end

        extreme_count = extreme_count + sum(ch_extreme);
    end

    fprintf('Extreme values: %d (%.3f%% of data)\n', ...
            extreme_count, 100 * extreme_count / (n_samples * n_channels));
    %% 2. Gradient-based detection
    fprintf('Detecting rapid changes...\n');
    gradient_count = 0;

    for ch = 1:n_channels
        ch_gradient = abs([0; diff(X(:, ch))]);

        % Use adaptive threshold
        gradient_threshold = channel_stats(ch).grad_threshold;

        % Physiological cap for a raw-voltage input: 500 microvolts per ms, that is 500 * 1000 / fs per sample.
        % On a standardized input the empirical threshold is orders of magnitude smaller, so the cap never binds there.
        max_physiological_gradient = 500 * (1000 / fs);
        gradient_threshold = min(gradient_threshold, max_physiological_gradient);

        ch_gradient_artifact = ch_gradient > gradient_threshold;

        artifact_mask(:, ch) = artifact_mask(:, ch) | ch_gradient_artifact;
        artifact_scores(ch_gradient_artifact, ch) = artifact_scores(ch_gradient_artifact, ch) + 0.5;

        % Mark type
        grad_idx = find(ch_gradient_artifact);
        for idx = grad_idx'
            artifact_types{idx, ch} = [artifact_types{idx, ch}, 'GRAD;'];
        end

        gradient_count = gradient_count + sum(ch_gradient_artifact);
    end

    fprintf('Rapid changes: %d (%.3f%% of data)\n', ...
            gradient_count, 100 * gradient_count / (n_samples * n_channels));
    %% 3. Flatline detection
    fprintf('Detecting flatlines...\n');
    flatline_count = 0;
    flatline_window_samples = round(params.flatline_window * fs);

    for ch = 1:n_channels
        if ~isempty(params.flatline_relative_std)
            validateattributes(params.flatline_relative_std, {'numeric'}, ...
                               {'scalar', 'positive', 'finite', '<', 1}, mfilename, ...
                               'flatline_relative_std');
            flatline_threshold = params.flatline_relative_std * ...
                channel_stats(ch).robust_std;
        else
            validateattributes(params.flatline_std, {'numeric'}, ...
                               {'scalar', 'nonnegative', 'finite'}, mfilename, 'flatline_std');
            flatline_threshold = params.flatline_std;
        end
        for t = 1:flatline_window_samples: ...
                n_samples - flatline_window_samples + 1
            window_data = X(t:t + flatline_window_samples - 1, ch);
            window_std = std(window_data);

            if window_std <= flatline_threshold
                artifact_mask(t:t + flatline_window_samples - 1, ch) = true;
                artifact_scores(t:t + flatline_window_samples - 1, ch) = ...
                    artifact_scores(t:t + flatline_window_samples - 1, ch) + 2.0; % High confidence

                % Mark type
                for idx = t:t + flatline_window_samples - 1
                    artifact_types{idx, ch} = [artifact_types{idx, ch}, 'FLAT;'];
                end

                flatline_count = flatline_count + flatline_window_samples;
            end
        end
    end

    fprintf('Flatlines: %d samples (%.2f%% of data)\n', ...
            flatline_count, 100 * flatline_count / (n_samples * n_channels));
    %% 4. Sustained artifact detection
    fprintf('Checking for sustained artifacts...\n');
    sustained_window = round(0.05 * fs);  % 50 ms

    for ch = 1:n_channels
        ch_artifacts = artifact_mask(:, ch);

        if any(ch_artifacts)
            % Dilate with a 10 ms structuring element to connect nearby artifacts
            ch_artifacts_dilated = imdilate(ch_artifacts, ones(round(0.01 * fs), 1));

            % Find continuous regions
            diff_artifacts = diff([0; ch_artifacts_dilated; 0]);
            starts = find(diff_artifacts == 1);
            ends = find(diff_artifacts == -1) - 1;

            % Boost score for sustained artifacts
            for i = 1:length(starts)
                duration = ends(i) - starts(i) + 1;
                if duration > sustained_window
                    % Add bonus score for sustained artifacts
                    artifact_scores(starts(i):ends(i), ch) = ...
                        artifact_scores(starts(i):ends(i), ch) + 0.5;

                    % Mark as sustained
                    for idx = starts(i):ends(i)
                        artifact_types{idx, ch} = [artifact_types{idx, ch}, 'SUST;'];
                    end
                end
            end
        end
    end
    %% 5. Multi-channel coherent artifacts
    fprintf('Detecting multi-channel artifacts...\n');
    correlation_window = round(params.correlation_window * fs);
    step_size = round(correlation_window / 2);
    multi_channel_times = [];

    for t = 1:step_size:n_samples - correlation_window + 1
        % A candidate sample needs more than multi_channel_ratio of the channels flagged at that same sample.
        % Requiring simultaneity keeps strongly correlated physiological LFP from being marked as one continuous global artifact.
        window_artifacts = artifact_mask(t:t + correlation_window - 1, :);
        candidate_offsets = find(sum(window_artifacts, 2) ...
                                 > n_channels * params.multi_channel_ratio);

        if ~isempty(candidate_offsets)
            % Check correlation among affected channels
            affected_ch_idx = find(any( ...
                                       window_artifacts(candidate_offsets, :), 1));

            if length(affected_ch_idx) >= 3
                % Up to five affected channels, drawn from the private stream when random_seed is set
                sample_size = min(5, length(affected_ch_idx));
                if isempty(detector_stream)
                    sample_order = randperm(length(affected_ch_idx), sample_size);
                else
                    sample_order = randperm(detector_stream, ...
                                            length(affected_ch_idx), sample_size);
                end
                sample_idx = affected_ch_idx(sample_order);

                window_data = X(t:t + correlation_window - 1, sample_idx);

                % Remove mean to focus on correlation
                window_data_centered = bsxfun(@minus, window_data, mean(window_data, 1));

                if std(window_data_centered(:)) > 0 % Avoid division by zero
                    corr_matrix = corrcoef(window_data_centered);

                    % Calculate mean absolute correlation (excluding diagonal)
                    corr_matrix_abs = abs(corr_matrix);
                    off_diagonal = ~eye(size(corr_matrix_abs), 'logical');
                    mean_corr = mean(corr_matrix_abs(off_diagonal));

                    if mean_corr > params.correlation_threshold
                        candidate_times = t + candidate_offsets - 1;
                        multi_channel_times = [multi_channel_times, ...
                                               candidate_times(:).']; %#ok<AGROW>

                        % Confirm only the simultaneous candidate samples, not every sample of the window.
                        artifact_scores(candidate_times, :) = ...
                            artifact_scores(candidate_times, :) + 0.5;

                        % Mark as multi-channel
                        for idx = candidate_times(:).'
                            for ch = 1:n_channels
                                artifact_types{idx, ch} = [artifact_types{idx, ch}, 'MULTI;'];
                            end
                        end
                    end
                end
            end
        end
    end

    multi_channel_times = unique(multi_channel_times);
    fprintf('Found %d samples with correlated multi-channel artifacts\n', ...
            length(multi_channel_times));
    %% 6. Powerline interference
    fprintf('Checking powerline interference...\n');
    powerline_channels = [];
    powerline_info = struct();
    welch_window_samples = min(n_samples, round(fs));
    welch_overlap_samples = floor(welch_window_samples / 2);
    welch_nfft = max(round(fs), welch_window_samples);

    for ch = 1:n_channels
        [psd, f] = pwelch(X(:, ch), hamming(welch_window_samples), ...
                          welch_overlap_samples, welch_nfft, fs);

        for pf = params.powerline_freq
            [~, f_idx] = min(abs(f - pf));

            % SNR of the bin nearest pf against the median of up to five bins on each side
            signal_power = psd(f_idx);
            noise_indices = max(1, f_idx - 5):min(length(f), f_idx + 5);
            noise_indices(noise_indices == f_idx) = [];
            noise_power = median(psd(noise_indices));

            snr_db = 10 * log10(signal_power / noise_power);

            if snr_db > params.powerline_snr
                powerline_channels = [powerline_channels, ch];
                powerline_info(ch).freq = pf;
                powerline_info(ch).snr = snr_db;
                fprintf('Channel %d: %.0f Hz SNR = %.1f dB\n', ch, pf, snr_db);
            end
        end
    end

    powerline_channels = unique(powerline_channels);
    %% 7. Final artifact determination
    fprintf('\nFinalizing artifact detection...\n');

    % Use score threshold
    final_artifact_mask = artifact_scores >= params.score_threshold;

    % Apply minimum duration constraint
    for ch = 1:n_channels
        ch_artifacts = final_artifact_mask(:, ch);

        if any(ch_artifacts)
            diff_artifacts = diff([0; ch_artifacts; 0]);
            starts = find(diff_artifacts == 1);
            ends = find(diff_artifacts == -1) - 1;

            % Remove very short segments
            min_samples = params.min_artifact_duration * fs;
            for i = 1:length(starts)
                if (ends(i) - starts(i) + 1) < min_samples
                    final_artifact_mask(starts(i):ends(i), ch) = false;
                end
            end
        end
    end
    %% 8. Extract artifact regions
    fprintf('Extracting artifact regions...\n');

    % Global artifacts: samples where more than multi_channel_ratio of the channels are marked
    global_artifact_mask = sum(final_artifact_mask, 2) > n_channels * params.multi_channel_ratio;
    artifact_regions.global = {};

    if any(global_artifact_mask)
        diff_mask = diff([0; global_artifact_mask; 0]);
        region_starts = find(diff_mask == 1);
        region_ends = find(diff_mask == -1) - 1;

        % Merge nearby regions
        if ~isempty(region_starts)
            merged_starts = region_starts(1);
            merged_ends = [];

            for i = 1:length(region_starts) - 1
                if region_starts(i + 1) - region_ends(i) > params.merge_gap * fs
                    % Save current region and start new one
                    merged_ends = [merged_ends; region_ends(i)];
                    merged_starts = [merged_starts; region_starts(i + 1)];
                end
            end
            merged_ends = [merged_ends; region_ends(end)];

            % Convert to time
            for i = 1:length(merged_starts)
                artifact_regions.global{i} = [time_vector(merged_starts(i)), ...
                                              time_vector(merged_ends(i))];
            end
        end
    end

    % Per-channel artifacts and the five reason flags (low_power, abnormal_kurt, isolated, redundant, temporal_rate).
    % The flags let the cleaner apply its removal policy by reason, which the detector score cannot express once a channel-level flag has set it to 100.
    artifact_regions.channel = cell(n_channels, 1);
    channel_artifact_pct = zeros(n_channels, 1);
    flag_low_power     = false(1, n_channels);
    flag_abnormal_kurt = false(1, n_channels);
    flag_isolated      = false(1, n_channels);
    flag_redundant     = false(1, n_channels);
    flag_temporal_rate = false(1, n_channels);

    for ch = 1:n_channels
        ch_artifacts = final_artifact_mask(:, ch);
        channel_artifact_pct(ch) = sum(ch_artifacts) / n_samples * 100;
        % flag_temporal_rate reads the true marked fraction, before Section 8b sets channel_artifact_pct to 100 for a flagged channel.
        if channel_artifact_pct(ch) > params.temporal_rate_threshold_pct
            flag_temporal_rate(ch) = true;
        end

        if any(ch_artifacts) && params.output_channel_artifacts
            diff_artifacts = diff([0; ch_artifacts; 0]);
            ch_starts = find(diff_artifacts == 1);
            ch_ends = find(diff_artifacts == -1) - 1;

            % Merge nearby regions
            if ~isempty(ch_starts)
                merged_starts = ch_starts(1);
                merged_ends = [];

                for i = 1:length(ch_starts) - 1
                    if ch_starts(i + 1) - ch_ends(i) > params.merge_gap * fs
                        merged_ends = [merged_ends; ch_ends(i)];
                        merged_starts = [merged_starts; ch_starts(i + 1)];
                    end
                end
                merged_ends = [merged_ends; ch_ends(end)];

                % Store as time ranges
                channel_regions = zeros(length(merged_starts), 2);
                for i = 1:length(merged_starts)
                    channel_regions(i, :) = [time_vector(merged_starts(i)), ...
                                             time_vector(merged_ends(i))];
                end
                artifact_regions.channel{ch} = channel_regions;
            end
        end
    end
    % Keep the true marked fraction for the summary below, because Section 8b sets channel_artifact_pct to 100 for every channel it flags.
    temporal_artifact_pct = channel_artifact_pct;

    %% 8b. Channel-level quality assessment
    % Flag channels whose whole-recording diagnostics are atypical among the recording's channels.
    % Unlike the temporal checks of Sections 1-6, these are channel-level indicators, not fractions of affected samples.
    % A flagged channel gets the detector score 100 and its reason flag; whether it is removed is decided by the cleaner's reason-specific policy.
    fprintf('\nAssessing channel-level quality...\n');

    % --- 1. Robust std below low_power_fraction of the channels' median ---
    robust_power = arrayfun(@(s) s.robust_std, channel_stats)';  % [n_channels x 1]
    median_power = median(robust_power);
    low_power_channels = find(robust_power < params.low_power_fraction * median_power);

    if ~isempty(low_power_channels)
        fprintf('Low power channels (%d): %s\n', length(low_power_channels), mat2str(low_power_channels'));
        for i = 1:length(low_power_channels)
            channel_artifact_pct(low_power_channels(i)) = 100;
            flag_low_power(low_power_channels(i)) = true;
        end
    end

    % --- 2. Kurtosis more than kurtosis_mad_factor scaled MADs from the median channel kurtosis ---
    kurt_values = kurtosis(X)';  % [n_channels x 1]
    finite_kurtosis = isfinite(kurt_values);
    if any(finite_kurtosis)
        finite_kurtosis_values = kurt_values(finite_kurtosis);
        median_kurt = median(finite_kurtosis_values);
        mad_kurt = 1.4826 * median( ...
                                   abs(finite_kurtosis_values - median_kurt));
    else
        median_kurt = NaN;
        mad_kurt = NaN;
    end
    if isfinite(mad_kurt) && mad_kurt > 0
        abnormal_kurt_channels = find(finite_kurtosis ...
                                      & abs(kurt_values - median_kurt) ...
                                      > params.kurtosis_mad_factor * mad_kurt);
    else
        abnormal_kurt_channels = [];
    end
    if any(~finite_kurtosis)
        fprintf(['Non-finite kurtosis diagnostics (%d channels) are excluded from ' ...
                 'the recording-level median/MAD reference and are not themselves ' ...
                 'classified as abnormal-kurtosis outliers.\n'], nnz(~finite_kurtosis));
    end

    if ~isempty(abnormal_kurt_channels)
        fprintf('Abnormal kurtosis channels (%d): %s\n', length(abnormal_kurt_channels), mat2str(abnormal_kurt_channels'));
        for i = 1:length(abnormal_kurt_channels)
            channel_artifact_pct(abnormal_kurt_channels(i)) = 100;
            flag_abnormal_kurt(abnormal_kurt_channels(i)) = true;
        end
    end

    % --- 3. Low mean absolute correlation with the other channels ---
    % The mean runs over the other channels only.
    % An undefined correlation, as from a constant channel that the power and flatline checks already catch, counts as zero, so one NaN cannot void the isolation check of every channel.
    R = abs(corr(X));
    R(~isfinite(R)) = 0;
    off_diagonal = ~eye(n_channels, 'logical');
    mean_abs_corr = sum(R .* off_diagonal, 2) ./ (n_channels - 1);
    median_corr = median(mean_abs_corr);
    mad_corr = 1.4826 * median(abs(mean_abs_corr - median_corr));
    if mad_corr > 0
        isolated_channels = find(mean_abs_corr < median_corr - params.isolation_mad_factor * mad_corr ...
                                 | mean_abs_corr < params.isolation_abs_threshold);
    else
        isolated_channels = find(mean_abs_corr < params.isolation_abs_threshold);
    end

    if ~isempty(isolated_channels)
        fprintf('Isolated channels (%d): %s\n', length(isolated_channels), mat2str(isolated_channels'));
        for i = 1:length(isolated_channels)
            channel_artifact_pct(isolated_channels(i)) = 100;
            flag_isolated(isolated_channels(i)) = true;
        end
    end

    % --- 4. Redundancy candidates (near-duplicate signals) ---
    % Pairs with |corr| above redundancy_threshold share nearly all their variance.
    % A greedy pass flags one channel per pair, preferring the channel in the most remaining pairs and breaking ties by the higher detector score.
    % The flag is diagnostic; automatic cleaning keeps a redundancy candidate.
    [ri, rj] = find(R > params.redundancy_threshold & triu(true(n_channels), 1));
    redundant_pairs_idx = [ri, rj];
    n_redundant_pairs = size(redundant_pairs_idx, 1);

    if n_redundant_pairs > 0
        fprintf('Redundant channel pairs (%d) with |corr| > %.2f:\n', ...
                n_redundant_pairs, params.redundancy_threshold);
        for p = 1:min(n_redundant_pairs, 10)
            fprintf('  Ch %d -- Ch %d  (|r| = %.4f)\n', ...
                    redundant_pairs_idx(p, 1), redundant_pairs_idx(p, 2), ...
                    R(redundant_pairs_idx(p, 1), redundant_pairs_idx(p, 2)));
        end
        if n_redundant_pairs > 10
            fprintf('  ... and %d more pairs\n', n_redundant_pairs - 10);
        end

        redundant_channels = [];
        remaining = redundant_pairs_idx;
        while ~isempty(remaining)
            all_ch = remaining(:);
            [unique_ch, ~, ic] = unique(all_ch);
            counts = accumarray(ic, 1);
            max_count = max(counts);
            tied = unique_ch(counts == max_count);
            if length(tied) > 1
                [~, worst] = max(channel_artifact_pct(tied));
                ch_remove = tied(worst);
            else
                ch_remove = tied;
            end
            redundant_channels = [redundant_channels; ch_remove]; %#ok<AGROW>
            keep = remaining(:, 1) ~= ch_remove & remaining(:, 2) ~= ch_remove;
            remaining = remaining(keep, :);
        end
        redundant_channels = sort(redundant_channels);

        fprintf(['Redundancy diagnostic candidates (%d; retained automatically ' ...
                 'unless separately abnormal-kurtosis flagged): %s\n'], ...
                length(redundant_channels), mat2str(redundant_channels'));
        for i = 1:length(redundant_channels)
            channel_artifact_pct(redundant_channels(i)) = 100;
            flag_redundant(redundant_channels(i)) = true;
        end
    else
        redundant_channels = [];
        fprintf('No redundant channel pairs (threshold |corr| > %.2f)\n', ...
                params.redundancy_threshold);
    end

    % Every channel with a channel-level flag; the field name all_quality_rejected does not mean these channels are removed.
    all_quality_rejected = unique([low_power_channels; abnormal_kurt_channels; isolated_channels; redundant_channels]);
    fprintf('Total channels flagged by quality assessment: %d\n', length(all_quality_rejected));

    %% 9. Summary report
    fprintf('\n=== Artifact Detection Summary ===\n');
    fprintf('Total recording time: %.2f seconds\n', n_samples / fs);

    % Global artifacts
    fprintf('\nGlobal artifact regions: %d\n', length(artifact_regions.global));
    total_global_artifact_time = 0;
    for i = 1:length(artifact_regions.global)
        duration = artifact_regions.global{i}(2) - artifact_regions.global{i}(1);
        total_global_artifact_time = total_global_artifact_time + duration;
        fprintf('Region %d: %.3f - %.3f s (%.3f s)\n', ...
                i, artifact_regions.global{i}(1), artifact_regions.global{i}(2), duration);
    end

    fprintf('\nGlobal artifact time: %.2f s (%.1f%%)\n', ...
            total_global_artifact_time, total_global_artifact_time / (n_samples / fs) * 100);

    % Temporal artifact load from the true marked fraction, not the detector score.
    fprintf(['\nPer-channel temporal artifact load (fraction of samples marked by ' ...
             'the temporal detector):\n']);
    fprintf('Min: %.1f%%, Max: %.1f%%, Mean: %.1f%%, Median: %.1f%%\n', ...
            min(temporal_artifact_pct), max(temporal_artifact_pct), ...
            mean(temporal_artifact_pct), median(temporal_artifact_pct));

    % Highest temporal loads
    [sorted_pct, sorted_idx] = sort(temporal_artifact_pct, 'descend');
    fprintf('\nHighest temporal artifact loads:\n');
    for i = 1:min(params.max_channel_artifacts, n_channels)
        if sorted_pct(i) > 0
            ch = sorted_idx(i);
            n_regions = size(artifact_regions.channel{ch}, 1);
            fprintf('Channel %3d: %5.1f%% artifact samples (%d regions)', ...
                    ch, sorted_pct(i), n_regions);

            % Add artifact type summary
            ch_types = artifact_types(:, ch);
            % Filter out empty cells and convert to cell array of strings
            ch_types_nonempty = ch_types(~cellfun(@isempty, ch_types));
            type_summary = {};
            if ~isempty(ch_types_nonempty) && any(contains(ch_types_nonempty, 'EXT'))
                type_summary{end + 1} = 'Extreme';
            end
            if ~isempty(ch_types_nonempty) && any(contains(ch_types_nonempty, 'GRAD'))
                type_summary{end + 1} = 'Gradient';
            end
            if ~isempty(ch_types_nonempty) && any(contains(ch_types_nonempty, 'FLAT'))
                type_summary{end + 1} = 'Flatline';
            end
            if ~isempty(ch_types_nonempty) && any(contains(ch_types_nonempty, 'SUST'))
                type_summary{end + 1} = 'Sustained';
            end
            if ~isempty(type_summary)
                fprintf('[%s]', strjoin(type_summary, ', '));
            end
            fprintf('\n');
        end
    end

    % Powerline summary
    if ~isempty(powerline_channels)
        fprintf('\nChannels with powerline interference: %d\n', length(powerline_channels));
        fprintf('Channels: %s\n', mat2str(powerline_channels));
    end

    % Channel quality assessment summary
    fprintf('\n--- Channel Quality Assessment ---\n');
    fprintf('Low power channels (%d):', length(low_power_channels));
    if ~isempty(low_power_channels)
        fprintf(' %s\n', mat2str(low_power_channels'));
    else
        fprintf(' none\n');
    end
    fprintf('Abnormal kurtosis channels (%d):', length(abnormal_kurt_channels));
    if ~isempty(abnormal_kurt_channels)
        fprintf(' %s\n', mat2str(abnormal_kurt_channels'));
    else
        fprintf(' none\n');
    end
    fprintf('Isolated channels (%d):', length(isolated_channels));
    if ~isempty(isolated_channels)
        fprintf(' %s\n', mat2str(isolated_channels'));
    else
        fprintf(' none\n');
    end
    fprintf('Redundant channels (%d):', length(redundant_channels));
    if ~isempty(redundant_channels)
        fprintf(' %s\n', mat2str(redundant_channels'));
    else
        fprintf(' none\n');
    end
    fprintf('Total unique channels flagged: %d\n', length(all_quality_rejected));

    % Store all results
    artifact_info = struct();
    artifact_info.final_artifact_mask = final_artifact_mask;
    artifact_info.artifact_scores = artifact_scores;
    artifact_info.artifact_types = artifact_types;
    artifact_info.channel_stats = channel_stats;
    artifact_info.channel_artifact_pct = channel_artifact_pct;
    artifact_info.powerline_channels = powerline_channels;
    artifact_info.powerline_info = powerline_info;
    artifact_info.total_global_time = total_global_artifact_time;
    artifact_info.params = params;
    artifact_info.extreme_threshold_convention = ...
        'centered_abs_deviation_configured_percentile_or_mad_v1';
    artifact_info.multichannel_correlation_convention = ...
        'simultaneous_candidates_mean_abs_off_diagonal_v1';
    artifact_info.channel_isolation_correlation_convention = ...
        'mean_abs_off_diagonal_undefined_as_zero_v1';

    % Quality assessment results
    artifact_info.quality_flags.low_power_channels = low_power_channels;
    artifact_info.quality_flags.abnormal_kurt_channels = abnormal_kurt_channels;
    artifact_info.quality_flags.isolated_channels = isolated_channels;
    artifact_info.quality_flags.redundant_channels = redundant_channels;
    artifact_info.quality_flags.redundant_pairs = redundant_pairs_idx;
    artifact_info.quality_flags.all_quality_rejected = all_quality_rejected;
    artifact_info.quality_metrics.robust_power = robust_power;
    artifact_info.quality_metrics.kurtosis_values = kurt_values;
    artifact_info.quality_metrics.mean_abs_correlation = mean_abs_corr;

    % Per-channel reason flags, 1 x n_channels each; a channel can carry several.
    % They let the cleaner apply its policy by reason instead of reading a detector score of 100 as a temporal percentage or a removal rule.
    artifact_info.flag_low_power     = flag_low_power;
    artifact_info.flag_abnormal_kurt = flag_abnormal_kurt;
    artifact_info.flag_isolated      = flag_isolated;
    artifact_info.flag_redundant     = flag_redundant;
    artifact_info.flag_temporal_rate = flag_temporal_rate;

end
