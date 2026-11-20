function params = peArtifactDetectionDefaults()
    % PEARTIFACTDETECTIONDEFAULTS Return the defaults of every setting detectSEEGArtifacts reads from its params struct.
    %
    % The tunable settings live in this one function, so the Phase 1 settings identity covers every detector setting a caller can change.
    % detectSEEGArtifacts also fixes some constants in its own code, outside the identity, such as the 500 microvolt-per-ms gradient cap, the 50 ms sustained-run window, the 10 ms dilation and the score weight of each detector.
    % Phase 1 overrides only the flatline threshold of the recording's amplitude profile and the private random seed.

    params = struct( ...
                    'percentile_threshold', 99.5, ...  % percentile of |x - median| for the extreme-value threshold
                    'mad_factor', 8, ...  % robust standard deviations for the extreme-value threshold
                    'gradient_percentile', 99, ...  % percentile of |diff(x)| for the rapid-change threshold
                    'min_artifact_duration', 0.01, ...  % s; shorter marked runs are dropped
                    'merge_gap', 0.1, ...  % s; regions closer than this are merged
                    'multi_channel_ratio', 0.2, ...  % an event on more than this fraction of the channels is multi-channel or global
                    'correlation_window', 0.05, ...  % s; window of the multi-channel coherence check
                    'correlation_threshold', 0.9, ...  % mean |corr| that confirms a multi-channel event
                    'powerline_freq', [50, 60], ...  % Hz
                    'powerline_snr', 20, ...  % dB of the line-frequency bin over the median of its neighbouring bins
                    'flatline_std', 1.0, ...  % absolute flatline threshold on a window's std, input units
                    'flatline_relative_std', [], ...  % flatline threshold as a fraction of robust std; replaces flatline_std when set
                    'random_seed', [], ...  % seed of the private stream for channel sampling; [] uses the global stream
                    'flatline_window', 0.05, ...  % s; window of the flatline check
                    'score_threshold', 0.5, ...  % per-sample score from which a sample is an artifact
                    'output_channel_artifacts', true, ...  % return per-channel artifact regions
                    'max_channel_artifacts', 10, ...  % highest-load channels listed in the console summary
                    'low_power_fraction', 0.1, ...  % a robust std below this fraction of the channels' median flags low power
                    'kurtosis_mad_factor', 5, ...  % a kurtosis more than this many scaled MADs from the channels' median flags abnormal kurtosis
                    'isolation_mad_factor', 3, ...  % a mean |corr| more than this many scaled MADs below the channels' median flags isolation
                    'isolation_abs_threshold', 0.02, ...  % mean |corr| below this also flags isolation
                    'redundancy_threshold', 0.98, ...  % |corr| of a near-duplicate channel pair
                    'temporal_rate_threshold_pct', 10);  % percent of marked samples that flags a high temporal rate
end
