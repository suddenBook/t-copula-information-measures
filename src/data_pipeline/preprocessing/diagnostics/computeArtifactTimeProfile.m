function profile = computeArtifactTimeProfile(final_artifact_mask, fs, window_sec)
    % COMPUTEARTIFACTTIMEPROFILE Reduce a per-sample artifact mask to a [channels x windows] occupancy matrix.
    %
    % detectSEEGArtifacts returns per-sample diagnostics (final_artifact_mask, artifact_scores and the cell array artifact_types) as large as the signal itself.
    % The cell array also inflates a -v7.3 file about tenfold, so Phase 1 drops all three before saving.
    % This function keeps the one quantity the artifact-timing figure needs: the fraction of artifact samples per channel in each fixed-length time window.
    % At 1 s windows that is one double per channel per second of recording, a small fraction of the per-sample mask.
    % The profile is indexed by the pre-cleaning channel set, like artifact_info.channel_artifact_pct, so the two can be plotted against each other and against clean_info.excluded_channels.
    % The last window is usually partial, and its fraction divides by the samples it contains rather than the nominal window length, so a short trailing window is not reported as low-artifact.
    %
    % Inputs:
    %   final_artifact_mask - [samples x channels] logical (or numeric 0/1).
    %   fs                  - sampling rate in Hz.
    %   window_sec          - window length in s (default 1).
    %
    % Outputs:
    %   profile - struct with fields
    %     fraction           - [channels x windows] double in [0, 1].
    %     window_sec         - window length used, in s, rounded to whole samples.
    %     window_centers_sec - [1 x windows] window midpoints in s, for the figure's time axis.
    %     n_windows, n_channels, n_samples - sizes of the profile and of the mask.
    %     note               - provenance string.

    if nargin < 3 || isempty(window_sec)
        window_sec = 1;
    end
    validateattributes(fs, {'numeric'}, {'scalar', 'positive', 'finite'}, mfilename, 'fs');
    validateattributes(window_sec, {'numeric'}, {'scalar', 'positive', 'finite'}, ...
                       mfilename, 'window_sec');

    mask = double(logical(final_artifact_mask));
    [n_samples, n_channels] = size(mask);

    win = max(1, round(window_sec * fs));
    n_windows = max(1, ceil(n_samples / win));

    % Zero-pad to whole windows, sum each window, and divide by its true sample count so the partial last window is not diluted.
    pad = n_windows * win - n_samples;
    if pad > 0
        mask = [mask; zeros(pad, n_channels)];
    end
    counts = reshape(sum(reshape(mask, win, n_windows, n_channels), 1), n_windows, n_channels);

    samples_per_window = repmat(win, n_windows, 1);
    samples_per_window(end) = win - pad;

    fraction = (counts ./ samples_per_window).';   % [channels x windows]

    edges = (0:n_windows) * win;
    edges(end) = n_samples;
    centers = (edges(1:end - 1) + edges(2:end)) / (2 * fs);

    profile = struct( ...
                     'fraction',           fraction, ...
                     'window_sec',         win / fs, ...
                     'window_centers_sec', centers, ...
                     'n_windows',          n_windows, ...
                     'n_channels',         n_channels, ...
                     'n_samples',          n_samples, ...
                     'note', sprintf(['fraction of artifact samples per %g s window, from ' ...
                         'detectSEEGArtifacts final_artifact_mask; original ' ...
                         '(pre-cleaning) channel indexing'], win / fs));
end
