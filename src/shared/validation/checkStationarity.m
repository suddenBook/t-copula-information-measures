%CHECKSTATIONARITY Descriptive sliding-window amplitude QC of a multichannel signal.
%
% Computes the channel-averaged variance and absolute mean in sliding windows; Phase 1 applies it to the raw input, before artifact cleaning and band-pass filtering.
% The summaries make amplitude drift and state changes visible, but they are not a stationarity test, give no verdict and never decide inclusion or exclusion.
% render_phase1_stationarity.py draws the diagnostic figures from the returned struct.
%
% Inputs:
%   X       - [samples x channels] real, finite numeric matrix.
%   fs      - sampling rate in Hz.
%   options - optional struct with fields
%     windowSec - window length in s (default 60), capped at the recording length.
%     stepSec   - step between window starts in s (default 30).
%
% Outputs:
%   info - struct with fields
%     window_times       - [nWindows x 1] window centre times in s, sample 1 at t = 0.
%     window_var         - [nWindows x 1] channel-averaged variance per window.
%     window_mean        - [nWindows x 1] channel-averaged absolute mean per window.
%     cv_var             - coefficient of variation of window_var, NaN with fewer than two windows or a non-positive mean variance.
%     nWindows           - number of windows.
%     totalDurationSec   - recording length in s.
%     windowSec, stepSec - window length and step actually used, in s, rounded to whole samples.
%     input_stage        - provenance label of the summarized signal.
%     used_for_exclusion - always false.

function info = checkStationarity(X, fs, options)

    if nargin < 3
        options = struct();
    end

    windowSec = fieldOrDefault(options, 'windowSec', 60);
    stepSec   = fieldOrDefault(options, 'stepSec', 30);

    [nSamples, ~] = size(X);
    validateattributes(X, {'numeric'}, {'2d', 'nonempty', 'real', 'finite'}, ...
        mfilename, 'X');
    validateattributes(fs, {'numeric'}, {'scalar', 'positive', 'finite'}, ...
        mfilename, 'fs');
    validateattributes(windowSec, {'numeric'}, {'scalar', 'positive', 'finite'}, ...
        mfilename, 'options.windowSec');
    validateattributes(stepSec, {'numeric'}, {'scalar', 'positive', 'finite'}, ...
        mfilename, 'options.stepSec');

    windowLen = min(nSamples, max(1, round(windowSec * fs)));
    stepLen   = max(1, round(stepSec * fs));

    % Sliding window positions
    windowStarts = 1:stepLen:(nSamples - windowLen + 1);
    nWindows = length(windowStarts);

    window_var   = zeros(nWindows, 1);
    window_mean  = zeros(nWindows, 1);
    window_times = zeros(nWindows, 1);

    for w = 1:nWindows
        i0 = windowStarts(w);
        i1 = i0 + windowLen - 1;
        seg = X(i0:i1, :);

        ch_var  = var(seg, 0, 1);
        ch_mean = mean(seg, 1);

        window_var(w)   = mean(ch_var);
        window_mean(w)  = mean(abs(ch_mean));
        % Sample 1 is at t = 0, so both endpoints are converted to zero-based sample times before averaging.
        window_times(w) = ((i0 - 1) + (i1 - 1)) / (2 * fs);
    end

    % With one window, or a zero mean variance, the coefficient of variation has no finite meaning and is NaN.
    mean_window_var = mean(window_var);
    if nWindows < 2 || ~isfinite(mean_window_var) || mean_window_var <= 0
        cv_var = NaN;
    else
        cv_var = std(window_var) / mean_window_var;
    end

    % Console output
    fprintf('\n--- Raw-input sliding-window amplitude diagnostic ---\n');
    fprintf('  Stage: before artifact cleaning and band-pass filtering.\n');
    fprintf('  Duration: %.1f sec | %d windows (%.0fs window, %.0fs step)\n', ...
        nSamples / fs, nWindows, windowLen / fs, stepLen / fs);
    fprintf('  Variance — mean: %.1f, range: [%.1f, %.1f], CV: %.3f\n', ...
        mean(window_var), min(window_var), max(window_var), cv_var);
    fprintf(['  Interpretation: descriptive non-exclusionary QC only; no ' ...
        'stationarity test, verdict, or inclusion threshold is applied.\n']);
    fprintf('---------------------------------------------------------\n\n');

    % Pack output
    info.window_times     = window_times;
    info.window_var       = window_var;
    info.window_mean      = window_mean;
    info.cv_var           = cv_var;
    info.nWindows         = nWindows;
    info.totalDurationSec = nSamples / fs;
    info.windowSec        = windowLen / fs;
    info.stepSec          = stepLen / fs;
    info.input_stage      = 'raw_input_pre_clean_pre_bandpass';
    info.used_for_exclusion = false;

end

%% Local helper

function val = fieldOrDefault(s, name, default)
    % s.(name) when the field exists, otherwise default.
    if isfield(s, name)
        val = s.(name);
    else
        val = default;
    end
end
