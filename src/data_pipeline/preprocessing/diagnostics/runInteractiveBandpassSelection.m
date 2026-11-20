% RUNINTERACTIVEBANDPASSSELECTION Show the log-log PSD, prompt for a band-pass range and save the decision.
%
% Draws an interactive figure used as a human input surface: the operator reads the log-log PSD and types the [low high] band at the input() prompt.
%
%   [bandpass_range, decision] = runInteractiveBandpassSelection(X, fs, dataset_name, sidecar_path)
%
% Inputs:
%   X            - [samples x channels] numeric matrix.
%   fs           - sampling rate in Hz.
%   dataset_name - char or string label shown in the figure title and written to the sidecar.
%   sidecar_path - path of bandpass_decision.json; '' (the default) skips the file, for example in tests.
%
% Behaviour:
%   1. Summarize the per-channel Welch PSD as the across-channel mean of |PSD|^0.5 with a +/- 1 sd band (computeBandSelectionPsd).
%   2. Plot it on log-log axes, with dataset_name, fs, the channel count and the sample count in the title.
%   3. Prompt for the band as a MATLAB array such as [1 70]; there is no default band, so an empty answer is an error.
%   4. Validate a 2-element numeric range with 0 < low < high < fs/2.
%   5. Write mode, range, fs, channel_count, dataset_name and timestamp to sidecar_path when one is given.
%
% Outputs:
%   bandpass_range - 1 x 2 [low high] in Hz.
%   decision       - struct with the fields written to the sidecar.
%
% Without the MATLAB desktop (for example under matlab -batch) input() cannot be relied on, so the function raises runInteractiveBandpassSelection:nonInteractive after drawing the figure.
% A numeric [low high] preset in the Phase 1 recording table, or a replayed saved decision, avoids the prompt.

function [bandpass_range, decision] = runInteractiveBandpassSelection( ...
                                                                      X, fs, dataset_name, sidecar_path)

    if nargin < 4
        sidecar_path = '';
    end

    if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
        error('runInteractiveBandpassSelection:badX', ...
              'X must be a non-empty numeric [samples x channels] matrix.');
    end
    if ~isscalar(fs) || ~isnumeric(fs) || fs <= 0
        error('runInteractiveBandpassSelection:badFs', ...
              'fs must be a positive scalar (Hz).');
    end
    dataset_name = char(string(dataset_name));

    [n_samples, n_channels] = size(X);

    %% Across-channel PSD summary (the band-choice evidence)
    % computeBandSelectionPsd also produces the psd_band_selection evidence that Phase 1 saves, so the plot and the saved summary are the same numbers.
    psd = computeBandSelectionPsd(X, fs);
    freq = psd.freq;
    mean_amp = psd.mean_amp;
    sd_amp = psd.sd_amp;

    %% Log-log PSD figure
    figure('Name', sprintf('Bandpass interactive: %s', dataset_name), ...
           'Position', [120, 120, 900, 600]);
    valid = freq > 0 & mean_amp > 0;
    f = freq(valid);
    m = mean_amp(valid);
    s = sd_amp(valid);

    upper = m + s;
    lower = max(m - s, eps);
    fill_x = [f; flipud(f)];
    fill_y = [lower; flipud(upper)];

    fill(fill_x, fill_y, [0.85 0.85 0.95], 'EdgeColor', 'none', ...
         'DisplayName', '\pm 1 sd across channels');
    hold on;
    loglog(f, m, 'b-', 'LineWidth', 1.4, ...
           'DisplayName', 'mean |PSD|^{0.5} across channels');
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Frequency (Hz)');
    ylabel('|PSD|^{0.5} (a.u.)');
    grid on;
    legend('Location', 'best', 'Interpreter', 'tex');
    title(sprintf('PSD (log-log) — %s — fs=%g Hz, %d channels, %d samples', ...
                  dataset_name, fs, n_channels, n_samples), 'Interpreter', 'none');
    drawnow;

    %% Prompt
    if ~usejava('desktop')
        error('runInteractiveBandpassSelection:nonInteractive', ...
              ['Bandpass interactive mode requires a desktop MATLAB ', ...
               'session. Set PHASE2_BANDPASS=''preset'' to bypass the ', ...
               'prompt for unattended re-runs, or supply a numeric ', ...
               '[low high] preset on the dataset_configs row.']);
    end

    fprintf('Bandpass interactive: enter the [low_hz high_hz] range as MATLAB array (e.g. [1 70]):\n');
    raw = input('> ');

    bandpass_range = validateBandpassRange(raw, fs);

    fprintf('runInteractiveBandpassSelection: chosen bandpass = [%.3f %.3f] Hz.\n', ...
            bandpass_range(1), bandpass_range(2));

    %% Save the decision (an empty sidecar_path skips the file)
    decision = struct( ...
                      'bandpass_mode',   'interactive', ...
                      'bandpass_range',  double(bandpass_range), ...
                      'fs',              double(fs), ...
                      'channel_count',   double(n_channels), ...
                      'dataset_name',    dataset_name, ...
                      'timestamp',       char(datetime('now', 'TimeZone', 'UTC', ...
                                         'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')));

    if ~isempty(sidecar_path)
        writeBandpassDecisionSidecar(sidecar_path, decision);
    end
end

function range = validateBandpassRange(raw, fs)
    % The typed range as a 1 x 2 double, or an error unless it holds two numbers with 0 < low < high < fs/2.
    if ~isnumeric(raw) || numel(raw) ~= 2
        error('runInteractiveBandpassSelection:badInput', ...
              'Bandpass range must be a 2-element numeric array, got class=%s numel=%d.', ...
              class(raw), numel(raw));
    end
    range = double(raw(:)).';
    nyquist = fs / 2;
    if ~(range(1) > 0 && range(2) > range(1) && range(2) < nyquist)
        error('runInteractiveBandpassSelection:rangeOutOfBounds', ...
              ['Bandpass range must satisfy 0 < low < high < fs/2 = %.6g, ', ...
               'got [%.6g %.6g].'], nyquist, range(1), range(2));
    end
end

function writeBandpassDecisionSidecar(path, decision)
    % Write the decision as pretty-printed JSON, reporting an I/O failure on stderr instead of raising it.
    parent_dir = fileparts(path);
    if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
        try
            mkdir(parent_dir);
        catch
            fprintf(2, 'Could not create sidecar dir %s; skipping bandpass decision JSON.\n', ...
                    parent_dir);
            return
        end
    end
    try
        fid = fopen(path, 'w');
        if fid == -1
            fprintf(2, 'Could not open sidecar %s for writing.\n', path);
            return
        end
        cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(decision, 'PrettyPrint', true));
    catch sidecar_err
        fprintf(2, 'Failed to write bandpass_decision.json: %s\n', sidecar_err.message);
    end
end
