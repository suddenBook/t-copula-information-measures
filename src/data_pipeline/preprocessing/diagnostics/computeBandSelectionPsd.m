%COMPUTEBANDSELECTIONPSD Across-channel PSD summary from which the band-pass range is chosen.
%
% Computes the quantity the interactive band-pass picker plots and Phase 1 saves as the band-choice evidence: the per-channel Welch PSD reduced to the across-channel mean of |PSD|^0.5 with a +/- 1 sd envelope.
% runInteractiveBandpassSelection plots the output of this function, so the saved evidence and the plotted figure are the same numbers, and Phase 1 saves the summary on the preset path too.
% It is a QC diagnostic: the analysis input is the saved signal (X, X_filtered), not this PSD.
% pwelch runs with its default segmentation; render_phase1_psd uses a hamming(2*fs) window for a separate channel-1 view in dB.
%
% Inputs:
%   X  - [samples x channels] numeric matrix, non-empty.
%   fs - sampling rate in Hz, positive scalar.
%
% Outputs:
%   psd - struct with fields
%     freq       - [nFreq x 1] frequency vector in Hz, from pwelch.
%     mean_amp   - [nFreq x 1] across-channel mean of sqrt(abs(PSD)).
%     sd_amp     - [nFreq x 1] across-channel standard deviation of sqrt(abs(PSD)).
%     fs         - sampling rate in Hz.
%     n_channels - channel count.
%     n_samples  - sample count.
%     method     - provenance string describing the computation.
%
% The [nFreq x channels] per-channel PSD matrix is formed internally and not returned, because only the across-channel summary is the band-choice evidence.

function psd = computeBandSelectionPsd(X, fs)

    if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
        error('computeBandSelectionPsd:badX', ...
            'X must be a non-empty numeric [samples x channels] matrix.');
    end
    if ~isscalar(fs) || ~isnumeric(fs) || fs <= 0
        error('computeBandSelectionPsd:badFs', ...
            'fs must be a positive scalar (Hz).');
    end

    [n_samples, n_channels] = size(X);

    % Welch PSD per channel with the default segmentation; the first call also returns the shared frequency vector.
    [psd_first, freq] = pwelch(double(X(:, 1)), [], [], [], fs);
    psd_matrix = zeros(numel(freq), n_channels);
    psd_matrix(:, 1) = psd_first;
    for ch = 2:n_channels
        psd_matrix(:, ch) = pwelch(double(X(:, ch)), [], [], [], fs);
    end

    amp = sqrt(abs(psd_matrix));

    psd = struct( ...
        'freq',       freq, ...
        'mean_amp',   mean(amp, 2), ...
        'sd_amp',     std(amp, 0, 2), ...
        'fs',         double(fs), ...
        'n_channels', n_channels, ...
        'n_samples',  n_samples, ...
        'method',     'pwelch default segmentation; amp=sqrt(abs(PSD)); across-channel mean +/- 1 sd');
end
