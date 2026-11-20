% FILTERDATA Zero-phase Butterworth band-pass filter applied to each column.
%
%   X_filt = filterData(X, fs, freqRange, order)
%
% Designs a band-pass IIR Butterworth filter with designfilt and applies it forward and backward with filtfilt, so the output has no phase shift.
%
% Inputs:
%   X         - [samples x channels] signal.
%   fs        - sampling rate in Hz.
%   freqRange - [low high] half-power frequencies of the designed filter in Hz (default [7 13]).
%   order     - designfilt filter order (default 4).
%
% Outputs:
%   X_filt - filtered signal, the same size as X.

function X_filt = filterData(X, fs, freqRange, order)

    if nargin < 3 || isempty(freqRange)
        freqRange = [7, 13];
    end

    if nargin < 4 || isempty(order)
        order = 4;
    end

    filterObj = designfilt('bandpassiir', ...
                           'FilterOrder', order, ...
                           'HalfPowerFrequency1', freqRange(1), ...
                           'HalfPowerFrequency2', freqRange(2), ...
                           'SampleRate', fs, ...
                           'DesignMethod', 'butter');

    X_filt = filtfilt(filterObj, X);

end
