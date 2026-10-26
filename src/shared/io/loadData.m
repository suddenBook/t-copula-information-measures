%LOADDATA Load a .mat or EEGLAB .set recording as a [samples x channels] matrix.
%
%   [data, metadata] = loadData(filePath, options)
%
% Dispatches on the file extension to loadMatFile or loadEEGFile, checks the result with validateData (a numeric matrix without NaN or Inf), and prints a summary with displayDataInfo.
%
% Inputs:
%   filePath - path to the data file.
%   options  - optional struct with fields
%     fs       - sampling rate in Hz, used when a .mat file stores none (default []).
%     channels - column indices to keep (default [], all).
%
% Outputs:
%   data     - [samples x channels] data matrix.
%   metadata - struct with filePath, fileType, loadTime, fs and the fields the loader adds.

function [data, metadata] = loadData(filePath, options)

    % Default options
    if nargin < 2
        options = struct();
    end

    defaultOptions = struct( ...
        'fs', [], ...
        'channels', [] ...
        );

    options = mergeOptions(defaultOptions, options);

    % Validate input file
    if ~exist(filePath, 'file')
        error('File not found: %s', filePath);
    end

    % Get file extension
    [~, ~, ext] = fileparts(filePath);

    % Initialize metadata
    metadata = struct( ...
        'filePath', filePath, ...
        'fileType', ext(2:end), ... % Remove leading dot
        'loadTime', datetime('now'), ...
        'fs', options.fs ...
        );

    % Load data based on file type
    switch lower(ext)
        case '.mat'
            [data, metadata] = loadMatFile(filePath, options, metadata);

        case '.set'
            [data, metadata] = loadEEGFile(filePath, options, metadata);

        otherwise
            error('Unsupported file format: %s', ext);
    end

    % Validate output data
    validateData(data);

    % Display information
    displayDataInfo(data, metadata);
end
