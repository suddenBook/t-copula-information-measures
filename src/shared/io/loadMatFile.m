%LOADMATFILE Load the data matrix of a .mat file by its variable name.
%
% The data are the first of the variables X, sampledData, X_gauss and X_pvii that the file holds; a file with none of them raises an error.
% The sampling rate is the file's fs, else options.fs, else 250 Hz with a warning.
% A struct named metadata in the file is kept as metadata.source.
%
% Inputs:
%   filePath - path to the .mat file.
%   options  - options struct with fs and channels (column indices to keep; empty keeps all).
%   metadata - metadata struct to extend.
%
% Outputs:
%   data     - [samples x channels] data matrix.
%   metadata - the extended metadata struct.

function [data, metadata] = loadMatFile(filePath, options, metadata)

    matData = load(filePath);

    % Check for expected variable names
    if isfield(matData, 'X')
        data = matData.X;
    elseif isfield(matData, 'sampledData')
        data = matData.sampledData;
    elseif isfield(matData, 'X_gauss')
        data = matData.X_gauss;
    elseif isfield(matData, 'X_pvii')
        data = matData.X_pvii;
    else
        error('MAT file must contain variable named ''X'' or ''sampledData'' or ''X_gauss'' or ''X_pvii''');
    end

    % Apply channel selection if specified
    if ~isempty(options.channels)
        data = data(:, options.channels);
        metadata.selectedChannels = options.channels;
    end

    % Update metadata
    metadata.originalSize = size(data);
    metadata.variableNames = fieldnames(matData);
    if isfield(matData, 'fs')
        metadata.fs = matData.fs;
    elseif ~isempty(options.fs)
        metadata.fs = options.fs;
    else
        warning('loadMatFile:noFs', 'No sampling rate found in file or options, defaulting to 250 Hz.');
        metadata.fs = 250;
    end

    % Carry over metadata struct from file if present
    if isfield(matData, 'metadata')
        metadata.source = matData.metadata;
    end

end
