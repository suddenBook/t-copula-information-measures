% DISPLAYDATAINFO Print a summary of loaded data and their metadata.
%
% Prints the file, its type, the matrix size, and, when the metadata carry them, the sampling rate, the duration and the epoch layout.
%
% Inputs:
%   data     - [samples x channels] data matrix.
%   metadata - struct from loadData, with at least filePath and fileType.

function displayDataInfo(data, metadata)

    fprintf('\n=== Data Loading Summary ===\n');
    fprintf('File: %s\n', metadata.filePath);
    fprintf('Type: %s\n', metadata.fileType);
    fprintf('Size: %d samples × %d channels\n', size(data));

    if isfield(metadata, 'fs') && ~isempty(metadata.fs)
        fprintf('Sampling rate: %.1f Hz\n', metadata.fs);
        fprintf('Duration: %.2f seconds\n', size(data, 1) / metadata.fs);
    end

    if isfield(metadata, 'epochInfo') && metadata.epochInfo.hasEpochs
        fprintf('Epochs: %d × %d samples\n', ...
                metadata.epochInfo.numEpochs, ...
                metadata.epochInfo.epochLength);
    end
    fprintf('========================\n\n');

end
