%LOADEEGFILE Load an EEGLAB .set file as a [samples x channels] matrix.
%
% Requires EEGLAB on the path.
% Epoched data are concatenated along time, and the sampling rate, original size, epoch layout and channel labels are added to the metadata.
%
% Inputs:
%   filePath - path to the .set file.
%   options  - options struct; options.channels selects columns (empty keeps all).
%   metadata - metadata struct to extend.
%
% Outputs:
%   data     - [samples x channels] data matrix.
%   metadata - the extended metadata struct.

function [data, metadata] = loadEEGFile(filePath, options, metadata)

    % Check for EEGLAB
    if ~exist('eeglab', 'file')
        error('EEGLAB is required for loading .set files');
    end

    % Load EEGLAB dataset
    EEG = pop_loadset(filePath);

    % Extract data and reshape if necessary
    if ndims(EEG.data) == 3
        % Epoched data, concatenated along time
        data = reshape(EEG.data, EEG.nbchan, [])';
    else
        % Continuous data
        data = EEG.data';
    end

    % Apply channel selection
    if ~isempty(options.channels)
        data = data(:, options.channels);
        metadata.selectedChannels = options.channels;
    end

    % Update metadata
    metadata.fs = EEG.srate;
    metadata.originalSize = size(EEG.data);
    metadata.epochInfo = struct( ...
        'hasEpochs', ndims(EEG.data) == 3, ...
        'numEpochs', EEG.trials, ...
        'epochLength', EEG.pnts ...
        );
    metadata.channelLabels = {EEG.chanlocs.labels};

end
