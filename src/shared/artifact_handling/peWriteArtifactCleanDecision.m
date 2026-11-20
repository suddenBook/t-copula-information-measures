function [payload, wrote] = peWriteArtifactCleanDecision(sidecar_path, decision)
    % PEWRITEARTIFACTCLEANDECISION Write one artifact_clean_decision.json sidecar.
    %
    % The sidecar is a reporting record: which channels the cleaning policy removed, which channels the detector listed as candidates (detector score above severe_threshold), and the reason string behind each.
    % No fitted quantity depends on it.
    % cleanSEEGArtifacts writes it through this function.
    %
    % The function guarantees two properties of the file:
    %   1. Pairing: reasons{k} describes flagged_channels(k), the removed set, and auto_detected_reasons{k} describes auto_detected_channels(k), the candidate list; the two sets differ in length, so each has its own reason array.
    %   2. Array form: every channel-index and per-channel flag field is a JSON array, because jsonencode writes a 1 x 1 numeric as a bare scalar and a one-channel list would otherwise read as a number.
    % A decision with a missing field, an unknown mode, invalid channel indices or a reason count that does not match its channels raises an error.
    %
    % Inputs:
    %   sidecar_path - char, destination path; missing parent folders are created.
    %   decision     - struct with these fields, channel indices 1-based on the pre-cleaning channel axis:
    %     mode                 - char, 'auto' or 'interactive'.
    %     severe_threshold_pct - scalar, the detector-score threshold of the candidate list.
    %     removed_channels     - numeric vector of the removed channels (may be empty).
    %     removed_reasons      - cell of char, one per removed channel.
    %     candidate_channels   - numeric vector of the candidate list (may be empty).
    %     candidate_reasons    - cell of char, one per candidate channel.
    %   Optional fields:
    %     channel_labels - cell of labels for every channel; a missing or empty entry falls back to the numeric index.
    %     flag_payload   - struct of extra per-channel arrays appended to the file (flag_low_power, detected_isolated, ...).
    %     timestamp      - char; defaults to the current UTC time.
    %
    % Outputs:
    %   payload - struct exactly as encoded, so a caller or test can inspect it without reading the file.
    %   wrote   - logical, true when the file was written; an I/O failure is reported on stderr and returns false instead of raising, so a sidecar problem never stops the phase.

    arguments
        sidecar_path (1, :) char {mustBeNonempty}
        decision (1, 1) struct
    end

    required = {'mode', 'severe_threshold_pct', 'removed_channels', ...
                'removed_reasons', 'candidate_channels', 'candidate_reasons'};
    for k = 1:numel(required)
        assert(isfield(decision, required{k}), ...
               'peWriteArtifactCleanDecision:MissingField', ...
               'decision is missing required field "%s" (sidecar %s).', ...
               required{k}, sidecar_path);
    end
    assert(ismember(decision.mode, {'auto', 'interactive'}), ...
           'peWriteArtifactCleanDecision:BadMode', ...
           'decision.mode must be ''auto'' or ''interactive'', got "%s" (sidecar %s).', ...
           decision.mode, sidecar_path);

    removed = localChannelRow(decision.removed_channels, 'removed_channels', sidecar_path);
    candidates = localChannelRow(decision.candidate_channels, 'candidate_channels', sidecar_path);
    removed_reasons = localReasonRow(decision.removed_reasons, numel(removed), ...
                                     'removed_reasons', 'removed_channels', sidecar_path);
    candidate_reasons = localReasonRow(decision.candidate_reasons, numel(candidates), ...
                                       'candidate_reasons', 'candidate_channels', sidecar_path);

    channel_labels = {};
    if isfield(decision, 'channel_labels') && iscell(decision.channel_labels)
        channel_labels = decision.channel_labels;
    end
    flag_payload = struct();
    if isfield(decision, 'flag_payload') && isstruct(decision.flag_payload) && ...
            isscalar(decision.flag_payload)
        flag_payload = decision.flag_payload;
    end
    if isfield(decision, 'timestamp') && ~isempty(decision.timestamp)
        timestamp = char(string(decision.timestamp));
    else
        timestamp = char(datetime('now', 'TimeZone', 'UTC', ...
                                  'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    end

    % The key order is fixed, so two sidecars of the same recording diff line by line.
    % severe_threshold_pct is the score threshold of the candidate list, not a severity verdict.
    payload = struct();
    payload.mode = decision.mode;
    payload.severe_threshold_pct = double(decision.severe_threshold_pct);
    payload.auto_detected_channels = localJsonNumericArray(candidates);
    payload.flagged_channels = localJsonNumericArray(removed);
    payload.flagged_channel_labels = localChannelLabels(removed, channel_labels);
    payload.reasons = removed_reasons;
    payload.auto_detected_reasons = candidate_reasons;
    payload.timestamp = timestamp;

    % The per-reason classification (the five reason flags and the detected-but-kept lists) is appended field by field, numeric fields in JSON array form, so the record covers all five reasons.
    payload_fields = fieldnames(flag_payload);
    for k = 1:numel(payload_fields)
        field = payload_fields{k};
        value = flag_payload.(field);
        if isnumeric(value) || islogical(value)
            payload.(field) = localJsonNumericArray(value);
        else
            payload.(field) = value;
        end
    end

    wrote = localWriteJson(sidecar_path, payload);
end

function row = localChannelRow(value, name, sidecar_path)
    % Normalise a channel-index list to a double row vector of 1-based integers.
    assert(isnumeric(value) || islogical(value), ...
           'peWriteArtifactCleanDecision:BadChannelList', ...
           'decision.%s must be numeric, got %s (sidecar %s).', ...
           name, class(value), sidecar_path);
    row = double(value(:)');
    assert(all(row == fix(row) & row >= 1), ...
           'peWriteArtifactCleanDecision:BadChannelIndex', ...
           'decision.%s must hold 1-based integer channel indices (sidecar %s).', ...
           name, sidecar_path);
end

function text = localReasonRow(value, n_expected, name, channel_field, sidecar_path)
    % Normalise a reason list to a 1 x N cell of char, one entry per channel it describes.
    if isempty(value)
        value = {};
    end
    if isstring(value) || ischar(value)
        value = cellstr(value);
    end
    assert(iscell(value), 'peWriteArtifactCleanDecision:BadReasonList', ...
           'decision.%s must be a cell array of strings, got %s (sidecar %s).', ...
           name, class(value), sidecar_path);
    assert(numel(value) == n_expected, ...
           'peWriteArtifactCleanDecision:ReasonCountMismatch', ...
           ['decision.%s has %d entries but decision.%s has %d channels; ' ...
            'reasons{k} must describe channel k (sidecar %s).'], ...
           name, numel(value), channel_field, n_expected, sidecar_path);

    text = cell(1, numel(value));
    for k = 1:numel(value)
        if isempty(value{k})
            text{k} = '';
        else
            text{k} = char(string(value{k}));
        end
    end
end

function labels = localChannelLabels(channels, channel_labels)
    % Human label per channel where one exists, numeric index otherwise.
    labels = cell(1, numel(channels));
    for k = 1:numel(channels)
        ch = channels(k);
        if iscell(channel_labels) && ch >= 1 && ch <= numel(channel_labels) && ...
                ~isempty(channel_labels{ch})
            labels{k} = char(string(channel_labels{ch}));
        else
            labels{k} = sprintf('%d', ch);
        end
    end
end

function out = localJsonNumericArray(value)
    % Force the JSON array form for a channel-index or per-channel flag list.
    %
    % jsonencode writes a 1 x 1 numeric as a bare scalar (24) but a one-element cell as an array ([24]), so a single element is wrapped in a cell.
    % Multi-element and empty numerics already encode as arrays and pass through unchanged.
    row = double(value(:)');
    if isscalar(row)
        out = {row};
    else
        out = row;
    end
end

function wrote = localWriteJson(path, payload)
    % Write the encoded payload, reporting rather than raising on I/O failure.
    wrote = false;
    parent_dir = fileparts(path);
    if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
        try
            mkdir(parent_dir);
        catch
            fprintf(2, 'Could not create sidecar dir %s; skipping decision JSON.\n', ...
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
        fwrite(fid, jsonencode(payload, 'PrettyPrint', true));
        wrote = true;
    catch sidecar_err
        fprintf(2, 'Failed to write artifact_clean_decision.json: %s\n', ...
                sidecar_err.message);
    end
end
