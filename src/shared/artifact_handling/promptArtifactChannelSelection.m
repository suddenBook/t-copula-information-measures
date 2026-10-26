%PROMPTARTIFACTCHANNELSELECTION Ask the operator which detector candidates to remove.
%
% Lists the candidates with their reasons and reads the answer at the command line.
% Enter or 'all' removes every candidate, 'none' (or 'keep', '[]') keeps them all, and a list such as [3 7 12] or 3 7 12 removes that subset.
% An answer that cannot be parsed, or that names a channel outside the list, is asked again.
%
% Inputs:
%   candidate_channels - detector candidate indices (detector score above severe_threshold).
%   reasons            - optional cell of reason strings, one per candidate in ascending channel order.
%   severe_threshold   - optional detector-score threshold that defines the candidates, shown in the prompt.
%
% Outputs:
%   selected_channels - row vector of the channels chosen for removal.

function selected_channels = promptArtifactChannelSelection(candidate_channels, reasons, severe_threshold)

    if nargin < 2 || isempty(reasons)
        reasons = cell(numel(candidate_channels), 1);
    end

    if nargin < 3
        severe_threshold = [];
    end

    candidate_channels = unique(candidate_channels(:)');
    selected_channels = candidate_channels;

    if isempty(candidate_channels)
        fprintf('\nInteractive channel selection: the detector candidate list is empty.\n');
        return;
    end

    fprintf('\nInteractive channel selection enabled.\n');
    if ~isempty(severe_threshold)
        fprintf(['Review the channel plot, then choose which candidates from the ' ...
            'legacy score > %.1f ledger to exclude.\n'], ...
            severe_threshold);
    else
        fprintf('Review the channel plot, then choose which detected channels to exclude.\n');
    end

    fprintf('Candidate channels:\n');
    for idx = 1:numel(candidate_channels)
        ch = candidate_channels(idx);
        if idx <= numel(reasons) && ~isempty(reasons{idx})
            fprintf('  %d: %s\n', ch, reasons{idx});
        else
            fprintf('  %d\n', ch);
        end
    end

    fprintf(['Enter ''all'' to remove every listed channel, ''none'' to keep them all,\n' ...
        'or type a subset such as [3 7 12] or 3 7 12.\n']);

    while true
        user_input = strtrim(input('Channel selection [all]: ', 's'));

        if isempty(user_input) || strcmpi(user_input, 'all')
            selected_channels = candidate_channels;
            break;
        end

        if any(strcmpi(user_input, {'none', 'keep', '[]'}))
            selected_channels = [];
            break;
        end

        sanitized_input = regexprep(user_input, '[\[\],;]', ' ');
        sanitized_input = regexprep(sanitized_input, '\s+', ' ');
        parsed_channels = sscanf(strtrim(sanitized_input), '%f')';

        if isempty(parsed_channels) || any(~isfinite(parsed_channels)) || ...
                any(parsed_channels ~= round(parsed_channels))
            fprintf('Input not understood. Type ''all'', ''none'', or integer channel numbers.\n');
            continue;
        end

        parsed_channels = unique(parsed_channels);
        invalid_channels = setdiff(parsed_channels, candidate_channels);
        if ~isempty(invalid_channels)
            fprintf('Only detected channels can be selected here. Valid choices: %s\n', ...
                mat2str(candidate_channels));
            fprintf('Invalid channel numbers: %s\n', mat2str(invalid_channels));
            continue;
        end

        selected_channels = parsed_channels;
        break;
    end

    fprintf('Final exclusion set: %s\n', mat2str(selected_channels));

end
