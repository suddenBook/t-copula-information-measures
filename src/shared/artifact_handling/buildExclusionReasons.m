% BUILDEXCLUSIONREASONS Build a readable reason string for each listed channel.
%
% Each string names the channel-level quality flags the channel carries (low power, abnormal kurtosis, isolated, redundant) and counts each per-sample artifact tag (EXT, GRAD, FLAT, SUST, MULTI) on it.
% A channel with neither falls back to its detector score, channel_artifact_pct.
% The result depends only on the indices and artifact_info, so the same function describes the candidate list and the removed set.
%
% Inputs:
%   severe_channels - channel indices to describe, on the pre-cleaning channel axis.
%   artifact_info   - struct from detectSEEGArtifacts.
%
% Outputs:
%   reasons - k x 1 cell of char, one reason string per listed channel.

function reasons = buildExclusionReasons(severe_channels, artifact_info)

    reasons = cell(length(severe_channels), 1);

    % Quality flag labels
    flag_names = {'low_power_channels', 'abnormal_kurt_channels', ...
                  'isolated_channels', 'redundant_channels'};
    flag_labels = {'low power', 'abnormal kurtosis', 'isolated', 'redundant'};

    has_quality_flags = isfield(artifact_info, 'quality_flags');
    has_artifact_types = isfield(artifact_info, 'artifact_types');

    for i = 1:length(severe_channels)
        ch = severe_channels(i);
        parts = {};

        % Check quality flags
        if has_quality_flags
            for f = 1:length(flag_names)
                if isfield(artifact_info.quality_flags, flag_names{f})
                    flagged = artifact_info.quality_flags.(flag_names{f});
                    if ismember(ch, flagged)
                        parts{end + 1} = flag_labels{f}; %#ok<AGROW>
                    end
                end
            end
        end

        % Summarize artifact types from the per-sample cell array
        if has_artifact_types && ch <= size(artifact_info.artifact_types, 2)
            ch_types = artifact_info.artifact_types(:, ch);
            ch_types_nonempty = ch_types(~cellfun(@isempty, ch_types));
            if ~isempty(ch_types_nonempty)
                all_tags = strjoin(ch_types_nonempty', '');
                tag_map = {'EXT', 'extreme value'; 'GRAD', 'gradient'; ...
                           'FLAT', 'flatline'; 'SUST', 'sustained'; 'MULTI', 'multi-channel'};
                for t = 1:size(tag_map, 1)
                    count = length(strfind(all_tags, tag_map{t, 1}));
                    if count > 0
                        parts{end + 1} = sprintf('%s x%d', tag_map{t, 2}, count); %#ok<AGROW>
                    end
                end
            end
        end

        % Fall back to the detector score when no flag or tag applies
        if isempty(parts)
            pct = artifact_info.channel_artifact_pct(ch);
            parts = {sprintf('legacy detector score %.1f', pct)};
        end

        reasons{i} = strjoin(parts, ', ');
    end

end
