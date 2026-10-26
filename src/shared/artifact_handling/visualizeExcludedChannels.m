%VISUALIZEEXCLUDEDCHANNELS Plot the detector candidates for interactive review.
%
% Draws an interactive figure used as a human input surface: cleanSEEGArtifacts calls it only in interactive mode, before the promptArtifactChannelSelection prompt.
% In automatic mode render_phase1_excluded_channels.py renders the same traces instead.
% Each candidate's full time series is shown with its detector score and reasons, at most eight channels per figure.
% The display supports the operator's decision; it does not remove a channel or judge it bad.
%
% Inputs:
%   X                    - [samples x channels] signal before cleaning.
%   severe_channels      - candidate channel indices, valid columns of X.
%   channel_artifact_pct - detector score per channel; 100 also marks a channel-level flag.
%   fs                   - sampling rate in Hz.
%   reasons              - optional cell of reason strings, one per candidate.

function visualizeExcludedChannels(X, severe_channels, channel_artifact_pct, fs, reasons)

    if nargin < 5
        reasons = {};
    end

    n_excluded = length(severe_channels);
    if n_excluded == 0
        return;
    end

    t = (0:size(X, 1) - 1) / fs;

    % At most eight channels per figure; more candidates spread over several figures
    max_per_figure = 8;
    n_figures = ceil(n_excluded / max_per_figure);

    for fig_idx = 1:n_figures
        ch_start = (fig_idx - 1) * max_per_figure + 1;
        ch_end = min(fig_idx * max_per_figure, n_excluded);
        n_plots = ch_end - ch_start + 1;

        figure('Position', [100, 50, 1200, 180 * n_plots], ...
            'Name', sprintf('Excluded Channels (%d/%d)', fig_idx, n_figures));

        for i = 1:n_plots
            idx = ch_start + i - 1;
            ch = severe_channels(idx);
            subplot(n_plots, 1, i);
            plot(t, X(:, ch), 'Color', [0.8 0.2 0.2]);
            ylabel(sprintf('Ch %d', ch));

            % Title with the detector score and the reasons
            title_str = sprintf('Channel %d — legacy detector score %.1f', ...
                ch, channel_artifact_pct(ch));
            if ~isempty(reasons) && idx <= length(reasons)
                title_str = sprintf('%s  |  %s', title_str, reasons{idx});
            end
            title(title_str, 'FontSize', 10, 'Interpreter', 'none');

            xlim([t(1), t(end)]);
            box off;

            if i < n_plots
                set(gca, 'XTickLabel', []);
            end
        end

        xlabel('Time (s)');
        sgtitle('Detector candidates — interactive review, no automatic verdict', ...
            'FontSize', 13, 'FontWeight', 'bold');
    end

end
