%CHECKMARGINALDISTRIBUTIONS Per-channel kurtosis and skewness of one or more signal matrices.
%
% Descriptive QC only: ranks remove the marginals before the copula analysis, so there is no marginal-normality test, no copula goodness-of-fit test and no inclusion or exclusion decision.
% A channel with fewer than two samples, any non-finite value or a constant signal gets NaN.
% render_phase1_marginals.py draws the diagnostic figures (histograms, QQ plots, kurtosis and skewness bars, pooled-z comparison) from the returned stats and the saved data.
%
%   stats = checkMarginalDistributions(X)
%   stats = checkMarginalDistributions({X_a, X_b}, {'Condition A', 'Condition B'})
%
% Inputs:
%   datasets - [samples x channels] matrix, or a cell array of such matrices with equal channel counts.
%   labels   - optional char, string or cell array of dataset names; defaults to 'Data' for one matrix and 'Dataset 1', 'Dataset 2', ... otherwise.
%
% Outputs:
%   stats - 1 x n_datasets struct array with label, n_samples, per-channel kurtosis and skewness, and the provenance fields analysis_relevance and used_for_exclusion (always false).

function stats = checkMarginalDistributions(datasets, labels)

    arguments
        datasets
        labels = {}
    end

    %% Normalise inputs

    if ~iscell(datasets)
        datasets = {datasets};
    end
    n_datasets = numel(datasets);

    if isempty(labels)
        if n_datasets == 1
            labels = {'Data'};
        else
            labels = arrayfun(@(k) sprintf('Dataset %d', k), 1:n_datasets, 'UniformOutput', false);
        end
    elseif ischar(labels) || isstring(labels)
        labels = {char(labels)};
    end

    n_channels = size(datasets{1}, 2);
    for k = 2:n_datasets
        if size(datasets{k}, 2) ~= n_channels
            error('All datasets must have the same number of channels (%d vs %d).', ...
                n_channels, size(datasets{k}, 2));
        end
    end

    %% Compute per-channel statistics

    stats = struct();
    for k = 1:n_datasets
        X = datasets{k};
        stats(k).label = labels{k};
        stats(k).n_samples = size(X, 1);
        stats(k).kurtosis = zeros(1, n_channels);
        stats(k).skewness = zeros(1, n_channels);
        stats(k).analysis_relevance = 'margins_removed_by_rank_transform';
        stats(k).used_for_exclusion = false;

        for ch = 1:n_channels
            x = X(:, ch);
            if numel(x) < 2 || any(~isfinite(x)) || max(x) == min(x)
                stats(k).kurtosis(ch) = NaN;
                stats(k).skewness(ch) = NaN;
            else
                stats(k).kurtosis(ch) = kurtosis(x);
                stats(k).skewness(ch) = skewness(x);
            end
        end
    end

    %% Console summary

    fprintf('\n=== Marginal Distribution Summary ===\n');
    for k = 1:n_datasets
        fprintf('\n--- %s (%d samples) ---\n', stats(k).label, stats(k).n_samples);
        localPrintFiniteMomentSummary('Kurtosis', stats(k).kurtosis, ...
            n_channels, '  (Gaussian reference=3)');
        localPrintFiniteMomentSummary('Skewness', stats(k).skewness, ...
            n_channels, '');
        fprintf(['  Interpretation: descriptive marginal-shape QC only; ranks ' ...
            'remove these marginals before copula analysis. There is no ' ...
            'marginal-normality or copula goodness-of-fit test and no inclusion rule.\n']);
    end

end


function localPrintFiniteMomentSummary(name, values, nChannels, suffix)
    % Print the mean and range of the finite per-channel values and how many channels they cover.
    finiteValues = values(isfinite(values));
    if isempty(finiteValues)
        fprintf('  %s: no finite nonconstant-channel values (0/%d channels)%s\n', ...
            name, nChannels, suffix);
        return;
    end
    fprintf('  %s: mean=%.2f, range=[%.2f, %.2f] (%d/%d finite channels)%s\n', ...
        name, mean(finiteValues), min(finiteValues), max(finiteValues), ...
        numel(finiteValues), nChannels, suffix);
end
