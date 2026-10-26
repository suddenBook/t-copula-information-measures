function [tau, acf, lags] = rankIntegratedAutocorrTime(X, maxLag)
    %RANKINTEGRATEDAUTOCORRTIME Integrated autocorrelation time of rank-normal scores.
    %
    %   [tau, acf, lags] = rankIntegratedAutocorrTime(X, maxLag)
    %
    % Maps every column of X ([samples x channels], finite) to normal scores of its ranks, forms the sample autocorrelation up to maxLag (capped at n - 2), and sums it up to the lag before the first non-positive value, a single-lag form of Geyer's initial positive sequence.
    % It is a diagnostic of how far the rows are from independent and decides nothing.
    % The rows must form one continuous segment, so a trial recording is passed one trial segment at a time, as Phase 1 does.
    %
    % Outputs:
    %   tau  - 1 x d integrated autocorrelation time 1 + 2 * sum_{k>=1} acf(k) per channel, in samples (1 for an independent series).
    %   acf  - d x numel(lags) autocorrelation at the returned lags.
    %   lags - 1 x L lags computed, 1:min(maxLag, n - 2).

    arguments
        X (:, :) double {mustBeReal, mustBeFinite}
        maxLag (1, 1) double {mustBeInteger, mustBePositive}
    end
    [n, d] = size(X);
    maxLag = min(maxLag, n - 2);
    lags = 1:maxLag;
    Z = zeros(n, d);
    for c = 1:d
        Z(:, c) = norminv(tiedrank(X(:, c)) / (n + 1));
    end
    Z = Z - mean(Z, 1);
    v = mean(Z .^ 2, 1);
    acf = zeros(d, maxLag);
    for k = 1:maxLag
        acf(:, k) = (sum(Z(1:n - k, :) .* Z(k + 1:n, :), 1) / (n - k)) ./ v;
    end
    tau = ones(1, d);
    for c = 1:d
        firstNonPositive = find(acf(c, :) <= 0, 1);
        if isempty(firstNonPositive)
            used = maxLag;
        else
            used = firstNonPositive - 1;
        end
        tau(c) = 1 + 2 * sum(acf(c, 1:used));
    end
end
