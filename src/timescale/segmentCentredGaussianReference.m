function reference = segmentCentredGaussianReference(score_acf, segment_ends, max_lag)
    % SEGMENTCENTREDGAUSSIANREFERENCE Gaussian reference of the square's autocorrelation under within-segment centring.
    %
    % For each channel (column of SCORE_ACF), returns the autocorrelation at lags 0..MAX_LAG that averagedRankAutocorrelation(X, SEGMENT_ENDS, MAX_LAG, 'square', 'segment') estimates, in expectation, when the channel is a stationary Gaussian series with the autocorrelation SCORE_ACF.
    % It is the Gaussian reference of the within-trial square.
    % Centring within each segment pulls every lag of the square's autocorrelation down by about 1 / n_eff of a segment.
    % The square of the likewise-centred score autocorrelation is not pulled down with it, and reads the fitted amplitude excess of stationary Gaussian trials of about 800 samples as 0.885 instead of 1 (autocorrelationTimescaleTest); this reference carries the same pull.
    %
    % For a zero-mean unit-variance Gaussian series with autocorrelation rho, the squared series y = Z.^2 - 1 has autocovariance gamma(h) = 2 rho(h)^2 (Isserlis).
    % Centred by its mean over a segment of L samples, its lag-k product sum S_L(k) has the expectation below, with A_L(t) the covariance of sample t with the segment mean and V_L the variance of that mean:
    %     E[S_L(k)] = (L - k) (gamma(k) + V_L) - sum_{t=1}^{L-k} (A_L(t) + A_L(t + k))
    %     A_L(t) = (1/L) sum_{u=1}^{L} gamma(t - u)
    %     V_L = (1/L^2) sum_{u,v=1}^{L} gamma(u - v)
    % The estimator pools the segments longer than k as E[c(k)] = sum_s E[S_{L_s}(k)] / sum_s (L_s - k), and REFERENCE(k) = E[c(k)] / E[c(0)], a ratio of expectations in which the factor 2 of gamma cancels.
    % A_L and V_L depend on the length alone, so the segments are grouped by length, and cumulative sums of gamma and of A_L make each length O(L) per channel.
    %
    % Inputs:
    %   SCORE_ACF    - [H x C] each channel's score autocorrelation at lags 0..H-1, 1 at lag 0, with H at least the longest segment, since a segment's mean couples all of its samples.
    %                  Phase 6 passes the whole-recording-centred estimate of averagedRankAutocorrelation out to the longest segment.
    %                  A NaN column (a constant channel) gives a NaN column.
    %   SEGMENT_ENDS - cumulative segment end indices, strictly increasing positive integers.
    %   MAX_LAG      - nonnegative integer, in samples, smaller than the longest segment.
    %
    % Outputs:
    %   REFERENCE - [MAX_LAG+1 x C] the reference autocorrelation of each channel, 1 at lag 0.

    arguments
        score_acf (:, :) double {mustBeReal}
        segment_ends (:, 1) double {mustBeInteger, mustBePositive}
        max_lag (1, 1) double {mustBeInteger, mustBeNonnegative}
    end

    lengths = diff([0; segment_ends]);
    assert(~isempty(lengths) && all(lengths > 0), ...
           'segmentCentredGaussianReference:InvalidSegments', ...
           'Segment ends must be strictly increasing positive integers; got %d ends.', ...
           numel(segment_ends));
    longest = max(lengths);
    assert(max_lag < longest, 'segmentCentredGaussianReference:LagTooLong', ...
           'MAX_LAG = %d must be smaller than the longest segment (%d samples).', max_lag, longest);
    assert(size(score_acf, 1) >= longest, 'segmentCentredGaussianReference:AcfTooShort', ...
           ['SCORE_ACF must reach lag %d, one short of the longest segment (%d samples), because ' ...
            'a segment''s mean couples all of its samples; it reaches lag %d.'], ...
           longest - 1, longest, size(score_acf, 1) - 1);
    valid = ~isnan(score_acf(1, :));
    assert(all(isfinite(score_acf(1:longest, valid)), 'all') && all(score_acf(1, valid) == 1), ...
           'segmentCentredGaussianReference:BadAcf', ...
           ['Every column of SCORE_ACF that is not NaN must be finite out to lag %d and 1 at ' ...
            'lag 0.'], longest - 1);

    gamma = 2 * score_acf(1:longest, :).^2;   % autocovariance of Z.^2 - 1 at lags 0..longest-1
    cumulative = cumsum(gamma, 1);               % row m + 1: the sum of gamma over lags 0..m
    n_channels = size(score_acf, 2);
    sums = zeros(max_lag + 1, n_channels);
    pairs = zeros(max_lag + 1, 1);
    [distinct, ~, group] = unique(lengths);
    for g = 1:numel(distinct)
        L = distinct(g);
        count = nnz(group == g);
        t = (1:L)';
        % sum over u of gamma(t - u) = (lags 0..t-1) + (lags 0..L-t) - gamma(0)
        A = (cumulative(t, :) + cumulative(L - t + 1, :) - gamma(1, :)) / L;
        A_sums = [zeros(1, n_channels); cumsum(A, 1)];   % row m + 1: A_L(1) + ... + A_L(m)
        V = A_sums(L + 1, :) / L;
        k = (0:min(max_lag, L - 1))';
        expected = (L - k) .* (gamma(k + 1, :) + V) ...
            - (A_sums(L - k + 1, :) + A_sums(L + 1, :) - A_sums(k + 1, :));
        sums(k + 1, :) = sums(k + 1, :) + count * expected;
        pairs(k + 1) = pairs(k + 1) + count * (L - k);
    end
    covariance = sums ./ pairs;
    reference = covariance ./ covariance(1, :);
end
