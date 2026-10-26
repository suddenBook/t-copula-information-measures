function out = averagedRankAutocorrelation(X, segment_ends, max_lag, transform, centring)
    %AVERAGEDRANKAUTOCORRELATION Channel-averaged autocorrelation of rank-normal scores within segments.
    %
    % Each column of the signal matrix X [samples x channels] is ranked over the whole recording and mapped to normal scores Z = norminv(rank / (n + 1)).
    % Every channel's autocorrelation at lags 0..MAX_LAG samples is estimated from lagged products taken within each segment only, so no pair of samples straddles a declared trial boundary.
    % Rank-normal scores are used so that recordings and simulated Student paths are compared on the same marginal-free transform.
    %
    % TRANSFORM selects the series:
    %   'score'  - Z itself, the signal's own memory.
    %   'square' - Z.^2, the amplitude series, whose memory is that of the local signal power.
    %
    % CENTRING selects the mean each series is centred by before its lagged products:
    %   'recording' - its mean over the whole recording (the default).
    %   'segment'   - its mean within each segment, so that a level differing between segments is not read as memory at every lag inside a segment.
    % On a trial recording that level is the between-trial variance of the signal power, which enters the square.
    % A mean estimated from the segment itself pulls every lag down by about 1 / n_eff, with n_eff the segment's length over the series' integrated autocorrelation time.
    % Where the true autocorrelation has decayed, the 'segment' estimate therefore reads slightly negative, a few hundredths for the square at the long lags of sub-second trials.
    % With one segment the two centrings are the same computation.
    %
    % The lag-k autocovariance pools the within-segment products over segments and divides by their number of pairs, and the autocorrelation divides it by the lag-0 value.
    % The channel average is the mean over the channels whose series is not constant.
    %
    % Inputs:
    %   X            - [n x C] finite real double, samples x channels, with n >= 2.
    %   SEGMENT_ENDS - cumulative segment end indices, strictly increasing integers with the last equal to n; [] means one segment.
    %   MAX_LAG      - nonnegative integer lag window, in samples, smaller than the longest segment.
    %   TRANSFORM    - 'score' (default) or 'square'.
    %   CENTRING     - 'recording' (default) or 'segment'.
    %
    % Outputs:
    %   OUT - struct with fields
    %     lags                - [L+1 x 1] lags in samples, 0..MAX_LAG.
    %     acf_mean            - [L+1 x 1] channel-averaged autocorrelation, 1 at lag 0.
    %     acf_by_channel      - [L+1 x C] per-channel autocorrelations, NaN for a constant series.
    %     pair_counts         - [L+1 x 1] within-segment sample pairs at each lag.
    %     n_valid_channels    - number of channels entering the average.
    %     transform           - the transform used.
    %     acf_square_gaussian - ('score' only) [L+1 x 1] channel mean of the squared per-channel autocorrelations.
    % For a stationary Gaussian series the autocorrelation of Z.^2 is the square of that of Z (Isserlis), so acf_square_gaussian is what the 'square' channel average would be if every channel were Gaussian with its own autocorrelation.
    % It is the Gaussian reference that amplitude memory is measured against, taken under the same CENTRING as the scores.
    % Under 'segment' centring it is not the square's reference, which carries the pull of that centring (segmentCentredGaussianReference).

    arguments
        X (:,:) double {mustBeFinite, mustBeReal}
        segment_ends (:,1) double = []
        max_lag (1,1) double {mustBeInteger, mustBeNonnegative} = 0
        transform (1,:) char {mustBeMember(transform, {'score', 'square'})} = 'score'
        centring (1,:) char {mustBeMember(centring, {'recording', 'segment'})} = 'recording'
    end

    [n, n_channels] = size(X);
    assert(n >= 2 && n_channels >= 1, 'averagedRankAutocorrelation:EmptyMatrix', ...
        'X must have at least 2 samples and 1 channel; got %d x %d.', n, n_channels);
    if isempty(segment_ends)
        segment_ends = n;
    end
    assert(all(segment_ends == round(segment_ends)) && all(diff(segment_ends) > 0) ...
        && segment_ends(1) >= 1 && segment_ends(end) == n, ...
        'averagedRankAutocorrelation:InvalidSegments', ...
        ['Segment ends must be strictly increasing integers ending at the sample count %d; ' ...
        'got %d ends ending at %g.'], n, numel(segment_ends), segment_ends(end));
    segment_lengths = diff([0; segment_ends]);
    assert(max_lag < max(segment_lengths), 'averagedRankAutocorrelation:LagTooLong', ...
        'MAX_LAG = %d must be smaller than the longest segment (%d samples).', ...
        max_lag, max(segment_lengths));

    Z = norminv(tiedrank(X) ./ (n + 1));
    if strcmp(transform, 'square')
        S = Z .^ 2;
    else
        S = Z;
    end
    if strcmp(centring, 'segment')
        segment_start = 1;
        for s = 1:numel(segment_ends)
            rows = segment_start:segment_ends(s);
            S(rows, :) = S(rows, :) - mean(S(rows, :), 1);
            segment_start = segment_ends(s) + 1;
        end
    else
        S = S - mean(S, 1);
    end

    n_lags = max_lag + 1;
    autocov_sum = zeros(n_lags, n_channels);
    pair_counts = zeros(n_lags, 1);
    % debt: channels are transformed in blocks of 16 to bound the FFT buffer at about 16 times the padded segment length, and a segment far longer than 1e6 samples would want a smaller block.
    block_size = 16;
    segment_start = 1;
    for s = 1:numel(segment_ends)
        rows = segment_start:segment_ends(s);
        segment_start = segment_ends(s) + 1;
        len = numel(rows);
        n_used = min(max_lag, len - 1) + 1;
        pair_counts(1:n_used) = pair_counts(1:n_used) + (len - (0:n_used - 1))';
        n_fft = 2 ^ nextpow2(2 * len - 1);
        for first = 1:block_size:n_channels
            cols = first:min(first + block_size - 1, n_channels);
            spectrum = fft(S(rows, cols), n_fft);
            lagged = real(ifft(abs(spectrum) .^ 2));
            autocov_sum(1:n_used, cols) = autocov_sum(1:n_used, cols) + lagged(1:n_used, :);
        end
    end

    autocov = autocov_sum ./ pair_counts;
    acf_by_channel = autocov ./ autocov(1, :);
    constant = ~(autocov(1, :) > 0);
    acf_by_channel(:, constant) = NaN;
    n_valid = nnz(~constant);
    assert(n_valid >= 1, 'averagedRankAutocorrelation:AllChannelsConstant', ...
        'Every one of the %d channels has a constant %s series; no autocorrelation exists.', ...
        n_channels, transform);

    out = struct( ...
        'lags', (0:max_lag)', ...
        'acf_mean', mean(acf_by_channel(:, ~constant), 2), ...
        'acf_by_channel', acf_by_channel, ...
        'pair_counts', pair_counts, ...
        'n_valid_channels', n_valid, ...
        'transform', transform);
    if strcmp(transform, 'score')
        out.acf_square_gaussian = mean(acf_by_channel(:, ~constant) .^ 2, 2);
    end
end
