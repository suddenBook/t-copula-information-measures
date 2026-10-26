function tau = kendallTauKnight(x, y)
    %KENDALLTAUKNIGHT Kendall's tau-b by a vectorised version of Knight's merge-sort algorithm, in O(n log^2 n).
    %
    % An exact drop-in for MATLAB's corr(x, y, 'type', 'Kendall'): it returns the tie-corrected tau-b and agrees with corr to floating precision (<= 1e-12), including on heavily tied data.
    % The cost is O(n log^2 n) (a vectorised block merge sort) instead of corr's O(n^2), so it uses every row of a full recording (n ~ 3e5) in well under a second, where the O(n^2) form takes hours.
    %
    % Tie handling (tau-b) uses these counts:
    %     n0 = n(n-1)/2                                (all unordered pairs)
    %     n1 = sum_g t_g (t_g - 1)/2   over x-tie groups of size t_g
    %     n2 = sum_h u_h (u_h - 1)/2   over y-tie groups of size u_h
    %     n3 = sum_k v_k (v_k - 1)/2   over joint (x,y)-tie groups of size v_k
    %     nd = number of discordant pairs
    % The concordant-minus-discordant count is nc - nd = n0 - n1 - n2 + n3 - 2*nd (the pairs tied in neither x nor y, minus twice the discordances), and tau_b = (nc - nd) / sqrt((n0 - n1) (n0 - n2)).
    % For continuous data with no ties (n1 = n2 = n3 = 0) this reduces to the familiar tau-a = (nc - nd)/n0.
    %
    % Algorithm: sort the observations by x ascending, with ties broken by y ascending.
    % Under that order the discordant count nd equals the number of strict inversions of the y-sequence (pairs i < j with y_i > y_j).
    % A pair tied in x sits inside one x-group where y is already ascending, so it contributes no inversion, and a pair tied in y is a non-strict equality, so it contributes none either.
    % Knight's algorithm counts those inversions with a merge sort; here it is a vectorised bottom-up merge sort (local_count_inversions) of O(log n) passes, each sorting blocks of growing width, so the whole routine is O(n log^2 n).
    % The classic pointer-based linear merge is O(n log n), but the block sort is far simpler to vectorise and still far below corr's O(n^2).
    %
    % Conventions (the same as corr's on R2025b):
    %   * n < 2 gives NaN, as in corr.
    %   * A constant (zero-variance) column gives NaN, because its denominator (n0 - n1) or (n0 - n2) is 0 when all pairs tie; corr returns NaN there too.
    %   * Non-finite input gives NaN, matching corr's default 'Rows','all', which propagates any NaN row to NaN; callers strip non-finite rows first.
    %
    % Inputs:
    %   x, y - numeric vectors of equal length; unequal lengths raise kendallTauKnight:SizeMismatch.
    % Outputs:
    %   tau  - Kendall's tau-b in [-1, 1], or NaN per the conventions above.

    x = x(:);
    y = y(:);
    n = numel(x);
    if numel(y) ~= n
        error('kendallTauKnight:SizeMismatch', ...
            'x and y must have equal length (got %d and %d).', n, numel(y));
    end
    if n < 2
        tau = NaN; return;                 % corr(...,'Kendall') is NaN for n < 2
    end
    if ~(all(isfinite(x)) && all(isfinite(y)))
        tau = NaN; return;                 % NaN/Inf row -> NaN, as corr's default does
    end

    n0 = n * (n - 1) / 2;

    % Order by x ascending, ties broken by y ascending.
    S = sortrows([x, y]);
    xs = S(:, 1);
    ys = S(:, 2);

    % Tie corrections (n1 x-ties, n2 y-ties, n3 joint (x,y)-ties).
    n1 = local_run_pairs(diff(xs) ~= 0, n);
    [~, ~, yr] = unique(ys);                           % dense ranks of y (ties equal)
    yc = accumarray(yr, 1);
    n2 = sum(yc .* (yc - 1) / 2);
    n3 = local_run_pairs((diff(xs) ~= 0) | (diff(ys) ~= 0), n);

    % Discordant pairs = strict inversions of y under the (x,y) ordering.
    nd = local_count_inversions(yr);

    num   = n0 - n1 - n2 + n3 - 2 * nd;                % nc - nd
    denom = sqrt((n0 - n1) * (n0 - n2));
    if denom == 0
        tau = NaN;                                     % constant column, as corr returns
    else
        tau = num / denom;
    end
end


function s = local_run_pairs(change, n)
    % Sum of t(t-1)/2 over the maximal runs of equal consecutive elements, where `change` (logical, length n-1) is true wherever consecutive elements differ.
    b = find(change);
    ends   = [b; n];
    starts = [1; b + 1];
    t = ends - starts + 1;
    s = sum(t .* (t - 1) / 2);
end


function cnt = local_count_inversions(v)
    % Number of strict inversions (pairs i < j with v(i) > v(j)) of the finite numeric column v, by a vectorised bottom-up merge sort.
    % The cost is O(n log^2 n): O(log n) passes, each a column-wise sort (O(n log blk)) plus a cumsum over the padded array.
    n = numel(v);
    if n < 2, cnt = 0; return; end
    N = 2 ^ ceil(log2(n));
    a = [v; inf(N - n, 1)];        % +Inf sentinels (>= any value) add zero inversions
    cnt = 0;
    w = 1;                         % current run half-width
    while w < N
        blk = 2 * w;
        M = reshape(a, blk, N / blk);   % each column is one block: [left(w); right(w)]
        [Ms, ord] = sort(M, 1);         % stable ascending within each block
        isR  = ord > w;                 % this sorted element came from the right half
        cumR = cumsum(isR, 1);          % right-elements seen at/above each position
        % A left-origin element with k right-origin elements above it in sorted order contributes k strict inversions.
        % The stable sort puts an equal-valued left element before the right one, so only a strict left > right counts.
        cnt = cnt + sum(cumR(~isR));
        a = Ms(:);                      % merged runs of width blk for the next pass
        w = blk;
    end
end
