%ESTIMATEMIKSG Model-free copula mutual information (d = 2) or total correlation (d >= 3) by the Kraskov-Stoegbauer-Grassberger estimator.
%
% Implements the first Kraskov-Stoegbauer-Grassberger estimator (KSG algorithm 1), which makes no parametric copula assumption.
% The function removes the marginal scales itself, with the complete-case, average-rank pseudo-observations the copula fits use (computeCopulaPseudoObservations), and applies KSG to their normal scores.
% This matters because the finite-sample max-norm KSG estimate on raw channel amplitudes is not invariant to rescaling or other strictly monotone marginal transforms.
%
% Algorithm:
%   1. knnsearch in the d-dimensional joint space (Chebyshev or max norm) gives the k-th neighbour distance eps(i).
%   2. For each one-dimensional margin m, n_m(i) = #{j ~= i : |X(j,m) - X(i,m)| < eps(i)}.
%   3. MI or TC = (d-1) psi(N) + psi(k) - mean over i of sum_m psi(n_m(i) + 1).
%
% Inputs:
%   X - [N x d] data matrix, samples down the rows.
%       Rows with a non-finite entry are removed, and each column is transformed to norminv(rank/(N_complete+1)).
%   k - number of nearest neighbours (default 6).
%   Name-value:
%     'jitter'       - logical, default false; adds i.i.d. Gaussian noise, scaled by jitter_sigma times each column's standard deviation, to the normal scores before knnsearch.
%                      It is needed when X may contain duplicate rows (for example resampled inputs), where the unperturbed estimator can have eps(i) = 0, which raises estimateMIKSG:ZeroKthDistance.
%                      It is off by default, so point estimates stay deterministic on tie-free inputs.
%     'jitter_sigma' - double, default 1e-9; scale of the jitter noise relative to the per-column standard deviation, small enough to leave tie-free estimates unchanged at machine precision.
%     'jitter_seed'  - default []; seed of a private RandStream for reproducible jitter, which leaves MATLAB's global generator untouched.
%                      When empty, the jitter draws from the global stream.
%
% Outputs:
%   MI - mutual information (d = 2) or total correlation (d >= 3) in nats.

function MI = estimateMIKSG(X, k, varargin)

    if nargin < 2 || isempty(k)
        k = 6;
    end

    p = inputParser;
    p.FunctionName = 'estimateMIKSG';
    addParameter(p, 'jitter',       false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'jitter_sigma', 1e-9,  @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'jitter_seed',  [],    @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    parse(p, varargin{:});
    opts = p.Results;

    validateattributes(X, {'numeric'}, {'2d', 'real'}, mfilename, 'X', 1);
    if ~(isnumeric(k) && isreal(k) && isscalar(k) && isfinite(k) ...
            && k >= 1 && k == fix(k))
        error('estimateMIKSG:InvalidK', ...
            'The neighbour count k must be a finite positive integer.');
    end
    [~, d] = size(X);
    assert(d >= 2, 'estimateMIKSG:TooFewDimensions', ...
        'Input must have at least 2 columns.');

    % A copula estimate must not depend on electrode units, so the ranks come from the same pseudo-observation function as the copula fits.
    [U, ~, rank_diag] = computeCopulaPseudoObservations(double(X), ...
        'Rows', 'complete', 'MinSamples', k + 2);
    N = size(U, 1);
    assert(N > k + 1 && ~rank_diag.too_few_samples, ...
        'estimateMIKSG:TooFewCompleteRows', ...
        'Need at least k+2=%d jointly finite rows; observed %d.', k + 2, N);
    assert(~rank_diag.has_constant_column, ...
        'estimateMIKSG:ConstantColumn', ...
        'KSG copula information is undefined when a fitted column is constant.');
    X = norminv(U);
    assert(all(isfinite(X), 'all'), 'estimateMIKSG:RankTransformFailure', ...
        'Rank-normal pseudo-observations must be finite.');

    % The optional jitter before knnsearch breaks ties in resampled or duplicated data.
    % It is off by default, so estimates on tie-free data are unchanged.
    if opts.jitter && opts.jitter_sigma > 0
        col_std = std(X, 0, 1);
        if isempty(opts.jitter_seed)
            noise = randn(size(X));
        else
            rs = RandStream('twister', 'Seed', double(opts.jitter_seed));
            noise = randn(rs, size(X));
        end
        X = X + opts.jitter_sigma .* col_std .* noise;
    end

    % Step 1: the k-th neighbour distance in the joint space (Chebyshev or max norm).
    % Column 1 of D is the point itself (distance 0), and column k+1 is its k-th neighbour.
    [~, D] = knnsearch(X, X, 'K', k + 1, 'Distance', 'chebychev');
    epsilon = D(:, k + 1);
    if any(epsilon == 0)
        error('estimateMIKSG:ZeroKthDistance', ...
            ['The k-th joint-neighbour distance is zero for %d/%d rows. ' ...
            'KSG Algorithm 1 is undefined for this tied sample; use an ' ...
            'explicit, reproducibly seeded jitter policy or remove duplicates.'], ...
            nnz(epsilon == 0), N);
    end

    % Step 2: count the marginal neighbours within epsilon in each dimension.
    % Sorted-merge counting costs O(N log N) per dimension.
    psi_sum = zeros(N, 1);

    for dim = 1:d
        x_col = X(:, dim);
        x_sorted = sort(x_col);

        lower = x_col - epsilon;
        upper = x_col + epsilon;

        % Elements of x_sorted strictly below each upper bound.
        cnt_lt_upper = mergeSortedRankLT(x_sorted, upper);
        % Elements of x_sorted at or below each lower bound.
        cnt_le_lower = mergeSortedRankLE(x_sorted, lower);

        % Points in the open interval (lower, upper), excluding the point itself.
        n_marginal = max(cnt_lt_upper - cnt_le_lower - 1, 0);

        psi_sum = psi_sum + psi(n_marginal + 1);
    end

    % Step 3: MI or TC = (d-1) psi(N) + psi(k) - mean over i of sum_m psi(n_m(i) + 1).
    MI = (d - 1) * psi(N) + psi(k) - mean(psi_sum);

end


%% Local helper: count of elements strictly less than each query value
function ranks = mergeSortedRankLT(x_sorted, queries)
    % For each query q, the number of elements of the sorted vector x_sorted strictly below q.
    % A sort-merge walk costs O(M log M) to sort the queries plus O(N + M) for the merge.

    N = numel(x_sorted);
    M = numel(queries);

    [q_sorted, q_order] = sort(queries);
    ranks_sorted = zeros(M, 1);

    j = 1;
    for i = 1:M
        while j <= N && x_sorted(j) < q_sorted(i)
            j = j + 1;
        end
        ranks_sorted(i) = j - 1;
    end

    ranks = zeros(M, 1);
    ranks(q_order) = ranks_sorted;
end


%% Local helper: count of elements less than or equal to each query value
function ranks = mergeSortedRankLE(x_sorted, queries)
    % As mergeSortedRankLT, but counting the elements at or below each query.

    N = numel(x_sorted);
    M = numel(queries);

    [q_sorted, q_order] = sort(queries);
    ranks_sorted = zeros(M, 1);

    j = 1;
    for i = 1:M
        while j <= N && x_sorted(j) <= q_sorted(i)
            j = j + 1;
        end
        ranks_sorted(i) = j - 1;
    end

    ranks = zeros(M, 1);
    ranks(q_order) = ranks_sorted;
end
