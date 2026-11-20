function ll = tCopulaLoglik(p, plan, Rinfo, cache)
    % TCOPULALOGLIK Total Student-t copula log-likelihood of a pseudo-observation matrix.
    %
    % ll = tCopulaLoglik(p, plan, Rinfo) evaluates the t-copula log-likelihood at shape parameter p = 1/nu (p = 0 is the Gaussian copula), summed over all rows, for the pseudo-observations prepared by tCopulaQuantilePlan and the fixed correlation matrix packed in Rinfo:
    %   Rinfo.Lr       upper-triangular Cholesky factor with R = Lr' * Lr
    %   Rinfo.logdetR  log det R
    %   Rinfo.d        dimension
    %
    % Let z_ij be the t quantile of u_ij, S_i = z_i' R^{-1} z_i and G the gamma function.
    % The log-likelihood and its constant term are:
    %   ll = n c(nu, d) - (n/2) log det R - ((nu+d)/2) sum_i log(1 + S_i/nu) + ((nu+1)/2) sum_ij log(1 + z_ij^2/nu)
    %   c(nu, d) = log G((nu+d)/2) + (d-1) log G(nu/2) - d log G((nu+1)/2)
    % The log(nu pi) terms of the joint and the marginal densities cancel.
    % The marginal sum runs over the distinct |z| values with their multiplicities.
    % At p = 0 the closed-form Gaussian copula log-likelihood is used exactly.
    % The total is not divided by n; the fitter's tolerances are per row and are multiplied by n there.
    %
    % A non-finite value raises an error instead of returning -Inf, so a numerical failure is never mistaken for a low-likelihood basin.
    %
    % ll = tCopulaLoglik(p, plan, Rinfo, cache) reads the tail quantiles from a tCopulaQuantileCache instead of computing them, which is the same arithmetic on the same numbers, because the cache is filled by the same tTailQuantile calls.
    % The table is used only when the cache's row count matches the plan's, p is exactly one of the cached nodes and the plan carries the cached rank plan.
    % Any other case (another row count, a refinement step between nodes, or a plan whose ties moved its distinct tail probabilities) takes the direct route.

    arguments
        p (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
        plan (1, 1) struct
        Rinfo (1, 1) struct
        cache = []
    end
    n = plan.n;
    d = plan.d;
    assert(Rinfo.d == d, 'tCopulaLoglik:DimensionMismatch', ...
           'The correlation factor is %d-dimensional but the plan has d = %d.', Rinfo.d, d);

    absz = [];
    % The cache is an exact shortcut for the recording's row count at the grid nodes; any other row count (a channel set with fewer complete rows) or an off-grid p takes the direct route below, the same computation without the table.
    if ~isempty(cache) && cache.n == n
        node = find(cache.p_grid == p, 1);
        if ~isempty(node) && isequal(cache.q_unique, plan.q_unique)
            absz = cache.absz(:, node);
        end
    end
    if isempty(absz)
        absz = tTailQuantile(plan.q_unique, p);
    end
    Z = absz(plan.q_index);
    Z(plan.negate) = -Z(plan.negate);
    W = Z / Rinfo.Lr;                       % rows z_i Lr^{-1}, so sum(W.^2, 2) = z_i R^{-1} z_i'
    S = sum(W.^2, 2);

    if p == 0
        ll = -0.5 * n * Rinfo.logdetR - 0.5 * (sum(S) - sum(plan.q_count .* absz.^2));
    else
        nu = 1 / p;
        % c(nu, d) = [log G(nu/2 + d/2) - log G(nu/2)] - d [log G(nu/2 + 1/2) - log G(nu/2)], assembled from log-gamma ratios to avoid the large-nu cancellation of three gammaln values.
        cNu = tLogGammaRatio(nu / 2, d) - d * tLogGammaRatio(nu / 2, 1);
        ll = n * cNu - 0.5 * n * Rinfo.logdetR ...
            - (nu + d) / 2 * sum(log1p(S / nu)) ...
            + (nu + 1) / 2 * sum(plan.q_count .* log1p(absz.^2 / nu));
    end

    if ~isfinite(ll)
        error('tCopulaLoglik:NonFiniteLoglik', ...
              ['The t-copula log-likelihood is not finite at p = %.6g (nu = %.6g), ' ...
               'n = %d, d = %d: %d non-finite radii and %d non-finite quantiles.'], ...
              p, 1 / p, n, d, nnz(~isfinite(S)), nnz(~isfinite(absz)));
    end
end
