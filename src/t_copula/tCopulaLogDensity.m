function logc = tCopulaLogDensity(Z, R, p)
    %TCOPULALOGDENSITY Per-row log density of the Student-t copula at latent points.
    %
    % logc = tCopulaLogDensity(Z, R, p) returns the n x 1 log copula density c(u) = f_d(z; R, nu) / prod_j f_1(z_j; nu) at the latent points Z (n x d) on the t scale with nu = 1/p, or on the Gaussian scale when p = 0.
    % R is the d x d correlation matrix and must be positive definite.
    %
    % tCopulaLoglik computes the fitted likelihood.
    % This row-by-row form, which needs no quantile plan, is the reference the tests use for the sampler, the density panels and the Monte-Carlo copula entropy.

    arguments
        Z (:, :) double {mustBeReal, mustBeFinite}
        R (:, :) double {mustBeReal, mustBeFinite}
        p (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
    end
    [n, d] = size(Z);
    assert(isequal(size(R), [d d]), 'tCopulaLogDensity:DimensionMismatch', ...
        'R must be %d x %d to match Z; got %d x %d.', d, d, size(R, 1), size(R, 2));
    if d == 1
        logc = zeros(n, 1);
        return;
    end
    [Lr, flag] = chol(R);
    assert(flag == 0, 'tCopulaLogDensity:RNotPositiveDefinite', ...
        'R is not positive definite (Cholesky failed at column %d).', flag);
    logdetR = 2 * sum(log(diag(Lr)));
    S = sum((Z / Lr) .^ 2, 2);

    if p == 0
        logc = -0.5 * logdetR - 0.5 * (S - sum(Z .^ 2, 2));
        return;
    end
    nu = 1 / p;
    logJoint = gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * log(nu * pi) ...
        - 0.5 * logdetR - (nu + d) / 2 * log1p(S / nu);
    logUnivariate = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * log(nu * pi) ...
        - (nu + 1) / 2 * log1p(Z .^ 2 / nu);
    logc = logJoint - sum(logUnivariate, 2);
end
