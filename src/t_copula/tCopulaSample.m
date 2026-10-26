function X = tCopulaSample(n, d, R, p, seed)
    %TCOPULASAMPLE Latent sample from the Student-t copula with continuous margins.
    %
    % X = tCopulaSample(n, d, R, p, seed) draws n rows of the d-variate Student-t distribution with correlation matrix R and nu = 1/p degrees of freedom as the scale mixture X = Z ./ sqrt(W/nu), Z ~ N(0, R), W ~ chi-square(nu).
    % The draw uses a private RandStream seeded by seed, so the global RNG is untouched and the sample is reproducible.
    % p = 0 returns the Gaussian sample itself.
    % The rows have continuous margins; ranking each column (tiedrank/(n+1)) gives pseudo-observations from the t copula.
    % The chi-square variate is drawn through gammaincinv, so only base MATLAB is needed.

    arguments
        n (1, 1) double {mustBeInteger, mustBePositive}
        d (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(d, 2)}
        R (:, :) double {mustBeReal, mustBeFinite}
        p (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
        seed (1, 1) double {mustBeInteger, mustBeNonnegative}
    end
    assert(isequal(size(R), [d d]), 'tCopulaSample:DimensionMismatch', ...
        'R must be %d x %d; got %d x %d.', d, d, size(R, 1), size(R, 2));
    [Lr, flag] = chol(R);
    assert(flag == 0, 'tCopulaSample:RNotPositiveDefinite', ...
        'R is not positive definite (Cholesky failed at column %d).', flag);

    stream = RandStream('twister', 'Seed', seed);
    X = randn(stream, n, d) * Lr;           % rows ~ N(0, R) because Lr' * Lr = R
    if p == 0
        return;
    end
    nu = 1 / p;
    u = rand(stream, n, 1);
    u = min(max(u, realmin), 1 - eps);
    chiSquare = 2 * gammaincinv(u, nu / 2); % chi-square(nu) is Gamma(nu/2) with scale 2
    X = X ./ sqrt(chiSquare / nu);
end
