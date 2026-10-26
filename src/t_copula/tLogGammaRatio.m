function r = tLogGammaRatio(a, m)
    %TLOGGAMMARATIO log Gamma(a + m/2) - log Gamma(a) without cancellation at large a.
    %
    % r = tLogGammaRatio(a, m) returns log G(a + m/2) - log G(a) for a > 0 (array) and a non-negative integer m (scalar).
    % Below a = 1e4 it subtracts two gammaln values directly; that difference of two values of order a*log(a) loses absolute precision as a grows, to about 1e-11 near a = 1e4.
    % From a = 1e4 on, the ratio is assembled from the exact integer steps G(a + k) / G(a) = a (a+1) ... (a+k-1), summed as logarithms.
    % When m is odd it adds the half step log G(a + 1/2) - log G(a), whose asymptotic series 1/2 log a - 1/(8a) + 1/(192 a^3) - 1/(640 a^5) is exact to double precision for a >= 1e4.
    %
    % The t-copula entropy and log-likelihood use it, because their shape terms at nu = 1/p near the Gaussian point are small differences of large numbers.

    arguments
        a double {mustBeReal, mustBePositive}
        m (1, 1) double {mustBeInteger, mustBeNonnegative}
    end
    r = zeros(size(a));
    if m == 0
        return;
    end
    direct = a < 1e4;
    if any(direct(:))
        r(direct) = gammaln(a(direct) + m / 2) - gammaln(a(direct));
    end
    large = ~direct;
    if any(large(:))
        aL = a(large);
        k = floor(m / 2);
        if mod(m, 2) == 0
            offset = 0;
            half = 0;
        else
            offset = 0.5;
            half = 0.5 * log(aL) - 1 ./ (8 * aL) + 1 ./ (192 * aL .^ 3) - 1 ./ (640 * aL .^ 5);
        end
        steps = zeros(size(aL));
        for j = 0:k - 1
            steps = steps + log(aL + offset + j);
        end
        r(large) = steps + half;
    end
end
