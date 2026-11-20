function r = tDigammaDiff(a, m)
    % TDIGAMMADIFF psi(a + m/2) - psi(a) without cancellation at large a.
    %
    % r = tDigammaDiff(a, m) returns the digamma difference for a > 0 (array) and a non-negative integer m (scalar).
    % Below a = 1e4 the two psi values are subtracted directly.
    % From a = 1e4 on, the integer steps are summed exactly, psi(a + k) - psi(a) = sum_{j=0}^{k-1} 1/(a + j), and an odd m adds the half step psi(a + 1/2) - psi(a), whose asymptotic series 1/(2a) + 1/(8a^2) - 1/(64a^4) is exact to double precision for a >= 1e4.
    %
    % It is the companion of tLogGammaRatio in the t-copula entropy.

    arguments
        a double {mustBeReal, mustBePositive}
        m (1, 1) double {mustBeInteger, mustBeNonnegative}
    end
    r = zeros(size(a));
    if m == 0
        return
    end
    direct = a < 1e4;
    if any(direct(:))
        r(direct) = psi(a(direct) + m / 2) - psi(a(direct));
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
            half = 1 ./ (2 * aL) + 1 ./ (8 * aL.^2) - 1 ./ (64 * aL.^4);
        end
        steps = zeros(size(aL));
        for j = 0:k - 1
            steps = steps + 1 ./ (aL + offset + j);
        end
        r(large) = steps + half;
    end
end
