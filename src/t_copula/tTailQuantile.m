function absz = tTailQuantile(q, p)
    %TTAILQUANTILE Magnitude of the Student-t quantile for a two-sided tail probability.
    %
    % absz = tTailQuantile(q, p) returns |z| with P(|T| > |z|) = q for a Student-t variable T with nu = 1/p degrees of freedom.
    % p = 0 selects the standard normal limit.
    % q is an array of two-sided tail probabilities in (0, 1], absz has the same shape, and q = 1 gives exactly 0.
    %
    % It needs base MATLAB only.
    % For T ~ t_nu the variable x = nu/(nu + T^2) is Beta(nu/2, 1/2) distributed, so P(|T| > z) = I_x(nu/2, 1/2) with I the regularized incomplete beta function.
    % The inversion runs in the branch that keeps full relative precision: for q < 1/2 it solves for x directly, and for q >= 1/2 (small |z|, x close to 1) it solves for y = 1 - x through the symmetry I_x(a, b) = 1 - I_{1-x}(b, a).
    % For nu >= 1e5 the incomplete-beta inversion loses accuracy and fails outright near nu = 1e8, so the Cornish-Fisher expansion of the t quantile around the normal quantile z0 is used instead:
    %   z = z0 + (z0^3 + z0)/(4 nu) + (5 z0^5 + 16 z0^3 + 3 z0)/(96 nu^2)
    %          + (3 z0^7 + 19 z0^5 + 17 z0^3 - 15 z0)/(384 nu^3)
    % Its truncation error is of order nu^-4 and stays below double-precision resolution for q >= 1e-12, which keeps the likelihood smooth all the way to the Gaussian point.
    %
    % This is the only place the margin quantile of the t copula is computed; tCopulaLoglik, tCopulaQuantileCache and tCopulaPairDensityGrid call it.

    arguments
        q double {mustBeReal}
        p (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
    end
    assert(all(q(:) > 0 & q(:) <= 1), 'tTailQuantile:TailProbabilityOutOfRange', ...
        'Two-sided tail probabilities must lie in (0, 1]; got min %.3g and max %.3g.', ...
        min(q(:)), max(q(:)));

    if p == 0
        absz = sqrt(2) * erfcinv(q);
        return;
    end
    if p <= 1e-5
        z0 = sqrt(2) * erfcinv(q);
        z2 = z0 .^ 2;
        absz = z0 .* (1 + p * (z2 + 1) / 4 ...
            + p ^ 2 * (5 * z2 .^ 2 + 16 * z2 + 3) / 96 ...
            + p ^ 3 * (3 * z2 .^ 3 + 19 * z2 .^ 2 + 17 * z2 - 15) / 384);
        return;
    end

    nu = 1 / p;
    absz = zeros(size(q));
    lower = q < 0.5;
    if any(lower(:))
        x = betaincinv(q(lower), nu / 2, 0.5);          % x = nu / (nu + z^2)
        absz(lower) = sqrt(nu * (1 - x) ./ x);
    end
    if any(~lower(:))
        y = betaincinv(1 - q(~lower), 0.5, nu / 2);     % y = 1 - x = z^2 / (nu + z^2)
        absz(~lower) = sqrt(nu * y ./ (1 - y));
    end
end
