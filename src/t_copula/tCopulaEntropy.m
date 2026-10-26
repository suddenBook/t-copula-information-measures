function ce = tCopulaEntropy(d, p)
    %TCOPULAENTROPY Closed-form copula-entropy shape terms of the Student-t copula.
    %
    % ce = tCopulaEntropy(d, p) returns the shape terms for dimension d and shape parameter p = 1/nu (p may be an array; p = 0 is the Gaussian copula) as a struct with fields:
    %   kappa        h_d(nu) - d h_1(nu), the shape term of H_c = 1/2 log det R + kappa
    %   kappa_margin the shape term of every (d-1)-dimensional margin, kappa_t(d-1, nu)
    %   H_joint      h_d(nu), the entropy of the standard d-variate t (R = I)
    %   H_marginal   h_1(nu), the entropy of the univariate t
    %   d, p, nu     the inputs and nu = 1/p (Inf at p = 0)
    %   valid        true when every kappa and kappa_margin is finite
    % Here h_m(nu) is the entropy of the standard m-variate t:
    %   h_m(nu) = log G(nu/2) - log G((nu+m)/2) + (m/2) log(nu pi)
    %             + ((nu+m)/2) (psi((nu+m)/2) - psi(nu/2))
    %
    % The t family is closed under marginalisation with the same nu, so the leave-one-out shape term is exactly kappa at d-1.
    % For d = 2 the margin is univariate and kappa_margin = 0.
    % At p = 0 every shape term is exactly zero and the entropies are those of the standard normal.
    % kappa <= 0 with equality only at p = 0, because the Gaussian copula maximizes copula entropy at fixed R; tests/oracles/ellipticalEntropyMaxCheck.py checks this independently of this file.
    %
    % Numerical note: at large nu, h_m(nu) is a small difference of large log-gamma and digamma values, so both differences go through tLogGammaRatio and tDigammaDiff.
    % Those switch to exact integer steps plus an asymptotic half step from nu = 2e4 on, and kappa then stays accurate to about 1e-13 absolute up to nu = 1e6 and beyond.
    % The (m/2) log(nu pi) terms cancel between h_d and d h_1 and are not carried into kappa.

    arguments
        d (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(d, 2)}
        p double {mustBeReal, mustBeFinite, mustBeNonnegative}
    end

    kappa = zeros(size(p));
    kappaMargin = zeros(size(p));
    hJoint = repmat(d / 2 * log(2 * pi * exp(1)), size(p));
    hMarginal = repmat(0.5 * log(2 * pi * exp(1)), size(p));

    positive = p > 0;
    if any(positive(:))
        nu = 1 ./ p(positive);
        hJoint(positive) = localTEntropy(d, nu);
        hMarginal(positive) = localTEntropy(1, nu);
        kappa(positive) = localShapeTerm(d, nu);
        if d >= 3
            kappaMargin(positive) = localShapeTerm(d - 1, nu);
        end
    end

    ce = struct( ...
        'd', d, ...
        'p', p, ...
        'nu', 1 ./ p, ...
        'kappa', kappa, ...
        'kappa_margin', kappaMargin, ...
        'H_joint', hJoint, ...
        'H_marginal', hMarginal, ...
        'valid', all(isfinite(kappa(:))) && all(isfinite(kappaMargin(:))));
end

function h = localTEntropy(m, nu)
    % Differential entropy of the standard m-variate Student-t with nu degrees of freedom; nu may be an array.
    a = nu / 2;
    h = -tLogGammaRatio(a, m) + m / 2 * log(nu * pi) + (a + m / 2) .* tDigammaDiff(a, m);
end

function kappa = localShapeTerm(m, nu)
    % h_m(nu) - m h_1(nu) with the log(nu pi) terms cancelled algebraically, so that only the well-conditioned ratio terms remain.
    a = nu / 2;
    kappa = -(tLogGammaRatio(a, m) - m * tLogGammaRatio(a, 1)) ...
        + (a + m / 2) .* tDigammaDiff(a, m) - m * (a + 0.5) .* tDigammaDiff(a, 1);
end
