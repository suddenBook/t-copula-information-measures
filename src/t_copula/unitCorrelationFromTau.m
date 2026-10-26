function [R, corr] = unitCorrelationFromTau(tau_sub, fit_opts)
    %UNITCORRELATIONFROMTAU Correlation reference and its audit for one channel set.
    %
    % [R, corr] = unitCorrelationFromTau(tau_sub, fit_opts) maps the [d x d] Kendall tau-b matrix of one channel set to the elliptical-copula correlation matrix R through the sine relation rho = sin(pi tau / 2), which is exact for every elliptical copula and therefore for the Student-t working model.
    % It also returns the audit the unit row reports:
    %
    %   d = 2   one free parameter and no repair.
    %           A pair with 1 - |rho| < 1e-6 is a duplicate or near-duplicate channel whose R is (nearly) singular; it is reported as corr.degenerate_pair rather than clamped to a barely positive-definite matrix with a meaningless log-determinant.
    %   d >= 3  when the smallest eigenvalue is at most 1e-8, the eigenvalues are floored there and the diagonal renormalised.
    %           The repair is a numerical construction, not evidence that the empirical rank correlations are mutually compatible, so its RMS off-diagonal shift is reported and flagged as corr.repair_gross above fit_opts.max_projection_shift_rms (0.02).
    %
    % Inputs:
    %   tau_sub  - [d x d] Kendall tau-b matrix of the channel set, d >= 2.
    %   fit_opts - (optional) tFitDefaultOptions overrides; only max_projection_shift_rms is read here.
    %
    % Outputs:
    %   R    - [d x d] correlation matrix, returned even when the audit fails so the repair stays inspectable; NaN when tau itself is not finite.
    %   corr - struct with fields
    %          rho, tau                 the pair's correlation and tau (d = 2 only)
    %          max_abs_rho, mean_abs_rho   over the off-diagonal entries of R
    %          numerical_valid          kendallToCorrelation accepted the matrix
    %          repair_applied           the eigenvalue floor moved R (d >= 3)
    %          repair_shift_rms         RMS off-diagonal change of the repair
    %          repair_gross             repair_shift_rms above the threshold
    %          degenerate_pair          d = 2 duplicate-channel condition
    %          min_eig, cond_R          smallest eigenvalue and condition number
    %          logdet_R                 log det R (NaN when R is not PD)
    %          sum_logdet_minors        sum_i log det of R with row and column i removed; 0 at d = 2, NaN when R is not PD
    %          reason                   '' or the audit's failure code

    arguments
        tau_sub (:, :) double {mustBeReal}
        fit_opts (1, 1) struct = struct()
    end

    d = size(tau_sub, 1);
    if size(tau_sub, 2) ~= d || d < 2
        error('unitCorrelationFromTau:NotASquareTauMatrix', ...
            ['tau_sub must be a square [d x d] Kendall matrix with d >= 2; ' ...
            'got %d x %d.'], size(tau_sub, 1), size(tau_sub, 2));
    end
    fit_opts = tFitDefaultOptions(fit_opts);

    corr = struct('rho', NaN, 'tau', NaN, ...
        'max_abs_rho', NaN, 'mean_abs_rho', NaN, ...
        'numerical_valid', false, 'repair_applied', false, ...
        'repair_shift_rms', NaN, 'repair_gross', false, ...
        'degenerate_pair', false, 'min_eig', NaN, 'cond_R', NaN, ...
        'logdet_R', NaN, 'sum_logdet_minors', NaN, 'reason', '');

    if d == 2
        corr.tau = tau_sub(1, 2);
    end
    if any(~isfinite(tau_sub), 'all')
        % A constant or all-tied channel gives NaN tau-b, so the correlation reference does not exist, and inventing one would put a fabricated dependence into the fit.
        R = nan(d);
        corr.reason = 'nonfinite_kendall_tau';
        return;
    end

    [R, audit] = kendallCorrelationReference(tau_sub, ...
        struct('max_projection_shift_rms', fit_opts.max_projection_shift_rms));

    corr.numerical_valid  = audit.numerical_valid;
    corr.repair_applied   = audit.projection_applied;
    corr.repair_shift_rms = audit.off_diag_shift_rms;
    corr.repair_gross     = audit.projection_gross;
    corr.degenerate_pair  = (d == 2) && ~audit.numerical_valid;
    corr.reason           = audit.reason;

    offdiag = R(~eye(d));
    corr.max_abs_rho  = max(abs(offdiag));
    corr.mean_abs_rho = mean(abs(offdiag));
    if d == 2
        corr.rho = R(1, 2);
    end

    corr.min_eig = min(eig((R + R.') / 2));
    corr.cond_R = cond(R);

    % The log-determinant and the leave-one-out log-determinants are the Gaussian reference's own Cholesky quantities, so the unit row and Phase 4 read one implementation.
    % A non-positive-definite R leaves both terms NaN.
    G = gaussianInformationMeasures(R);
    corr.logdet_R = G.logdet_R;
    corr.sum_logdet_minors = sum(G.logdet_R_minor);
end
