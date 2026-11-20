function G = gaussianInformationMeasures(R)
    % GAUSSIANINFORMATIONMEASURES Closed-form Gaussian-copula information measures from the log-determinants of R.
    %
    % For a Gaussian copula the copula entropy is 1/2 log det R, and the measures reduce to determinant expressions:
    %   MI_G  = -1/2 log(1 - rho^2)                              (d = 2 only)
    %   TC_G  = -1/2 log det R
    %   DTC_G = -1/2 log( (det R)^(d-1) / prod_j det R_{-j} )
    %   O_G   =  1/2 log( (det R)^(d-2) / prod_j det R_{-j} )
    % Here R_{-j} is R with coordinate j removed.
    % At d = 2, MI_G = TC_G, DTC_G = TC_G and O_G = 0.
    % These are the Gaussian reference values reported beside the Student-t and the model-free (KSG) values.
    % The ratios above are the definitions, not the computation: the measures are computed in the log domain from Cholesky log-determinants, so they stay finite at large d and strong correlation, where (det R)^(d-1) and prod_j det R_{-j} underflow to 0.
    %
    % Inputs:
    %   R - [d x d] copula correlation matrix (unit diagonal, positive definite).
    % Outputs:
    %   G - struct with fields d, valid, MI, TC, DTC, O, logdet_R and logdet_R_minor (1 x d).
    %       A malformed or non-positive-definite input fails closed with valid = false and NaN measures.

    d = size(R, 1);
    G = struct('d', d, 'valid', false, 'MI', NaN, 'TC', NaN, ...
               'DTC', NaN, 'O', NaN, 'logdet_R', NaN, 'logdet_R_minor', NaN);
    if ~isnumeric(R) || ~isreal(R) || ~ismatrix(R) ...
            || size(R, 2) ~= d || d < 2 || any(~isfinite(R(:)))
        return
    end

    % Symmetry and the unit diagonal are checked to a tolerance, not bit for bit.
    % An exchangeable matrix built as rho*ones(d) + (1-rho)*eye(d) has the diagonal 0.9999999999999999 at rho = -0.15, -0.4 and -0.9, one ulp below 1, so an exact test would return all-NaN measures for a valid correlation matrix.
    % The two tolerances are the ones the t-copula fit applies to the same two properties (r_sym_tol and r_diag_tol of tFitDefaultOptions), so one concept carries one tolerance.
    % They are deliberately not parameters: the input is by definition a correlation matrix, and a looser notion would answer a different question.
    %
    % R is then used as given, without renormalisation.
    % A deviation of order 1e-16 moves a Cholesky log-determinant by about d*1e-16, more than ten orders of magnitude below the millinat scale the measures are reported at, while renormalising would also move the off-diagonals.
    R_SYM_TOL  = 1e-10;   % = r_sym_tol of tFitDefaultOptions
    R_DIAG_TOL = 1e-8;    % = r_diag_tol of tFitDefaultOptions
    if norm(R - R.', 'fro') > R_SYM_TOL ...
            || max(abs(diag(R) - 1)) > R_DIAG_TOL
        return
    end

    % The log-determinants come from Cholesky factors, not from det().
    % At large d the determinant of a strongly correlated R underflows to 0, and the raw ratio det_R^(d-1) / prod(det_minor) is then 0/0 = NaN, or +Inf when only the numerator survives as a subnormal.
    % A Cholesky log-determinant stays finite whenever the matrix is positive definite, and validity is Cholesky success on R and on every leave-one-out minor.
    [Lr, pr] = chol(R);
    if pr ~= 0
        return                                   % R not positive-definite
    end
    logdet_R = 2 * sum(log(diag(Lr)));

    logdet_minor = zeros(1, d);
    for j = 1:d
        keep = true(1, d);
        keep(j) = false;
        [Lm, pm] = chol(R(keep, keep));
        if pm ~= 0
            return                               % a leave-one-out minor is not PD
        end
        logdet_minor(j) = 2 * sum(log(diag(Lm)));
    end

    % The measures use the exact identity below, so they equal the determinant ratios to rounding wherever those ratios are representable, and stay finite where the ratios underflow:
    %   log( det_R^(d-1) / prod_j det_minor_j ) = (d-1) logdet_R - sum_j logdet_minor_j
    G.logdet_R       = logdet_R;
    G.logdet_R_minor = logdet_minor;
    G.TC  = -0.5 * logdet_R;
    G.DTC = -0.5 * ((d - 1) * logdet_R - sum(logdet_minor));
    G.O   =  0.5 * ((d - 2) * logdet_R - sum(logdet_minor));
    if d == 2
        G.MI = -0.5 * log(1 - R(1, 2)^2);          % = TC at d = 2
    end
    G.valid = true;
end
