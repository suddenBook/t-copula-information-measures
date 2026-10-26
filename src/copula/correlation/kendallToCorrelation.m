%KENDALLTOCORRELATION Map a Kendall's tau matrix to an elliptical-copula correlation matrix.
%
% Uses the elliptical-copula identity rho = sin(pi * tau / 2), applied elementwise (the Gaussian copula is one member of that class).
% For d >= 3 the resulting matrix need not be positive definite, so it is symmetrized and, when its smallest eigenvalue is at most EigenFloor, its eigenvalues are floored and its diagonal renormalized.
% This is an explicit positive-definite repair, not the nearest-correlation-matrix solution, and callers must inspect and gate the reported off-diagonal shift.
%
% Inputs:
%   tau_matrix - [d x d] Kendall's tau matrix, symmetric, with a diagonal of 1 or 0.
%   varargin   - optional name-value pair:
%       'EigenFloor' (default 1e-8) : the eigenvalue floor, applied before the diagonal is renormalized, so the smallest eigenvalue of the repaired matrix can differ from it.
%
% Outputs:
%   Sigma    - [d x d] correlation matrix (symmetric, unit diagonal), positive definite except possibly for a degenerate pair at d = 2.
%   Sigma_m1 - [d x d] inverse of Sigma, through the Cholesky factor at d >= 3, in closed form at d = 2, and inf(2) for an exactly singular pair.
%   info     - struct with fields pd_before, min_eig_before, min_eig_after, projection_applied, off_diag_shift_l2, valid and reason.
%              off_diag_shift_l2 is the Frobenius norm of the change the repair makes to the off-diagonal entries.
%              valid is false, with a reason, only at d = 2 when the pair is degenerate (1 - |rho| < 1e-6, a duplicate or near-duplicate channel).
%              Such a pair is returned as the raw, possibly singular matrix rather than clamped, so callers gate on info.valid.
%              At d = 1 and d >= 3 valid is always true; at d >= 3 a matrix that still fails Cholesky after the repair raises kendallToCorrelation:NotPD.

function [Sigma, Sigma_m1, info] = kendallToCorrelation(tau_matrix, varargin)

    p = inputParser;
    addRequired(p, 'tau_matrix', @(M) isnumeric(M) && ismatrix(M) && size(M, 1) == size(M, 2));
    addParameter(p, 'EigenFloor', 1e-8, @(x) isnumeric(x) && x > 0);
    parse(p, tau_matrix, varargin{:});

    eig_floor = p.Results.EigenFloor;
    d = size(tau_matrix, 1);

    % d = 1: the correlation matrix is the scalar 1.
    if d == 1
        Sigma = 1;
        Sigma_m1 = 1;
        info = struct('pd_before', true, 'min_eig_before', 1, ...
            'min_eig_after', 1, 'projection_applied', false, ...
            'off_diag_shift_l2', 0, 'valid', true, 'reason', '');
        return;
    end

    % Elementwise tau -> rho via the elliptical-copula identity.
    rho_matrix = sin(pi / 2 * tau_matrix);
    rho_matrix(1:(d + 1):end) = 1;                  % force diag = 1
    rho_matrix = (rho_matrix + rho_matrix.') / 2;   % symmetrize

    if d == 2
        % One free parameter rho, which is not clamped, because clamping would only give a meaningless, barely positive-definite matrix.
        % A degenerate pair (|rho| near 1, a duplicate or near-duplicate channel) is flagged invalid instead, and the raw, possibly singular matrix is returned for callers to gate on info.valid.
        rho = rho_matrix(1, 2);
        min_eig = 1 - abs(rho);
        valid = min_eig >= 1e-6;
        if valid, reason = ''; else, reason = 'degenerate_d2_duplicate_channel'; end
        Sigma = [1, rho; rho, 1];
        denom = 1 - rho^2;
        if denom > 0
            Sigma_m1 = [1, -rho; -rho, 1] / denom;
        else
            Sigma_m1 = inf(2);   % exactly singular: a sentinel; callers gate on info.valid
        end
        info = struct('pd_before', valid, 'min_eig_before', min_eig, ...
            'min_eig_after', min_eig, 'projection_applied', false, ...
            'off_diag_shift_l2', 0, 'valid', valid, 'reason', reason);
        return;
    end

    % d >= 3: unless already positive definite above the floor, eigen-decompose, floor the eigenvalues, rebuild and renormalize the diagonal.
    pre_off = rho_matrix - diag(diag(rho_matrix));
    lambda_min_before = min(eig(rho_matrix));
    pd_before = lambda_min_before > eig_floor;

    if pd_before
        Sigma = rho_matrix;
        projection_applied = false;
    else
        [V, D] = eig((rho_matrix + rho_matrix.') / 2, 'vector');
        D = max(D, eig_floor);
        Sigma_raw = V * diag(D) * V.';
        Sigma_raw = (Sigma_raw + Sigma_raw.') / 2;

        % Renormalize to unit diagonal so it's a correlation matrix.
        scale = sqrt(diag(Sigma_raw));
        Sigma = Sigma_raw ./ (scale * scale.');
        projection_applied = true;
    end

    % Enforce exact diagonal 1 after any floating-point drift.
    Sigma(1:(d + 1):end) = 1;
    Sigma = (Sigma + Sigma.') / 2;

    % Compute Sigma_m1 via Cholesky for numerical stability.
    [L, chol_flag] = chol(Sigma, 'lower');
    if chol_flag ~= 0
        error('kendallToCorrelation:NotPD', ...
            'Correlation matrix failed Cholesky after projection.');
    end
    Sigma_m1 = L.' \ (L \ eye(d));
    Sigma_m1 = (Sigma_m1 + Sigma_m1.') / 2;

    post_off = Sigma - diag(diag(Sigma));
    off_diag_shift_l2 = norm(post_off - pre_off, 'fro');

    info = struct( ...
        'pd_before',          pd_before, ...
        'min_eig_before',     lambda_min_before, ...
        'min_eig_after',      min(eig(Sigma)), ...
        'projection_applied', projection_applied, ...
        'off_diag_shift_l2',  off_diag_shift_l2, ...
        'valid',              true, ...
        'reason',             '');
end
