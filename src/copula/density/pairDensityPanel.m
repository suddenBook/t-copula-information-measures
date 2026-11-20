function panel = pairDensityPanel(x_i, x_j, rho, p_hat, u_edges, v_edges)
    % PAIRDENSITYPANEL The six d = 2 density grids of one channel pair.
    %
    % panel = pairDensityPanel(x_i, x_j, rho, p_hat, u_edges, v_edges) builds the three cell-averaged copula densities of one pair on one grid (the empirical one from the pair's own pseudo-observations, the Gaussian one at rho, the Student-t one at (rho, p_hat)), plus the three signed residual maps and their RMS over cells.
    % All six grids live on the same edges, so a cell of one panel is the same cell of every other panel and the residuals are defined cell by cell.
    %
    % The cell average, not a pointwise density at the cell centres, is what a histogram of pseudo-observations estimates.
    % tCopulaPairDensityGrid and gaussianCopulaDensityGrid return exact cell masses from mvtcdf and mvncdf divided by the cell area, and computePairEmpiricalCopulaDensity divides the counts the same way.
    % The three panels are therefore on one scale, and each integrates to 1 over the unit square.
    %
    % Nothing here is fitted: rho and p_hat come from the caller's Phase 2 unit row, so a panel illustrates a fit and can never disagree with it.
    %
    % Inputs:
    %   x_i, x_j - the two channels' samples, equal-length vectors; non-finite values are allowed, and their rows are dropped by the pseudo-observation step.
    %   rho      - the pair's copula correlation, strictly inside (-1, 1).
    %   p_hat    - the pair's fitted shape parameter p = 1/nu, finite and >= 0; p_hat = 0 makes the t panel the Gaussian panel exactly.
    %   u_edges  - grid edges for the first margin, spanning exactly [0, 1].
    %   v_edges  - grid edges for the second margin, spanning exactly [0, 1].
    %
    % Output panel (every grid [numel(u_edges)-1 x numel(v_edges)-1]):
    %   u_edges, v_edges  the edges actually used, as row vectors
    %   n                 complete rows behind the empirical grid
    %   rho, p_hat, nu_hat
    %   empirical, gaussian, t          the three cell-averaged densities
    %   resid_emp_gauss   empirical - gaussian
    %   resid_emp_t       empirical - t
    %   resid_t_gauss     t - gaussian
    %   rms_emp_gauss, rms_emp_t, rms_t_gauss   RMS of each residual over cells
    %   ratio_gauss_over_t  rms_emp_gauss / rms_emp_t (Inf when the t panel matches the empirical one exactly, NaN when both RMS values are 0)

    arguments
        x_i (:, 1) double {mustBeReal}
        x_j (:, 1) double {mustBeReal}
        rho (1, 1) double {mustBeReal, mustBeFinite}
        p_hat (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
        u_edges (1, :) double {mustBeReal, mustBeFinite}
        v_edges (1, :) double {mustBeReal, mustBeFinite}
    end

    if numel(x_i) ~= numel(x_j)
        error('pairDensityPanel:LengthMismatch', ...
              ['The two channels of a pair must have the same number of ' ...
               'samples; got %d and %d.'], numel(x_i), numel(x_j));
    end
    if abs(rho) >= 1
        error('pairDensityPanel:CorrelationOutOfRange', ...
              'rho must lie strictly inside (-1, 1); got %.6g.', rho);
    end

    empirical = computePairEmpiricalCopulaDensity(x_i, x_j, u_edges, v_edges);
    gaussian = gaussianCopulaDensityGrid(rho, u_edges, v_edges);
    t_density = tCopulaPairDensityGrid(rho, p_hat, u_edges, v_edges);

    resid_emp_gauss = empirical - gaussian;
    resid_emp_t = empirical - t_density;
    resid_t_gauss = t_density - gaussian;

    rms_emp_gauss = localRms(resid_emp_gauss);
    rms_emp_t = localRms(resid_emp_t);

    panel = struct( ...
                   'u_edges', u_edges, ...
                   'v_edges', v_edges, ...
                   'n', sum(all(isfinite([x_i, x_j]), 2)), ...
                   'rho', rho, ...
                   'p_hat', p_hat, ...
                   'nu_hat', 1 / p_hat, ...
                   'empirical', empirical, ...
                   'gaussian', gaussian, ...
                   't', t_density, ...
                   'resid_emp_gauss', resid_emp_gauss, ...
                   'resid_emp_t', resid_emp_t, ...
                   'resid_t_gauss', resid_t_gauss, ...
                   'rms_emp_gauss', rms_emp_gauss, ...
                   'rms_emp_t', rms_emp_t, ...
                   'rms_t_gauss', localRms(resid_t_gauss), ...
                   'ratio_gauss_over_t', rms_emp_gauss / rms_emp_t);
end

function value = localRms(residual)
    % RMS over every cell of the grid; the residual carries the density's units.
    value = sqrt(mean(residual(:) .* residual(:)));
end
