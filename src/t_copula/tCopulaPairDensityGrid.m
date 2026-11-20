function [density, mass] = tCopulaPairDensityGrid(rho, p, u_edges, v_edges)
    % TCOPULAPAIRDENSITYGRID Cell-averaged Student-t copula density on a rectangular grid.
    %
    % [density, mass] = tCopulaPairDensityGrid(rho, p, u_edges, v_edges) returns the probability mass the bivariate Student-t copula with correlation rho and nu = 1/p degrees of freedom puts on each cell of the grid u_edges x v_edges, and the cell-averaged density, which is that mass divided by the cell area.
    %
    % The cell average is the quantity a histogram of pseudo-observations estimates, so this panel can be compared with an empirical density cell by cell; a pointwise density at the cell centres is a different number and is not what this returns.
    % The masses come from mvtcdf through the exact rectangle rule of copulaCellMassFromCdf, so a grid covering [0, 1]^2 sums to 1.
    %
    % p = 0 is the Gaussian copula and is delegated to gaussianCopulaDensityGrid: mvtcdf has no nu = Inf, and the Gaussian copula is a member of the family, evaluated exactly rather than approximated with a large nu.
    %
    % The margin quantiles come from tTailQuantile, as in the likelihood, so a density panel and the fit it illustrates share one quantile route.
    %
    % Inputs:
    %   rho     - correlation in (-1, 1).
    %   p       - shape parameter p = 1/nu, real, finite, >= 0.
    %   u_edges - ascending edges inside [0, 1], at least two.
    %   v_edges - ascending edges inside [0, 1], at least two.
    % Outputs:
    %   density - [numel(u_edges)-1 x numel(v_edges)-1] cell-averaged density.
    %   mass    - the same size, the cell probabilities.

    arguments
        rho (1, 1) double {mustBeReal, mustBeFinite}
        p (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
        u_edges (1, :) double {mustBeReal, mustBeFinite}
        v_edges (1, :) double {mustBeReal, mustBeFinite}
    end
    if abs(rho) >= 1
        error('tCopulaPairDensityGrid:CorrelationOutOfRange', ...
              'rho must lie strictly inside (-1, 1); got %.6g.', rho);
    end
    if p == 0
        [density, mass] = gaussianCopulaDensityGrid(rho, u_edges, v_edges);
        return
    end

    nu = 1 / p;
    R = [1, rho; rho, 1];
    mass = copulaCellMassFromCdf(u_edges, v_edges, ...
                                 localTQuantile(u_edges, p), localTQuantile(v_edges, p), ...
                                 @(Z) mvtcdf(Z, R, nu));
    density = mass ./ (diff(u_edges).' * diff(v_edges));
end

function z = localTQuantile(u, p)
    % Signed t quantile from the two-sided tail magnitude, finite inside (0, 1); the 0 and 1 edges are handled in closed form by copulaCellMassFromCdf, which never reads their placeholder value.
    z = zeros(size(u));
    inside = u > 0 & u < 1;
    ui = u(inside);
    z(inside) = sign(ui - 0.5) .* tTailQuantile(2 * min(ui, 1 - ui), p);
end
