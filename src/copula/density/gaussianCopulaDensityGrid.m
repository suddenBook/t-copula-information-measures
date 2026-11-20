function [density, mass] = gaussianCopulaDensityGrid(rho, u_edges, v_edges)
    % GAUSSIANCOPULADENSITYGRID Cell-averaged Gaussian copula density on a rectangular grid.
    %
    % [density, mass] = gaussianCopulaDensityGrid(rho, u_edges, v_edges) returns the probability mass the bivariate Gaussian copula with correlation rho puts on each cell of the grid u_edges x v_edges, and the cell-averaged density, which is that mass divided by the cell area.
    %
    % The cell average is the quantity a histogram of pseudo-observations estimates, so this panel can be compared with an empirical density cell by cell; a pointwise density at the cell centres is a different number and is not what this returns.
    % The masses come from mvncdf through the exact rectangle rule of copulaCellMassFromCdf, so a grid covering [0, 1]^2 sums to 1.
    %
    % Inputs:
    %   rho     - correlation in (-1, 1).
    %   u_edges - ascending edges inside [0, 1], at least two.
    %   v_edges - ascending edges inside [0, 1], at least two.
    % Outputs:
    %   density - [numel(u_edges)-1 x numel(v_edges)-1] cell-averaged density.
    %   mass    - the same size, the cell probabilities.

    arguments
        rho (1, 1) double {mustBeReal, mustBeFinite}
        u_edges (1, :) double {mustBeReal, mustBeFinite}
        v_edges (1, :) double {mustBeReal, mustBeFinite}
    end
    if abs(rho) >= 1
        error('gaussianCopulaDensityGrid:CorrelationOutOfRange', ...
              'rho must lie strictly inside (-1, 1); got %.6g.', rho);
    end

    R = [1, rho; rho, 1];
    mass = copulaCellMassFromCdf(u_edges, v_edges, ...
                                 localNormalQuantile(u_edges), localNormalQuantile(v_edges), ...
                                 @(Z) mvncdf(Z, [0, 0], R));
    density = mass ./ (diff(u_edges).' * diff(v_edges));
end

function z = localNormalQuantile(u)
    % Standard normal quantile, finite inside (0, 1); the 0 and 1 edges are handled in closed form by copulaCellMassFromCdf, which never reads their placeholder value.
    z = zeros(size(u));
    inside = u > 0 & u < 1;
    z(inside) = norminv(u(inside));
end
