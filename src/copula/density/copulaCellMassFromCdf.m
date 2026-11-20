function mass = copulaCellMassFromCdf(u_edges, v_edges, z_u, z_v, cdfFun)
    % COPULACELLMASSFROMCDF Exact cell probabilities of a bivariate copula on a grid.
    %
    % mass = copulaCellMassFromCdf(u_edges, v_edges, z_u, z_v, cdfFun) evaluates the copula's distribution function on the full edge grid and applies the rectangle rule:
    %   mass(i, j) = C(u_{i+1}, v_{j+1}) - C(u_i, v_{j+1})
    %              - C(u_{i+1}, v_j) + C(u_i, v_j)
    % Each cell is therefore exact rather than quadrature-approximated, and the masses of a grid covering [0, 1]^2 sum to 1.
    %
    % C(u, v) = F_2(z_u, z_v), with F_2 the latent bivariate distribution function supplied as cdfFun (mvncdf for the Gaussian copula, mvtcdf for the t copula) and z the margin quantiles of the edges.
    % The edges u = 0 and u = 1 are handled in closed form, C(0, v) = 0 and C(1, v) = v, because their quantiles are infinite; cdfFun only sees edges strictly inside (0, 1).
    %
    % Inputs:
    %   u_edges, v_edges - strictly ascending edges inside [0, 1], at least two each.
    %   z_u, z_v         - the latent quantiles of those edges; the entries at an edge of exactly 0 or 1 are never read.
    %   cdfFun           - handle taking an [m x 2] matrix of latent points and returning the [m x 1] bivariate distribution function.
    % Output:
    %   mass - [numel(u_edges)-1 x numel(v_edges)-1] cell probabilities.

    arguments
        u_edges (1, :) double {mustBeReal, mustBeFinite}
        v_edges (1, :) double {mustBeReal, mustBeFinite}
        z_u (1, :) double {mustBeReal}
        z_v (1, :) double {mustBeReal}
        cdfFun (1, 1) function_handle
    end
    localCheckEdges(u_edges, 'u_edges', numel(z_u));
    localCheckEdges(v_edges, 'v_edges', numel(z_v));

    C = zeros(numel(u_edges), numel(v_edges));
    insideU = u_edges > 0 & u_edges < 1;
    insideV = v_edges > 0 & v_edges < 1;
    if any(insideU) && any(insideV)
        [ZU, ZV] = ndgrid(z_u(insideU), z_v(insideV));
        C(insideU, insideV) = reshape(cdfFun([ZU(:), ZV(:)]), ...
                                      nnz(insideU), nnz(insideV));
    end
    % The uniform margins of the copula give the boundary rows and columns exactly; C(0, .) and C(., 0) are the zeros already in place.
    atOneU = u_edges == 1;
    atOneV = v_edges == 1;
    C(atOneU, :) = repmat(v_edges, nnz(atOneU), 1);
    C(:, atOneV) = repmat(u_edges.', 1, nnz(atOneV));

    mass = diff(diff(C, 1, 1), 1, 2);
end

function localCheckEdges(edges, name, n_quantiles)
    % Error unless the edges ascend strictly inside [0, 1] and come with one quantile each.
    if numel(edges) < 2 || any(diff(edges) <= 0) ...
            || edges(1) < 0 || edges(end) > 1
        error('copulaCellMassFromCdf:InvalidEdges', ...
              ['%s must be at least two strictly ascending edges inside ' ...
               '[0, 1]; got %d edges spanning [%.6g, %.6g].'], ...
              name, numel(edges), min(edges), max(edges));
    end
    if n_quantiles ~= numel(edges)
        error('copulaCellMassFromCdf:QuantileCountMismatch', ...
              'One quantile per %s edge is required; got %d for %d edges.', ...
              name, n_quantiles, numel(edges));
    end
end
