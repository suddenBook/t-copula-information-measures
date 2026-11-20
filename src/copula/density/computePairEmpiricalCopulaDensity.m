% COMPUTEPAIREMPIRICALCOPULADENSITY Empirical cell-averaged copula density of one channel pair.
%
% [density, pmf] = computePairEmpiricalCopulaDensity(x_i, x_j, u_edges, v_edges) returns the cell-averaged density first (cell probability divided by cell area) and the cell probabilities second.
% tCopulaPairDensityGrid and gaussianCopulaDensityGrid also divide their exact cell masses by the cell area, so the empirical, Gaussian and t panels are comparable cell by cell without each caller doing the division.
%
% Each channel is rank-transformed to uniform margins, which leaves the copula unchanged, and the pairs are counted in a 2-D histogram on the given grid.
% The pseudo-observations come from computeCopulaPseudoObservations, as in the copula fit, so both use the same U: rows with a non-finite value are dropped before ranking, and ties take average ranks.
%
% Inputs:
%   x_i, x_j - vectors of equal length.
%   u_edges  - increasing histogram edges for U, spanning exactly [0, 1].
%   v_edges  - increasing histogram edges for V, spanning exactly [0, 1].
%
% Outputs:
%   density - [numel(u_edges)-1 x numel(v_edges)-1] cell-averaged density; sum(density .* cell_area) = 1.
%   pmf     - same size, the cell probabilities, which sum to 1.

function [density, pmf] = computePairEmpiricalCopulaDensity(x_i, x_j, u_edges, v_edges)

    validateattributes(x_i, {'numeric'}, {'vector', 'real'}, mfilename, 'x_i', 1);
    validateattributes(x_j, {'numeric'}, {'vector', 'real'}, mfilename, 'x_j', 2);
    validateattributes(u_edges, {'numeric'}, ...
                       {'vector', 'real', 'finite', 'increasing'}, mfilename, 'u_edges', 3);
    validateattributes(v_edges, {'numeric'}, ...
                       {'vector', 'real', 'finite', 'increasing'}, mfilename, 'v_edges', 4);
    if numel(x_i) ~= numel(x_j)
        error('computePairEmpiricalCopulaDensity:LengthMismatch', ...
              'x_i and x_j must have equal length; got %d and %d.', ...
              numel(x_i), numel(x_j));
    end
    if u_edges(1) ~= 0 || u_edges(end) ~= 1 || ...
            v_edges(1) ~= 0 || v_edges(end) ~= 1
        error('computePairEmpiricalCopulaDensity:IncompleteGrid', ...
              'u_edges and v_edges must each span exactly [0,1].');
    end

    pair_data = [x_i(:), x_j(:)];
    % Complete-case rows and average ranks.
    % computeCopulaPseudoObservations filters rows before ranking, so uniform_pair holds only complete-case rows.
    [uniform_pair, row_mask, ~] = computeCopulaPseudoObservations(pair_data, ...
                                                                  'Rows', 'complete', 'MinSamples', 2); %#ok<ASGLU>

    pmf = histcounts2(uniform_pair(:, 1), uniform_pair(:, 2), u_edges, v_edges);
    pmf_sum = sum(pmf(:));
    n_complete = size(uniform_pair, 1);
    if pmf_sum ~= n_complete
        error('computePairEmpiricalCopulaDensity:HistogramCoverage', ...
              ['The full [0,1]^2 grid counted %d of %d complete-case ranks; ' ...
               'refusing to conditionally renormalize a truncated histogram.'], ...
              pmf_sum, n_complete);
    end
    pmf = pmf / n_complete;

    cell_area = reshape(diff(u_edges), [], 1) * reshape(diff(v_edges), 1, []);
    density = pmf ./ cell_area;
end
