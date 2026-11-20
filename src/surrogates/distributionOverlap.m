function [ovl, info] = distributionOverlap(a, b, n_bins, atoms)
    % DISTRIBUTIONOVERLAP Overlap coefficient of two samples on shared equal-count cells.
    %
    % [ovl, info] = distributionOverlap(a, b, n_bins, atoms) returns the overlap coefficient OVL = sum over cells of min(P_a, P_b) of two samples a and b, where P_a and P_b are the shares of each sample that fall in each cell.
    % OVL is 1 when the two samples put the same share of their values in every cell and 0 when no cell holds values of both; 1 - OVL is the mass outside the overlap.
    % It is a descriptive number about two samples, not a test, and carries no p-value.
    %
    % The cells:
    %   Atoms - every value in atoms is a cell of its own and holds exactly the values equal to it.
    %           The shape estimate p_hat has two atoms, the domain endpoints: p = 0, the Gaussian copula, onto which about half of the fits of Gaussian data are snapped, and p = 3, the heavy end.
    %           A value next to an atom but not on it is not in the atom's cell.
    %   Bins  - the remaining (non-atom) values of the pooled sample [a; b] define n_bins equal-count bins through their quantiles at (1:n_bins-1)/n_bins: edges = unique(quantile(pooled, levels)).
    %           Tied quantiles merge, so a pooled sample with few distinct values gets fewer bins.
    %           The J bins are [-Inf, e1), [e1, e2), ..., [e(J-1), Inf], so every finite value lands in exactly one cell.
    % Each sample's cell counts are divided by that sample's own total count, atoms included, so the masses of either sample sum to one and two samples of different sizes are compared as distributions.
    %
    % Invariance: the bin of a value depends only on its rank among the pooled values.
    % A quantile interpolates between two neighbouring order statistics, and under a strictly increasing map it interpolates between their images, which sit in the same order.
    % So OVL is unchanged by any strictly increasing map of the non-atom values that keeps them off the atoms: p_hat, log p_hat or any other increasing scale gives the same number.
    % A decreasing map such as nu_hat = 1/p_hat reverses the order, and the interpolated quantile edges then move slightly: OVL changes by at most about 0.008 at 500 to 1200 values and 0.0006 at 4000 to 10,000.
    %
    % Non-finite values are not values of a sample: NaN (no estimate) and +-Inf are dropped from each sample before anything is counted, and info reports how many.
    % The overlap of an empty sample with anything is undefined, so ovl is then NaN and info.status names the empty side ('empty_a', 'empty_b', 'empty_both'); otherwise info.status is 'ok'.
    %
    % Inputs:
    %   a, b   - real numeric vectors of any length, including zero; the two samples.
    %   n_bins - positive integer, the number of equal-count bins of the pooled non-atom values (fewer when their quantiles tie).
    %   atoms  - real finite vector of the exact values with a cell of their own; [] for none.
    % Outputs:
    %   ovl  - scalar in [0, 1], or NaN when either sample is empty.
    %   info - struct with fields
    %            status                    'ok' | 'empty_a' | 'empty_b' | 'empty_both'
    %            n_a, n_b                  finite values counted in each sample
    %            n_dropped_a, n_dropped_b  non-finite values dropped
    %            atoms                     1 x K, the atom values in ascending order
    %            edges                     1 x (J-1), the interior bin edges in ascending order
    %            mass_a, mass_b            1 x (K + J) cell masses: the K atoms first, in the order of info.atoms, then the J bins in ascending order (NaN when that sample is empty)

    arguments
        a (:, 1) double {mustBeReal}
        b (:, 1) double {mustBeReal}
        n_bins (1, 1) double {mustBeInteger, mustBePositive}
        atoms (1, :) double {mustBeReal, mustBeFinite}
    end

    atoms = unique(atoms);
    finite_a = isfinite(a);
    finite_b = isfinite(b);
    info = struct('status', 'ok', ...
                  'n_a', nnz(finite_a), 'n_b', nnz(finite_b), ...
                  'n_dropped_a', nnz(~finite_a), 'n_dropped_b', nnz(~finite_b), ...
                  'atoms', atoms, 'edges', zeros(1, 0), ...
                  'mass_a', zeros(1, 0), 'mass_b', zeros(1, 0));
    a = a(finite_a);
    b = b(finite_b);

    % The bins come from the pooled non-atom values, so both samples are counted on the same cells and the comparison is cell by cell.
    pooled = [a(~ismember(a, atoms)); b(~ismember(b, atoms))];
    if ~isempty(pooled)
        info.edges = reshape(unique(quantile(pooled, (1:n_bins - 1) / n_bins)), 1, []);
    end
    info.mass_a = localCellMass(a, atoms, info.edges);
    info.mass_b = localCellMass(b, atoms, info.edges);

    if isempty(a) && isempty(b)
        info.status = 'empty_both';
    elseif isempty(a)
        info.status = 'empty_a';
    elseif isempty(b)
        info.status = 'empty_b';
    end
    if ~strcmp(info.status, 'ok')
        ovl = NaN;
        return
    end
    ovl = sum(min(info.mass_a, info.mass_b));
end

function mass = localCellMass(x, atoms, edges)
    % The share of x in each cell: the atoms first, then the bins [-Inf, e1), ..., [e(J-1), Inf].
    % An empty sample has no shares and gets NaN in every cell.
    n_cells = numel(atoms) + numel(edges) + 1;
    if isempty(x)
        mass = nan(1, n_cells);
        return
    end
    % Every index vector is forced to a column, because logical indexing of a one-value sample returns 0 x 0 rather than 0 x 1, which accumarray refuses.
    x = x(:);
    [on_atom, atom_cell] = ismember(x, atoms);
    atom_counts = accumarray(reshape(atom_cell(on_atom), [], 1), 1, [numel(atoms), 1]);
    bin = reshape(discretize(x(~on_atom), [-Inf, edges, Inf]), [], 1);
    bin_counts = accumarray(bin, 1, [numel(edges) + 1, 1]);
    mass = [atom_counts; bin_counts].' / numel(x);
end
