function [R, provenance] = simulationRMatrix(spec, d)
    %SIMULATIONRMATRIX Resolve and validate one simulated correlation design.
    %
    % Supported specs (spec.name is a free label, spec.type selects the branch):
    %   type="equi",   rho=<scalar>
    %   type="factor", loading_range=[low high], loading_order="ascending" (the default) or "spread"
    %                  The d loadings are linspace(low, high, d), in ascending order or ends first and then in van der Corput order.
    %                  The spread order makes every leading block of two or more channels span [low, high], as a study that fits the leading d x d block of one parent needs.
    %   type="block",  rho_within=<scalar>, rho_between=<scalar>, n_blocks=<positive integer>
    %                  Channels are assigned to the blocks round-robin.
    %
    % Inputs:
    %   spec - scalar struct with at least name and type; see the branches above.
    %   d    - dimension (positive integer).
    %
    % Outputs:
    %   R          - [d x d] symmetric correlation matrix, unit diagonal, positive definite (Cholesky is asserted).
    %   provenance - struct with the resolved name and type.

    arguments
        spec (1, 1) struct
        d (1, 1) double {mustBeInteger, mustBePositive}
    end
    assert(isfield(spec, 'name') && isfield(spec, 'type'), ...
        'simulationRMatrix:BadSpec', 'R spec needs name and type fields.');

    type = string(spec.type);
    provenance = struct('name', string(spec.name), 'type', type);
    switch type
        case "equi"
            validateattributes(spec.rho, {'numeric'}, ...
                {'scalar', 'real', 'finite', '>', -1 / max(d - 1, 1), '<', 1});
            R = (1 - spec.rho) * eye(d) + spec.rho * ones(d);
        case "factor"
            assert(isfield(spec, 'loading_range') ...
                && isnumeric(spec.loading_range) ...
                && numel(spec.loading_range) == 2, ...
                'simulationRMatrix:BadFactorSpec', ...
                'A factor R spec needs loading_range=[low high].');
            loadingRange = double(spec.loading_range(:).');
            assert(all(isfinite(loadingRange)) ...
                && all(abs(loadingRange) < 1), ...
                'simulationRMatrix:BadFactorSpec', ...
                'Every factor loading endpoint must have magnitude below one.');
            loadings = linspace(loadingRange(1), loadingRange(2), d).';
            loadings = loadings(localLoadingOrder(spec, d));
            R = diag(1 - loadings.^2) + loadings*loadings.';
        case "block"
            required = {'rho_within', 'rho_between', 'n_blocks'};
            assert(all(isfield(spec, required)), ...
                'simulationRMatrix:BadBlockSpec', ...
                'A block R spec needs rho_within, rho_between and n_blocks.');
            validateattributes(spec.rho_within, {'numeric'}, ...
                {'scalar', 'real', 'finite', '>=', 0, '<', 1});
            validateattributes(spec.rho_between, {'numeric'}, ...
                {'scalar', 'real', 'finite', '>=', 0, ...
                '<=', spec.rho_within});
            validateattributes(spec.n_blocks, {'numeric'}, ...
                {'scalar', 'integer', 'positive'});
            blockIndex = mod((0:(d - 1))', spec.n_blocks) + 1;
            sameBlock = blockIndex == blockIndex.';
            R = spec.rho_between*ones(d) ...
                + (spec.rho_within - spec.rho_between)*double(sameBlock) ...
                + (1 - spec.rho_within)*eye(d);
        otherwise
            error('simulationRMatrix:UnknownType', ...
                'Unknown R spec type "%s".', type);
    end

    R = (R + R.') / 2;
    assert(max(abs(diag(R) - 1)) <= 1e-8, ...
        'simulationRMatrix:NotCorrelation', ...
        'Resolved matrix does not have a unit diagonal.');
    [~, p] = chol(R);
    assert(p == 0, 'simulationRMatrix:NotPositiveDefinite', ...
        'Resolved matrix "%s" is not positive definite at d=%d.', spec.name, d);
end


function order = localLoadingOrder(spec, d)
    % The order of the d factor loadings: ascending, or spread, whose every prefix of two or more covers the range.
    % The spread order takes the two ends first, then the positions round(1 + v_k (d - 1)) of the van der Corput sequence v_k, skipping a position already taken.
    % The multiset of loadings is the same in both orders.
    choice = "ascending";
    if isfield(spec, 'loading_order') && ~isempty(spec.loading_order) && ...
            strlength(string(spec.loading_order)) > 0
        choice = string(spec.loading_order);
    end
    switch choice
        case "ascending"
            order = 1:d;
        case "spread"
            order = unique([1, d], 'stable');
            k = 0;
            while numel(order) < d
                k = k + 1;
                assert(k <= 64 * d, 'simulationRMatrix:SpreadOrder', ...
                    'The spread order of %d loadings did not complete.', d);
                position = round(1 + localVanDerCorput(k) * (d - 1));
                if ~ismember(position, order)
                    order(end + 1) = position; %#ok<AGROW>
                end
            end
        otherwise
            error('simulationRMatrix:BadFactorSpec', ...
                'loading_order must be "ascending" or "spread"; got "%s".', choice);
    end
end


function v = localVanDerCorput(k)
    % The k-th base-2 van der Corput number: the binary digits of k mirrored about the binary point (1 -> 0.5, 2 -> 0.25, 3 -> 0.75, 4 -> 0.125, ...).
    v = 0;
    scale = 0.5;
    while k > 0
        v = v + scale * mod(k, 2);
        k = floor(k / 2);
        scale = scale / 2;
    end
end
