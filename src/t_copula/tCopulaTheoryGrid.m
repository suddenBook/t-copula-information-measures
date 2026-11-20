function T = tCopulaTheoryGrid(dList, pGrid)
    % TCOPULATHEORYGRID Deterministic Student-t copula shape-term and delta grid.
    %
    % T = tCopulaTheoryGrid(dList, pGrid) evaluates the copula-entropy shape terms of the Student-t copula and the beyond-Gaussian information differences they imply on the grid of dimensions dList and shape parameters pGrid (p = 1/nu, p = 0 the Gaussian copula).
    % It involves no data, no randomness and no fitted quantity: every row is a function of (d, p) alone, so the table is reproducible anywhere and is a reference table rather than a result.
    %
    % The differences do not depend on R.
    % kappa and kappa_margin carry no correlation, and the Gaussian level cancels in every difference, so no correlation matrix appears in this file.
    % That is why the deltas can be tabulated against (d, p) alone while the full measures cannot.
    %
    % Inputs:
    %   dList - interaction orders, integers >= 2, any shape (used as a list).
    %   pGrid - shape parameters p = 1/nu, real, finite, >= 0, any shape (used as a list); p = 0 gives zero shape terms and differences.
    %
    % Output:
    %   T - table, one row per (d, p) with p varying fastest, with columns d, p, nu, kappa_t, kappa_t_dm1, MI_delta_nats, TC_delta_nats, DTC_delta_nats, O_delta_nats, kappa_over_p2.
    %
    %   kappa_t       = kappa_t(d, nu), the shape term of H_c = 1/2 log det R + kappa
    %   kappa_t_dm1   = kappa_t(d-1, nu), the leave-one-out margin shape term, exact for the t family because it is closed under marginalisation with the same nu (0 at d = 2)
    %   MI_delta_nats = -kappa at d = 2, NaN otherwise, because mutual information is a bivariate measure and no value is invented for it at d >= 3
    %   TC_delta_nats = -kappa
    %   DTC_delta_nats= d*kappa_t_dm1 - (d-1)*kappa
    %   O_delta_nats  = (d-2)*kappa - d*kappa_t_dm1
    %   kappa_over_p2 = kappa/p^2, the small-p scaling diagnostic, NaN at p = 0, where it is 0/0 (its finite limit is not filled in)
    %
    % The three multivariate deltas satisfy O = TC - DTC identically, and the function asserts that identity so that an edit to any one of the three expressions cannot pass silently.

    arguments
        dList double {mustBeInteger, mustBeGreaterThanOrEqual(dList, 2)}
        pGrid double {mustBeReal, mustBeFinite, mustBeNonnegative}
    end

    dValues = reshape(dList, 1, []);
    pValues = reshape(pGrid, 1, []);
    vars = {'d', 'p', 'nu', 'kappa_t', 'kappa_t_dm1', 'MI_delta_nats', ...
            'TC_delta_nats', 'DTC_delta_nats', 'O_delta_nats', 'kappa_over_p2'};
    M = nan(numel(dValues) * numel(pValues), numel(vars));

    row = 0;
    for d = dValues
        ce = tCopulaEntropy(d, pValues);
        if ~ce.valid
            error('tCopulaTheoryGrid:InvalidShapeTerm', ...
                  ['tCopulaEntropy returned a non-finite shape term at d = %d ' ...
                   'over a p grid of %d values spanning [%.6g, %.6g].'], ...
                  d, numel(pValues), min(pValues), max(pValues));
        end
        kappa = reshape(ce.kappa, 1, []);
        kappaMargin = reshape(ce.kappa_margin, 1, []);

        tcDelta = -kappa;
        dtcDelta = d * kappaMargin - (d - 1) * kappa;
        oDelta = (d - 2) * kappa - d * kappaMargin;
        if d == 2
            miDelta = tcDelta;
        else
            miDelta = nan(1, numel(pValues));
        end
        kappaOverP2 = nan(1, numel(pValues));
        positive = pValues > 0;
        kappaOverP2(positive) = kappa(positive) ./ pValues(positive).^2;

        % O = TC - DTC is an algebraic identity in (kappa, kappa_margin), so only floating-point roundoff separates the two sides.
        % A residual above 1e-12 means one of the three expressions above does not match the decomposition, not that the arithmetic is imprecise.
        residual = max(abs(oDelta - (tcDelta - dtcDelta)));
        assert(residual <= 1e-12, 'tCopulaTheoryGrid:DeltaIdentityViolated', ...
               ['O = TC - DTC failed at d = %d: max residual %.3e exceeds 1e-12 ' ...
                'over %d p values.'], d, residual, numel(pValues));

        block = row + (1:numel(pValues));
        M(block, :) = [repmat(d, numel(pValues), 1), pValues(:), ...
                       1 ./ pValues(:), kappa(:), kappaMargin(:), miDelta(:), ...
                       tcDelta(:), dtcDelta(:), oDelta(:), kappaOverP2(:)];
        row = row + numel(pValues);
    end

    T = array2table(M, 'VariableNames', vars);
end
