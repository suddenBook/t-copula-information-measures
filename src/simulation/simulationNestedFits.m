function [rows, R_hat, prep_seconds] = simulationNestedFits(X, d_list, fit_opts, cache)
    % SIMULATIONNESTEDFITS Fit the nested channel sets 1..d of one simulated path, for every d.
    %
    % [rows, R_hat, prep_seconds] = simulationNestedFits(X, d_list, fit_opts, cache) runs the study's unit recipe on the channel sets 1:d of one path, for each d in d_list, from one pseudo-observation matrix and one Kendall matrix of the leading max(d_list) channels.
    % Channel set 1:d takes their first d columns and their leading d x d block.
    % Ranks are taken per column and Kendall's tau pairwise, and a simulated path has no incomplete row, so every row is bit-identical to what tFitChannelSet(X(:, 1:d), 1:d, fit_opts, cache) computes on its own.
    % The Kendall matrix, which dominates the cost at long windows, is computed once instead of once per dimension.
    %
    % X may instead be simulationNestedPreparation(X, d_list), the same two matrices computed first, so that a caller holding the only reference to the path can release it before the fits; the rows are the same, bit for bit.
    %
    % Inputs:
    %   X        - [n_obs x D] simulated path, samples down the rows, every value finite, or the struct simulationNestedPreparation(X, d_list) returned for it.
    %   d_list   - 1 x k increasing channel-set sizes in [2, min(D, 20)].
    %   fit_opts - tFitDefaultOptions overrides.
    %   cache    - tCopulaQuantileCache built for n_obs rows at the fit's profile grid, or []; it changes no number, only the cost of the coarse profile.
    %
    % Outputs:
    %   rows         - 1 x k unit rows as tFitUnit returns them, with channels = 1:d NaN-padded to 20 as tFitChannelSet fills them; dataset_key and unit_index are left for the caller.
    %   R_hat        - 1 x k cell, the d x d correlation matrix of each fit.
    %   prep_seconds - wall seconds of the shared ranks and Kendall matrix.

    arguments
        X       % validated by simulationNestedPreparation, or already prepared by it
        d_list (1, :) double {mustBeInteger, mustBePositive}
        fit_opts (1, 1) struct = struct()
        cache = []
    end

    fit_opts = tFitDefaultOptions(fit_opts);
    if isstruct(X)
        prepared = X;
        assert(isscalar(prepared) && isfield(prepared, 'd_list') ...
               && isequal(prepared.d_list, d_list), ...
               'simulationNestedFits:NotPreparedForTheseDimensions', ...
               ['The prepared ranks and Kendall matrix are not ' ...
                'simulationNestedPreparation''s for d_list = [%s].'], num2str(d_list));
    else
        prepared = simulationNestedPreparation(X, d_list);
    end

    k = numel(d_list);
    R_hat = cell(1, k);
    rows = cell(1, k);
    for c = 1:k
        d = d_list(c);
        [R_hat{c}, corr] = unitCorrelationFromTau(prepared.tau(1:d, 1:d), fit_opts);
        row = tFitUnit(prepared.U(:, 1:d), R_hat{c}, corr, fit_opts, cache);
        row.channels = [1:d, nan(1, 20 - d)];
        rows{c} = row;
    end
    rows = [rows{:}];
    prep_seconds = prepared.prep_seconds;
end
