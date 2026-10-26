function tau = kendallTauMatrixParallel(X, n_workers)
    %KENDALLTAUMATRIXPARALLEL d x d Kendall tau-b matrix, one channel pair per worker.
    %
    % The same matrix as kendallTauMatrix, computed by the same kendallTauKnight call on the same rows, with the nested pair loop flattened into one parfor over the d(d-1)/2 pairs.
    % The worker count is a pure scheduling choice: each pair is an independent call, so the matrix cannot depend on it.
    %
    % It serves Phase 3, which needs one full-recording Kendall matrix per surrogate realization and then reads its submatrices.
    % At 128 channels that is 8128 pairs, which is worth a pool, while a 20-channel unit's 190 pairs is not, so kendallTauMatrix is the serial entry point that Phase 2 uses.
    %
    % X is a parfor broadcast variable, so the whole [samples x channels] matrix is sent to every worker once per call.
    % That cost is paid once per surrogate realization and sits far below the pair scan itself, whereas a call per unit would be dominated by it.
    %
    % Inputs:
    %   X         - [n_samples x n_channels] data, samples down the rows, with at least two channels; the caller removes non-finite rows, since kendallTauKnight returns NaN for a pair if any remain, as in the serial entry point.
    %   n_workers - optional worker ceiling of the loop, the M of `parfor (k = 1:n, M)`.
    %               0 runs the pair scan in the client even when a pool is open, as a run that asked for zero workers requires, and the default Inf leaves the loop unrestricted.
    %               It is a scheduling choice and changes no number.
    % Outputs:
    %   tau - [d x d] symmetric Kendall tau-b matrix, unit diagonal.

    arguments
        X (:, :) double {mustBeReal}
        n_workers (1, 1) double {mustBeNonnegative} = Inf
    end

    d = size(X, 2);
    if d < 2
        error('kendallTauMatrixParallel:TooFewChannels', ...
            ['A Kendall matrix needs at least two channels; X is %d x %d ' ...
            '([samples x channels]).'], size(X, 1), d);
    end

    pairs = nchoosek(1:d, 2);
    n_pairs = size(pairs, 1);
    left = pairs(:, 1);
    right = pairs(:, 2);
    values = zeros(n_pairs, 1);
    parfor (k = 1:n_pairs, n_workers)
        values(k) = kendallTauKnight(X(:, left(k)), X(:, right(k))); %#ok<PFBNS>
    end

    tau = eye(d);
    tau(sub2ind([d, d], pairs(:, 1), pairs(:, 2))) = values;
    tau(sub2ind([d, d], pairs(:, 2), pairs(:, 1))) = values;
end
