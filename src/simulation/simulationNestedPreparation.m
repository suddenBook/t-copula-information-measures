function prepared = simulationNestedPreparation(X, d_list)
    % SIMULATIONNESTEDPREPARATION The ranks and Kendall matrix the nested fits of a path read.
    %
    % prepared = simulationNestedPreparation(X, d_list) computes one pseudo-observation matrix and one Kendall tau-b matrix of the leading max(d_list) channels of one simulated path.
    % It is the first half of simulationNestedFits, which fits every channel set 1:d from these two and reads nothing else of the path.
    % It is split out so that a caller holding the only reference to the path can release it before the fits; simulationPathJob does this, so a worker never holds the path and the fits' working set at once.
    % simulationNestedFits(prepared, d_list, ...) equals simulationNestedFits(X, d_list, ...) bit for bit.
    %
    % Inputs:
    %   X      - [n_obs x D] simulated path, samples down the rows, every value finite.
    %   d_list - 1 x k increasing channel-set sizes in [2, min(D, 20)].
    %
    % Output:
    %   prepared - struct with
    %                U             [n_obs x max(d_list)] pseudo-observations
    %                tau           [max(d_list) x max(d_list)] Kendall tau-b
    %                d_list        the d_list they are prepared for
    %                prep_seconds  wall seconds of the two

    arguments
        X (:, :) double {mustBeReal}
        d_list (1, :) double {mustBeInteger, mustBePositive}
    end

    % simulationNestedFits's own checks, raised under its identifiers, which its callers see.
    D = size(X, 2);
    assert(~isempty(d_list) && all(d_list >= 2) && all(diff(d_list) > 0) ...
           && max(d_list) <= min(D, 20), 'simulationNestedFits:BadDimensions', ...
           ['d_list must be increasing channel-set sizes in [2, %d] for a %d x %d path; ' ...
            'got [%s].'], min(D, 20), size(X, 1), D, num2str(d_list));
    % The shared ranks equal each channel set's own ranks only when no row is dropped as incomplete; a simulated path has none, and this assertion keeps it so.
    assert(all(isfinite(X), 'all'), 'simulationNestedFits:NonFinitePath', ...
           'The %d x %d path holds %d non-finite value(s); a simulated path has none.', ...
           size(X, 1), D, nnz(~isfinite(X)));

    width = max(d_list);
    prep_timer = tic;
    U = computeCopulaPseudoObservations(X(:, 1:width), 'MinSamples', 1);
    tau = kendallTauMatrix(X(:, 1:width));
    prep_seconds = toc(prep_timer);
    prepared = struct('U', U, 'tau', tau, 'd_list', d_list, 'prep_seconds', prep_seconds);
end
