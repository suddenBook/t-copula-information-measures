function cache = tCopulaQuantileCache(n, p_grid)
    %TCOPULAQUANTILECACHE Tail-quantile table for one rank plan at a fixed p grid.
    %
    % cache = tCopulaQuantileCache(n, p_grid) precomputes, for the rank plan of n complete untied rows (ranks 1..n, u = k/(n+1)), the magnitudes |z| of the Student-t quantiles at every shape parameter in p_grid.
    % The table is keyed on the distinct two-sided tail probabilities q = 2*min(u, 1-u) that tCopulaQuantilePlan builds, of which there are about n/2 whatever the dimension, so one cache serves every channel set of a recording.
    %
    % The table is filled by exactly the tTailQuantile calls the direct route makes, on exactly the q_unique vector tCopulaQuantilePlan produces for that rank plan, so a cached and a direct evaluation are bit-identical rather than merely close.
    % tCopulaLoglik takes the direct route whenever the row count differs, the plan it is handed does not carry this rank plan (ties move q_unique), or the shape parameter is not exactly one of the cached nodes, as the fminbnd refinement steps are not.
    %
    % Inputs:
    %   n      - number of complete rows, integer >= 2.
    %   p_grid - shape parameters p = 1/nu to tabulate, real, finite and >= 0; the intended use is the coarse profile grid of tFitDefaultOptions (41 nodes by default).
    %
    % Outputs:
    %   cache - struct with fields
    %     n            the rank plan's row count
    %     p_grid       1 x K the tabulated shape parameters
    %     q_unique     m x 1 the distinct two-sided tail probabilities, ascending
    %     num_distinct m
    %     absz         m x K, absz(i, k) = tTailQuantile(q_unique(i), p_grid(k))
    %
    % Size: m is about n/2, so the table takes about 4*n*K bytes, which is 66 MB at n = 4e5 and K = 41.
    % Build one per recording, not one per unit.

    arguments
        n (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(n, 2)}
        p_grid (1, :) double {mustBeReal, mustBeFinite, mustBeNonnegative}
    end
    assert(~isempty(p_grid), 'tCopulaQuantileCache:EmptyGrid', ...
        'p_grid must hold at least one shape parameter.');

    % The plan itself defines which tail probabilities are distinct (it merges the mirror ranks that differ only by rounding), so the cache asks it rather than reimplementing the merge and drifting from it.
    % Any matrix whose columns are the untied ranks 1..n gives the same q_unique, because unique() sees the same value set; two columns is the plan's minimum.
    plan = tCopulaQuantilePlan(repmat(((1:n).') / (n + 1), 1, 2));

    absz = zeros(plan.num_distinct, numel(p_grid));
    for k = 1:numel(p_grid)
        absz(:, k) = tTailQuantile(plan.q_unique, p_grid(k));
    end

    cache = struct( ...
        'n', n, ...
        'p_grid', p_grid, ...
        'q_unique', plan.q_unique, ...
        'num_distinct', plan.num_distinct, ...
        'absz', absz);
end
