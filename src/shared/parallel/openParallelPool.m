function n_workers = openParallelPool(use_parallel, workers)
    % OPENPARALLELPOOL Open (or reuse) a local parpool of the requested size.
    %
    % Pool size is a pure scheduling choice: the units of every phase are independent and every worker runs NumThreads = 1 (the parcluster('local') default, not overridden here), so the worker count cannot move a result.
    %
    % A shortfall, an absent toolbox and a fallback to serial execution are therefore printed notes and never errors.
    % Serial execution does change the last bits of a long likelihood sum, because the client uses maxNumCompThreads rather than a worker's one thread and the reduction order differs; that floor sits orders of magnitude below any reported effect and is a wall-clock decision, not a scientific one.
    %
    % Inputs:
    %   use_parallel - false to run serially (returns 0).
    %   workers      - requested worker count; [] (default) means the local profile's NumWorkers, which also caps a larger request.
    %
    % Output:
    %   n_workers - workers actually available; 0 when running serially or when the Parallel Computing Toolbox is absent, in which case a bare `parfor` runs as a `for` loop, which is correct, just slower.

    arguments
        use_parallel (1, 1) logical
        workers double {mustBeScalarOrEmpty, mustBeInteger, mustBeNonnegative} = []
    end

    n_workers = 0;
    if ~use_parallel
        return
    end
    if ~license('test', 'Distrib_Computing_Toolbox')
        fprintf(['openParallelPool: Parallel Computing Toolbox unavailable; ' ...
                 'continuing serially.\n']);
        return
    end

    try
        clus = parcluster('local');
        if isempty(workers)
            target = clus.NumWorkers;
        else
            target = min(workers, clus.NumWorkers);
        end
        pool = gcp('nocreate');
        if ~isempty(pool) && pool.NumWorkers == target
            n_workers = pool.NumWorkers;
            return
        end
        if ~isempty(pool)
            delete(pool);
        end
        pool = parpool(clus, target);
        n_workers = pool.NumWorkers;
    catch ME
        fprintf('openParallelPool: parpool init failed; continuing serially: %s\n', ...
                ME.message);
        n_workers = 0;
        return
    end

    if ~isempty(workers) && n_workers ~= workers
        fprintf(['openParallelPool: %d workers requested, pool opened with %d; ' ...
                 'continuing at %d (scheduling only, no result depends on it).\n'], ...
                workers, n_workers, n_workers);
    end
end
