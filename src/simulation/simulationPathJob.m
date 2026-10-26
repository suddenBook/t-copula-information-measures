function job_out = simulationPathJob(job, fit_opts, cache, n_workers)
    %SIMULATIONPATHJOB Every replicate of one simulation job: simulate, fit every d, score.
    %
    % One job is one (p_true, T_over_tau, samples_per_tau) of the recovery arm or one correlation design of the factorial.
    % Each replicate simulates one parent path of the Student-t diffusion (studentDiffusionPath, from the replicate's own seed) and runs the same estimator as the real-data phases on its channel sets 1:d, for every d of the job.
    % simulationNestedPreparation computes the pseudo-observations and Kendall tau-b, after which the path is released, and simulationNestedFits applies the sine relation with repair and the profile fit in p = 1/nu.
    % Each fit is then scored against the truth of its own d x d block (simulationInformationError).
    % The d cells of one replicate share its path, as the channel sets of a recording share the recording.
    %
    % Inputs:
    %   job       - one element of simulationJobs: arm, job_index, label, design, p_true, T_over_tau, samples_per_tau, step_samples_per_tau, n_obs, refine, parent_dim, d, cell_index, reps, r_spec, seeds.
    %               step_samples_per_tau is the rate whose substep every path is integrated on, so the job's rate observes the study's one path of each replicate.
    %   fit_opts  - (optional) tFitDefaultOptions overrides.
    %   cache     - (optional) tCopulaQuantileCache for job.n_obs rows at the fit's profile grid; it changes no number, only the cost of the coarse profile.
    %               A path whose observations repeat (a Metropolis interval with no accepted substep) has tied ranks, and its fit evaluates the quantiles directly instead of through the cache.
    %   n_workers - (optional) the worker ceiling of the replicate loop, `parfor (rep = 1:reps, n_workers)`; 0 runs the replicates in the client, and the default Inf is unrestricted.
    %               Every replicate owns its seed, so the worker count changes no number.
    %
    % Output job_out:
    %   arm, job_index, label, design, p_true, nu_true, T_over_tau, samples_per_tau, step_samples_per_tau, n_obs, refine, parent_dim, substeps, h, reps, seeds, d, cell_index
    %   R, r_provenance                the parent truth correlation
    %   cells      1 x k struct: cell_index, d, cond_R, min_eig_R (of the d x d truth), G_true (gaussianInformationMeasures), shape_true (tInformationDifferences)
    %   units      one column struct over the reps x k fitted rows, d-major (all replicates of the first d, then the next)
    %              It holds every unit-row field (unitRowsToColumns, unit_index = replicate), every simulationInformationError column, and cell_index, acceptance, repeated_observation_fraction, sim_seconds and prep_seconds (the last four per replicate, repeated across its d).
    %   error_columns                  the names of the information-error columns
    %   wall_seconds, median_sim_seconds, median_prep_seconds
    %   median_fit_seconds             the median over replicates of the summed fit seconds of all d
    %   n_ok                           1 x k, the numerically ok fits per d
    %   fit_opts                       the materialised fit options

    arguments
        job (1, 1) struct
        fit_opts (1, 1) struct = struct()
        cache = []
        n_workers (1, 1) double {mustBeNonnegative} = Inf
    end

    required = {'arm', 'job_index', 'label', 'p_true', 'T_over_tau', ...
        'samples_per_tau', 'step_samples_per_tau', 'n_obs', 'refine', 'parent_dim', ...
        'd', 'cell_index', 'reps', 'r_spec', 'seeds'};
    missing = required(~isfield(job, required));
    assert(isempty(missing), 'simulationPathJob:BadJob', ...
        'The job has no %s field; simulationJobs supplies it.', strjoin(missing, ', '));
    assert(numel(job.seeds) == job.reps, 'simulationPathJob:SeedCountMismatch', ...
        'Job %d (%s) declares %d replicates but carries %d seeds.', ...
        job.job_index, job.label, job.reps, numel(job.seeds));
    assert(numel(job.cell_index) == numel(job.d), 'simulationPathJob:CellCountMismatch', ...
        'Job %d (%s) has %d fitted dimensions but %d cell numbers.', ...
        job.job_index, job.label, numel(job.d), numel(job.cell_index));

    fit_opts = tFitDefaultOptions(fit_opts);
    [R, r_provenance] = simulationRMatrix(job.r_spec, job.parent_dim);
    d_list = reshape(job.d, 1, []);
    k = numel(d_list);
    cells = localTruthCells(job, R, d_list);
    G_true = {cells.G_true};

    p_true = job.p_true;
    nu = 1 / p_true;                % Inf at the Gaussian truth: the exact OU branch
    n_obs = job.n_obs;
    samples_per_tau = job.samples_per_tau;
    path_options = struct('step_samples_per_tau', job.step_samples_per_tau);
    refine = job.refine;
    seeds = job.seeds;
    reps = job.reps;

    per_rep = cell(1, reps);
    wall_timer = tic;
    parfor (rep = 1:reps, n_workers)
        stream = RandStream('twister', 'Seed', seeds(rep));
        sim_timer = tic;
        [X, info] = studentDiffusionPath(R, nu, n_obs, samples_per_tau, refine, stream, ...
            path_options);
        sim_seconds = toc(sim_timer);
        % The fits read only the ranks and the Kendall matrix, so the path is released once they exist.
        % This body holds its only reference, so a worker never holds the path and the fits' working set at once.
        prepared = simulationNestedPreparation(X, d_list);
        X = []; %#ok<NASGU> released, never read again
        [rows, R_hat, prep_seconds] = simulationNestedFits(prepared, d_list, fit_opts, ...
            cache);
        errors = cell(1, k);
        for c = 1:k
            rows(c).unit_index = rep;
            errors{c} = simulationInformationError(d_list(c), p_true, rows(c).p_hat, ...
                G_true{c}, gaussianInformationMeasures(R_hat{c}));
        end
        per_rep{rep} = struct('rows', {rows}, 'errors', {[errors{:}]}, ...
            'acceptance', info.acceptance, ...
            'repeated_observation_fraction', info.repeated_observation_fraction, ...
            'substeps', info.substeps, 'h', info.h, ...
            'sim_seconds', sim_seconds, 'prep_seconds', prep_seconds);
    end
    wall_seconds = toc(wall_timer);
    per_rep = [per_rep{:}];

    % The substep count and h are functions of (nu, D, the two rates, refine) only.
    substeps = unique([per_rep.substeps]);
    h = unique([per_rep.h]);
    assert(isscalar(substeps) && isscalar(h), 'simulationPathJob:StepNotConstant', ...
        'Job %d (%s): the replicates used %d substep counts; one job has one step.', ...
        job.job_index, job.label, numel(substeps));

    [units, error_columns, n_ok, fit_totals] = localUnits(per_rep, d_list, ...
        job.cell_index);

    job_out = struct( ...
        'arm', string(job.arm), ...
        'job_index', job.job_index, ...
        'label', string(job.label), ...
        'design', string(localField(job, 'design', "")), ...
        'p_true', p_true, ...
        'nu_true', nu, ...
        'T_over_tau', job.T_over_tau, ...
        'samples_per_tau', samples_per_tau, ...
        'step_samples_per_tau', job.step_samples_per_tau, ...
        'n_obs', n_obs, ...
        'refine', refine, ...
        'parent_dim', job.parent_dim, ...
        'substeps', substeps, ...
        'h', h, ...
        'reps', reps, ...
        'seeds', reshape(seeds, 1, []), ...
        'd', d_list, ...
        'cell_index', reshape(job.cell_index, 1, []), ...
        'R', R, ...
        'wall_seconds', wall_seconds, ...
        'median_sim_seconds', median([per_rep.sim_seconds]), ...
        'median_prep_seconds', median([per_rep.prep_seconds]), ...
        'median_fit_seconds', median(fit_totals, 'omitnan'), ...
        'n_ok', n_ok);
    job_out.r_provenance = r_provenance;
    job_out.cells = cells;
    job_out.units = units;
    job_out.error_columns = error_columns;
    job_out.fit_opts = fit_opts;
end


function cells = localTruthCells(job, R, d_list)
    % The truth of each cell is the leading d x d block of the parent correlation.
    cells = struct('cell_index', {}, 'd', {}, 'cond_R', {}, 'min_eig_R', {}, ...
        'G_true', {}, 'shape_true', {});
    for c = 1:numel(d_list)
        d = d_list(c);
        R_d = R(1:d, 1:d);
        G = gaussianInformationMeasures(R_d);
        assert(G.valid, 'simulationPathJob:TruthNotMeasurable', ...
            ['The %d x %d truth block of job %d (%s) has no Gaussian information ' ...
            'measures; its smallest eigenvalue is %.3g.'], d, d, job.job_index, ...
            job.label, min(eig((R_d + R_d.') / 2)));
        cells(c) = struct('cell_index', job.cell_index(c), 'd', d, ...
            'cond_R', cond(R_d), 'min_eig_R', min(eig((R_d + R_d.') / 2)), ...
            'G_true', G, 'shape_true', tInformationDifferences(d, job.p_true));
    end
end


function [units, error_columns, n_ok, fit_totals] = localUnits(per_rep, d_list, cell_index)
    % One column struct over every fitted row of the job, d-major, with the error columns and the replicate's simulator facts beside the unit-row fields.
    % A struct array would cost one HDF5 dataset per field per row in a -v7.3 checkpoint.
    reps = numel(per_rep);
    k = numel(d_list);
    all_rows = [per_rep.rows];                  % replicate-major: rep 1 d 1..k, rep 2 ...
    all_errors = [per_rep.errors];
    order = reshape(reshape(1:reps * k, k, reps).', 1, []);   % d-major
    units = unitRowsToColumns(all_rows(order));
    error_columns = reshape(fieldnames(all_errors), 1, []);
    for c = 1:numel(error_columns)
        values = [all_errors.(error_columns{c})];
        units.(error_columns{c}) = reshape(values(order), [], 1);
    end
    units.cell_index = reshape(repmat(reshape(cell_index, 1, []), reps, 1), [], 1);
    for name = {'acceptance', 'repeated_observation_fraction', 'sim_seconds', ...
            'prep_seconds'}
        units.(name{1}) = repmat(reshape([per_rep.(name{1})], [], 1), k, 1);
    end
    ok = reshape(logical(units.numerical_ok), reps, k);
    n_ok = sum(ok, 1);
    fit_totals = sum(reshape(units.fit_seconds, reps, k), 2, 'omitnan');
end


function value = localField(s, name, fallback)
    if isfield(s, name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = fallback;
    end
end
