function jobs = simulationJobs(arm, declared, selected)
    %SIMULATIONJOBS Enumerate one arm of the diffusion simulation study: jobs, cells and seeds.
    %
    % A job is one truth simulated as one parent path per replicate (parent_dim channels, the Student-t diffusion of studentDiffusionPath), observed at one rate and fitted at every selected dimension d on the path's first d channels, because the channel sets of a recording are subsets of one recording.
    % A cell is one (truth, d) output of a job.
    %
    %   recovery   job  = one (p_true, T_over_tau, samples_per_tau)
    %              cell = (p_true, d, T_over_tau, samples_per_tau)
    %   factorial  job  = one correlation design (parent_dim x parent_dim, dimension d using its leading d x d block), at one T_over_tau and rate
    %              cell = (design, d)
    %
    % The job number, the cell number and every seed are properties of the declared grid, not of the run.
    % Each is a mixed-radix index over the declared axes, slowest first:
    %
    %   recovery   job  over [n_p n_T n_s]        (p_true, T_over_tau, samples_per_tau)
    %              cell over [n_p n_d n_T n_s]    (p_true, d, T_over_tau, samples_per_tau)
    %              seed(rep) = studySeed('recovery', [ip, iT, rep], [n_p n_T reps])
    %   factorial  job  over [n_g]                (design)
    %              cell over [n_g n_d]            (design, d)
    %              seed(rep) = studySeed('factorial', [ig, rep], [n_g reps])
    %
    % A pilot that selects a subset of the declared values therefore runs the production jobs under their production numbers and draws the production paths.
    % The seed has no d and no samples_per_tau subscript: the d cells of one replicate share its path, and so do its rates.
    % Every path of both arms is integrated on one substep set by the finest declared rate (step_samples_per_tau, studentDiffusionPath's nesting option), each interval of that rate holding refine * max(1, ceil((1/step_samples_per_tau)/delta0)) substeps.
    % The replicate observed at a coarser rate s is therefore the finest path read every step_samples_per_tau / s observations, exactly for the whitened states and to rounding for the coloured observations (6.7e-16 relative at the production grid).
    % The recovery cells of one (p_true, T/tau, replicate) at different rates are thus nested observations of one process, as the channel sets are nested subsets of one parent.
    %
    % Inputs:
    %   arm      - "recovery" or "factorial".
    %   declared - the production grid of the arm:
    %              recovery  : p_true (1 x n_p), T_over_tau (1 x n_T), samples_per_tau (1 x n_s), d (1 x n_d, the fitted dimensions), reps, r_spec (one simulationRMatrix spec), parent_dim, refine, step_samples_per_tau
    %              factorial : designs (1 x n_g struct array of simulationRMatrix specs, each with a name), p_true (scalar), T_over_tau (scalar), samples_per_tau (scalar), d, reps, parent_dim, refine, step_samples_per_tau
    %              step_samples_per_tau must be an integer multiple of every rate the arm declares, so each rate observes the common substep grid.
    %   selected - the subset to run, by value (p_true, T_over_tau, samples_per_tau, d) or by name (designs); a missing or empty field means the whole declared axis.
    %              A value that is not declared is an error, because it has no number.
    %              `selected.reps` runs the first that many replicates of every job; the declared count stays the seed radix.
    %
    % Output:
    %   jobs - 1 x K struct array ordered by job_index, with fields
    %          arm, job_index
    %          subscripts       the job's subscripts on the declared job axes, [ip iT is] or [ig]
    %          sizes            the declared seed radices, [n_p n_T reps] or [n_g reps]
    %                           The recovery seed has no rate subscript, so subscripts and sizes are not the arguments of one studySeed call; seeds holds the seeds.
    %          label
    %          design           "" in the recovery arm
    %          p_true, nu_true  nu_true is Inf at p = 0
    %          T_over_tau, samples_per_tau, step_samples_per_tau
    %          n_obs            T_over_tau * samples_per_tau
    %          refine, parent_dim
    %          d                1 x k selected fitted dimensions, ascending
    %          cell_index       1 x k
    %          cell_sizes       the declared cell radices
    %          reps             the number of replicates run
    %          r_spec
    %          seeds            1 x reps

    arguments
        arm (1, 1) string
        declared (1, 1) struct
        selected (1, 1) struct = struct()
    end

    switch arm
        case "recovery"
            jobs = localRecoveryJobs(declared, selected);
        case "factorial"
            jobs = localFactorialJobs(declared, selected);
        otherwise
            error('simulationJobs:UnknownArm', ...
                'arm must be "recovery" or "factorial"; got "%s".', arm);
    end
end


function jobs = localRecoveryJobs(declared, selected)
    localRequire(declared, {'p_true', 'T_over_tau', 'samples_per_tau', 'd', 'reps', ...
        'r_spec', 'parent_dim', 'refine', 'step_samples_per_tau'}, 'recovery');
    localCheckDimensions(declared);
    localCheckRates(declared.samples_per_tau, declared.step_samples_per_tau, 'recovery');
    n_p = numel(declared.p_true);
    n_T = numel(declared.T_over_tau);
    n_s = numel(declared.samples_per_tau);
    n_d = numel(declared.d);
    sizes = [n_p, n_T, declared.reps];
    reps_run = localRepsRun(declared, selected);

    ip = localAxisSubscripts(declared.p_true, selected, 'p_true');
    iT = localAxisSubscripts(declared.T_over_tau, selected, 'T_over_tau');
    is = localAxisSubscripts(declared.samples_per_tau, selected, 'samples_per_tau');
    id = localAxisSubscripts(declared.d, selected, 'd');

    jobs = localEmptyJobs();
    for a = 1:numel(ip)
        for b = 1:numel(iT)
            % One seed vector per (p_true, T/tau): every rate of it observes the same paths.
            seeds = localSeeds("recovery", [ip(a), iT(b)], sizes, reps_run);
            for c = 1:numel(is)
                p_true = declared.p_true(ip(a));
                T_over_tau = declared.T_over_tau(iT(b));
                samples_per_tau = declared.samples_per_tau(is(c));
                job = localJob(declared, "recovery", ...
                    localMixedRadix([ip(a), iT(b), is(c)], [n_p, n_T, n_s]), ...
                    [ip(a), iT(b), is(c)], sizes, ...
                    sprintf('p%.4g_T%g_s%g', p_true, T_over_tau, samples_per_tau), "", ...
                    p_true, T_over_tau, samples_per_tau, id, ...
                    arrayfun(@(k) localMixedRadix([ip(a), k, iT(b), is(c)], ...
                    [n_p, n_d, n_T, n_s]), id), ...
                    [n_p, n_d, n_T, n_s], reps_run, declared.r_spec, seeds);
                jobs(end + 1) = job; %#ok<AGROW>
            end
        end
    end
    jobs = localSortByJob(jobs);
end


function jobs = localFactorialJobs(declared, selected)
    localRequire(declared, {'designs', 'p_true', 'T_over_tau', 'samples_per_tau', 'd', ...
        'reps', 'parent_dim', 'refine', 'step_samples_per_tau'}, 'factorial');
    localCheckDimensions(declared);
    assert(isscalar(declared.samples_per_tau), 'simulationJobs:BadRate', ...
        'The factorial runs at ONE rate; got %d samples_per_tau values.', ...
        numel(declared.samples_per_tau));
    localCheckRates(declared.samples_per_tau, declared.step_samples_per_tau, 'factorial');
    names = string({declared.designs.name});
    n_g = numel(names);
    n_d = numel(declared.d);
    sizes = [n_g, declared.reps];
    reps_run = localRepsRun(declared, selected);

    ig = localNameSubscripts(names, selected);
    id = localAxisSubscripts(declared.d, selected, 'd');

    jobs = localEmptyJobs();
    for a = 1:numel(ig)
        job = localJob(declared, "factorial", ig(a), ig(a), sizes, ...
            sprintf('%s_T%g', names(ig(a)), declared.T_over_tau), names(ig(a)), ...
            declared.p_true, declared.T_over_tau, declared.samples_per_tau, id, ...
            arrayfun(@(k) localMixedRadix([ig(a), k], [n_g, n_d]), id), ...
            [n_g, n_d], reps_run, declared.designs(ig(a)), ...
            localSeeds("factorial", ig(a), sizes, reps_run));
        jobs(end + 1) = job; %#ok<AGROW>
    end
    jobs = localSortByJob(jobs);
end


function job = localJob(declared, arm, job_index, subscripts, sizes, label, design, ...
        p_true, T_over_tau, samples_per_tau, id, cell_index, cell_sizes, reps_run, ...
        r_spec, seeds)
    % One job; the observation count is the declared window times the job's rate.
    n_obs = round(T_over_tau * samples_per_tau);
    assert(abs(n_obs - T_over_tau * samples_per_tau) < 1e-9, ...
        'simulationJobs:OffGrid', ...
        ['T/tau = %g at %g samples per tau is not a whole number of observation ' ...
        'intervals; the window would be silently rounded.'], ...
        T_over_tau, samples_per_tau);
    assert(n_obs >= 2, 'simulationJobs:WindowTooShort', ...
        ['T/tau = %g at %g samples per tau gives %d observation(s); a fit needs ' ...
        'at least two.'], T_over_tau, samples_per_tau, n_obs);
    job = struct( ...
        'arm', arm, ...
        'job_index', job_index, ...
        'subscripts', subscripts, ...
        'sizes', sizes, ...
        'label', label, ...
        'design', design, ...
        'p_true', p_true, ...
        'nu_true', 1 / p_true, ...
        'T_over_tau', T_over_tau, ...
        'samples_per_tau', samples_per_tau, ...
        'step_samples_per_tau', declared.step_samples_per_tau, ...
        'n_obs', n_obs, ...
        'refine', declared.refine, ...
        'parent_dim', declared.parent_dim, ...
        'd', declared.d(id), ...
        'cell_index', cell_index, ...
        'cell_sizes', cell_sizes, ...
        'reps', reps_run, ...
        'r_spec', r_spec, ...
        'seeds', seeds);
end


function localCheckRates(rates, step_rate, arm)
    % Every declared rate observes the common substep grid: step_rate / s is an integer.
    ratio = step_rate ./ reshape(double(rates), 1, []);
    assert(isscalar(step_rate) && step_rate > 0 && all(rates > 0) && ...
        all(abs(ratio - round(ratio)) < 1e-9) && all(round(ratio) >= 1), ...
        'simulationJobs:RateNotNested', ...
        ['The %s rates [%s] samples per tau must each divide step_samples_per_tau = %g, ' ...
        'so that every rate observes the one substep grid of the study.'], ...
        arm, num2str(rates), step_rate);
end


function localCheckDimensions(declared)
    % Every fitted dimension is a channel set of the parent path, so it must fit inside.
    d = declared.d;
    assert(isnumeric(d) && ~isempty(d) && all(d == round(d)) && all(d >= 2) ...
        && all(diff(d) > 0) && max(d) <= declared.parent_dim, ...
        'simulationJobs:BadDimensions', ...
        ['The fitted dimensions must be increasing integers in [2, parent_dim = %d]; ' ...
        'got [%s].'], declared.parent_dim, num2str(d));
end


function jobs = localEmptyJobs()
    jobs = struct('arm', {}, 'job_index', {}, 'subscripts', {}, 'sizes', {}, ...
        'label', {}, 'design', {}, 'p_true', {}, 'nu_true', {}, 'T_over_tau', {}, ...
        'samples_per_tau', {}, 'step_samples_per_tau', {}, 'n_obs', {}, 'refine', {}, ...
        'parent_dim', {}, 'd', {}, 'cell_index', {}, 'cell_sizes', {}, 'reps', {}, ...
        'r_spec', {}, 'seeds', {});
end


function jobs = localSortByJob(jobs)
    if isempty(jobs)
        return;
    end
    [~, order] = sort([jobs.job_index]);
    jobs = jobs(order);
end


function seeds = localSeeds(namespace, subscripts, sizes, reps_run)
    % One seed per replicate actually run, from the declared sizes.
    % rep is the fastest radix of studySeed's mixed-radix index, so a job's seeds are consecutive and two studySeed calls fill the vector.
    % The last seed is checked against studySeed, so an edit of studySeed's indexing fails here instead of silently renumbering a study's paths.
    first = studySeed(namespace, [subscripts, 1], sizes);
    seeds = first + (0:(reps_run - 1));
    last = studySeed(namespace, [subscripts, reps_run], sizes);
    assert(last == seeds(end), 'simulationJobs:SeedsNotConsecutive', ...
        ['studySeed no longer numbers a job''s replicates consecutively: ' ...
        'replicate %d of [%s] is %.0f, not %.0f.'], reps_run, ...
        num2str(subscripts, '%d '), last, seeds(end));
end


function reps_run = localRepsRun(declared, selected)
    % A pilot runs the first selected.reps replicates of every job; a request above the declared count, which is the seed radix, is capped at that count.
    reps_run = declared.reps;
    if isfield(selected, 'reps') && ~isempty(selected.reps)
        reps_run = min(double(selected.reps), declared.reps);
    end
    validateattributes(reps_run, {'numeric'}, ...
        {'scalar', 'integer', 'positive'}, mfilename, 'reps');
end


function index = localMixedRadix(subscripts, sizes)
    % studySeed's own mixed-radix index, one-based, so job, cell and seed share one rule.
    index = 0;
    for k = 1:numel(sizes)
        index = index * sizes(k) + (subscripts(k) - 1);
    end
    index = index + 1;
end


function subscripts = localAxisSubscripts(axis_values, selected, name)
    % Where the selected values sit on the declared axis, ascending by declared position.
    if ~isfield(selected, name) || isempty(selected.(name))
        subscripts = 1:numel(axis_values);
        return;
    end
    wanted = reshape(double(selected.(name)), 1, []);
    subscripts = zeros(1, numel(wanted));
    for k = 1:numel(wanted)
        hit = find(axis_values == wanted(k), 1);
        assert(~isempty(hit), 'simulationJobs:ValueNotDeclared', ...
            ['%s = %.10g is not on the declared axis [%s], so it has no job or ' ...
            'cell number; add it to the declared grid or select a declared value.'], ...
            name, wanted(k), num2str(axis_values, '%.10g '));
        subscripts(k) = hit;
    end
    subscripts = unique(subscripts);
end


function subscripts = localNameSubscripts(names, selected)
    % The factorial designs are selected by name.
    if ~isfield(selected, 'designs') || isempty(selected.designs)
        subscripts = 1:numel(names);
        return;
    end
    wanted = reshape(string(selected.designs), 1, []);
    subscripts = zeros(1, numel(wanted));
    for k = 1:numel(wanted)
        hit = find(names == wanted(k), 1);
        assert(~isempty(hit), 'simulationJobs:DesignNotDeclared', ...
            ['Design "%s" is not among the declared designs (%s), so it has no ' ...
            'job number.'], wanted(k), strjoin(names, ', '));
        subscripts(k) = hit;
    end
    subscripts = unique(subscripts);
end


function localRequire(declared, fields, arm)
    missing = fields(~isfield(declared, fields));
    assert(isempty(missing), 'simulationJobs:MissingGridField', ...
        'The declared %s grid has no %s field.', arm, strjoin(missing, ', '));
end
