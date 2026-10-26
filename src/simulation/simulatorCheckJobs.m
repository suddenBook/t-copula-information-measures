function jobs = simulatorCheckJobs(declared, selected)
    %SIMULATORCHECKJOBS Enumerate the jobs of Phase 7's simulator check, with their seeds.
    %
    % The simulator check asks whether studentDiffusionPath delivers what the study needs: the Student one-time law, a time scale of tau, and a discretisation error small against the estimator's own sampling spread.
    % It reports and gates nothing.
    % Its six parts, and what one job of each is:
    %
    %   law         one nu: independent exact-start paths advanced horizon_tau at the production step, one observation of each at 0 and at horizon_tau
    %   refinement  one (nu, T_over_tau): replicate paths at every refine factor, each fitted at every d; the factors share each replicate's seed
    %   relaxation  one nu: paths from one fixed start, the conditional mean at times t, at every refine factor; the factors share each path's seed
    %   timescale   one nu: long paths at the production step, their channel-averaged rank autocorrelations and envelope time scales over several lag windows
    %   sampling    one (nu, T_over_tau): replicate paths at every samples-per-tau value, each fitted at every d; the rates share each replicate's seed and the common substep, so a coarser rate observes the finer rate's path
    %   coarse      one parent dimension D: independent exact-start paths advanced horizon_tau in forced steps of h_tau, twice from each seed (the Metropolis-corrected chain and the uncorrected frozen-multiplier chain, the negative control with the wrong radial law), one observation of each at 0 and at horizon_tau
    %
    % The job number and every seed are properties of the declared check.
    % Jobs are numbered over the declared lists in the part order above, with T_over_tau fastest in refinement and sampling.
    % Item k of job j (a path or a replicate) draws from studySeed('check', [j, k], [n_jobs, n_items]), with n_jobs the declared job count and n_items the largest declared item count of any part.
    % So a pilot that selects parts or fewer items runs the production jobs under their production numbers and draws their first items.
    % A part appended to the order leaves the job numbers of the parts before it unchanged, and their seeds too as long as its item count does not exceed n_items, the radix of the item subscript (20,000 in Phase 7's check, the law, relaxation and coarse path counts).
    %
    % Inputs:
    %   declared - struct with law, refinement, relaxation, timescale, sampling and coarse, each the part's declared settings from the Phase 7 script, plus samples_per_tau and refine, the production observation rate and refine factor of the parts that do not vary them, and step_samples_per_tau, the rate whose substep every path is integrated on (every rate a part uses must divide it; the coarse part forces its own step instead).
    %   selected - (optional) parts (cellstr/string subset of the six names), paths (cap on the path count of law, relaxation, timescale and coarse) and reps (cap on the replicate count of refinement and sampling); empty means all.
    %
    % Output:
    %   jobs - 1 x K struct array ordered by job_index, with fields part, job_index, label, nu, D (the coarse part's parent dimension; NaN, the study's parent, for the others), T_over_tau, samples_per_tau (1 x s), refine (1 x f), step_samples_per_tau (recorded on every job; the coarse part forces its own step and does not use it), n_obs (the observations of the longest path one item holds, which sets the job's worker ceiling in Phase 7), items (the number run), spec (the part's declared settings) and seeds (1 x items).

    arguments
        declared (1, 1) struct
        selected (1, 1) struct = struct()
    end

    parts = {'law', 'refinement', 'relaxation', 'timescale', 'sampling', 'coarse'};
    missing = parts(~isfield(declared, parts));
    assert(isempty(missing), 'simulatorCheckJobs:MissingPart', ...
        'The declared check has no %s part.', strjoin(missing, ', '));
    assert(all(isfield(declared, {'samples_per_tau', 'refine', 'step_samples_per_tau'})), ...
        'simulatorCheckJobs:MissingProduction', ...
        ['The declared check needs the production samples_per_tau and refine, and ' ...
        'the step_samples_per_tau of the common substep.']);

    all_jobs = localEnumerate(declared);
    n_jobs = numel(all_jobs);
    n_items = max([all_jobs.declared_items]);

    wanted = parts;
    if isfield(selected, 'parts') && ~isempty(selected.parts)
        wanted = cellstr(string(selected.parts));
        unknown = setdiff(wanted, parts);
        assert(isempty(unknown), 'simulatorCheckJobs:UnknownPart', ...
            'Unknown check part(s) %s; the parts are %s.', strjoin(unknown, ', '), ...
            strjoin(parts, ', '));
    end

    jobs = struct('part', {}, 'job_index', {}, 'label', {}, 'nu', {}, 'D', {}, ...
        'T_over_tau', {}, 'samples_per_tau', {}, 'refine', {}, ...
        'step_samples_per_tau', {}, 'n_obs', {}, 'items', {}, 'spec', {}, 'seeds', {});
    for j = 1:n_jobs
        job = all_jobs(j);
        if ~ismember(job.part, wanted)
            continue;
        end
        items = job.declared_items;
        cap = localCap(selected, job.part);
        if ~isempty(cap)
            items = min(items, cap);
        end
        seeds = studySeed('check', [j, 1], [n_jobs, n_items]) + (0:items - 1);
        assert(seeds(end) == studySeed('check', [j, items], [n_jobs, n_items]), ...
            'simulatorCheckJobs:SeedsNotConsecutive', ...
            'studySeed no longer numbers the items of check job %d consecutively.', j);
        jobs(end + 1) = struct('part', job.part, 'job_index', j, ...
            'label', job.label, 'nu', job.nu, 'D', job.D, 'T_over_tau', job.T_over_tau, ...
            'samples_per_tau', job.samples_per_tau, 'refine', job.refine, ...
            'step_samples_per_tau', declared.step_samples_per_tau, 'n_obs', job.n_obs, ...
            'items', items, 'spec', job.spec, 'seeds', seeds); %#ok<AGROW>
    end
end


function jobs = localEnumerate(declared)
    % Every declared job, in the documented order, with its declared item count.
    jobs = struct('part', {}, 'label', {}, 'nu', {}, 'D', {}, 'T_over_tau', {}, ...
        'samples_per_tau', {}, 'refine', {}, 'n_obs', {}, 'declared_items', {}, ...
        'spec', {});
    s0 = declared.samples_per_tau;
    f0 = declared.refine;
    step = declared.step_samples_per_tau;

    spec = declared.law;
    for nu = localRow(spec.nu)
        jobs(end + 1) = localJob('law', sprintf('law_nu%g', nu), nu, NaN, ...
            spec.horizon_tau, s0, f0, spec.paths, spec, step); %#ok<AGROW>
    end
    spec = declared.refinement;
    for nu = localRow(spec.nu)
        for T = localRow(spec.T_over_tau)
            jobs(end + 1) = localJob('refinement', sprintf('refinement_nu%g_T%g', ...
                nu, T), nu, NaN, T, s0, localRow(spec.refine), spec.reps, spec, ...
                step); %#ok<AGROW>
        end
    end
    spec = declared.relaxation;
    for nu = localRow(spec.nu)
        jobs(end + 1) = localJob('relaxation', sprintf('relaxation_nu%g', nu), nu, NaN, ...
            max(spec.t), s0, localRow(spec.refine), spec.paths, spec, step); %#ok<AGROW>
    end
    spec = declared.timescale;
    for nu = localRow(spec.nu)
        jobs(end + 1) = localJob('timescale', sprintf('timescale_nu%g', nu), nu, NaN, ...
            spec.T_over_tau, s0, f0, spec.paths, spec, step); %#ok<AGROW>
    end
    spec = declared.sampling;
    for nu = localRow(spec.nu)
        for T = localRow(spec.T_over_tau)
            jobs(end + 1) = localJob('sampling', sprintf('sampling_nu%g_T%g', nu, T), ...
                nu, NaN, T, localRow(spec.samples_per_tau), f0, spec.reps, spec, ...
                step); %#ok<AGROW>
        end
    end
    spec = declared.coarse;
    for D = localRow(spec.D)
        jobs(end + 1) = localJob('coarse', sprintf('coarse_nu%g_D%d', spec.nu, D), ...
            spec.nu, D, spec.horizon_tau, 1 / spec.h_tau, 1, spec.paths, spec, ...
            []); %#ok<AGROW>
    end
end


function job = localJob(part, label, nu, D, T_over_tau, samples_per_tau, refine, items, ...
        spec, step)
    % One job of the enumeration, after checking its item count and that its durations sit on the observation grid.
    validateattributes(items, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, [part ' item count']);
    localAssertOnGrid(part, label, localDurations(part, T_over_tau, spec), ...
        samples_per_tau, step);
    % The longest path one item holds, as simulatorCheckJob sizes it: the duration at the densest rate, plus the start row for the parts that also observe time 0.
    n_obs = round(T_over_tau * max(samples_per_tau)) + ...
        ismember(part, {'law', 'relaxation', 'coarse'});
    job = struct('part', part, 'label', label, 'nu', nu, 'D', D, ...
        'T_over_tau', T_over_tau, 'samples_per_tau', samples_per_tau, ...
        'refine', refine, 'n_obs', n_obs, 'declared_items', items, 'spec', spec);
end


function durations = localDurations(part, T_over_tau, spec)
    % Every duration, in tau, that simulatorCheckJob turns into a whole number of observation intervals for this part.
    switch part
        case 'relaxation'
            durations = localRow(spec.t);
        case 'timescale'
            durations = [T_over_tau, localRow(spec.max_lag_tau)];
        otherwise
            durations = T_over_tau;
    end
end


function localAssertOnGrid(part, label, durations, samples_per_tau, step)
    % The check refuses, at enumeration, a duration that is not a whole number of observation intervals at a declared rate, or a rate that does not divide the common substep rate.
    % A bad setting then stops the phase before any shard runs instead of aborting a shard's check job halfway, with its recovery jobs behind it.
    % STEP is [] for a part that forces its own step.
    for s = localRow(samples_per_tau)
        steps = durations * s;
        bad = abs(steps - round(steps)) >= 1e-9 | round(steps) < 1;
        assert(~any(bad), 'simulatorCheckJobs:OffGrid', ...
            ['Check job %s (%s): the duration(s) %s tau are not positive whole ' ...
            'numbers of observation intervals at %g samples per tau; choose ' ...
            'samples_per_tau (and the sampling list) so that every check duration ' ...
            'times it is an integer.'], label, part, mat2str(durations(bad)), s);
        if ~isempty(step)
            ratio = step / s;
            assert(abs(ratio - round(ratio)) < 1e-9 && round(ratio) >= 1, ...
                'simulatorCheckJobs:RateNotNested', ...
                ['Check job %s (%s): %g samples per tau does not divide the common ' ...
                'substep rate %g.'], label, part, s, step);
        end
    end
end


function cap = localCap(selected, part)
    % Paths cap the path-ensemble parts, reps the replicate parts.
    cap = [];
    if ismember(part, {'law', 'relaxation', 'timescale', 'coarse'}) && ...
            isfield(selected, 'paths')
        cap = selected.paths;
    elseif ismember(part, {'refinement', 'sampling'}) && isfield(selected, 'reps')
        cap = selected.reps;
    end
    if ~isempty(cap)
        validateattributes(cap, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
            mfilename, 'item cap');
    end
end


function row = localRow(values)
    row = reshape(double(values), 1, []);
end
