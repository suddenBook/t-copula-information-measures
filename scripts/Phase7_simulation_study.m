%PHASE7_SIMULATION_STUDY Phase 7: the data-free simulation study that certifies the estimator on autocorrelated paths.
%
% No recording enters this phase.
% Every number comes from a truth this script declares, so the study shows what the pipeline does when the model is correctly specified, and nothing the data happen to do can confound it.
% That includes the Gaussian member p = 0, which is a boundary of the domain and therefore the hardest case.
%
% The truth has memory: each replicate is one path of the Student-t diffusion, whose unique stationary law is exactly the multivariate t_nu(0, R):
%
%   dX = -X/tau dt + sqrt(2 (nu + X' R^-1 X) / (tau (nu + D - 2))) R^(1/2) dW
%
% Each path starts from an exact stationary draw and is integrated by studentDiffusionPath, whose Metropolis-corrected steps keep that law exactly at every step size.
% With the drift matrix A = -I/tau the process is exactly self-similar in time, so the study runs at tau = 1 and its results depend only on the observation window T/tau and the number of samples per tau, which are both axes of the recovery grid.
% The grid is declared independently of the data and covers the recordings' own values (Phase 6) and a wider range; the cross-recording renderer places the recordings on it afterwards, outside this run.
%
% One parent path per replicate: a replicate simulates parent_dim = 20 channels once and fits every dimension d on the path's first d channels, because the channel sets of a recording are subsets of one recording.
% Every path is integrated on the substep of the finest declared rate, so the replicate observed at a coarser rate is exactly the same path read less often (nested rates).
% The fit is the full Phase 2 estimator (pseudo-observations, Kendall tau-b, the sine relation with repair, the profile fit in p = 1/nu), computed from one rank matrix and one Kendall matrix per path and rate (simulationNestedFits).
%
% Three arms:
%
%   check      The simulator check, which reports and gates nothing (simulatorCheckJob, simulatorCheckSummaries).
%              It reads the one-time law of the paths, also at a coarse forced step against the frozen chain, the same proposals without the Metropolis correction.
%              It compares refinements of the integration step against the estimator's own spread, read as one pooled interval per step pair.
%              Its conditional-mean relaxation bounds the discretisation error on p_hat.
%              It fits and integrates the model's time scales and amplitude excess with the Phase 6 estimator, over lag windows that cover the recordings' own.
%              It measures the saturation of the estimate in the samples per tau.
%   recovery   (p_true, d, T_over_tau, samples_per_tau) over the whole grid at one equicorrelated parent R.
%              It answers how well p is recovered, how often the Gaussian bound is returned when it is the truth, how large the information error at the Gaussian truth is (the floor, a function of T/tau and of the sampling rate), and what observation time and rate an accurate estimate needs.
%   factorial  One (p, T/tau, rate) over six correlation designs at every d.
%              It answers whether any of the above depends on the conditioning of R.
%
% The information error is attributed, not only measured.
% For a t copula every measure splits as M(p, R) = M_Gaussian(R) + Delta(d, p), so the estimator's error M(p_hat, R_hat) - M(p_true, R_true) decomposes exactly into a p-only term, an R-only term and an interaction that is identically zero (simulationInformationError).
%
% Output (<results_root>/simulation/t_copula/all/Phase7/):
%   mat/checkpoints/check_jobNNN.mat, recovery_jobNNN.mat, factorial_jobNNN.mat
%                            One file per job.
%                            A recovery or factorial job holds its parent R, the truth of each of its d cells, and `units`, every fitted row of the job stored column-wise (reps x d rows, d-major) with its error columns and the path's simulator facts.
%                            A check job holds its raw per-path or per-replicate numbers.
%   mat/simulator_check.mat, recovery_all.mat, factorial_all.mat
%                            The merged projections: per-cell metadata plus [cells x reps] matrices of every replicate number, and the check's raw jobs.
%   text/recovery_long.csv, recovery_summary.csv, floor_by_d_T_s.csv,
%   text/spread_law_by_d_s.csv, r_factorial_long.csv, r_factorial_summary.csv,
%   text/check_law.csv, check_law_pit_histogram.csv, check_refinement.csv,
%   text/check_refinement_equivalence.csv, check_refinement_pooled.csv,
%   text/check_relaxation.csv, check_discretisation.csv,
%   text/check_timescale.csv, check_sampling.csv,
%   text/timing_jobs.csv, run_config.json, log.txt, images/
%
% Resumability and sharding: there is one checkpoint per job.
% A run computes the jobs of its memory class (TC_SIM_CLASS, see the scheduling settings) that its shard TC_SHARD=k/N owns: the recovery jobs are split by longest processing time on T/tau x samples per tau, the check and factorial jobs by mod(job - 1, N) == k - 1.
% A later 1/1 pass of the class, or of every class, covers every shard's jobs.
% A job whose checkpoint exists is skipped unless TC_OVERWRITE=1.
% An unrestricted single-shard run (N = 1, TC_SIM_CLASS unset) that finds every selected job checkpointed merges and publishes.
% A sharded run never does, and neither does a run restricted to one class, even as a single shard, because several shards, or the runs of the two classes, finishing together would publish the same files at the same moment.
% That merge belongs to the TC_FINALIZE=1 job that scripts/hpc/submit.sh schedules after the array jobs.
%
% Publication is all or nothing.
% TC_FINALIZE=1 writes the study under its ordinary file names, so it refuses to run with a job missing: a summary quietly short of a job is a wrong result, not a partial one.
% The wrapper schedules the finalizer with afterany, which is exactly when jobs go missing.
%
% Every reuse is checked, and the study is done only when it is published.
% Every checkpoint and every merged mat carries the settings identity of its own arm (settingsIdentity), a hash of the arm's declared grid, the shared path settings, the pilot selection and the fit options.
% A file whose identity is not this run's is refused, not merged.
% The arms have separate identities, so that editing the check settings leaves the recovery checkpoints valid, and the reverse.
% Publication order is checkpoints -> the merged mats -> the CSVs -> the figures -> run_config.json, which only the publishing run writes.
% The study counts as complete only when every merged mat of the run's arms carries this run's identity and every table the mats list and run_config.json exist.
% Otherwise a publishing run rebuilds the publication from the checkpoints, never from a merged mat as it stands.
% TC_OVERWRITE=1 first withdraws the existing publication in every shard, so the finalizer that follows publishes the recomputed jobs.
%
% Environment: TC_RESULTS_ROOT, TC_WORKERS, TC_SHARD, TC_OVERWRITE, TC_RENDER_FIGURES, TC_FINALIZE and TC_SIM_CLASS (light | heavy, scheduling only).
% Pilot overrides (comma lists or integers; empty means production):
%   TC_SIM_REPS          the first N replicates of every recovery and factorial job
%   TC_SIM_P_LIST        recovery p_true values (declared values only)
%   TC_SIM_T_LIST        recovery T/tau values (declared values only)
%   TC_SIM_S_LIST        recovery samples-per-tau values (declared values only)
%   TC_SIM_D_LIST        the fitted dimensions of both arms (declared values only)
%   TC_SIM_STEPS         which arms run, among check, recovery, factorial
%   TC_SIM_CHECK_PARTS   which check parts run, among law, refinement, relaxation, timescale, sampling, coarse
%   TC_SIM_CHECK_PATHS   the first N paths of the law, relaxation, timescale and coarse jobs
%   TC_SIM_CHECK_REPS    the first N replicates of the refinement and sampling jobs
% A pilot keeps the production job and cell numbers and the production seeds: it runs a subset of the declared grid and the first items of each job.
% The Python figure renderers start through system(), so MATLAB must be launched with SHELL set to an executable path such as /bin/bash.

% Find the repository root (the first folder above this script that holds src/ and scripts/) and put src/ on the path.
script_dir = fileparts(mfilename('fullpath'));
worktree_root = script_dir;
while ~(isfolder(fullfile(worktree_root, 'src')) && isfolder(fullfile(worktree_root, 'scripts')))
    parent = fileparts(worktree_root);
    if strcmp(parent, worktree_root)
        error('bootstrap:RepoRootNotFound', ...
            'repository root (a dir with src/ and scripts/) not found above %s', script_dir);
    end
    worktree_root = parent;
end
addpath(genpath(fullfile(worktree_root, 'src')));

%% ============================================================ settings
% The declared grids.
% Job numbers, cell numbers and path seeds are mixed-radix indices over exactly these axes (simulationJobs, simulatorCheckJobs), so editing an axis renumbers the study, while selecting a subset through the TC_SIM_* pilot variables does not.
settings = struct();
settings.rel_base = 'simulation/t_copula/all';

% The parent path, in units of tau: parent_dim channels, integrated with refine times the starting substep rule of studentDiffusionPath on the interval of step_samples_per_tau, the finest declared rate.
% Every path of the recovery and factorial arms therefore runs on one substep, and a replicate observed at a coarser rate s is exactly the finest path read every step_samples_per_tau / s observations: the rates are nested, as the channel sets are.
% The check integrates on the same substep except where it varies the step: its refinement and relaxation parts use the refine factors they declare, and its coarse part forces a step of h_tau.
%   refine           2, the refinement factor f that divides each substep.
%                    The p_hat distributions at f = 1, 2 and 4 differ by at most 0.11 sd, within about 1.7 standard errors of zero.
%                    The clock of the Metropolis chain runs 2.5% slow at f = 2 against 3.4% at f = 1, for about 3% more cost.
%                    The check reads the discretisation bound from the exact relaxation and the pooled f-against-2f interval at 1000 replicates.
settings.parent_dim = 20;
settings.refine = 2;

% Recovery: 9 x 8 x 6 = 432 jobs (p_true, T/tau, samples per tau) and 9 x 19 x 8 x 6 = 8208 cells.
% p = 0 is the Gaussian truth and carries the floor; p = 3 is the heavy end of the domain and carries the boundary flag.
% The window is n = T/tau x samples per tau rows.
% The rates are an axis of their own, and both axes are declared independently of the data, wide enough to cover every recording's samples per tau_env and T/tau_env (Phase 6) and a wider range.
% The rates span [5, 160] samples per tau, the top a factor of two above the densest recording, and the windows span T/tau in [10, 3e4].
settings.recovery = struct( ...
    'p_true',          [0 0.02 0.05 0.1 0.2 0.5 1 2 3], ...
    'T_over_tau',      [10 30 100 300 1000 3000 1e4 3e4], ...
    'samples_per_tau', [5 10 20 40 80 160], ...
    'd',               2:20, ...
    'reps',            1000);
settings.recovery.r_spec = localDesign('equi_0.40', 'equi', 0.40, [], [], [], [], "");
settings.step_samples_per_tau = max(settings.recovery.samples_per_tau);

% Factorial: 6 jobs (the designs, built as parent_dim x parent_dim parents whose leading d x d block is fitted at dimension d) and 6 x 19 = 114 cells, at one mildly non-Gaussian truth, one window and one rate, so that only the correlation structure varies.
% factor_hetero orders its loadings so that every leading block spans the loading range (simulationRMatrix, loading_order "spread"); block_four keeps its round-robin blocks.
settings.factorial = struct( ...
    'p_true',          0.1, ...
    'T_over_tau',      3000, ...
    'samples_per_tau', 10, ...
    'd',               2:20, ...
    'reps',            1000);
settings.factorial.designs = [ ...
    localDesign('equi_0.10',     'equi',   0.10, [], [], [], [], ""), ...
    localDesign('equi_0.40',     'equi',   0.40, [], [], [], [], ""), ...
    localDesign('equi_0.75',     'equi',   0.75, [], [], [], [], ""), ...
    localDesign('equi_0.97',     'equi',   0.97, [], [], [], [], ""), ...
    localDesign('factor_hetero', 'factor', [], [0.25 0.85], [], [], [], "spread"), ...
    localDesign('block_four',    'block',  [], [], 0.75, 0.12, 4, "")];

% The simulator check, on the recovery parent R, at samples_per_tau (the rate of the parts that do not vary it): 6 + 6 + 3 + 6 + 6 + 2 = 29 jobs.
%   law         one-time law after horizon_tau from exact starts, per nu
%   refinement  p_hat at the refine factors 1, 2 and 4, per (nu, T/tau), fitted at d
%   relaxation  E[y_1(t)] / (exp(-t) y_1(0)) - 1 from the start_quantile radius, per nu and refine factor: the exact reference that bounds the discretisation error
%   timescale   tau_env, tau_amp and its Gaussian reference of the model, read by the Phase 6 estimator per nu at every lag window of max_lag_tau, with the integrated amplitude excess beside the fitted one.
%               The lag windows cover the recordings' own windows in units of tau_env, the shortest of which lie below 5 tau, and a wider range, hence the 2 tau window.
%   sampling    p_hat over the recovery rates, inherited from settings.recovery, at nu = 2, 5 and Inf, per T/tau
%   coarse      the one-time law at a forced step h of 0.5 tau, D = 2 and 20, nu = 1/3: the Metropolis chain against the frozen chain, the negative control
settings.check = struct( ...
    'samples_per_tau', 10, ...
    'law',        struct('nu', [1/3 1 2 4 10 Inf], 'paths', 20000, ...
    'horizon_tau', 50), ...
    'refinement', struct('nu', [1/3 2 10], 'T_over_tau', [300 3000], ...
    'reps', 1000, 'refine', [1 2 4], 'd', [2 5 10 20]), ...
    'relaxation', struct('nu', [1/3 2 10], 'paths', 20000, 't', [0.5 1 2], ...
    'refine', [1 2 4], 'start_quantile', 0.5), ...
    'timescale',  struct('nu', [1/3 1 2 4 10 Inf], 'paths', 20, ...
    'T_over_tau', 1000, 'max_lag_tau', [2 5 10 20 50 100], ...
    'floor', 0.05, 'envelope_method', 'score_mode_peak_envelope'), ...
    'sampling',   struct('nu', [2 5 Inf], 'T_over_tau', [300 3000], 'reps', 200, ...
    'samples_per_tau', settings.recovery.samples_per_tau, ...
    'd', [2 5 10 20]), ...
    'coarse',     struct('nu', 1/3, 'D', [2 20], 'paths', 20000, 'horizon_tau', 50, ...
    'h_tau', 0.5));

% The check's reading settings, applied when its tables are built at the merge (simulatorCheckSummaries), so they change no path and no checkpoint and belong to no identity.
%   tolerance_sd     the refinement equivalence margin, in sd of p_hat, against which the pooled interval of each (nu, f against 2f) is read
%   ks_family_alpha  the family-wise level of the law KS distances (Bonferroni over the Metropolis rows)
%   bootstrap_reps, bootstrap_seed  the paired bootstrap of the pooled sd ratio
settings.check_reading = struct('tolerance_sd', 0.1, 'ks_family_alpha', 0.05, ...
    'bootstrap_reps', 2000, 'bootstrap_seed', 20260927);

% The production fit options except store_profile = false, which drops the fitter's candidate and basin diagnostic vectors, per-unit debugging aids the study does not need.
% fitTCopulaShape always returns the 41-node coarse profile, so every unit row still carries it.
settings.fit_opts = struct('store_profile', false);

settings.workers = getEnvInteger("TC_WORKERS", [], [0, 1024]);
settings.shard = getEnvShard("TC_SHARD");

% Scheduling settings, never part of an identity: which jobs this run computes and how many workers a job may use.
% TC_SIM_CLASS = light | heavy restricts a run to one memory class, so that the heavy recovery jobs (more than heavy_n_obs observations per path: the longest windows at the densest rates) can run on nodes with more memory and every other job on nodes with more cores, each class sharded k/N on its own; unset, a run computes every class.
% The memory model charges every worker of the pool worker_gb(1) GiB whether a job uses it or not, each active worker worker_gb(2) GiB more per 1e6 observations of the path it holds (a recovery replicate at D = 20 with its quantile cache), and the MATLAB client client_gb.
% A job of n_obs observations per path may therefore use min(pool, M) workers, with memory_gb the job's own allocation (SLURM_MEM_PER_NODE):
%   M = floor((memory_gb - client_gb - pool x worker_gb(1)) / (worker_gb(2) x n_obs / 1e6))
% A check job is charged for the longest path one of its items holds.
% A pool whose baseline alone does not fit stops the run at its first job, a job of which not even one path fits (M < 1) stops it before that job starts, and outside Slurm there is no ceiling.
% An idle worker holds about 1.2 GiB and a worker's peak is about 1.85 + 0.86 x n_obs / 1e6 GiB at any nu, which worker_gb rounds up to 1.9 + 0.9 x n_obs / 1e6, charging the whole pool the 1.9 GiB baseline because a worker keeps memory after a larger job.
% The client peaks at about 4 GiB and is reserved 8 GiB.
schedule = struct( ...
    'class',       localClass(getenv('TC_SIM_CLASS')), ...
    'heavy_n_obs', 1.2e6, ...
    'worker_gb',   [1.9, 0.9], ...
    'client_gb',   8, ...
    'memory_gb',   localMemoryBudget());

steps = struct( ...
    'check',          true, ...
    'recovery',       true, ...
    'factorial',      true, ...
    'finalize',       true, ...
    'render_figures', getEnvFlag("TC_RENDER_FIGURES", true), ...
    'overwrite',      getEnvFlag("TC_OVERWRITE", false));

finalize_only = getEnvFlag("TC_FINALIZE", false);
results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));

% Pilot overrides; empty means production.
% A listed value must lie on the declared axis, or it would have no job or cell number.
pilot = struct( ...
    'reps',            getEnvInteger("TC_SIM_REPS", [], [1, 1e6]), ...
    'p_true',          localNumberList("TC_SIM_P_LIST"), ...
    'T_over_tau',      localNumberList("TC_SIM_T_LIST"), ...
    'samples_per_tau', localNumberList("TC_SIM_S_LIST"), ...
    'd',               localNumberList("TC_SIM_D_LIST"));
pilot_check = struct( ...
    'parts', {getEnvList("TC_SIM_CHECK_PARTS")}, ...
    'paths', getEnvInteger("TC_SIM_CHECK_PATHS", [], [1, 1e7]), ...
    'reps',  getEnvInteger("TC_SIM_CHECK_REPS", [], [1, 1e6]));
steps = localSelectArms(steps, getEnvList("TC_SIM_STEPS"));

%% ============================================================ run
started_utc = datetime('now', 'TimeZone', 'UTC');
ctx = openPhaseOutput(settings.rel_base, 7, results_root);
checkpoint_dir = fullfile(ctx.mat_dir, 'checkpoints');
if ~isfolder(checkpoint_dir)
    mkdir(checkpoint_dir);
end
timing_path = fullfile(ctx.text_dir, 'timing_jobs.csv');

% The path settings every arm shares, merged into each arm's declared grid: the parent, the refine factor and the rate whose interval carries the substep rule.
% Each arm declares its own observation rate(s).
shared = struct( ...
    'parent_dim',           settings.parent_dim, ...
    'step_samples_per_tau', settings.step_samples_per_tau, ...
    'refine',               settings.refine);
fit_opts = tFitDefaultOptions(settings.fit_opts);
check_R = simulationRMatrix(settings.recovery.r_spec, settings.parent_dim);

arms = struct('name', {'check', 'recovery', 'factorial'}, ...
    'merged_name', {'simulator_check.mat', 'recovery_all.mat', 'factorial_all.mat'}, ...
    'enabled', {steps.check, steps.recovery, steps.factorial}, ...
    'jobs', {[], [], []}, 'identity', {'', '', ''});
arms(1).jobs = simulatorCheckJobs(localWithFields(settings.check, ...
    rmfield(shared, 'parent_dim')), pilot_check);
arms(2).jobs = simulationJobs("recovery", ...
    localWithFields(settings.recovery, shared), pilot);
arms(3).jobs = simulationJobs("factorial", ...
    localWithFields(settings.factorial, shared), ...
    struct('d', pilot.d, 'reps', pilot.reps));

% The settings identity of each arm covers its declared grid, the shared path settings, the pilot selection that says which of its jobs and how many of their items this run wants, and the materialised fit options; the check's identity also names the parent R it uses.
% The shard, the worker count and the step switches are left out, because they schedule the work and change no number.
% A pilot override is part of it, because it changes which study is run.
arms(1).identity = settingsIdentity(struct('phase', 'Phase7', 'arm', 'check', ...
    'shared', shared, 'check', settings.check, 'r_spec', settings.recovery.r_spec, ...
    'pilot', pilot_check, 'fit_opts', fit_opts));
arms(2).identity = settingsIdentity(struct('phase', 'Phase7', 'arm', 'recovery', ...
    'shared', shared, 'recovery', settings.recovery, 'pilot', pilot, ...
    'fit_opts', fit_opts));
arms(3).identity = settingsIdentity(struct('phase', 'Phase7', 'arm', 'factorial', ...
    'shared', shared, 'factorial', settings.factorial, ...
    'pilot', struct('d', pilot.d, 'reps', pilot.reps), 'fit_opts', fit_opts));
arms = arms([arms.enabled]);

merged_paths = fullfile(ctx.mat_dir, {arms.merged_name});
merged = ~steps.overwrite && ~isempty(arms) && all(cellfun(@isfile, merged_paths));
if merged
    % The study is merged.
    % Each arm's identity must be this run's, or the mats describe a different study under the same names.
    % Every table the mats list and run_config.json must also exist; otherwise the run that wrote the mats stopped before its publication finished, or a table is missing.
    for a = 1:numel(arms)
        localAssertIdentity(merged_paths{a}, arms(a).identity);
    end
    if ~finalize_only && localIsPublished(merged_paths, ctx.text_dir)
        fprintf('[Phase7] already done, skipping (%s)\n', strjoin(merged_paths, ', '));
        return;
    end
end
if steps.overwrite
    localWithdrawPublication(ctx);
end

% A finalize-only run merges checkpoints and computes nothing in parallel.
% A requested pool whose baseline cannot fit the allocation is refused before it opens, because the cgroup could otherwise kill the job while the pool starts, before any message is printed.
if finalize_only
    n_workers = 0;
else
    if ~isempty(settings.workers) && settings.workers > 0
        localWorkerCeiling(struct('label', "the requested pool", 'n_obs', 0), ...
            settings.workers, schedule);
    end
    n_workers = openParallelPool(isempty(settings.workers) || settings.workers > 0, ...
        settings.workers);
end
pool_guard = onCleanup(@() delete(gcp('nocreate')));

memory_text = 'no per-job ceiling';
if ~isempty(schedule.memory_gb)
    memory_text = sprintf(['%.1f GiB for the job ceilings (client %.1f GiB, pool %d x ' ...
        '%.2f GiB, %.2f GiB per 1e6 observations per active worker, measured 2026-09-28)'], ...
        schedule.memory_gb, schedule.client_gb, n_workers, schedule.worker_gb(1), ...
        schedule.worker_gb(2));
end
fprintf(['[Phase7] shard %d/%d | class %s | workers %d | memory %s | ' ...
    'overwrite %d | finalize_only %d\n' ...
    '[Phase7] parent D = %d | recovery rates %s samples per tau | substep on ' ...
    'the %g-per-tau grid | refine %d\n'], ...
    settings.shard(1), settings.shard(2), schedule.class, n_workers, ...
    memory_text, steps.overwrite, finalize_only, settings.parent_dim, ...
    mat2str(settings.recovery.samples_per_tau), settings.step_samples_per_tau, ...
    settings.refine);
for a = 1:numel(arms)
    fprintf('[Phase7] %-9s %3d job(s) | identity %s\n', arms(a).name, ...
        numel(arms(a).jobs), arms(a).identity);
end

if ~finalize_only
    for a = 1:numel(arms)
        localRunArm(arms(a), settings, steps, checkpoint_dir, timing_path, ...
            n_workers, check_R, schedule);
    end
end

missing = arrayfun(@(arm) localMissingJobs(arm, checkpoint_dir), arms, ...
    'UniformOutput', false);
complete = all(cellfun(@isempty, missing));
fprintf('[Phase7] checkpoints missing: %s\n', strjoin(arrayfun(@(a) ...
    sprintf('%s %d', arms(a).name, numel(missing{a})), 1:numel(arms), ...
    'UniformOutput', false), ', '));
if finalize_only
    localAssertComplete(arms, missing, checkpoint_dir);
end

% Only a publishing run publishes: the TC_FINALIZE=1 job that scripts/hpc/submit.sh schedules after the array jobs, or an unrestricted single-shard run, which is its own finalizer.
% A shard of a sharded run never publishes, whatever it finds on disk, and neither does a run of one class, even at 1/1, because several shards, or the runs of the two classes, finishing together would race over the same files.
% The publisher always rebuilds the study from the checkpoints and never republishes a merged mat as it stands, so a merge older than a rewritten checkpoint can never be published.
if steps.finalize && complete && ~isempty(arms) && (finalize_only || ...
        (settings.shard(2) == 1 && strcmp(schedule.class, 'all')))
    if merged
        fprintf(['[Phase7] rebuilding the publication of the merged study from ' ...
            'its checkpoints WITHOUT RECOMPUTING A JOB\n']);
    end
    localFinalize(ctx, arms, checkpoint_dir, settings, steps);

    run_settings = settings;
    run_settings.schedule = schedule;
    run_settings.steps = steps;
    run_settings.pilot = pilot;
    run_settings.pilot_check = pilot_check;
    run_settings.finalize_only = finalize_only;
    run_settings.settings_identity = cell2struct({arms.identity}, {arms.name}, 2);
    writeRunConfig(ctx, run_settings, struct( ...
        'n_workers',     n_workers, ...
        'started_utc',   started_utc, ...
        'finished_utc',  datetime('now', 'TimeZone', 'UTC'), ...
        'shard',         settings.shard, ...
        'jobs_complete', complete));
elseif complete
    fprintf(['[Phase7] every job is on disk; the merge belongs to the ' ...
        'TC_FINALIZE=1 job, not to shard %d of %d of class %s (only an ' ...
        'unrestricted single-shard run publishes on its own)\n'], ...
        settings.shard(1), settings.shard(2), schedule.class);
else
    fprintf(['[Phase7] not finalising: %d job(s) still missing and ' ...
        'TC_FINALIZE is not set.\n'], sum(cellfun(@numel, missing)));
end

fprintf('[Phase7] finished %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC', 'Format', 'uuuu-MM-dd''T''HH:mm:ss''Z''')));


% ===========================================================================
% Local functions
% ===========================================================================
function localRunArm(arm, settings, steps, checkpoint_dir, timing_path, n_workers, ...
        check_R, schedule)
    % Run every job of one arm that this run owns (localOwnedJobs), skipping the ones already checkpointed.
    %
    % The quantile cache of the recovery and factorial jobs is built lazily on the client, once per distinct n_obs, and kept for the whole arm because the job order revisits every window and rate; the parfor broadcasts it.
    % It changes no number, only the cost of the 41-node profile.
    % The check jobs build their own caches.
    % debt: a cache is 175 bytes per observation, about 0.84 GB at n_obs = 4.8e6, each parfor worker holds a copy (inside the worker memory model), and the client keeps every cache it built, about 2.4 GB if one shard visits every (T/tau, rate).
    % Rebuild the cache per job if that ever exceeds the memory.
    cache_store = struct('n', [], 'cache', {{}});
    p_grid = localProfileGrid(settings.fit_opts);
    owned = localOwnedJobs(arm, settings.shard, schedule);
    % An overwrite run first removes every checkpoint it owns.
    % The identity hashes the settings, not the code, so a checkpoint that different code wrote under the same settings would otherwise pass as done for a resuming run if this run stopped before rewriting it.
    if steps.overwrite
        for j = find(owned)
            path = localCheckpointPath(checkpoint_dir, arm.name, arm.jobs(j).job_index);
            if isfile(path)
                delete(path);
            end
        end
    end
    for j = find(owned)
        job = arm.jobs(j);
        path = localCheckpointPath(checkpoint_dir, arm.name, job.job_index);
        if ~steps.overwrite && isfile(path)
            % A job file is keyed by its job number, a property of the declared grid, so a file written under another grid, another item count or other fit options describes other paths under the same number.
            localAssertIdentity(path, arm.identity);
            fprintf('[Phase7] %s job %3d (%s): checkpoint present, skipping\n', ...
                arm.name, job.job_index, job.label);
            continue;
        end

        workers = localWorkerCeiling(job, n_workers, schedule);
        if strcmp(arm.name, 'check')
            job_out = simulatorCheckJob(job, check_R, settings.fit_opts, workers);
            fprintf(['[Phase7] check job %3d (%s): %d item(s), paths of up to %d ' ...
                'observations on %d of %d worker(s), %.1f s wall\n'], ...
                job.job_index, job.label, job.items, job.n_obs, workers, n_workers, ...
                job_out.wall_seconds);
        else
            [cache, cache_store] = localQuantileCache(cache_store, job.n_obs, p_grid);
            job_out = simulationPathJob(job, settings.fit_opts, cache, workers);
            fprintf(['[Phase7] %s job %3d (%s): %d rep(s) x %d d of %d observations ' ...
                'on %d of %d worker(s), %d/%d ok | p_hat median at d = %d: %.4f | ' ...
                '%.1f s wall, %.3g s sim, %.3g s ranks+Kendall, %.3g s fits per ' ...
                'replicate | acceptance %.4f\n'], ...
                arm.name, job.job_index, job.label, job.reps, numel(job.d), job.n_obs, ...
                workers, n_workers, sum(job_out.n_ok), job.reps * numel(job.d), job.d(end), ...
                median(job_out.units.p_hat(end - job.reps + 1:end), 'omitnan'), ...
                job_out.wall_seconds, job_out.median_sim_seconds, ...
                job_out.median_prep_seconds, job_out.median_fit_seconds, ...
                median(job_out.units.acceptance));
        end
        % The identity is a separate top-level variable, so the reuse check above reads it without loading the job's numbers.
        localSaveAtomic(path, struct('job', job_out, ...
            'settings_identity', arm.identity));
        localAppendTiming(timing_path, arm.name, job, job_out, workers);
    end
    fprintf('[Phase7] %s arm: %d of %d job(s) owned by shard %d/%d of class %s\n', ...
        arm.name, nnz(owned), numel(arm.jobs), settings.shard(1), settings.shard(2), ...
        schedule.class);
end


function owned = localOwnedJobs(arm, shard, schedule)
    % The jobs of one arm this run computes: those of its memory class, split among the class's N shards.
    % The recovery jobs are split by longest processing time on their cost, T/tau x samples per tau observations per replicate: heaviest first, each to the least-loaded shard, the lowest on a tie, so every shard computes the same assignment.
    % The light check and factorial jobs are split by their job number modulo N.
    % A later 1/1 pass of the class, or of every class, covers every shard's jobs.
    in_class = localInClass(arm, schedule);
    owned = false(1, numel(arm.jobs));
    index = find(in_class);
    if isempty(index)
        return;
    end
    k = shard(1);
    N = shard(2);
    if strcmp(arm.name, 'recovery')
        weight = reshape([arm.jobs(index).n_obs], [], 1);
        [~, order] = sortrows([-weight, reshape([arm.jobs(index).job_index], [], 1)]);
        assigned = zeros(1, N);
        owner = zeros(1, numel(index));
        for o = reshape(order, 1, [])
            [~, target] = min(assigned);
            owner(o) = target;
            assigned(target) = assigned(target) + weight(o);
        end
        owned(index) = owner == k;
    else
        owned(index) = mod([arm.jobs(index).job_index] - 1, N) == k - 1;
    end
end


function in_class = localInClass(arm, schedule)
    % Which jobs of an arm belong to the run's memory class; every job for class all.
    n = numel(arm.jobs);
    switch schedule.class
        case 'all'
            in_class = true(1, n);
        case {'light', 'heavy'}
            if strcmp(arm.name, 'check')
                heavy = false(1, n);
            else
                heavy = [arm.jobs.n_obs] > schedule.heavy_n_obs;
            end
            in_class = heavy == strcmp(schedule.class, 'heavy');
    end
end


function workers = localWorkerCeiling(job, n_workers, schedule)
    % The workers a job may use: the pool, unless the job's own memory allocation cannot hold that many paths of job.n_obs observations at once.
    % Every worker of the pool is resident whether the job uses it or not, so the whole pool is charged its baseline and only the active workers their path.
    % A pool whose baseline alone does not fit, or a job of which not even one path fits beside it, is an error before the job starts, because such a job can only be killed, with nothing checkpointed, and killed again on every resume.
    workers = n_workers;
    if n_workers == 0 || isempty(schedule.memory_gb)
        return;
    end
    pool_gb = n_workers * schedule.worker_gb(1);
    free_gb = schedule.memory_gb - schedule.client_gb - pool_gb;
    assert(free_gb > 0, 'Phase7:PoolExceedsMemory', ...
        ['The pool of %d workers holds %.1f GiB at rest (%.2f GiB per worker, ' ...
        'schedule.worker_gb(1)) and the client %.1f GiB, which leaves nothing of ' ...
        'the job''s %.1f GiB allocation (SLURM_MEM_PER_NODE) for a path. Open a ' ...
        'smaller pool (TC_WORKERS) or request more memory.'], ...
        n_workers, pool_gb, schedule.worker_gb(1), schedule.client_gb, ...
        schedule.memory_gb);
    path_gb = schedule.worker_gb(2) * job.n_obs / 1e6;
    fit = floor(free_gb / path_gb);
    assert(fit >= 1, 'Phase7:JobExceedsMemory', ...
        ['Job %s holds paths of %d observations, %.2f GiB per active worker ' ...
        '(schedule.worker_gb(2) = %.2f GiB per 1e6), but the job''s %.1f GiB ' ...
        'allocation (SLURM_MEM_PER_NODE), less the %.1f GiB client and the pool of ' ...
        '%d workers at %.2f GiB each, leaves %.2f GiB, not one path: the job would ' ...
        'be killed, and killed again on every resume. Open a smaller pool ' ...
        '(TC_WORKERS) or request more memory.'], ...
        char(job.label), job.n_obs, path_gb, schedule.worker_gb(2), ...
        schedule.memory_gb, schedule.client_gb, n_workers, schedule.worker_gb(1), ...
        free_gb);
    workers = min(n_workers, fit);
end


function class = localClass(text)
    % TC_SIM_CLASS: empty means every class.
    class = lower(strtrim(char(string(text))));
    if isempty(class)
        class = 'all';
    end
    assert(ismember(class, {'all', 'light', 'heavy'}), 'Phase7:BadClass', ...
        'TC_SIM_CLASS must be light or heavy, or unset for every job; got "%s".', text);
end


function gb = localMemoryBudget()
    % The job's memory allocation in GiB from Slurm (SLURM_MEM_PER_NODE, in MiB), or [] outside a Slurm allocation that sets it.
    gb = [];
    mib = str2double(getenv('SLURM_MEM_PER_NODE'));
    if isfinite(mib) && mib > 0
        gb = mib / 1024;
    end
end


function [cache, store] = localQuantileCache(store, n, p_grid)
    % One tail-quantile table per distinct n_obs, built on first use.
    hit = find(store.n == n, 1);
    if ~isempty(hit)
        cache = store.cache{hit};
        return;
    end
    fprintf('[Phase7] building quantile cache for n = %d at %d grid node(s)\n', ...
        n, numel(p_grid));
    cache = tCopulaQuantileCache(n, p_grid);
    store.n(end + 1) = n;
    store.cache{end + 1} = cache;
end


function p_grid = localProfileGrid(fit_opts)
    % The fitter's own coarse profile grid, so the cache is exact at every node the fitter asks for.
    opts = tFitDefaultOptions(fit_opts);
    p_grid = linspace(opts.p_bounds(1), opts.p_bounds(2), opts.profile_grid_size);
end


function path = localCheckpointPath(checkpoint_dir, arm, job_index)
    path = fullfile(checkpoint_dir, sprintf('%s_job%03d.mat', arm, job_index));
end


function missing = localMissingJobs(arm, checkpoint_dir)
    % The jobs of the whole selected grid, not just this shard's, with no checkpoint.
    missing = zeros(1, 0);
    for j = 1:numel(arm.jobs)
        if ~isfile(localCheckpointPath(checkpoint_dir, arm.name, arm.jobs(j).job_index))
            missing(end + 1) = arm.jobs(j).job_index; %#ok<AGROW>
        end
    end
end


function localAssertComplete(arms, missing, checkpoint_dir)
    % TC_FINALIZE=1 publishes the study under its ordinary file names, so it refuses a study with a missing job.
    % The wrapper schedules the finalizer with afterany, which is exactly when a worker failure or an exhausted wall clock leaves a job behind.
    names = strings(0, 1);
    for a = 1:numel(arms)
        if ~isempty(missing{a})
            names = [names; compose("%s job %d", string(arms(a).name), ...
                reshape(missing{a}, [], 1))]; %#ok<AGROW>
        end
    end
    if isempty(names)
        return;
    end
    shown = names(1:min(10, numel(names)));
    error('Phase7:IncompleteStudy', ...
        ['TC_FINALIZE=1 publishes the study under its ordinary file names, so ' ...
        'it refuses an incomplete one: %d of %d requested job(s) have no ' ...
        'checkpoint under %s. First %d: %s. Run the missing shard(s), or ' ...
        'select the grid this study actually covers.'], ...
        numel(names), sum(arrayfun(@(arm) numel(arm.jobs), arms)), checkpoint_dir, ...
        numel(shown), strjoin(shown, ', '));
end


function localAssertIdentity(path, identity)
    % Refuse an artifact written under another scientific configuration.
    stored = localStoredIdentity(path);
    if strcmp(stored, identity)
        return;
    end
    if isempty(stored)
        stored = '<none recorded>';
    end
    error('Phase7:SettingsIdentityMismatch', ...
        ['%s was written under the settings identity %s, but this run''s ' ...
        'settings are %s. Rerun with TC_OVERWRITE=1 to recompute it, or ' ...
        'restore the settings that produced it.'], path, stored, identity);
end


function identity = localStoredIdentity(path)
    % The settings identity a MAT file stores as a top-level variable, or '' when it stores none.
    identity = char(string(localStoredVariable(path, 'settings_identity', '')));
end


function value = localStoredVariable(path, name, default)
    % One top-level variable of a MAT file, or DEFAULT when the file does not contain it.
    value = default;
    if ismember(name, who('-file', path))
        loaded = load(path, name);
        value = loaded.(name);
    end
end


function localAppendTiming(path, arm, job, job_out, n_workers)
    % Append one line per finished job.
    % A short line written through O_APPEND is atomic, so the shards of one study may share this file.
    % debt: two shards that start on a missing file can each write the header.
    % The file is a wall-clock log for extrapolating the compute time of a run, not a scientific output, so a duplicate header line is tolerated.
    fid = fopen(path, 'a');
    assert(fid ~= -1, 'Phase7:TimingOpenFailed', ...
        'Could not open the timing log %s for appending.', path);
    closer = onCleanup(@() fclose(fid));
    if ftell(fid) == 0
        fprintf(fid, ['arm,job,label,nu,T_over_tau,n_obs,items,wall_seconds,' ...
            'median_sim_seconds,median_prep_seconds,median_fit_seconds,workers\n']);
    end
    if strcmp(arm, 'check')
        facts = [job.nu, job.T_over_tau, job.n_obs, job.items, job_out.wall_seconds, ...
            NaN, NaN, NaN];
    else
        facts = [job_out.nu_true, job.T_over_tau, job.n_obs, job.reps, ...
            job_out.wall_seconds, job_out.median_sim_seconds, ...
            job_out.median_prep_seconds, job_out.median_fit_seconds];
    end
    fprintf(fid, '%s,%d,%s,%.10g,%.10g,%.10g,%d,%.3f,%.6g,%.6g,%.6g,%d\n', ...
        arm, job.job_index, char(job.label), facts, n_workers);
end


function localFinalize(ctx, arms, checkpoint_dir, settings, steps)
    % Merge every job of the study into the merged mats and the CSVs; the caller has established that every requested job is present.
    %
    % The summaries are computed before the mats are saved, because each mat records which of the declared tables this publication writes.
    % A table with no row (a floor without a Gaussian-truth cell, a spread law without a p_true > 0 cell, a check part that did not run) is not written, and completeness asks only for the tables that are.
    fprintf('[Phase7] finalising\n');
    names = {arms.name};
    recovery_long = table();
    factorial_long = table();
    check_tables = simulatorCheckSummaries(struct('part', {}), ...
        settings.check_reading);
    mats = cell(1, numel(arms));
    for a = 1:numel(arms)
        switch names{a}
            case 'check'
                [check_tables, mats{a}] = localMergeCheck(arms(a), checkpoint_dir, ...
                    settings);
            case 'recovery'
                [recovery_long, mats{a}] = localMergeArm(arms(a), checkpoint_dir, ...
                    settings);
            case 'factorial'
                [factorial_long, mats{a}] = localMergeArm(arms(a), checkpoint_dir, ...
                    settings);
        end
    end

    summaries = simulationSummaries(recovery_long, factorial_long);
    outputs = [reshape(localDeclaredTables(), [], 1), {recovery_long; ...
        summaries.recovery_summary; summaries.floor_by_d_T_s; ...
        summaries.spread_law_by_d_s; factorial_long; summaries.r_factorial_summary; ...
        check_tables.law; check_tables.law_pit_histogram; check_tables.refinement; ...
        check_tables.refinement_equivalence; check_tables.refinement_pooled; ...
        check_tables.relaxation; check_tables.discretisation; ...
        check_tables.timescale; check_tables.sampling}];
    published_tables = reshape(outputs(~cellfun(@isempty, outputs(:, 2)), 1), 1, []);

    for a = 1:numel(arms)
        mats{a}.published_tables = published_tables;
        localSaveAtomic(fullfile(ctx.mat_dir, arms(a).merged_name), mats{a});
    end
    for k = 1:size(outputs, 1)
        localWriteTable(outputs{k, 2}, ctx.text_dir, outputs{k, 1});
    end

    if steps.render_figures
        status = renderPhaseFigures('render_phase7_figures', ctx.base, ...
            ctx.images_dir, ctx.text_dir);
        if isstruct(status) && isfield(status, 'ok') && status.ok
            fprintf('[Phase7] %d figure file(s)\n', status.n_figures);
        else
            fprintf(2, ['[Phase7] figures did not complete; the mats and the ' ...
                'CSVs are unaffected.\n']);
        end
    end
end


function [tables, merged] = localMergeCheck(arm, checkpoint_dir, settings)
    % Every check job's raw numbers, in job order, and the check tables built from them.
    jobs = cell(1, numel(arm.jobs));
    for j = 1:numel(arm.jobs)
        path = localCheckpointPath(checkpoint_dir, arm.name, arm.jobs(j).job_index);
        localAssertIdentity(path, arm.identity);
        loaded = load(path, 'job');
        jobs{j} = loaded.job;
    end
    jobs = [jobs{:}];
    tables = simulatorCheckSummaries(jobs, settings.check_reading);
    merged = struct('arm', 'check', 'check_jobs', jobs, ...
        'settings_identity', arm.identity);
    merged.check = settings.check;
    merged.check_reading = settings.check_reading;
    fprintf('[Phase7] check arm: merged %d job(s)\n', numel(jobs));
end


function [long, merged] = localMergeArm(arm, checkpoint_dir, settings)
    % Merge every checkpoint of one arm into the long per-replicate table and the mat projection ([cells x reps] matrices beside the per-cell metadata).
    %
    % The merged mat drops the 41-node profiles on purpose: at the production grid they are over a gigabyte of per-replicate diagnostics, and the job checkpoints keep them, so nothing is lost and the merged file stays readable.
    % debt: every checkpoint is held in memory at once, which at the production grid (8208 recovery cells x 1000 replicates) is about 8.2 million replicate rows, held as the checkpoints' columns and again as the long table, and the CSV written from it is about 6-7 GiB.
    % Stream job by job into an appending writer if the merge ever runs out of memory.
    blocks = cell(1, numel(arm.jobs));
    jobs = cell(1, numel(arm.jobs));
    for j = 1:numel(arm.jobs)
        path = localCheckpointPath(checkpoint_dir, arm.name, arm.jobs(j).job_index);
        localAssertIdentity(path, arm.identity);
        loaded = load(path, 'job');
        jobs{j} = loaded.job;
        blocks{j} = localJobLongTable(loaded.job, arm.name);
    end
    long = vertcat(blocks{:});
    long = sortrows(long, {'cell', 'rep'});
    merged = localMergedMat(jobs, long, arm, settings);
    fprintf('[Phase7] %s arm: merged %d job(s), %d cell(s), %d replicate row(s)\n', ...
        arm.name, numel(jobs), numel(unique(long.cell)), height(long));
end


function T = localJobLongTable(job_out, arm)
    % One job's fitted rows as a table, one row per (cell, replicate).
    %
    % job_out.units is the column struct: every quantity below is already a column over the reps x d rows (d-major), and the status codes are decoded through their label list.
    % The truth's columns come from the cell each row belongs to.
    units = job_out.units;
    n_rows = numel(units.p_hat);
    reps = job_out.reps;
    k = numel(job_out.d);
    [~, cell_of_row] = ismember(units.cell_index, [job_out.cells.cell_index]);
    truth_logdet = arrayfun(@(c) c.G_true.logdet_R, job_out.cells);
    constant = @(value) repmat(value, n_rows, 1);

    T = table( ...
        units.cell_index, constant(job_out.job_index), ...
        constant(job_out.p_true), constant(job_out.nu_true), units.d, ...
        constant(job_out.T_over_tau), constant(job_out.samples_per_tau), ...
        units.n, constant(job_out.refine), constant(job_out.h), ...
        constant(job_out.substeps), constant(job_out.parent_dim), ...
        repmat((1:reps).', k, 1), repmat(reshape(job_out.seeds, [], 1), k, 1), ...
        units.p_hat, units.nu_hat, ...
        units.numerical_ok, ...
        units.heavy_boundary, ...
        units.gaussian_snapped, ...
        units.ambiguous_profile, ...
        reshape(units.status_labels(double(units.status)), [], 1), ...
        reshape(truth_logdet(cell_of_row), [], 1), ...
        units.logdet_R, ...
        reshape([job_out.cells(cell_of_row).cond_R], [], 1), ...
        reshape([job_out.cells(cell_of_row).min_eig_R], [], 1), ...
        units.repair_shift_rms, ...
        units.loglik_t, units.loglik_gaussian, ...
        units.aic_diff, units.fit_seconds, ...
        units.acceptance, units.repeated_observation_fraction, ...
        units.sim_seconds, units.prep_seconds, ...
        'VariableNames', {'cell', 'job', 'p_true', 'nu_true', 'd', 'T_over_tau', ...
        'samples_per_tau', 'n', 'refine', 'h', 'substeps', 'parent_dim', ...
        'rep', 'seed', 'p_hat', 'nu_hat', 'numerical_ok', 'heavy_boundary', ...
        'gaussian_snapped', 'ambiguous_profile', 'status', ...
        'logdet_R_true', 'logdet_R_hat', 'cond_R', 'min_eig_R', ...
        'repair_shift_rms', 'loglik_t', 'loglik_gaussian', 'aic_diff', ...
        'fit_seconds', 'acceptance', 'repeated_observation_fraction', ...
        'sim_seconds', 'prep_seconds'});

    if strcmp(arm, 'factorial')
        T = addvars(T, repmat(cellstr(char(job_out.design)), n_rows, 1), ...
            'After', 'cell', 'NewVariableNames', 'design');
    end

    for name = job_out.error_columns
        T.(name{1}) = units.(name{1});
    end
end


function merged = localMergedMat(jobs, long, arm, settings)
    % Per-job and per-cell metadata plus one [cells x reps] matrix per replicate quantity, rows in ascending cell order.
    [cells, ~, cell_row] = unique(long.cell);
    reps_max = max(long.rep);
    n_cells = numel(cells);
    matrix = @(values) localStack(values, cell_row, long.rep, n_cells, reps_max);

    job_meta = cellfun(@(job) rmfield(job, {'units', 'error_columns', 'cells'}), ...
        jobs, 'UniformOutput', false);
    merged = struct('arm', arm.name, 'reps', reps_max);
    merged.jobs = [job_meta{:}];
    merged.cell_index = cells;
    first = accumarray(cell_row, (1:height(long)).', [], @min);
    for name = {'job', 'p_true', 'nu_true', 'd', 'T_over_tau', 'samples_per_tau', ...
            'n', 'refine', 'h', 'logdet_R_true', 'cond_R', 'min_eig_R'}
        merged.(['cell_' name{1}]) = long.(name{1})(first);
    end
    merged.settings = settings;
    merged.settings_identity = arm.identity;
    merged.fit_opts = jobs{1}.fit_opts;

    per_replicate = {'seed', 'p_hat', 'nu_hat', 'numerical_ok', 'heavy_boundary', ...
        'gaussian_snapped', 'ambiguous_profile', 'logdet_R_hat', ...
        'repair_shift_rms', 'loglik_t', 'loglik_gaussian', 'aic_diff', ...
        'fit_seconds', 'acceptance', 'repeated_observation_fraction', ...
        'sim_seconds', 'prep_seconds'};
    for name = [per_replicate, jobs{1}.error_columns]
        merged.(name{1}) = matrix(double(long.(name{1})));
    end

    % The long table is what the CSV carries; the mat keeps its column names so a renderer reading the mat alone knows what it has.
    merged.long_columns = long.Properties.VariableNames;
end


function M = localStack(values, cell_row, rep, n_cells, reps_max)
    % [cells x reps], NaN-padded when a resumed run mixes replicate counts.
    M = nan(n_cells, reps_max);
    M(sub2ind([n_cells, reps_max], cell_row, rep)) = values;
end


function localWriteTable(T, text_dir, name)
    % Write one table atomically under text/, or skip it when it has no row.
    path = fullfile(text_dir, name);
    if isempty(T)
        fprintf('[Phase7] %s: nothing to write\n', name);
        return;
    end
    writeTableAtomic(T, path);
    fprintf('[Phase7] wrote %s (%d rows x %d cols)\n', name, height(T), width(T));
end


function localSaveAtomic(path, payload)
    % Save every field of PAYLOAD as its own top-level variable, through a temporary file and a rename.
    % The staging name carries a per-call token, as saveStructAtomic's does, so two writers of one destination can never delete or overwrite each other's half-written file.
    % saveStructAtomic itself is not used because it stores one variable, and these mats are read field by field.
    [~, token] = fileparts(tempname);
    tmp = [path, '.', token, '.tmp.mat'];
    try
        save(tmp, '-struct', 'payload', '-v7.3');
    catch save_error
        localDeleteFiles({tmp});
        rethrow(save_error);
    end
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'Phase7:MoveFailed', 'Could not move %s into place: %s', tmp, msg);
end


function names = localDeclaredTables()
    % The fifteen tables the study publishes, declared here and nowhere else, in the order localFinalize writes them.
    % A table with no row is not written, and the merged mats record which tables are (published_tables).
    names = {'recovery_long.csv', 'recovery_summary.csv', 'floor_by_d_T_s.csv', ...
        'spread_law_by_d_s.csv', 'r_factorial_long.csv', 'r_factorial_summary.csv', ...
        'check_law.csv', 'check_law_pit_histogram.csv', 'check_refinement.csv', ...
        'check_refinement_equivalence.csv', 'check_refinement_pooled.csv', ...
        'check_relaxation.csv', 'check_discretisation.csv', ...
        'check_timescale.csv', 'check_sampling.csv'};
end


function done = localIsPublished(merged_paths, text_dir)
    % The study is published when each merged mat lists the tables of its publication, every one of them exists, and the provenance stamp run_config.json, written last, exists.
    % A mat that lists no table is not taken as published.
    done = isfile(fullfile(text_dir, 'run_config.json'));
    for k = 1:numel(merged_paths)
        tables = localStoredVariable(merged_paths{k}, 'published_tables', {});
        done = done && ~isempty(tables) && ...
            all(cellfun(@(name) isfile(fullfile(text_dir, name)), tables));
    end
end


function localWithdrawPublication(ctx)
    % TC_OVERWRITE=1 withdraws the study's existing publication before any job is recomputed: every merged mat, every table and both stamps (run_config.json and figures_manifest.json).
    % Every shard does it, since none of them owns the publication, so the finalizer that follows publishes the recomputed jobs and nothing on disk looks finished in between.
    % The job checkpoints are left to the shard that owns each (localRunArm removes and rewrites them), and the timing log stays, being a log and not a result.
    localDeleteFiles([ ...
        fullfile(ctx.mat_dir, {'simulator_check.mat', 'recovery_all.mat', ...
        'factorial_all.mat'}), ...
        fullfile(ctx.text_dir, [localDeclaredTables(), ...
        {'run_config.json', 'figures_manifest.json'}]), ...
        {fullfile(ctx.images_dir, 'figures_manifest.json')}]);
    fprintf('[Phase7] TC_OVERWRITE=1: the previous publication of %s was withdrawn\n', ...
        ctx.base);
end


function localDeleteFiles(paths)
    % Delete each listed file that exists.
    for k = 1:numel(paths)
        if isfile(paths{k})
            delete(paths{k});
        end
    end
end


function row = localDesign(name, type, rho, loading_range, rho_within, ...
        rho_between, n_blocks, loading_order)
    % One correlation design, with the full field set so that the six designs form one struct array.
    % simulationRMatrix reads only the fields its own branch needs; loading_order "" selects the factor branch's default, ascending.
    row = struct( ...
        'name',          name, ...
        'type',          type, ...
        'rho',           rho, ...
        'loading_range', loading_range, ...
        'rho_within',    rho_within, ...
        'rho_between',   rho_between, ...
        'n_blocks',      n_blocks, ...
        'loading_order', loading_order);
end


function merged = localWithFields(base, extra)
    % BASE with every field of EXTRA added: an arm's declared grid plus the shared path settings, as the job enumerators read them.
    merged = base;
    for name = reshape(fieldnames(extra), 1, [])
        merged.(name{1}) = extra.(name{1});
    end
end


function steps = localSelectArms(steps, wanted)
    % TC_SIM_STEPS: run only the listed arms.
    % An empty list keeps the declared switches.
    if isempty(wanted)
        return;
    end
    arms = {'check', 'recovery', 'factorial'};
    unknown = setdiff(wanted, arms);
    assert(isempty(unknown), 'Phase7:BadStepList', ...
        'TC_SIM_STEPS lists %s; the arms are %s.', strjoin(unknown, ', '), ...
        strjoin(arms, ', '));
    for k = 1:numel(arms)
        steps.(arms{k}) = steps.(arms{k}) && ismember(arms{k}, wanted);
    end
end


function values = localNumberList(name)
    % A comma-separated numeric environment list, or [] when unset.
    text = getEnvList(name);
    if isempty(text)
        values = [];
        return;
    end
    values = str2double(text);
    assert(all(isfinite(values)), 'Phase7:BadNumberList', ...
        '%s must be a comma-separated list of numbers; got "%s".', ...
        name, getenv(name));
end
