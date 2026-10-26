function job_out = simulatorCheckJob(job, R, fit_opts, n_workers)
    %SIMULATORCHECKJOB Run one job of Phase 7's simulator check and return its raw numbers.
    %
    % job_out = simulatorCheckJob(job, R, fit_opts, n_workers) runs one element of simulatorCheckJobs on the parent correlation R with the production simulator (studentDiffusionPath on the study's common substep job.step_samples_per_tau), each path or replicate from its own seed.
    % It returns the per-item numbers that simulatorCheckSummaries turns into the check tables.
    % Nothing here decides anything: the check reports.
    % What one job returns, by part:
    %
    %   law         per path, Q = |y|^2 (whitened with the known R), the first channel and the squared first coordinate of the whitened direction, at time 0 and at the horizon, plus the path's acceptance, substeps and h.
    %   refinement  per replicate, refine factor and d, p_hat, numerical_ok and heavy_boundary from simulationNestedFits (the study's own fitting function), plus the acceptance, repeated-observation fraction, substeps, h and seconds per path.
    %   relaxation  per path and refine factor, the first whitened coordinate at each time t from the fixed start y0 = sqrt(q) e_1, with q the start_quantile of the stationary |y|^2.
    %               For the SDE, E[y(t) | y0] = exp(-t) y0 exactly.
    %   timescale   the Phase 6 estimator applied to the model's own paths, over lag windows that cover the recordings' own, in tau.
    %               It returns the channel-averaged rank autocorrelations of the score and of its square and the square's Gaussian reference (averagedRankAutocorrelation, with the paths as separate segments).
    %               At every lag window of the spec it adds their envelope time constants, all three read the way the score decides (envelopeReading, fitEnvelopeTimescale), and the integrated autocorrelation times of the square and of its reference over the whole window (integratedAutocorrelationTime).
    %   sampling    as refinement, over the samples-per-tau values instead of the refine factors; the rates observe one path per replicate.
    %   coarse      as law, on the leading D x D block of R in forced substeps of h_tau, for the Metropolis-corrected chain and for the frozen chain from the same seed (its negative control), with a flag per path and chain that the final observation is finite.
    %
    % Inputs:
    %   job       - one element of simulatorCheckJobs.
    %   R         - [D x D] parent correlation of the study (the recovery truth).
    %   fit_opts  - tFitDefaultOptions overrides for the fitting parts.
    %   n_workers - worker ceiling of the item loop; 0 serial, Inf unrestricted.
    %
    % Output:
    %   job_out - struct with part, job_index, label, nu, D, T_over_tau, samples_per_tau, refine, step_samples_per_tau, items, seeds, spec, wall_seconds, and results, the part's struct of arrays (items down the first dimension).

    arguments
        job (1, 1) struct
        R (:, :) double {mustBeReal, mustBeFinite}
        fit_opts (1, 1) struct = struct()
        n_workers (1, 1) double {mustBeNonnegative} = Inf
    end

    fit_opts = tFitDefaultOptions(fit_opts);
    path_options = struct('step_samples_per_tau', job.step_samples_per_tau);
    wall_timer = tic;
    switch job.part
        case 'law'
            results = localLaw(job, R, n_workers, path_options);
        case 'refinement'
            results = localFitted(job, R, fit_opts, n_workers, 'refine', path_options);
        case 'relaxation'
            results = localRelaxation(job, R, n_workers, path_options);
        case 'timescale'
            results = localTimescale(job, R, n_workers, path_options);
        case 'sampling'
            results = localFitted(job, R, fit_opts, n_workers, 'samples_per_tau', ...
                path_options);
        case 'coarse'
            results = localCoarse(job, R, n_workers);
        otherwise
            error('simulatorCheckJob:UnknownPart', ...
                'Check job %d has the unknown part "%s".', job.job_index, job.part);
    end
    D = size(R, 1);
    if strcmp(job.part, 'coarse')
        D = job.D;
    end
    job_out = struct('part', job.part, 'job_index', job.job_index, ...
        'label', job.label, 'nu', job.nu, 'D', D, 'T_over_tau', job.T_over_tau, ...
        'samples_per_tau', job.samples_per_tau, 'refine', job.refine, ...
        'step_samples_per_tau', job.step_samples_per_tau, ...
        'items', job.items, 'seeds', job.seeds, 'spec', job.spec, ...
        'wall_seconds', toc(wall_timer));
    job_out.results = results;
end


function results = localLaw(job, R, n_workers, path_options)
    % One observation at time 0 and one at the horizon, from independent exact starts.
    L = chol(R, 'lower');
    s = job.samples_per_tau;
    n_obs = localSteps(job.T_over_tau, s, 'law horizon') + 1;
    [nu, refine, seeds, items] = deal(job.nu, job.refine, job.seeds, job.items);
    q = nan(items, 2);
    x1 = nan(items, 2);
    u1_squared = nan(items, 2);
    acceptance = nan(items, 1);
    step = nan(items, 2);
    parfor (k = 1:items, n_workers)
        stream = RandStream('twister', 'Seed', seeds(k));
        [X, info] = studentDiffusionPath(R, nu, n_obs, s, refine, stream, path_options);
        y = L \ X([1, end], :).';
        radius_squared = sum(y .^ 2, 1);
        q(k, :) = radius_squared;
        x1(k, :) = X([1, end], 1).';
        u1_squared(k, :) = y(1, :) .^ 2 ./ radius_squared;
        acceptance(k) = info.acceptance;
        step(k, :) = [info.substeps, info.h];
    end
    results = struct('times_tau', [0, job.T_over_tau], 'q', q, 'x1', x1, ...
        'u1_squared', u1_squared, 'acceptance', acceptance, ...
        'substeps', step(:, 1), 'h', step(:, 2));
end


function results = localCoarse(job, R, n_workers)
    % The law at a forced coarse step, for the Metropolis chain and for the uncorrected frozen chain, both from one seed per path.
    % The corrected chain must keep the Student radial law where the frozen one visibly loses it, which shows that the check can see the failure it guards against in the heavy corner, where the one-time law at the production step is weak evidence.
    D = job.D;
    R_D = R(1:D, 1:D);
    L = chol(R_D, 'lower');
    s = job.samples_per_tau;
    n_obs = localSteps(job.T_over_tau, s, 'coarse horizon') + 1;
    [nu, seeds, items] = deal(job.nu, job.seeds, job.items);
    schemes = {'metropolis', 'frozen'};
    q = nan(items, 2, 2);
    x1 = nan(items, 2, 2);
    u1_squared = nan(items, 2, 2);
    finite = false(items, 2);
    acceptance = nan(items, 1);
    parfor (k = 1:items, n_workers)
        q_k = nan(2, 2);
        x_k = nan(2, 2);
        u_k = nan(2, 2);
        finite_k = false(1, 2);
        accepted = NaN;
        for scheme = 1:2
            stream = RandStream('twister', 'Seed', seeds(k));
            [X, info] = studentDiffusionPath(R_D, nu, n_obs, s, 1, stream, ...
                struct('substeps', 1, 'frozen', scheme == 2));
            y = L \ X([1, end], :).';
            radius_squared = sum(y .^ 2, 1);
            q_k(scheme, :) = radius_squared;
            x_k(scheme, :) = X([1, end], 1).';
            u_k(scheme, :) = y(1, :) .^ 2 ./ radius_squared;
            finite_k(scheme) = all(isfinite(X(end, :)));
            if scheme == 1
                accepted = info.acceptance;
            end
        end
        q(k, :, :) = reshape(q_k.', [1, 2, 2]);
        x1(k, :, :) = reshape(x_k.', [1, 2, 2]);
        u1_squared(k, :, :) = reshape(u_k.', [1, 2, 2]);
        finite(k, :) = finite_k;
        acceptance(k) = accepted;
    end
    results = struct('times_tau', [0, job.T_over_tau], 'schemes', {schemes}, ...
        'h', 1 / s, 'q', q, 'x1', x1, 'u1_squared', u1_squared, 'finite', finite, ...
        'acceptance', acceptance);
end


function results = localFitted(job, R, fit_opts, n_workers, varied, path_options)
    % Replicate paths at every level of the varied setting, fitted at every d.
    d_list = reshape(job.spec.d, 1, []);
    switch varied
        case 'refine'
            refine_levels = reshape(job.refine, 1, []);
            rate_levels = repmat(job.samples_per_tau, size(refine_levels));
        case 'samples_per_tau'
            rate_levels = reshape(job.samples_per_tau, 1, []);
            refine_levels = repmat(job.refine, size(rate_levels));
    end
    n_levels = numel(rate_levels);
    n_obs = arrayfun(@(s) localSteps(job.T_over_tau, s, 'window'), rate_levels);
    opts = tFitDefaultOptions(fit_opts);
    p_grid = linspace(opts.p_bounds(1), opts.p_bounds(2), opts.profile_grid_size);
    caches = arrayfun(@(n) tCopulaQuantileCache(n, p_grid), n_obs, 'UniformOutput', false);

    [nu, seeds, items] = deal(job.nu, job.seeds, job.items);
    n_d = numel(d_list);
    p_hat = nan(items, n_levels, n_d);
    numerical_ok = false(items, n_levels, n_d);
    heavy_boundary = false(items, n_levels, n_d);
    acceptance = nan(items, n_levels);
    repeated = nan(items, n_levels);
    substeps = nan(items, n_levels);
    h = nan(items, n_levels);
    sim_seconds = nan(items, n_levels);
    fit_seconds = nan(items, n_levels);
    parfor (k = 1:items, n_workers)
        estimate = nan(n_levels, n_d);
        ok = false(n_levels, n_d);
        heavy = false(n_levels, n_d);
        facts = nan(n_levels, 6);
        for a = 1:n_levels
            % One seed per replicate at every level: each level starts from the same exact draw, which pairs the levels' estimates.
            % The rates of the sampling part also share the substep, so they observe one path.
            stream = RandStream('twister', 'Seed', seeds(k));
            sim_timer = tic;
            [X, info] = studentDiffusionPath(R, nu, n_obs(a), rate_levels(a), ...
                refine_levels(a), stream, path_options);
            simulated = toc(sim_timer);
            fit_timer = tic;
            rows = simulationNestedFits(X, d_list, fit_opts, caches{a});
            estimate(a, :) = [rows.p_hat];
            ok(a, :) = [rows.numerical_ok];
            heavy(a, :) = [rows.heavy_boundary];
            facts(a, :) = [info.acceptance, info.repeated_observation_fraction, ...
                info.substeps, info.h, simulated, toc(fit_timer)];
        end
        p_hat(k, :, :) = estimate;
        numerical_ok(k, :, :) = ok;
        heavy_boundary(k, :, :) = heavy;
        acceptance(k, :) = facts(:, 1).';
        repeated(k, :) = facts(:, 2).';
        substeps(k, :) = facts(:, 3).';
        h(k, :) = facts(:, 4).';
        sim_seconds(k, :) = facts(:, 5).';
        fit_seconds(k, :) = facts(:, 6).';
    end
    results = struct('varied', varied, 'samples_per_tau', rate_levels, ...
        'refine', refine_levels, 'n_obs', n_obs, 'd', d_list, 'p_hat', p_hat, ...
        'numerical_ok', numerical_ok, 'heavy_boundary', heavy_boundary, ...
        'acceptance', acceptance, 'repeated_observation_fraction', repeated, ...
        'substeps', substeps, 'h', h, 'sim_seconds', sim_seconds, ...
        'fit_seconds', fit_seconds);
end


function results = localRelaxation(job, R, n_workers, path_options)
    % The first whitened coordinate at the declared times from one fixed start, whose mean over the paths estimates its conditional mean.
    L = chol(R, 'lower');
    D = size(R, 1);
    s = job.samples_per_tau;
    t_list = reshape(job.spec.t, 1, []);
    index = arrayfun(@(t) localSteps(t, s, 'relaxation time'), t_list) + 1;
    q_start = localRadiusQuantile(job.spec.start_quantile, D, job.nu);
    y0 = [sqrt(q_start); zeros(D - 1, 1)];
    refine_levels = reshape(job.refine, 1, []);
    n_levels = numel(refine_levels);
    n_obs = max(index);
    options = path_options;
    options.y0 = y0;

    [nu, seeds, items] = deal(job.nu, job.seeds, job.items);
    y1 = nan(items, n_levels, numel(t_list));
    acceptance = nan(items, n_levels);
    parfor (k = 1:items, n_workers)
        values = nan(n_levels, numel(t_list));
        accepted = nan(1, n_levels);
        for a = 1:n_levels
            stream = RandStream('twister', 'Seed', seeds(k));
            [X, info] = studentDiffusionPath(R, nu, n_obs, s, refine_levels(a), ...
                stream, options);
            Y = L \ X(index, :).';
            values(a, :) = Y(1, :);
            accepted(a) = info.acceptance;
        end
        y1(k, :, :) = values;
        acceptance(k, :) = accepted;
    end
    results = struct('t', t_list, 'refine', refine_levels, 'y1_start', y0(1), ...
        'start_quantile', job.spec.start_quantile, 'y1', y1, 'acceptance', acceptance);
end


function results = localTimescale(job, R, n_workers, path_options)
    % Long paths as separate segments of one matrix, read by the Phase 6 estimator at every declared lag window.
    % The score decides the reading, and the square and its Gaussian reference are read the same way and integrated over the whole window, so the model's amplitude excess, fitted and integrated, is measured exactly as the recordings' is.
    s = job.samples_per_tau;
    n_obs = localSteps(job.T_over_tau, s, 'window');
    windows_tau = reshape(job.spec.max_lag_tau, 1, []);
    max_lags = arrayfun(@(w) localSteps(w, s, 'lag window'), windows_tau);
    assert(max(max_lags) < n_obs, 'simulatorCheckJob:LagTooLong', ...
        'The longest lag window (%d samples) must be shorter than one path (%d samples).', ...
        max(max_lags), n_obs);

    [nu, refine, seeds, items] = deal(job.nu, job.refine, job.seeds, job.items);
    paths = cell(1, items);
    acceptance = nan(items, 1);
    parfor (k = 1:items, n_workers)
        stream = RandStream('twister', 'Seed', seeds(k));
        [X_path, info] = studentDiffusionPath(R, nu, n_obs, s, refine, stream, ...
            path_options);
        paths{k} = X_path;
        acceptance(k) = info.acceptance;
    end
    X = vertcat(paths{:});
    segment_ends = (1:items).' * n_obs;
    score = averagedRankAutocorrelation(X, segment_ends, max(max_lags), 'score');
    square = averagedRankAutocorrelation(X, segment_ends, max(max_lags), 'square');
    curves = struct('score', score.acf_mean, 'square', square.acf_mean, ...
        'square_gaussian', score.acf_square_gaussian);

    windows = struct('window_tau', {}, 'mode', {}, 'fit_score', {}, 'fit_square', {}, ...
        'fit_square_gaussian', {}, 'tau_int_square', {}, 'tau_int_square_gaussian', {});
    for w = 1:numel(windows_tau)
        span = 1:(max_lags(w) + 1);
        lags = score.lags(span);
        reading = envelopeReading(curves.score(span), job.spec.floor);
        windows(w) = struct('window_tau', windows_tau(w), 'mode', reading.mode, ...
            'fit_score', localEnvelopeFit(curves.score(span), lags, s, job.spec, reading), ...
            'fit_square', localEnvelopeFit(curves.square(span), lags, s, job.spec, reading), ...
            'fit_square_gaussian', localEnvelopeFit(curves.square_gaussian(span), lags, s, ...
            job.spec, reading), ...
            'tau_int_square', integratedAutocorrelationTime(curves.square(span), lags) / s, ...
            'tau_int_square_gaussian', ...
            integratedAutocorrelationTime(curves.square_gaussian(span), lags) / s);
    end
    results = struct('lags_tau', score.lags / s, 'acf_score', curves.score, ...
        'acf_square', curves.square, 'acf_square_gaussian', curves.square_gaussian, ...
        'acceptance', acceptance);
    results.windows = windows;
end


function fit = localEnvelopeFit(curve, lags, samples_per_tau, spec, reading)
    % The envelope time constant in tau units, read as READING says, with the Ornstein-Uhlenbeck-equivalent time beside it, as Phase 6 records them for the recordings.
    % A fit the estimator refuses is recorded with its reason, not rethrown, because this is one number of a report.
    fit = struct('tau', NaN, 'mode', reading.mode, 'last_lag_tau', NaN, 'n_points', NaN, ...
        'rmse', NaN, 'floor_reached', NaN, 'message', '', ...
        'tau_ou', ouEquivalentTimescale(curve, lags) / samples_per_tau);
    try
        out = fitEnvelopeTimescale(curve, lags, spec.floor, reading);
    catch fit_error
        % Only the estimator's own refusal is a recorded outcome; any other error is a bug and stops the job.
        if ~strcmp(fit_error.identifier, 'fitEnvelopeTimescale:BoundaryFit')
            rethrow(fit_error);
        end
        fit.message = sprintf('%s: %s', fit_error.identifier, fit_error.message);
        return;
    end
    assert(strcmp(out.method, spec.envelope_method), ...
        'simulatorCheckJob:EnvelopeMethodMismatch', ...
        ['fitEnvelopeTimescale implements the %s envelope but the check declares ' ...
        '%s; update settings.check.timescale.envelope_method.'], ...
        out.method, spec.envelope_method);
    fit.tau = out.tau / samples_per_tau;
    fit.last_lag_tau = out.last_lag / samples_per_tau;
    fit.n_points = out.n_points;
    fit.rmse = out.rmse;
    fit.floor_reached = double(out.floor_reached);
end


function q = localRadiusQuantile(probability, D, nu)
    % The quantile of |y|^2 under the stationary law: Q/(nu+Q) ~ Beta(D/2, nu/2), and Q ~ chi-square(D) in the Gaussian limit.
    if isinf(nu)
        q = 2 * gammaincinv(probability, D / 2);
    else
        b = betaincinv(probability, D / 2, nu / 2);
        q = nu * b / (1 - b);
    end
end


function n = localSteps(duration_tau, samples_per_tau, what)
    % A duration in tau as a whole number of observation intervals.
    n = round(duration_tau * samples_per_tau);
    assert(abs(n - duration_tau * samples_per_tau) < 1e-9 && n >= 1, ...
        'simulatorCheckJob:OffGrid', ...
        ['The %s %g tau is not a positive whole number of observation intervals at ' ...
        '%g samples per tau.'], what, duration_tau, samples_per_tau);
end
