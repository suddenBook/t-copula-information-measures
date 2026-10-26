function tests = simulatorCheckSummariesTest
    %SIMULATORCHECKSUMMARIESTEST The simulator check's reading rules, on jobs built to known answers.
    %
    % simulatorCheckSummaries turns the raw check jobs into the tables the check is read from, and the suite checks these reading rules:
    %   - the law KS distances against a Bonferroni level over the family of Metropolis rows, with the frozen negative control read against the same value;
    %   - one pooled refinement interval per (nu, f against 2f), with the pooled sd ratio and its seeded bootstrap;
    %   - the clock error of the exact relaxation and the discretisation bound that carries it to p_hat, whose standard error accounts for every time being read on the same paths;
    %   - the integrated amplitude excess beside the fitted one.
    % The jobs below are constructed so that each of those numbers is known, in closed form or from its definition applied directly to the constructed values.
    % The suite uses constructed inputs only and runs no simulation.
    tests = functiontests(localfunctions);
end


function setupOnce(testCase)
    here = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(fileparts(here), 'src')));
    testCase.TestData.reading = struct('tolerance_sd', 0.1, 'ks_family_alpha', 0.05, ...
        'bootstrap_reps', 500, 'bootstrap_seed', 20260927);
    testCase.TestData.c = 0.04;         % standardized f-to-2f shift, per step pair
    testCase.TestData.kappa = [0.034, 0.025, 0.018];   % clock error at f = 1, 2, 4
    testCase.TestData.drift = 0.3;      % mean p_hat gain from T/tau 300 to 3000
end


function testPooledIntervalOnReplicateNoise(testCase)
    % Replicate-level noise around a known standardized shift per window.
    % The pooled mean is the equal-weight mean of the windows' shifts whatever their replicate counts.
    % Its standard error takes the replicate, not the (replicate, d) cell, as the unit and adds the windows' errors in quadrature, and the interval covers the known shift.
    shifts = [0.02, 0.08];
    n = [200, 120];
    [jobs, z] = localNoisyRefinementJobs(shifts, n);
    reading = testCase.TestData.reading;
    pooled = simulatorCheckSummaries(jobs, reading).refinement_pooled;
    verifyEqual(testCase, [pooled.refine_coarse, pooled.refine_fine], [1 2; 2 4]);
    verifyEqual(testCase, pooled.n_windows, [2; 2]);
    verifyEqual(testCase, pooled.n_pairs_min, [120; 120]);
    verifyEqual(testCase, pooled.pooled_difference_over_sd, mean(shifts) * [1; 1], ...
        'the windows enter with equal weight', AbsTol=1e-12);
    for pair = 1:2
        replicate_se = cellfun(@(zw) std(mean(zw, 2)) / sqrt(size(zw, 1)), z(:, pair));
        expected = sqrt(sum(replicate_se .^ 2)) / numel(replicate_se);
        verifyEqual(testCase, pooled.pooled_se_over_sd(pair), expected, ...
            'z is averaged over d within the replicate, the replicate is the unit', ...
            RelTol=1e-10);
        cell_se = cellfun(@(zw) std(zw(:)) / sqrt(numel(zw)), z(:, pair));
        verifyGreaterThan(testCase, pooled.pooled_se_over_sd(pair), ...
            1.2 * sqrt(sum(cell_se .^ 2)) / numel(cell_se), ...
            'the d cells of one replicate share its path and are not independent');
        verifyGreaterThan(testCase, abs(mean(replicate_se) - expected), 0.1 * expected, ...
            'the data tell errors added in quadrature from a mean of the errors');
    end
    low = pooled.ci95_low_over_sd;
    high = pooled.ci95_high_over_sd;
    verifyEqual(testCase, [low, high], pooled.pooled_difference_over_sd + ...
        1.96 * pooled.pooled_se_over_sd .* [-1, 1], AbsTol=1e-12);
    verifyTrue(testCase, all(low < mean(shifts) & mean(shifts) < high), ...
        'the pooled interval covers the known shift');
    verifyEqual(testCase, pooled.interval_within_tolerance, [true; true]);

    reading.tolerance_sd = 0.05;
    narrow = simulatorCheckSummaries(jobs, reading).refinement_pooled;
    verifyEqual(testCase, narrow.tolerance_sd, [0.05; 0.05], 'the reading''s margin');
    verifyEqual(testCase, narrow.interval_within_tolerance, [false; false], ...
        'an interval reaching past 0.05 sd is not within +-0.05 sd');
end


function testPooledAveragesWithinTheReplicateOverSuccessfulFits(testCase)
    % With failed fits, a replicate's z averages over the d where both fits succeeded, the sd of each d is taken over those pairs, a replicate with no successful pair drops out, and n_pairs_min counts the smallest window that is left.
    jobs = localNoisyRefinementJobs([0.02, 0.08], [200, 120]);
    jobs(1).results.numerical_ok(1:5, 3, :) = false;      % 5 lose every finest fit
    jobs(1).results.numerical_ok(6:15, 2, 3) = false;     % 10 lose one middle fit
    jobs(2).results.numerical_ok(1:4, 1, :) = false;      % 4 lose every coarsest fit
    pooled = simulatorCheckSummaries(jobs, testCase.TestData.reading).refinement_pooled;
    verifyEqual(testCase, pooled.n_pairs_min, [116; 120]);
    for pair = 1:2
        [expected_mean, expected_se] = localDirectPooled(jobs, pair, pair + 1);
        verifyEqual(testCase, pooled.pooled_difference_over_sd(pair), expected_mean, ...
            AbsTol=1e-12);
        verifyEqual(testCase, pooled.pooled_se_over_sd(pair), expected_se, AbsTol=1e-12);
    end
end


function testSdRatioBootstrapIsSeededAndReproducible(testCase)
    % The pooled sd ratio is the geometric mean of the (window, d) ratios, not 1 when the steps differ in spread, with a percentile interval from a bootstrap that draws from its own seeded stream.
    % Two calls give identical tables whatever the global stream, and another seed gives another interval around the same estimate.
    jobs = localNoisyRefinementJobs([0.02, 0.08], [200, 120]);
    reading = testCase.TestData.reading;
    testCase.addTeardown(@rng, rng);
    rng(1);
    first = simulatorCheckSummaries(jobs, reading).refinement_pooled;
    rng(2);
    second = simulatorCheckSummaries(jobs, reading).refinement_pooled;
    verifyEqual(testCase, second, first, 'the seeded bootstrap reproduces');
    for pair = 1:2
        log_ratio = [];
        for w = 1:numel(jobs)
            p = jobs(w).results.p_hat;
            log_ratio = [log_ratio, log(std(squeeze(p(:, pair, :))) ./ ...
                std(squeeze(p(:, pair + 1, :))))]; %#ok<AGROW>
        end
        verifyEqual(testCase, first.sd_ratio_coarse_over_fine(pair), ...
            exp(mean(log_ratio)), RelTol=1e-12);
    end
    verifyGreaterThan(testCase, abs(first.sd_ratio_coarse_over_fine - 1), 0.01, ...
        'replicate noise between the steps moves the spread');
    verifyTrue(testCase, all(first.sd_ratio_ci95_low < first.sd_ratio_coarse_over_fine & ...
        first.sd_ratio_coarse_over_fine < first.sd_ratio_ci95_high));

    reading.bootstrap_seed = reading.bootstrap_seed + 1;
    other = simulatorCheckSummaries(jobs, reading).refinement_pooled;
    verifyNotEqual(testCase, other.sd_ratio_ci95_low, first.sd_ratio_ci95_low, ...
        'the bootstrap draws from the reading''s seed');
    verifyEqual(testCase, other.sd_ratio_coarse_over_fine, ...
        first.sd_ratio_coarse_over_fine, 'the estimate itself draws nothing');
end


function testRelaxationClockAndDiscretisationBound(testCase)
    % Constructed relaxation means give the clock error kappa_f at every time, and the bound shifts the mean p_hat by -kappa_f times its slope in log T/tau.
    kappa = testCase.TestData.kappa;
    jobs = [localRefinementJobs(testCase.TestData.c, testCase.TestData.drift), ...
        localRelaxationJob(kappa, 4)];
    tables = simulatorCheckSummaries(jobs, testCase.TestData.reading);

    relaxation = tables.relaxation;
    for f = 1:3
        rows = relaxation(relaxation.refine == 2 ^ (f - 1), :);
        verifyEqual(testCase, rows.clock_error, kappa(f) * ones(3, 1), ...
            'E[y(t)] = exp(-(1 - kappa) t) y0 gives clock error kappa at every t', ...
            AbsTol=1e-12);
    end

    bound = tables.discretisation;
    verifyEqual(testCase, height(bound), 3 * 2 * 2, 'one row per (refine, d, T/tau)');
    slope = testCase.TestData.drift / log(10);
    for f = 1:3
        rows = bound(bound.refine == 2 ^ (f - 1), :);
        verifyEqual(testCase, rows.slope_mean_p_per_log_T, slope * ones(4, 1), ...
            AbsTol=1e-12);
        verifyEqual(testCase, rows.mean_shift, -kappa(f) * slope * ones(4, 1), ...
            'a slow clock shortens the window: -kappa times the slope', AbsTol=1e-12);
        verifyEqual(testCase, rows.mean_shift_over_sd, ...
            abs(rows.mean_shift) ./ rows.sd_p_hat, AbsTol=1e-12);
    end
end


function testClockErrorSeCountsTheSharedPaths(testCase)
    % Every time of the relaxation part is read on the same paths, so the standard error of the mean clock error is the delta-method one with the full covariance of the times, sqrt(g' cov(Y) g / n) / n_t with g_t = 1 / (t mean y(t)).
    % It exceeds the root sum of squares of the per-time errors when the readings of a path are positively correlated, and at a single time it is that time's own error.
    refinement = localRefinementJobs(testCase.TestData.c, testCase.TestData.drift);
    job = localRelaxationJob(testCase.TestData.kappa, 400);
    tables = simulatorCheckSummaries([refinement, job], testCase.TestData.reading);
    res = job.results;
    for f = 1:3
        Y = reshape(res.y1(:, f, :), size(res.y1, 1), []);
        g = 1 ./ (res.t(:) .* mean(Y, 1).');
        expected = sqrt(g.' * cov(Y) * g / size(Y, 1)) / numel(res.t);
        rows = tables.discretisation(tables.discretisation.refine == 2 ^ (f - 1), :);
        verifyEqual(testCase, rows.clock_error_se, expected * ones(height(rows), 1), ...
            RelTol=1e-10);
        per_time = tables.relaxation.clock_error_se(tables.relaxation.refine == 2 ^ (f - 1));
        verifyGreaterThan(testCase, expected, ...
            1.2 * sqrt(sum(per_time .^ 2)) / numel(per_time), ...
            'correlated times: the root sum of the per-time errors is too small');
    end

    single = job;
    single.results.t = res.t(2);
    single.results.y1 = res.y1(:, :, 2);
    tables = simulatorCheckSummaries([refinement, single], testCase.TestData.reading);
    for f = 1:3
        rows = tables.discretisation(tables.discretisation.refine == 2 ^ (f - 1), :);
        verifyEqual(testCase, rows.clock_error_se, ...
            tables.relaxation.clock_error_se(f) * ones(height(rows), 1), ...
            'at a single time the bound''s error is that time''s own', RelTol=1e-10);
    end
end


function testNonPositiveMeanGivesNaNNotComplex(testCase)
    % A pilot of a few paths can see a mean y_1(t) at or below zero: its clock error is NaN, never the complex logarithm, and so is every bound that uses it, while the other refine factors keep their numbers.
    job = localRelaxationJob(testCase.TestData.kappa, 4);
    job.results.y1(:, 2, 3) = -0.05 + [0.01; -0.01; 0.02; -0.02];   % refine 2, t = 2
    tables = simulatorCheckSummaries([localRefinementJobs(testCase.TestData.c, ...
        testCase.TestData.drift), job], testCase.TestData.reading);

    relaxation = tables.relaxation;
    verifyTrue(testCase, isreal(relaxation.clock_error) && ...
        isreal(relaxation.clock_error_se), 'no complex column');
    bad = relaxation.refine == 2 & relaxation.t_tau == 2;
    verifyTrue(testCase, isnan(relaxation.clock_error(bad)) && ...
        isnan(relaxation.clock_error_se(bad)));
    verifyTrue(testCase, all(isfinite(relaxation.clock_error(~bad))));

    bound = tables.discretisation;
    verifyTrue(testCase, isreal(bound.clock_error) && isreal(bound.clock_error_se) && ...
        isreal(bound.mean_shift_over_sd), 'no complex column');
    at_bad = bound.refine == 2;
    verifyTrue(testCase, all(isnan(bound.clock_error(at_bad)) & ...
        isnan(bound.clock_error_se(at_bad)) & isnan(bound.mean_shift_over_sd(at_bad))));
    verifyTrue(testCase, all(isfinite(bound.mean_shift_over_sd(~at_bad))));
end


function testLawBonferroniFamilyAndFrozenControl(testCase)
    % A law job and a coarse job whose frozen scheme has a quarter of its paths diverged: the family, its Bonferroni value on every row, and the flags.
    n = 400;
    jobs = [localLawJob(n), localCoarseJob(n)];
    reading = testCase.TestData.reading;
    law = simulatorCheckSummaries(jobs, reading).law;
    verifyEqual(testCase, height(law), 2 + 4, ...
        'two times of the law job, two schemes x two times of the coarse job');
    metropolis = law.scheme == "metropolis";
    verifyEqual(testCase, nnz(metropolis), 4);
    verifyEqual(testCase, unique(law.family_size), 8, ...
        'the family is both KS distances of every Metropolis row');
    % The asymptotic Kolmogorov quantile at the family's level, solved from the full series, stands on every row, because the frozen control is outside the family but is read against the same value.
    % The one-term tail the code uses is within 3e-9 of it at this level.
    verifyEqual(testCase, law.ks_critical_bonferroni, ...
        localKolmogorovQuantile(0.05 / 8) / sqrt(n) * ones(6, 1), RelTol=1e-8);
    verifyEqual(testCase, law.radial_exceeds_bonferroni, ...
        law.radial_ks > law.ks_critical_bonferroni);
    verifyEqual(testCase, law.marginal_exceeds_bonferroni, ...
        law.marginal_ks > law.ks_critical_bonferroni);
    frozen_end = law.scheme == "frozen" & law.time_tau == 50;
    verifyEqual(testCase, law.nonfinite_fraction(frozen_end), 0.25, ...
        'a diverged frozen path is counted');
    verifyTrue(testCase, law.radial_exceeds_bonferroni(frozen_end), ...
        'the diverged frozen control exceeds the family''s value and is flagged');
    verifyFalse(testCase, any(law.radial_exceeds_bonferroni(metropolis) | ...
        law.marginal_exceeds_bonferroni(metropolis)), 'exact draws are not flagged');

    reading.ks_family_alpha = 0.01;
    strict = simulatorCheckSummaries(jobs, reading).law;
    verifyEqual(testCase, strict.ks_critical_bonferroni, ...
        localKolmogorovQuantile(0.01 / 8) / sqrt(n) * ones(6, 1), ...
        'the reading''s family-wise level', RelTol=1e-8);
end


function testTimescaleIntegratedExcess(testCase)
    % The integrated amplitude excess: a square curve equal to its Gaussian reference gives 1 at every window.
    % A slow component of height h over the whole window W adds 2 h W tau to the square's integrated time, so the excess is 1 + 2 h W / tau_ref_int, where the fitted excess need not move.
    % The table has one row per (nu, window), with the three integrated columns last.
    s = 10;
    windows = [2 5];
    slow_height = 0.05;
    lags = (0:max(windows) * s).';
    reference = exp(-2 * lags / s);
    gaussian_job = localTimescaleJob(Inf, s, windows, reference, reference);
    slow_job = localTimescaleJob(2, s, windows, reference + slow_height * (lags > 0), ...
        reference);
    timescale = simulatorCheckSummaries([gaussian_job, slow_job], ...
        testCase.TestData.reading).timescale;
    verifyEqual(testCase, height(timescale), 2 * numel(windows), 'one row per (nu, window)');
    names = timescale.Properties.VariableNames;
    verifyEqual(testCase, names(end - 2:end), {'tau_amp_int', 'tau_amp_ref_int', ...
        'amp_excess_int'});
    verifyEqual(testCase, timescale.window_tau, [windows, windows].');
    verifyEqual(testCase, timescale.amp_excess, 1.2 * ones(4, 1), ...
        'the fitted excess, tau_amp 0.6 over its reference 0.5', AbsTol=1e-12);

    gaussian = timescale(isinf(timescale.nu), :);
    verifyEqual(testCase, gaussian.amp_excess_int, [1; 1], AbsTol=1e-12);
    slow = timescale(timescale.nu == 2, :);
    q = exp(-2 / s);
    for w = 1:numel(windows)
        n_lags = windows(w) * s;
        tau_ref = (1 + 2 * q * (1 - q ^ n_lags) / (1 - q)) / s;
        verifyEqual(testCase, slow.tau_amp_ref_int(w), tau_ref, RelTol=1e-12);
        verifyEqual(testCase, slow.tau_amp_int(w), tau_ref + 2 * slow_height * windows(w), ...
            RelTol=1e-12);
        verifyEqual(testCase, slow.amp_excess_int(w), ...
            1 + 2 * slow_height * windows(w) / tau_ref, RelTol=1e-12);
    end
end


function testReadingSettingsAreRequired(testCase)
    verifyError(testCase, @() simulatorCheckSummaries(localRefinementJobs(0.04, 0.3), ...
        struct('tolerance_sd', 0.1)), 'simulatorCheckSummaries:MissingReading');
end


% ---------------------------------------------------------------- constructed jobs

function jobs = localRefinementJobs(c, drift)
    % nu = 2 at T/tau 300 and 3000, refine [1 2 4], d [2 5]: at every replicate the level-f estimate is the level-2f one minus c sd, so every f-to-2f standardized difference is exactly c.
    % The T/tau = 3000 estimates are the 300 ones plus drift.
    items = 40;
    base = linspace(0.2, 0.8, items).';
    jobs = struct([]);
    for T = [300 3000]
        p = zeros(items, 3, 2);
        for c_d = 1:2
            x = base * c_d + drift * (T == 3000);
            s = std(x);
            p(:, 3, c_d) = x;
            p(:, 2, c_d) = x - c * s;
            p(:, 1, c_d) = x - 2 * c * s;
        end
        results = localRefinementResults(p, T);
        jobs = [jobs, localJob('refinement', 2, T, 10, [1 2 4], items, results)]; %#ok<AGROW>
    end
end


function [jobs, z] = localNoisyRefinementJobs(shifts, n)
    % nu = 2, one job per window (T/tau 300 and 3000) with n(w) replicates, refine [1 2 4], d [2 5 10].
    % The standardized f-to-2f difference of replicate k at d is shifts(w) + u_k + v_kd, with a common replicate noise u (sd 0.3) and a d-specific one v (sd 0.1).
    % Both noises are antithetic over the two halves of the replicates, so each window's mean difference is exactly shifts(w) while its spread is not zero.
    % The levels are built from the finest down, p_f = p_2f - z sd(p_2f), so the summaries' standardization recovers z; z{w, pair} holds it, pair 1 = (1, 2), pair 2 = (2, 4).
    windows = [300 3000];
    n_d = 3;
    stream = RandStream('twister', 'Seed', 11);
    jobs = struct([]);
    z = cell(numel(windows), 2);
    for w = 1:numel(windows)
        half = n(w) / 2;
        p = zeros(n(w), 3, n_d);
        p(:, 3, :) = reshape(0.5 + 0.2 * randn(stream, n(w), 1) + ...
            0.05 * randn(stream, n(w), n_d), n(w), 1, n_d);
        for pair = 2:-1:1
            noise = 0.3 * randn(stream, half, 1) + 0.1 * randn(stream, half, n_d);
            z{w, pair} = shifts(w) + [noise; -noise];
            fine = reshape(p(:, pair + 1, :), n(w), n_d);
            p(:, pair, :) = reshape(fine - z{w, pair} .* std(fine), n(w), 1, n_d);
        end
        results = localRefinementResults(p, windows(w));
        results.d = [2 5 10];
        jobs = [jobs, localJob('refinement', 2, windows(w), 10, [1 2 4], n(w), ...
            results)]; %#ok<AGROW>
    end
end


function results = localRefinementResults(p, T)
    % The result struct of a refinement job around the p_hat array [items x 3 x n_d].
    [items, ~, n_d] = size(p);
    results = struct('varied', 'refine', 'samples_per_tau', [10 10 10], ...
        'refine', [1 2 4], 'n_obs', T * 10 * [1 1 1], 'd', [2 5], 'p_hat', p, ...
        'numerical_ok', true(items, 3, n_d), 'heavy_boundary', false(items, 3, n_d), ...
        'acceptance', ones(items, 3), 'repeated_observation_fraction', zeros(items, 3), ...
        'substeps', ones(items, 3), 'h', 0.01 * ones(items, 3), ...
        'sim_seconds', ones(items, 3), 'fit_seconds', ones(items, 3));
end


function [pooled, se] = localDirectPooled(jobs, a, b)
    % The pooled refinement reading from its definition: per window, z_k = the mean over the d where both fits succeeded of (p_b - p_a) / sd(p_b over those pairs), then the mean and standard error of the replicates with a finite z_k.
    % The windows' means are averaged with equal weight and their errors added in quadrature.
    means = zeros(1, numel(jobs));
    ses = zeros(1, numel(jobs));
    for w = 1:numel(jobs)
        res = jobs(w).results;
        [n, ~, n_d] = size(res.p_hat);
        z = nan(n, n_d);
        for c = 1:n_d
            both = res.numerical_ok(:, a, c) & res.numerical_ok(:, b, c);
            fine = res.p_hat(both, b, c);
            z(both, c) = (fine - res.p_hat(both, a, c)) / std(fine);
        end
        z_k = mean(z, 2, 'omitnan');
        z_k = z_k(~isnan(z_k));
        means(w) = mean(z_k);
        ses(w) = std(z_k) / sqrt(numel(z_k));
    end
    pooled = mean(means);
    se = sqrt(sum(ses .^ 2)) / numel(ses);
end


function job = localRelaxationJob(kappa, n)
    % nu = 2, one fixed start y0 = 2, refine [1 2 4], t = [0.5 1 2], n paths.
    % The level-f value at time t is exp(-(1 - kappa_f) t) y0 plus a noise that one path shares across its times (a common term per path, as the real readings share their path).
    % The noise is antithetic over the two halves of the paths, so the mean is exactly the centre and the clock error exactly kappa_f.
    t = [0.5 1 2];
    y0 = 2;
    stream = RandStream('twister', 'Seed', 7);
    half = n / 2;
    y1 = zeros(n, 3, 3);
    for f = 1:3
        common = randn(stream, half, 1);
        for k = 1:3
            centre = exp(-(1 - kappa(f)) * t(k)) * y0;
            noise = 0.3 * centre * (common + 0.3 * randn(stream, half, 1));
            y1(:, f, k) = centre + [noise; -noise];
        end
    end
    results = struct('t', t, 'refine', [1 2 4], 'y1_start', y0, 'start_quantile', 0.5, ...
        'y1', y1, 'acceptance', ones(n, 3));
    job = localJob('relaxation', 2, 2, 10, [1 2 4], n, results);
end


function job = localTimescaleJob(nu, s, windows, square, reference)
    % One time-scale job whose windows carry what simulatorCheckJob records: the fitted envelope times (hand-set: tau_env 1, tau_amp 0.6 against a reference of 0.5) and the integrated times of the square and of its reference over each whole window, in tau, as integratedAutocorrelationTime gives them on the curves passed.
    lags = (0:numel(square) - 1).';
    fit = @(tau) struct('tau', tau, 'mode', 'monotone', 'last_lag_tau', 1, ...
        'n_points', 10, 'rmse', 0.01, 'floor_reached', 1, 'message', '', 'tau_ou', tau);
    rows = struct('window_tau', {}, 'mode', {}, 'fit_score', {}, 'fit_square', {}, ...
        'fit_square_gaussian', {}, 'tau_int_square', {}, 'tau_int_square_gaussian', {});
    for w = 1:numel(windows)
        span = 1:(windows(w) * s + 1);
        rows(w) = struct('window_tau', windows(w), 'mode', 'monotone', ...
            'fit_score', fit(1), 'fit_square', fit(0.6), 'fit_square_gaussian', fit(0.5), ...
            'tau_int_square', integratedAutocorrelationTime(square(span), lags(span)) / s, ...
            'tau_int_square_gaussian', ...
            integratedAutocorrelationTime(reference(span), lags(span)) / s);
    end
    results = struct('lags_tau', lags / s, 'acf_score', reference, 'acf_square', square, ...
        'acf_square_gaussian', reference, 'acceptance', ones(20, 1));
    results.windows = rows;
    job = localJob('timescale', nu, 1000, s, 2, 20, results);
    job.spec = struct('floor', 0.05);
end


function job = localLawJob(n)
    % One nu at D = 3, values drawn from the right laws with a private stream.
    stream = RandStream('twister', 'Seed', 3);
    nu = 4;
    D = 3;
    results = localLawResults(stream, nu, D, n);
    job = localJob('law', nu, 50, 10, 2, n, results);
    job.D = D;
end


function job = localCoarseJob(n)
    % The coarse part at D = 2: the Metropolis scheme from the right law, the frozen scheme with a quarter of its paths diverged at the horizon.
    stream = RandStream('twister', 'Seed', 4);
    nu = 1 / 3;
    D = 2;
    first = localLawResults(stream, nu, D, n);
    second = localLawResults(stream, nu, D, n);
    diverged = (1:n).' <= n / 4;
    second.q(diverged, 2) = Inf;
    second.x1(diverged, 2) = NaN;
    second.u1_squared(diverged, 2) = NaN;
    results = struct('times_tau', [0 50], 'schemes', {{'metropolis', 'frozen'}}, 'h', 0.5, ...
        'q', cat(3, first.q, second.q), 'x1', cat(3, first.x1, second.x1), ...
        'u1_squared', cat(3, first.u1_squared, second.u1_squared), ...
        'finite', [true(n, 1), ~diverged], 'acceptance', 0.9 * ones(n, 1));
    job = localJob('coarse', nu, 50, 2, 1, n, results);
    job.D = D;
end


function results = localLawResults(stream, nu, D, n)
    % Exact Student draws at both times: q = |y|^2, x1 = y_1, u1 = y_1^2 / |y|^2.
    q = zeros(n, 2);
    x1 = zeros(n, 2);
    u1 = zeros(n, 2);
    for t = 1:2
        g = randn(stream, n, D);
        V = 2 * gammaincinv(rand(stream, n, 1), nu / 2);
        y = g .* sqrt(nu ./ V);
        q(:, t) = sum(y .^ 2, 2);
        x1(:, t) = y(:, 1);
        u1(:, t) = y(:, 1) .^ 2 ./ q(:, t);
    end
    results = struct('times_tau', [0 50], 'q', q, 'x1', x1, 'u1_squared', u1, ...
        'acceptance', ones(n, 1), 'substeps', ones(n, 1), 'h', 0.1 * ones(n, 1));
end


function x = localKolmogorovQuantile(level)
    % The upper LEVEL quantile of the Kolmogorov distribution, from its full series P(sup |B| > x) = 2 sum_j (-1)^(j - 1) exp(-2 j^2 x^2).
    j = (1:100).';
    tail = @(x) 2 * sum((-1) .^ (j - 1) .* exp(-2 * j .^ 2 * x ^ 2)) - level;
    x = fzero(tail, [0.5, 4]);
end


function job = localJob(part, nu, T, samples_per_tau, refine, items, results)
    % A check job around RESULTS, with the fields simulatorCheckJob returns.
    job = struct('part', part, 'job_index', 1, 'label', part, 'nu', nu, 'D', 20, ...
        'T_over_tau', T, 'samples_per_tau', samples_per_tau, 'refine', refine, ...
        'step_samples_per_tau', 80, 'items', items, 'seeds', 1:items, 'spec', struct(), ...
        'wall_seconds', 0);
    job.results = results;
end
