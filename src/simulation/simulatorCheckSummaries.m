function tables = simulatorCheckSummaries(check_jobs, reading)
    % SIMULATORCHECKSUMMARIES The simulator check's tables, from the raw check jobs.
    %
    % tables = simulatorCheckSummaries(check_jobs, reading) turns the outputs of simulatorCheckJob into the tables Phase 7 publishes as text/check_*.csv.
    % The tables report: no column is a gate, and nothing downstream reads one as a decision.
    % The reading rules of the check are columns of these tables, and their settings come from READING and are applied here when the tables are built, so they change no path and no checkpoint.
    % The tables:
    %
    %   law                     one row per (part, scheme, nu, D, time).
    %                           It holds the radial beta-PIT of Q/(nu+Q) ~ Beta(D/2, nu/2) (chi-square(D) at nu = Inf) and its Kolmogorov-Smirnov distance, beside the asymptotic 95% and 99% critical values and the Bonferroni critical value at the family-wise level ks_family_alpha.
    %                           The family is every Metropolis KS distance of the table (radial and marginal, both times, the law and the coarse parts).
    %                           The row also holds the KS distance of the first channel's t_nu PIT, the direction moment E[U_1^2] against 1/D, the mean acceptance, the fraction of non-finite radii and the step.
    %                           Rows of the coarse part come in two schemes, 'metropolis' and 'frozen'.
    %                           The frozen negative control is outside the family but is read against the same critical value and flagged the same way: its time-0 rows are the same exact starts as the Metropolis rows, and at the horizon its wrong radial law is expected to exceed.
    %                           Each distance takes one value per independent path, so the one-time KS calibration is valid.
    %                           A non-finite radius (a diverging frozen chain) counts as a radial PIT of 1, and a NaN first channel as a marginal PIT of 1.
    %   law_pit_histogram       the radial PIT in 20 equal bins per law row, for a figure.
    %   refinement              one row per (nu, T/tau, refine, d): p_hat mean, sd and quantiles and the fractions at the Gaussian bound and at the heavy end, over the numerically successful replicates, with the medians of the paths' acceptance, repeated-observation fraction, substeps, h and seconds.
    %   refinement_equivalence  one row per (nu, T/tau, d, f against 2f): the difference of the p_hat means over the paired replicates in units of the finer factor's sd, with its paired standard error and 95% interval.
    %   refinement_pooled       the refinement reading, one row per (nu, f against 2f).
    %                           It holds the per-replicate standardized difference averaged over d, its mean pooled over the windows T/tau with its 95% interval against +-tolerance_sd, and the pooled ratio of the p_hat sds (coarse over fine) with a paired bootstrap interval of bootstrap_reps resamples drawn from bootstrap_seed, since a slow clock shows first in the spread.
    %   relaxation              one row per (nu, refine, t): the relative error E[y_1(t)] / (exp(-t) y_1(0)) - 1, its standard error, and the clock error it implies, kappa = log(1 + relative error) / t (the chain's time runs 1 - kappa as fast as the SDE's).
    %                           The clock error and its standard error are NaN where 1 + relative error <= 0, a mean at or below zero that only a pilot of a few paths can reach.
    %   discretisation          the discretisation bound, one row per (nu, refine, d, T/tau) where relaxation and refinement meet.
    %                           The clock error kappa of the relaxation part (the mean over its times t, with a standard error taken across the paths because every time is read on the same paths) is carried to p_hat through the slope of its mean in log T/tau over the refinement windows, as the mean shift -kappa * slope in units of p_hat's sd at that window.
    %                           To first order a chain whose time runs 1 - kappa as fast observes the SDE over (1 - kappa) T/tau at s / (1 - kappa) samples per tau, so the mean moves by -kappa dm/dln T + kappa dm/dln s.
    %                           The bound keeps the first term and omits the second, so it assumes the mean of p_hat is saturated in the samples per tau at the check's rate.
    %                           A step comparison f against 2f cannot bound the error of a scheme that converges like sqrt(h), but the exact conditional mean can.
    %   timescale               one row per (nu, lag window): the reading mode the score decides, tau_env, tau_amp and the Gaussian reference tau_amp_ref in tau (the Phase 6 estimator on simulated paths), the amplitude excess tau_amp / tau_amp_ref, the fit diagnostics, the Ornstein-Uhlenbeck-equivalent times and the model's closed-form radial time.
    %                           The integrated amplitude times tau_amp_int and tau_amp_ref_int follow, the integrated autocorrelation times of the square and of its reference over the whole window, with their ratio amp_excess_int.
    %                           That ratio is 1 for a Gaussian path at any window, and it counts a slow component's size and duration together, which the fitted excess does not.
    %   sampling                as refinement, per (nu, T/tau, samples per tau, d).
    %
    % Inputs:
    %   check_jobs - struct array of simulatorCheckJob outputs (any subset of parts).
    %   reading    - the reading settings (Phase 7's settings.check_reading), with fields
    %                  tolerance_sd     the refinement equivalence margin, in sd of p_hat
    %                  ks_family_alpha  the family-wise level of the law KS distances
    %                  bootstrap_reps   the paired bootstrap resamples of the pooled sd ratio
    %                  bootstrap_seed   the seed of that bootstrap's own stream, so the tables reproduce
    %
    % Output:
    %   tables - struct with the nine tables above; a part with no job gives table().

    arguments
        check_jobs struct
        reading (1, 1) struct
    end
    localAssertReading(reading);

    if isempty(check_jobs)
        parts = {};
    else
        parts = {check_jobs.part};
    end
    law_jobs = check_jobs(ismember(parts, {'law', 'coarse'}));
    refinement_jobs = check_jobs(strcmp(parts, 'refinement'));
    relaxation_jobs = check_jobs(strcmp(parts, 'relaxation'));
    [law, histogram] = localLaw(law_jobs, reading.ks_family_alpha);
    refinement = localFitted(refinement_jobs, 'refine');
    relaxation = localRelaxation(relaxation_jobs);
    tables = struct( ...
                    'law', law, ...
                    'law_pit_histogram', histogram, ...
                    'refinement', refinement, ...
                    'refinement_equivalence', localEquivalence(refinement_jobs, ...
                                                   reading.tolerance_sd), ...
                    'refinement_pooled', localPooled(refinement_jobs, reading), ...
                    'relaxation', relaxation, ...
                    'discretisation', localDiscretisation(relaxation, refinement, ...
                                              relaxation_jobs), ...
                    'timescale', localTimescale(check_jobs(strcmp(parts, 'timescale'))), ...
                    'sampling', localFitted(check_jobs(strcmp(parts, 'sampling')), ...
                                'samples_per_tau'));
end

function localAssertReading(reading)
    % Checks that the four reading settings the tables are built with are present and valid for the rules that use them.
    required = {'tolerance_sd', 'ks_family_alpha', 'bootstrap_reps', 'bootstrap_seed'};
    missing = required(~isfield(reading, required));
    assert(isempty(missing), 'simulatorCheckSummaries:MissingReading', ...
           ['The reading settings have no %s; Phase 7 declares all four in ' ...
            'settings.check_reading.'], strjoin(missing, ', '));
    validateattributes(reading.tolerance_sd, {'numeric'}, ...
                       {'scalar', 'positive', 'finite'}, mfilename, 'reading.tolerance_sd');
    validateattributes(reading.ks_family_alpha, {'numeric'}, ...
                       {'scalar', 'positive', '<', 1}, mfilename, 'reading.ks_family_alpha');
    validateattributes(reading.bootstrap_reps, {'numeric'}, ...
                       {'scalar', 'integer', 'positive'}, mfilename, 'reading.bootstrap_reps');
    validateattributes(reading.bootstrap_seed, {'numeric'}, ...
                       {'scalar', 'integer', 'nonnegative', '<', 2^32}, mfilename, ...
                       'reading.bootstrap_seed');
end

function [summary, histogram] = localLaw(jobs, alpha)
    % The one-time law rows of the law and coarse parts, then on every row the Bonferroni critical value of the family of every Metropolis KS distance of the table, at the family-wise level ALPHA.
    % The frozen control is outside the family and is read against the same value.
    summary = table();
    histogram = table();
    if isempty(jobs)
        return
    end
    rows = {};
    bins = {};
    edges = linspace(0, 1, 21);
    for j = 1:numel(jobs)
        job = jobs(j);
        res = job.results;
        if strcmp(job.part, 'coarse')
            schemes = res.schemes;
        else
            schemes = {'metropolis'};
        end
        for c = 1:numel(schemes)
            for t = 1:2
                if strcmp(job.part, 'coarse')
                    q = res.q(:, t, c);
                    x1 = res.x1(:, t, c);
                    u1 = res.u1_squared(:, t, c);
                    acceptance = res.acceptance;
                    if c == 2
                        acceptance = ones(size(acceptance));
                    end
                    substeps = 1;
                    h = res.h;
                else
                    q = res.q(:, t);
                    x1 = res.x1(:, t);
                    u1 = res.u1_squared(:, t);
                    acceptance = res.acceptance;
                    substeps = median(res.substeps);
                    h = median(res.h);
                end
                n = numel(q);
                radial = localRadialPit(q, job.nu, job.D);
                marginal = localMarginalPit(x1, job.nu);
                rows{end + 1} = table(string(job.part), string(schemes{c}), job.nu, ...
                                      job.D, res.times_tau(t), n, job.samples_per_tau, job.refine(1), h, ...
                                      substeps, mean(radial), localKs(radial), 1.3581 / sqrt(n), ...
                                      1.6276 / sqrt(n), localKs(marginal), localMean(u1), ...
                                      localSd(u1) / sqrt(n), 1 / job.D, mean(acceptance), ...
                                      mean(~isfinite(q)), ...
                                      'VariableNames', {'part', 'scheme', 'nu', 'D', 'time_tau', 'n_paths', ...
                                      'samples_per_tau', 'refine', 'h', 'substeps', 'radial_pit_mean', ...
                                      'radial_ks', 'ks_critical_95', 'ks_critical_99', 'marginal_ks', ...
                                      'direction_u1sq_mean', 'direction_u1sq_se', ...
                                      'direction_u1sq_target', 'acceptance_mean', ...
                                      'nonfinite_fraction'}); %#ok<AGROW>
                counts = histcounts(radial, edges).';
                bins{end + 1} = table(repmat(string(job.part), 20, 1), ...
                                      repmat(string(schemes{c}), 20, 1), repmat(job.nu, 20, 1), ...
                                      repmat(job.D, 20, 1), repmat(res.times_tau(t), 20, 1), ...
                                      edges(1:end - 1).', edges(2:end).', counts, repmat(n / 20, 20, 1), ...
                                      'VariableNames', {'part', 'scheme', 'nu', 'D', 'time_tau', ...
                                      'bin_lower', 'bin_upper', 'count', 'expected'}); %#ok<AGROW>
            end
        end
    end
    summary = vertcat(rows{:});
    histogram = vertcat(bins{:});

    % The family: every KS distance the corrected chain produced (two per row).
    % Its critical value solves the one-term asymptotic Kolmogorov tail 2 exp(-2 c^2) = level for c, scaled by 1/sqrt(n).
    in_family = summary.scheme == "metropolis";
    family = 2 * nnz(in_family);
    level = alpha / family;
    summary.family_size = repmat(family, height(summary), 1);
    summary.ks_critical_bonferroni = sqrt(-0.5 * log(level / 2)) ./ sqrt(summary.n_paths);
    summary.radial_exceeds_bonferroni = summary.radial_ks > summary.ks_critical_bonferroni;
    summary.marginal_exceeds_bonferroni = ...
        summary.marginal_ks > summary.ks_critical_bonferroni;
end

function summary = localFitted(jobs, varied)
    % One row per (job, level, d) over the numerically successful replicates.
    summary = table();
    if isempty(jobs)
        return
    end
    rows = {};
    for j = 1:numel(jobs)
        job = jobs(j);
        res = job.results;
        for a = 1:numel(res.samples_per_tau)
            for c = 1:numel(res.d)
                ok = res.numerical_ok(:, a, c);
                p = res.p_hat(ok, a, c);
                q = localQuantile(p, [0.05, 0.25, 0.5, 0.75, 0.95]);
                rows{end + 1} = table(job.nu, job.T_over_tau, ...
                                      res.samples_per_tau(a), res.n_obs(a), res.refine(a), res.d(c), ...
                                      numel(ok), nnz(ok), localMean(p), localSd(p), q(1), q(2), q(3), ...
                                      q(4), q(5), localMean(double(p == 0)), ...
                                      localMean(double(res.heavy_boundary(ok, a, c))), ...
                                      median(res.acceptance(:, a), 'omitnan'), ...
                                      median(res.repeated_observation_fraction(:, a), 'omitnan'), ...
                                      median(res.substeps(:, a), 'omitnan'), median(res.h(:, a), 'omitnan'), ...
                                      median(res.sim_seconds(:, a), 'omitnan'), ...
                                      median(res.fit_seconds(:, a), 'omitnan'), ...
                                      'VariableNames', {'nu', 'T_over_tau', 'samples_per_tau', 'n', ...
                                      'refine', 'd', 'reps', 'n_ok', 'mean', 'sd', 'q05', 'q25', ...
                                      'median', 'q75', 'q95', 'fraction_p0', 'fraction_heavy', ...
                                      'acceptance_median', 'repeated_median', 'substeps', 'h', ...
                                      'sim_seconds_median', 'fit_seconds_median'}); %#ok<AGROW>
            end
        end
    end
    summary = vertcat(rows{:});
    summary = addvars(summary, repmat(string(varied), height(summary), 1), ...
                      'Before', 1, 'NewVariableNames', 'varied');
end

function summary = localEquivalence(jobs, tolerance_sd)
    % f against 2f, paired on the replicate: each pair shares its seed and exact start.
    summary = table();
    if isempty(jobs)
        return
    end
    rows = {};
    for j = 1:numel(jobs)
        job = jobs(j);
        res = job.results;
        for a = 1:numel(res.refine)
            b = find(res.refine == 2 * res.refine(a), 1);
            if isempty(b)
                continue
            end
            for c = 1:numel(res.d)
                both = res.numerical_ok(:, a, c) & res.numerical_ok(:, b, c);
                coarse = res.p_hat(both, a, c);
                fine = res.p_hat(both, b, c);
                n = nnz(both);
                sd_fine = localSd(fine);
                difference = localMean(fine) - localMean(coarse);
                paired_se = localSd(fine - coarse) / sqrt(n);
                ratio = difference / sd_fine;
                se_ratio = paired_se / sd_fine;
                low = ratio - 1.96 * se_ratio;
                high = ratio + 1.96 * se_ratio;
                rows{end + 1} = table(job.nu, job.T_over_tau, res.d(c), ...
                                      res.refine(a), res.refine(b), n, localMean(coarse), ...
                                      localMean(fine), difference, sd_fine, ratio, se_ratio, low, high, ...
                                      tolerance_sd, abs(ratio) <= tolerance_sd, ...
                                      low >= -tolerance_sd && high <= tolerance_sd, ...
                                      'VariableNames', {'nu', 'T_over_tau', 'd', 'refine_coarse', ...
                                      'refine_fine', 'n_pairs', 'mean_coarse', 'mean_fine', ...
                                      'difference', 'sd_fine', 'difference_over_sd', ...
                                      'paired_se_over_sd', 'ci95_low_over_sd', 'ci95_high_over_sd', ...
                                      'tolerance_sd', 'within_tolerance', ...
                                      'interval_within_tolerance'}); %#ok<AGROW>
            end
        end
    end
    if ~isempty(rows)
        summary = vertcat(rows{:});
    end
end

function summary = localPooled(jobs, reading)
    % One interval per (nu, f against 2f), which is how the refinement is read.
    % The d cells of one replicate share its path, so they are averaged within the replicate (z_k = mean over d of (p_2f - p_f) / sd_2f(d), over the d where both fits succeeded).
    % The replicates of one window are independent, and so are the windows, which enter with equal weight.
    % The ratio of the sds, coarse over fine, is pooled as the mean of its log over the (window, d) cells, with a percentile interval from reading.bootstrap_reps paired bootstrap resamples of the replicates of each window.
    % The resamples are drawn from a stream seeded with reading.bootstrap_seed, so the table reproduces.
    summary = table();
    if isempty(jobs)
        return
    end
    tolerance_sd = reading.tolerance_sd;
    reps = reading.bootstrap_reps;
    rows = {};
    for nu = unique([jobs.nu], 'stable')
        at_nu = jobs([jobs.nu] == nu);
        levels = at_nu(1).results.refine;
        for a = 1:numel(levels)
            b = find(levels == 2 * levels(a), 1);
            if isempty(b)
                continue
            end
            means = nan(1, numel(at_nu));
            ses = nan(1, numel(at_nu));
            n_pairs = zeros(1, numel(at_nu));
            log_ratio = [];
            boot = zeros(reps, 0);
            stream = RandStream('twister', 'Seed', reading.bootstrap_seed);
            for j = 1:numel(at_nu)
                res = at_nu(j).results;
                [n_items, ~, n_d] = size(res.p_hat);
                both = reshape(res.numerical_ok(:, a, :) & res.numerical_ok(:, b, :), ...
                               n_items, n_d);
                coarse = reshape(res.p_hat(:, a, :), n_items, n_d);
                fine = reshape(res.p_hat(:, b, :), n_items, n_d);
                sd_fine = arrayfun(@(c) localSd(fine(both(:, c), c)), 1:size(fine, 2));
                standardized = (fine - coarse) ./ sd_fine;
                standardized(~both) = NaN;
                z = mean(standardized, 2, 'omitnan');
                z = z(isfinite(z));
                n_pairs(j) = numel(z);
                means(j) = localMean(z);
                ses(j) = localSd(z) / sqrt(numel(z));
                [cell_log_ratio, cell_boot] = localLogSdRatio(coarse, fine, both, ...
                                                              stream, reps);
                log_ratio = [log_ratio, cell_log_ratio]; %#ok<AGROW>
                boot = [boot, cell_boot]; %#ok<AGROW>
            end
            pooled = mean(means);
            se = sqrt(sum(ses.^2)) / numel(ses);
            low = pooled - 1.96 * se;
            high = pooled + 1.96 * se;
            ratio_boot = exp(mean(boot, 2, 'omitnan'));
            rows{end + 1} = table(nu, levels(a), levels(b), numel(at_nu), ...
                                  size(at_nu(1).results.p_hat, 3), min(n_pairs), pooled, se, low, high, ...
                                  tolerance_sd, low >= -tolerance_sd && high <= tolerance_sd, ...
                                  exp(mean(log_ratio, 'omitnan')), prctile(ratio_boot, 2.5), ...
                                  prctile(ratio_boot, 97.5), ...
                                  'VariableNames', {'nu', 'refine_coarse', 'refine_fine', 'n_windows', ...
                                  'n_d', 'n_pairs_min', 'pooled_difference_over_sd', ...
                                  'pooled_se_over_sd', 'ci95_low_over_sd', 'ci95_high_over_sd', ...
                                  'tolerance_sd', 'interval_within_tolerance', ...
                                  'sd_ratio_coarse_over_fine', 'sd_ratio_ci95_low', ...
                                  'sd_ratio_ci95_high'}); %#ok<AGROW>
        end
    end
    if ~isempty(rows)
        summary = vertcat(rows{:});
    end
end

function [log_ratio, boot] = localLogSdRatio(coarse, fine, both, stream, reps)
    % log(sd coarse / sd fine) per d over the paired successful replicates, and its REPS paired bootstrap replicates (the same resampled replicates for every d).
    [n, n_d] = size(coarse);
    log_ratio = nan(1, n_d);
    boot = nan(reps, n_d);
    draws = randi(stream, n, n, reps);
    for c = 1:n_d
        keep = both(:, c);
        log_ratio(c) = log(localSd(coarse(keep, c)) / localSd(fine(keep, c)));
        for r = 1:reps
            pick = draws(:, r);
            pick = pick(keep(pick));
            boot(r, c) = log(localSd(coarse(pick, c)) / localSd(fine(pick, c)));
        end
    end
end

function summary = localRelaxation(jobs)
    % One row per (nu, refine, t): the relative error of the mean first whitened coordinate against exp(-t) y_1(0), and the clock error it implies.
    summary = table();
    if isempty(jobs)
        return
    end
    rows = {};
    for j = 1:numel(jobs)
        job = jobs(j);
        res = job.results;
        for a = 1:numel(res.refine)
            for t = 1:numel(res.t)
                values = res.y1(:, a, t);
                n = numel(values);
                expected = exp(-res.t(t)) * res.y1_start;
                relative = mean(values) / expected - 1;
                relative_se = std(values) / sqrt(n) / expected;
                % A mean at or below zero has no clock error: NaN, not the complex logarithm, which would turn the whole column complex.
                growth = 1 + relative;
                clock_error = NaN;
                clock_error_se = NaN;
                if growth > 0
                    clock_error = log(growth) / res.t(t);
                    clock_error_se = relative_se / (growth * res.t(t));
                end
                rows{end + 1} = table(job.nu, job.D, res.refine(a), res.t(t), n, ...
                                      res.start_quantile, res.y1_start, mean(values), ...
                                      std(values) / sqrt(n), expected, relative, relative_se, ...
                                      clock_error, clock_error_se, mean(res.acceptance(:, a)), ...
                                      'VariableNames', {'nu', 'D', 'refine', 't_tau', 'n_paths', ...
                                      'start_quantile', 'y1_start', 'y1_mean', 'y1_se', ...
                                      'y1_expected', 'relative_error', 'relative_error_se', ...
                                      'clock_error', 'clock_error_se', 'acceptance_mean'}); %#ok<AGROW>
            end
        end
    end
    summary = vertcat(rows{:});
end

function summary = localDiscretisation(relaxation, refinement, relaxation_jobs)
    % The clock error of the relaxation part carried to p_hat.
    % A chain whose time runs 1 - kappa as fast as the SDE's observes a window of (1 - kappa) T/tau, so its mean p_hat sits at the mean of a window that much shorter: a shift of -kappa times the slope of the mean in log T/tau, read off the refinement part's windows at the same refine factor.
    % The same chain also observes that window at s / (1 - kappa) samples per tau, which moves the mean by +kappa dm/dln s.
    % That term is omitted, so the bound assumes the mean of p_hat is saturated in the samples per tau at the check's rate.
    % kappa is the mean of the clock errors over the times t, NaN when one of them is, and a (nu, refine) pair with fewer than two refinement windows gives no row.
    summary = table();
    if isempty(relaxation) || isempty(refinement)
        return
    end
    rows = {};
    for nu = reshape(intersect(unique(relaxation.nu), unique(refinement.nu)), 1, [])
        for f = reshape(intersect(unique(relaxation.refine(relaxation.nu == nu)), ...
                                  unique(refinement.refine(refinement.nu == nu))), 1, [])
            clock = relaxation(relaxation.nu == nu & relaxation.refine == f, :);
            kappa = mean(clock.clock_error);
            kappa_se = NaN;
            if isfinite(kappa)
                kappa_se = localClockErrorSe(relaxation_jobs, nu, f);
            end
            block = refinement(refinement.nu == nu & refinement.refine == f, :);
            windows = unique(block.T_over_tau);
            if numel(windows) < 2
                continue
            end
            for d = reshape(unique(block.d), 1, [])
                at_d = block(block.d == d, :);
                short = at_d(at_d.T_over_tau == min(windows), :);
                long = at_d(at_d.T_over_tau == max(windows), :);
                slope = (long.mean - short.mean) / log(max(windows) / min(windows));
                for T = reshape(windows, 1, [])
                    sd = at_d.sd(at_d.T_over_tau == T);
                    shift = -kappa * slope;
                    rows{end + 1} = table(nu, f, d, T, kappa, kappa_se, slope, sd, ...
                                          shift, abs(shift) / sd, ...
                                          'VariableNames', {'nu', 'refine', 'd', 'T_over_tau', ...
                                          'clock_error', 'clock_error_se', 'slope_mean_p_per_log_T', ...
                                          'sd_p_hat', 'mean_shift', 'mean_shift_over_sd'}); %#ok<AGROW>
                end
            end
        end
    end
    if ~isempty(rows)
        summary = vertcat(rows{:});
    end
end

function se = localClockErrorSe(jobs, nu, f)
    % The standard error of kappa = (1/n_t) sum_t log(mean y_1(t) / (exp(-t) y_1(0))) / t.
    % Every time t is read on the same paths, so the per-time errors are correlated and their root sum of squares understates the mean's error.
    % The delta method gives it instead as the standard error across paths of each path's own combination (1/n_t) sum_t y_1(t) / (t mean y_1(t)), which at a single time is the relaxation table's clock_error_se.
    job = jobs([jobs.nu] == nu);
    assert(isscalar(job), 'simulatorCheckSummaries:RelaxationNotUnique', ...
           ['The discretisation bound reads one relaxation job per nu; found %d at ' ...
            'nu = %g.'], numel(job), nu);
    res = job.results;
    level = find(res.refine == f, 1);
    Y = reshape(res.y1(:, level, :), size(res.y1, 1), []);
    per_path = mean(Y ./ (reshape(res.t, 1, []) .* mean(Y, 1)), 2);
    se = std(per_path) / sqrt(size(Y, 1));
end

function summary = localTimescale(jobs)
    % One row per (nu, lag window): the model's time scales as the Phase 6 estimator reads them, with the closed-form radial time beside them.
    summary = table();
    if isempty(jobs)
        return
    end
    rows = {};
    for j = 1:numel(jobs)
        job = jobs(j);
        res = job.results;
        for w = 1:numel(res.windows)
            window = res.windows(w);
            env = window.fit_score;
            amp = window.fit_square;
            ref = window.fit_square_gaussian;
            rows{end + 1} = table(job.nu, job.D, job.T_over_tau, job.items, ...
                                  job.samples_per_tau, job.refine, window.window_tau, job.spec.floor, ...
                                  string(window.mode), env.tau, amp.tau, ref.tau, amp.tau / ref.tau, ...
                                  amp.tau / env.tau, env.floor_reached, amp.floor_reached, ...
                                  ref.floor_reached, env.last_lag_tau, amp.last_lag_tau, ...
                                  ref.last_lag_tau, env.rmse, amp.rmse, ref.rmse, string(env.message), ...
                                  string(amp.message), string(ref.message), env.tau_ou, amp.tau_ou, ...
                                  ref.tau_ou, amp.tau_ou / env.tau_ou, ...
                                  diffusionRadialTime(job.nu, job.D), mean(res.acceptance), ...
                                  window.tau_int_square, window.tau_int_square_gaussian, ...
                                  window.tau_int_square / window.tau_int_square_gaussian, ...
                                  'VariableNames', {'nu', 'D', 'T_over_tau', 'n_paths', ...
                                  'samples_per_tau', 'refine', 'window_tau', 'floor', 'reading_mode', ...
                                  'tau_env', 'tau_amp', 'tau_amp_ref', 'amp_excess', 'amp_over_env', ...
                                  'env_floor_reached', 'amp_floor_reached', 'amp_ref_floor_reached', ...
                                  'env_last_lag_tau', 'amp_last_lag_tau', 'amp_ref_last_lag_tau', ...
                                  'env_rmse', 'amp_rmse', 'amp_ref_rmse', 'env_message', ...
                                  'amp_message', 'amp_ref_message', 'tau_env_ou', 'tau_amp_ou', ...
                                  'tau_amp_ref_ou', 'amp_over_env_ou', 'radial_time_tau', ...
                                  'acceptance_mean', 'tau_amp_int', 'tau_amp_ref_int', ...
                                  'amp_excess_int'}); %#ok<AGROW>
        end
    end
    summary = vertcat(rows{:});
end

function u = localRadialPit(q, nu, D)
    % P(Q/(nu+Q) <= b) under Beta(D/2, nu/2), from the stable complement nu/(nu+Q) = (sqrt(nu)/hypot(sqrt(nu), sqrt(Q)))^2 and its Beta(nu/2, D/2) upper tail, so a huge Q never forms 1 - Q/(nu+Q) by subtraction; chi-square(D) at Inf.
    % A non-finite Q (a diverged frozen chain) is beyond every quantile: PIT 1.
    if isinf(nu)
        u = gammainc(q / 2, D / 2);
    else
        w = (sqrt(nu) ./ hypot(sqrt(nu), sqrt(q))).^2;
        u = betainc(w, nu / 2, D / 2, 'upper');
    end
    u(~isfinite(q)) = 1;
end

function u = localMarginalPit(x, nu)
    % The PIT of the first channel, which is standard t_nu because R has a unit diagonal.
    % A NaN state (a diverged frozen chain) counts as beyond every quantile: PIT 1.
    if isinf(nu)
        u = normcdf(x);
    else
        u = tcdf(x, nu);
    end
    u(isnan(x)) = 1;
end

function ks = localKs(u)
    % Kolmogorov-Smirnov distance of a sample from the uniform law.
    u = sort(u(:));
    n = numel(u);
    ks = max(max((1:n).' / n - u), max(u - (0:n - 1).' / n));
end

function value = localMean(x)
    % The mean of the finite entries, NaN when there are none.
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function value = localSd(x)
    % The standard deviation of the finite entries, NaN when there are fewer than two.
    x = x(isfinite(x));
    if numel(x) < 2
        value = NaN;
    else
        value = std(x);
    end
end

function q = localQuantile(x, probabilities)
    % The quantiles of the finite entries, NaN when there are none.
    x = x(isfinite(x));
    if isempty(x)
        q = nan(size(probabilities));
    else
        q = quantile(x, probabilities);
    end
end
