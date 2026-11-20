function tests = simulationJobOrderTest
    % SIMULATIONJOBORDERTEST Jobs, cells and seeds of the diffusion study are the declared ones.
    %
    % Phase 7 simulates one parent path per replicate and fits every dimension on its leading channels, and every rate observes the same path on the common substep.
    % A job is therefore a (p_true, T/tau, samples per tau) triple or a design, a cell is a job's output at one d, and a seed belongs to a (p_true, T/tau) replicate and not to a d or a rate.
    % The suite checks that the numbers of the recovery, factorial and simulator-check jobs are mixed-radix indices over the declared grid, that a pilot subset runs the production jobs under their production numbers and draws the production paths, that no two distinct paths of the study share a seed, and that a rate off the substep grid is refused.
    % It also checks that the nested fits the jobs use equal tFitChannelSet bit for bit, and that the spread order of the factor loadings spans every leading block.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(here);
    addpath(genpath(fullfile(repo, 'src')));

    shared = struct('parent_dim', 20, 'refine', 1, 'step_samples_per_tau', 80);
    recovery = struct( ...
                      'p_true',          [0 0.02 0.05 0.1 0.2 0.5 1 2 3], ...
                      'T_over_tau',      [10 30 100 300 1000 3000 1e4 3e4], ...
                      'samples_per_tau', [5 10 20 40 80], ...
                      'd',               2:20, ...
                      'reps',            1000);
    recovery.r_spec = struct('name', 'equi_0.40', 'type', 'equi', 'rho', 0.40);
    testCase.TestData.recovery = localMerge(recovery, shared);

    factorial = struct('p_true', 0.1, 'T_over_tau', 3000, 'samples_per_tau', 10, ...
                       'd', 2:20, 'reps', 1000);
    factorial.designs = [ ...
                         struct('name', 'equi_0.10', 'type', 'equi', 'rho', 0.10), ...
                         struct('name', 'equi_0.40', 'type', 'equi', 'rho', 0.40), ...
                         struct('name', 'equi_0.75', 'type', 'equi', 'rho', 0.75), ...
                         struct('name', 'equi_0.97', 'type', 'equi', 'rho', 0.97), ...
                         struct('name', 'factor_hetero', 'type', 'equi', 'rho', 0.5), ...
                         struct('name', 'block_four', 'type', 'equi', 'rho', 0.6)];
    testCase.TestData.factorial = localMerge(factorial, shared);

    testCase.TestData.check = struct( ...
                                     'samples_per_tau', 10, 'refine', 1, 'step_samples_per_tau', 80, ...
                                     'law', struct('nu', [1 / 3 1 2 4 10 Inf], 'paths', 20000, 'horizon_tau', 50), ...
                                     'refinement', struct('nu', [1 / 3 2 10], 'T_over_tau', [300 3000], ...
                             'reps', 200, 'refine', [1 2 4], 'd', [2 5 10 20]), ...
                                     'relaxation', struct('nu', [1 / 3 2 10], 'paths', 20000, 't', [0.5 1 2], ...
                             'refine', [1 2 4], 'start_quantile', 0.5), ...
                                     'timescale', struct('nu', [1 / 3 1 2 4 10 Inf], 'paths', 20, ...
                            'T_over_tau', 1000, 'max_lag_tau', [2 5 10 20 50 100], 'floor', 0.05), ...
                                     'sampling', struct('nu', [2 5 Inf], 'T_over_tau', [300 3000], 'reps', 200, ...
                           'samples_per_tau', [5 10 20 40 80], 'd', [2 5 10 20]), ...
                                     'coarse', struct('nu', 1 / 3, 'D', [2 20], 'paths', 20000, 'horizon_tau', 50, ...
                         'h_tau', 0.5));
end

function testRecoveryJobsAndCellsAreTheDeclaredMixedRadix(testCase)
    declared = testCase.TestData.recovery;
    jobs = simulationJobs("recovery", declared);
    verifyEqual(testCase, numel(jobs), 9 * 8 * 5);
    verifyEqual(testCase, [jobs.job_index], 1:360);

    % p_true is the slowest job axis, then T/tau, and the rate the fastest.
    verifyEqual(testCase, [jobs(1:5).samples_per_tau], declared.samples_per_tau);
    verifyEqual(testCase, [jobs(1:5).T_over_tau], 10 * ones(1, 5));
    verifyEqual(testCase, [jobs(6:10).T_over_tau], 30 * ones(1, 5));
    verifyEqual(testCase, [jobs(1:40).p_true], zeros(1, 40));
    verifyEqual(testCase, [jobs(41:80).p_true], 0.02 * ones(1, 40));
    verifyEqual(testCase, [jobs.n_obs], [jobs.T_over_tau] .* [jobs.samples_per_tau]);
    verifyEqual(testCase, unique([jobs.step_samples_per_tau]), 80);

    % Every cell (p_true, d, T/tau, rate) of [9 19 8 5] belongs to exactly one job.
    cells = [jobs.cell_index];
    verifyEqual(testCase, sort(cells), 1:(9 * 19 * 8 * 5));
    ip = find(declared.p_true == 0.2);
    iT = find(declared.T_over_tau == 1000);
    is = find(declared.samples_per_tau == 20);
    job = jobs([jobs.job_index] == ((ip - 1) * 8 + (iT - 1)) * 5 + is);
    verifyEqual(testCase, [job.p_true, job.T_over_tau, job.samples_per_tau], ...
                [0.2, 1000, 20]);
    verifyEqual(testCase, job.n_obs, 20000);
    verifyEqual(testCase, job.d, 2:20);
    verifyEqual(testCase, job.cell_index, ...
                (((ip - 1) * 19 + (0:18)) * 8 + (iT - 1)) * 5 + is);
    expected = arrayfun(@(rep) studySeed("recovery", [ip, iT, rep], [9, 8, 1000]), ...
                        [1, 2, 1000]);
    verifyEqual(testCase, job.seeds([1, 2, 1000]), expected, ...
                'the seed radices are (p_true, T/tau, rep): no rate subscript');
end

function testEveryRateObservesTheSamePaths(testCase)
    % The rates of one (p_true, T/tau) draw the same replicate paths: the seed has no rate subscript, and the common substep makes the coarse path a subsample.
    declared = testCase.TestData.recovery;
    jobs = simulationJobs("recovery", declared, struct('p_true', 1, 'T_over_tau', 100));
    verifyEqual(testCase, numel(jobs), 5);
    verifyEqual(testCase, [jobs.samples_per_tau], [5 10 20 40 80]);
    for k = 2:5
        verifyEqual(testCase, jobs(k).seeds, jobs(1).seeds);
    end
    verifyEqual(testCase, numel(unique([jobs.job_index])), 5);
    verifyError(testCase, @() simulationJobs("recovery", ...
                                             localMerge(declared, struct('samples_per_tau', [5 7]))), ...
                'simulationJobs:RateNotNested');
end

function testPilotSubsetKeepsJobCellNumbersAndSeeds(testCase)
    declared = testCase.TestData.recovery;
    full = simulationJobs("recovery", declared);
    pilot = simulationJobs("recovery", declared, struct( ...
                                                        'p_true', [0 0.1 3], 'T_over_tau', [10 300], 'samples_per_tau', [5 80], ...
                                                        'd', [2 5 20], 'reps', 4));

    verifyEqual(testCase, numel(pilot), 3 * 2 * 2);
    for k = 1:numel(pilot)
        match = full([full.job_index] == pilot(k).job_index);
        verifyEqual(testCase, numel(match), 1, ...
                    'a pilot job number must name exactly one production job');
        verifyEqual(testCase, [pilot(k).p_true, pilot(k).T_over_tau, ...
                               pilot(k).samples_per_tau, pilot(k).n_obs], ...
                    [match.p_true, match.T_over_tau, match.samples_per_tau, match.n_obs]);
        verifyEqual(testCase, pilot(k).sizes, match.sizes, ...
                    'the seed radices must stay the DECLARED grid');
        verifyEqual(testCase, pilot(k).d, [2 5 20]);
        verifyEqual(testCase, pilot(k).cell_index, ...
                    match.cell_index(ismember(match.d, [2 5 20])), ...
                    'a pilot cell keeps its production number');
        verifyEqual(testCase, pilot(k).reps, 4);
        verifyEqual(testCase, pilot(k).seeds, match.seeds(1:4), ...
                    'a pilot draws the production paths');
    end
end

function testPathSeedDoesNotDependOnDimension(testCase)
    % One replicate is one parent path whatever dimensions are fitted on it.
    declared = testCase.TestData.recovery;
    low = simulationJobs("recovery", declared, struct('p_true', 1, 'T_over_tau', 100, ...
                                                      'samples_per_tau', 10, 'd', 2, 'reps', 5));
    high = simulationJobs("recovery", declared, struct('p_true', 1, 'T_over_tau', 100, ...
                                                       'samples_per_tau', 10, 'd', [7 20], 'reps', 5));
    verifyEqual(testCase, low.job_index, high.job_index);
    verifyEqual(testCase, low.seeds, high.seeds);
    verifyNotEqual(testCase, low.cell_index, high.cell_index(1));
end

function testFactorialJobsAreDesigns(testCase)
    declared = testCase.TestData.factorial;
    jobs = simulationJobs("factorial", declared);
    verifyEqual(testCase, numel(jobs), 6);
    verifyEqual(testCase, [jobs.job_index], 1:6);
    verifyEqual(testCase, sort([jobs.cell_index]), 1:(6 * 19));
    verifyEqual(testCase, jobs(3).design, "equi_0.75");
    verifyEqual(testCase, jobs(3).cell_index, 2 * 19 + (1:19));
    verifyEqual(testCase, jobs(3).seeds(1:3), ...
                arrayfun(@(rep) studySeed("factorial", [3, rep], [6, 1000]), 1:3));

    subset = simulationJobs("factorial", declared, ...
                            struct('designs', {{'block_four'}}, 'd', [2 20], 'reps', 2));
    verifyEqual(testCase, [subset.job_index], 6);
    verifyEqual(testCase, subset.cell_index, jobs(6).cell_index([1 19]));
    verifyEqual(testCase, subset.seeds, jobs(6).seeds(1:2));
end

function testCheckJobsKeepNumbersAndSeeds(testCase)
    declared = testCase.TestData.check;
    full = simulatorCheckJobs(declared);
    verifyEqual(testCase, numel(full), 6 + 6 + 3 + 6 + 6 + 2);
    verifyEqual(testCase, [full.job_index], 1:29);
    verifyEqual(testCase, {full([1 7 13 16 22 28]).part}, ...
                {'law', 'refinement', 'relaxation', 'timescale', 'sampling', 'coarse'});
    verifyEqual(testCase, [full(7:12).T_over_tau], repmat([300 3000], 1, 3), ...
                'refinement jobs run T/tau fastest');
    verifyEqual(testCase, [full(22:27).nu], [2 2 5 5 Inf Inf], ...
                'sampling jobs run nu slowest, T/tau fastest');
    verifyEqual(testCase, full(22).samples_per_tau, [5 10 20 40 80]);
    verifyEqual(testCase, [full(28:29).D], [2 20], 'the coarse part runs per D');
    verifyEqual(testCase, [full(28:29).samples_per_tau], [2 2], ...
                'the coarse part observes every forced step of h = 0.5 tau');
    verifyEqual(testCase, unique([full.step_samples_per_tau]), 80);
    verifyError(testCase, @() simulatorCheckJobs(localMerge(declared, struct( ...
                                                                             'sampling', struct('nu', 5, 'T_over_tau', 300, 'reps', 2, ...
                           'samples_per_tau', [5 7], 'd', 2)))), 'simulatorCheckJobs:RateNotNested');

    pilot = simulatorCheckJobs(declared, struct('parts', {{'refinement', 'law'}}, ...
                                                'paths', 100, 'reps', 3));
    verifyEqual(testCase, [pilot.job_index], 1:12);
    for k = 1:numel(pilot)
        match = full(pilot(k).job_index);
        verifyEqual(testCase, pilot(k).part, match.part);
        verifyEqual(testCase, pilot(k).seeds, match.seeds(1:pilot(k).items));
    end
    verifyEqual(testCase, pilot(1).items, 100);
    verifyEqual(testCase, pilot(7).items, 3);
    verifyError(testCase, @() simulatorCheckJobs(declared, ...
                                                 struct('parts', {{'laws'}})), 'simulatorCheckJobs:UnknownPart');
end

function testSeedsAreUniqueAndBelowTheStreamLimit(testCase)
    % No two paths of the study share a seed, across jobs, arms and the check.
    recovery = simulationJobs("recovery", testCase.TestData.recovery, ...
                              struct('samples_per_tau', 5));   % one job per path: the rates share them
    factorial = simulationJobs("factorial", testCase.TestData.factorial);
    check = simulatorCheckJobs(testCase.TestData.check);
    all_seeds = [[recovery.seeds], [factorial.seeds], [check.seeds]];
    verifyEqual(testCase, numel(unique(all_seeds)), numel(all_seeds), ...
                'two paths of the study must never share a seed');
    verifyLessThan(testCase, max(all_seeds), 2^32);
    verifyEqual(testCase, all_seeds, round(all_seeds));
end

function testUndeclaredValueIsAnError(testCase)
    verifyError(testCase, @() simulationJobs("recovery", ...
                                             testCase.TestData.recovery, struct('T_over_tau', 50)), ...
                'simulationJobs:ValueNotDeclared');
    verifyError(testCase, @() simulationJobs("recovery", ...
                                             testCase.TestData.recovery, struct('d', 21)), ...
                'simulationJobs:ValueNotDeclared');
    verifyError(testCase, @() simulationJobs("factorial", ...
                                             testCase.TestData.factorial, struct('designs', {{'block_six'}})), ...
                'simulationJobs:DesignNotDeclared');
end

function testNestedFitsEqualTheChannelSetFit(testCase)
    % The jobs fit channel sets 1:d from one shared rank matrix and one shared Kendall matrix; that must be exactly tFitChannelSet on X(:, 1:d), row for row.
    R = 0.6 * eye(6) + 0.4 * ones(6);
    X = studentDiffusionPath(R, 4, 600, 10, 1, RandStream('twister', 'Seed', 91));
    fit_opts = struct('store_profile', false);
    opts = tFitDefaultOptions(fit_opts);
    cache = tCopulaQuantileCache(600, ...
                                 linspace(opts.p_bounds(1), opts.p_bounds(2), opts.profile_grid_size));
    d_list = [2 3 6];
    [rows, R_hat] = simulationNestedFits(X, d_list, fit_opts, cache);
    verifyEqual(testCase, numel(rows), 3);
    for c = 1:numel(d_list)
        d = d_list(c);
        [expected, ~, expected_R] = tFitChannelSet(X(:, 1:d), 1:d, fit_opts, cache);
        row = rows(c);
        verifyGreaterThan(testCase, row.fit_seconds, 0);
        row.fit_seconds = NaN;          % wall clock, the one field that may differ
        expected.fit_seconds = NaN;
        verifyEqual(testCase, row, expected, sprintf( ...
                                                     'the nested route must equal tFitChannelSet at d = %d', d), AbsTol = 0);
        verifyEqual(testCase, R_hat{c}, expected_R, AbsTol = 0);
    end

    % The jobs prepare the ranks and the Kendall matrix first and release the path before the fits, which must give the same rows bit for bit, and a preparation for other dimensions is refused.
    prepared = simulationNestedPreparation(X, d_list);
    [prepared_rows, prepared_R] = simulationNestedFits(prepared, d_list, fit_opts, cache);
    for c = 1:numel(d_list)
        row = prepared_rows(c);
        expected = rows(c);
        row.fit_seconds = NaN;
        expected.fit_seconds = NaN;
        verifyEqual(testCase, row, expected, sprintf( ...
                                                     'the prepared route must equal the path route at d = %d', d_list(c)), AbsTol = 0);
        verifyEqual(testCase, prepared_R{c}, R_hat{c}, AbsTol = 0);
    end
    verifyError(testCase, @() simulationNestedFits(prepared, [2 3], fit_opts, cache), ...
                'simulationNestedFits:NotPreparedForTheseDimensions');
end

function testSpreadFactorLoadingsSpanEveryLeadingBlock(testCase)
    % The factorial fits the leading d x d block of one 20-channel parent, so the heterogeneous factor design orders its loadings ends first and then in van der Corput order.
    % The 20 loadings stay the same, and every leading block of two or more channels spans the whole range, whereas in ascending order the first five loadings span only 0.25-0.38.
    spec = struct('name', 'factor_hetero', 'type', 'factor', 'loading_range', ...
                  [0.25 0.85], 'loading_order', "spread");
    R = simulationRMatrix(spec, 20);
    loadings = zeros(1, 20);
    for i = 1:20
        j = mod(i, 20) + 1;
        k = mod(i + 1, 20) + 1;
        loadings(i) = sqrt(R(i, j) * R(i, k) / R(j, k));   % R_ij = l_i l_j
    end
    [~, order] = ismember(round(loadings, 10), round(linspace(0.25, 0.85, 20), 10));
    verifyEqual(testCase, order, [1 20 11 6 15 3 13 8 18 2 12 7 16 5 14 9 19 4 17 10]);
    for d = 2:20
        verifyEqual(testCase, max(loadings(1:d)) - min(loadings(1:d)), 0.6, ...
                    sprintf('the leading %d loadings must span [0.25, 0.85]', d), AbsTol = 1e-12);
    end
    ascending = simulationRMatrix(rmfield(spec, 'loading_order'), 20);
    verifyEqual(testCase, sort(eig(R)), sort(eig(ascending)), ...
                'a reorder of the channels keeps the spectrum', AbsTol = 1e-12);
    spec.loading_order = "";
    verifyEqual(testCase, simulationRMatrix(spec, 20), ascending, ...
                'empty text is the ascending default', AbsTol = 0);
end

function merged = localMerge(arm, shared)
    merged = arm;
    for name = reshape(fieldnames(shared), 1, [])
        merged.(name{1}) = shared.(name{1});
    end
end
