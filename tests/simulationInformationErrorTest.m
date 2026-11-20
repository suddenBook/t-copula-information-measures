function tests = simulationInformationErrorTest
    % SIMULATIONINFORMATIONERRORTEST The correction, the full measure, and the split.
    %
    % Two quantities come out of the same identity and are not interchangeable: the correction in nats, Delta(d, p), which is the study's estimand and depends on the fitted shape alone, and the full measure M_G(R) + Delta(d, p), whose error the p-only, R-only and interaction columns attribute.
    % The suite checks that the two stay distinct, that the attribution is exact (the parts add up to the whole error for every measure and every dimension), and that an estimate equal to the truth shows no error at all.
    % It also checks that simulationSummaries reads each floor from its own (d, T/tau, samples per tau) cell and each spread law from its own (d, samples per tau).
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(here);
    addpath(genpath(fullfile(repo, 'src')));
    testCase.TestData.measures = {'MI', 'TC', 'DTC', 'O'};
end

function testComponentsSumToTheWholeError(testCase)
    % err = err_p_only + err_R_only + err_interaction exactly, err = full_hat - full_true, and delta_err = err_p_only, at dimensions across the study's range.
    for d = [2 3 7 20]
        R_true = localEqui(d, 0.4);
        R_hat = localEqui(d, 0.37);
        err = simulationInformationError(d, 0.1, 0.17, ...
                                         gaussianInformationMeasures(R_true), ...
                                         gaussianInformationMeasures(R_hat));

        for k = 1:numel(testCase.TestData.measures)
            name = testCase.TestData.measures{k};
            whole = err.([name '_err']);
            parts = err.([name '_err_p_only']) + err.([name '_err_R_only']) ...
                + err.([name '_err_interaction']);
            if d == 2 || ~strcmp(name, 'MI')
                verifyEqual(testCase, parts, whole, 'AbsTol', 1e-15, ...
                            sprintf('%s at d = %d does not decompose', name, d));
                verifyEqual(testCase, whole, ...
                            err.([name '_full_hat']) - err.([name '_full_true']), ...
                            'AbsTol', 1e-15);
                verifyEqual(testCase, err.([name '_delta_err']), ...
                            err.([name '_err_p_only']), 'AbsTol', 0, ...
                            'the shape part of the full error IS the correction error');
                verifyTrue(testCase, isfinite(whole));
            else
                % MI is defined at d = 2 only; every MI column is NaN above it.
                verifyTrue(testCase, isnan(whole));
            end
        end
    end
end

function testInteractionIsIdenticallyZero(testCase)
    % M(p, R) = M_Gaussian(R) + Delta(d, p) leaves no cross term, so the remainder is zero whatever the two errors are.
    % A nonzero value would mean the additive identity the attribution rests on is broken.
    R_true = localEqui(8, 0.65);
    for rho_hat = [0.2 0.65 0.9]
        for p_hat = [0 0.05 3]
            err = simulationInformationError(8, 0.2, p_hat, ...
                                             gaussianInformationMeasures(R_true), ...
                                             gaussianInformationMeasures(localEqui(8, rho_hat)));
            for name = {'TC', 'DTC', 'O'}
                verifyEqual(testCase, err.([name{1} '_err_interaction']), 0, ...
                            'AbsTol', 1e-14);
            end
        end
    end
end

function testExactEstimateHasZeroError(testCase)
    % p_hat = p_true and R_hat = R_true: every error column is exactly zero and every full measure equals the truth.
    for d = [2 5 20]
        R = localEqui(d, 0.4);
        G = gaussianInformationMeasures(R);
        err = simulationInformationError(d, 0.2, 0.2, G, G);
        for k = 1:numel(testCase.TestData.measures)
            name = testCase.TestData.measures{k};
            if d > 2 && strcmp(name, 'MI')
                continue
            end
            verifyEqual(testCase, err.([name '_err']), 0, 'AbsTol', 0);
            verifyEqual(testCase, err.([name '_err_p_only']), 0, 'AbsTol', 0);
            verifyEqual(testCase, err.([name '_err_R_only']), 0, 'AbsTol', 0);
            verifyEqual(testCase, err.([name '_delta_err']), 0, 'AbsTol', 0);
            verifyEqual(testCase, err.([name '_full_hat']), ...
                        err.([name '_full_true']), 'AbsTol', 0);
            verifyEqual(testCase, err.([name '_delta_hat']), ...
                        err.([name '_delta_true']), 'AbsTol', 0);
        end
    end
end

function testGaussianTruthErrorIsTheShapeTermPlusTheRError(testCase)
    % At p_true = 0 the truth is the Gaussian value, so the whole error is the shape term at p_hat plus the R estimation error.
    d = 6;
    R_true = localEqui(d, 0.4);
    R_hat = localEqui(d, 0.42);
    G_true = gaussianInformationMeasures(R_true);
    G_hat = gaussianInformationMeasures(R_hat);
    err = simulationInformationError(d, 0, 0.08, G_true, G_hat);
    shape = tInformationDifferences(d, 0.08);

    for name = {'TC', 'DTC', 'O'}
        expected = shape.(name{1}) + (G_hat.(name{1}) - G_true.(name{1}));
        verifyEqual(testCase, err.([name{1} '_err']), expected, 'AbsTol', 1e-14);
        verifyEqual(testCase, err.([name{1} '_full_true']), ...
                    G_true.(name{1}), 'AbsTol', 1e-14);
        verifyEqual(testCase, err.([name{1} '_delta_true']), 0, 'AbsTol', 0, ...
                    'the truth''s own correction is exactly zero at p_true = 0');
        verifyEqual(testCase, err.([name{1} '_delta_hat']), shape.(name{1}), ...
                    'AbsTol', 0);
    end
end

function testFailedFitPropagatesNaN(testCase)
    % A replicate whose fit failed carries no shape estimate, so NaN travels through the columns that use it rather than a fabricated Gaussian substitute, and the truth's columns stay finite.
    G = gaussianInformationMeasures(localEqui(4, 0.3));
    err = simulationInformationError(4, 0.1, NaN, G, G);
    verifyTrue(testCase, isnan(err.TC_err));
    verifyTrue(testCase, isnan(err.TC_full_hat));
    verifyTrue(testCase, isfinite(err.TC_full_true));
    verifyTrue(testCase, isnan(err.TC_delta_hat));
    verifyTrue(testCase, isnan(err.TC_delta_err));
    verifyTrue(testCase, isfinite(err.TC_delta_true));
end

function testCorrectionIsNotTheFullMeasure(testCase)
    % A Gaussian truth fitted as Gaussian, with a wrong R.
    % The correction the estimator reports is then exactly zero, Delta(2, 0) = 0, and the whole error of the full measure is the error of the Gaussian correlation term.
    % Reporting the full measure under the name of the correction would make a perfect shape estimate look like a 0.136 nat departure.
    G_true = gaussianInformationMeasures([1 0.4; 0.4 1]);
    G_hat = gaussianInformationMeasures([1 0.6; 0.6 1]);
    err = simulationInformationError(2, 0, 0, G_true, G_hat);

    verifyEqual(testCase, err.MI_delta_true, 0, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_delta_hat, 0, 'AbsTol', 0, ...
                'the fitted correction at p_hat = 0 is exactly zero');
    verifyEqual(testCase, err.MI_delta_err, 0, 'AbsTol', 0);

    verifyEqual(testCase, err.MI_full_true, G_true.MI, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_full_hat, G_hat.MI, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_err, 0.13596685774182082, 'AbsTol', 1e-15, ...
                'the full-measure error is the Gaussian correlation error alone');
    verifyEqual(testCase, err.MI_err, G_hat.MI - G_true.MI, 'AbsTol', 0);

    % The p-only, R-only and interaction components still sum exactly to the full-measure error.
    verifyEqual(testCase, err.MI_err_p_only + err.MI_err_R_only ...
                + err.MI_err_interaction, err.MI_err, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_err_p_only, 0, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_err_R_only, err.MI_err, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_err_interaction, 0, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_full_p_only, err.MI_full_true, 'AbsTol', 0);
    verifyEqual(testCase, err.MI_full_R_only, err.MI_full_hat, 'AbsTol', 0);
end

function testGaussianFloorIsTheFittedCorrection(testCase)
    % The same replicate through simulationSummaries: the floor a real recording is read against is the correction the estimator fits on Gaussian data, which here is exactly zero.
    % The full-measure error travels beside it, in its own columns, and must not be what the floor reports.
    G_true = gaussianInformationMeasures([1 0.4; 0.4 1]);
    G_hat = gaussianInformationMeasures([1 0.6; 0.6 1]);
    err = simulationInformationError(2, 0, 0, G_true, G_hat);

    replicate = struct2table(err);
    replicate.cell = 1;
    replicate.p_true = 0;
    replicate.d = 2;
    replicate.T_over_tau = 10;
    replicate.samples_per_tau = 10;
    replicate.n = 100;
    replicate.refine = 1;
    replicate.h = 0.1;
    replicate.parent_dim = 20;
    replicate.acceptance = 1;
    replicate.rep = 1;
    replicate.p_hat = 0;
    replicate.numerical_ok = true;
    replicate.heavy_boundary = false;
    tables = simulationSummaries(replicate);

    floors = tables.floor_by_d_T_s;
    mi = floors(floors.measure == "MI", :);
    verifyEqual(testCase, height(mi), 1);
    verifyEqual(testCase, mi.q95, 0, 'AbsTol', 0, ...
                'a perfectly recovered shape has a zero floor');
    verifyEqual(testCase, mi.q99, 0, 'AbsTol', 0);
    verifyEqual(testCase, mi.q95_full_err, err.MI_err, 'AbsTol', 1e-15, ...
                'the full-measure error keeps its own floor columns');
    verifyEqual(testCase, mi.q99_full_err, err.MI_err, 'AbsTol', 1e-15);
    verifyEqual(testCase, [mi.T_over_tau, mi.samples_per_tau, mi.n], [10, 10, 100], ...
                'the floor is keyed on the observation window and the sampling rate');

    summary = tables.recovery_summary;
    verifyEqual(testCase, summary.MI_delta_true, 0, 'AbsTol', 0);
    verifyEqual(testCase, summary.MI_delta_median_error, 0, 'AbsTol', 0);
    verifyEqual(testCase, summary.MI_full_median_error, err.MI_err, ...
                'AbsTol', 1e-15);
    verifyTrue(testCase, isnan(summary.MI_delta_relative_error), ...
               'a ratio to a zero truth is withheld, not reported as a number');
    verifyEqual(testCase, summary.MI_full_relative_error, ...
                err.MI_err / abs(G_true.MI), 'AbsTol', 1e-15);
    verifyEqual(testCase, summary.radial_time_tau, 0.5, 'AbsTol', 0, ...
                'a Gaussian truth carries the OU squared-radius time');
end

function testSummariesKeepEveryWindowAndRateApart(testCase)
    % Two dimensions, two windows and two rates: the floor is one row per (d, T/tau, samples per tau) and measure, read from that cell's Gaussian-truth replicates only, and the spread law one row per (d, samples per tau), the median over its windows.
    % The synthetic spread of p_hat differs by window and by rate, so a table that pooled either axis would report other numbers.
    reps = 20;
    rows = {};
    cell_index = 0;
    for p_true = [0 0.5]
        for d = [2 3]
            G_true = gaussianInformationMeasures(localEqui(d, 0.4));
            for T = [10 100]
                for s = [5 10]
                    cell_index = cell_index + 1;
                    width = 0.3 / sqrt(T) * (1 + (s == 5));
                    p_hat = max(0, p_true + width * linspace(-1, 1, reps));
                    for rep = 1:reps
                        row = struct2table(simulationInformationError(d, p_true, ...
                                                                      p_hat(rep), G_true, gaussianInformationMeasures( ...
                                                                            localEqui(d, 0.4 + 0.001 * rep))));
                        row.cell = cell_index;
                        row.p_true = p_true;
                        row.d = d;
                        row.T_over_tau = T;
                        row.samples_per_tau = s;
                        row.n = T * s;
                        row.refine = 2;
                        row.h = 1 / 320;
                        row.parent_dim = 20;
                        row.acceptance = 1;
                        row.rep = rep;
                        row.p_hat = p_hat(rep);
                        row.numerical_ok = true;
                        row.heavy_boundary = false;
                        rows{end + 1} = row; %#ok<AGROW>
                    end
                end
            end
        end
    end
    long = vertcat(rows{:});
    tables = simulationSummaries(long);
    verifyEqual(testCase, height(tables.recovery_summary), 16, 'one row per cell');

    floors = tables.floor_by_d_T_s;
    verifyEqual(testCase, height(floors), 4 + 3 * 8, ...
                'MI at d = 2 only, TC, DTC and O at both, per (d, T/tau, samples per tau)');
    for d = [2 3]
        for T = [10 100]
            for s = [5 10]
                own = long.p_true == 0 & long.d == d & long.T_over_tau == T ...
                    & long.samples_per_tau == s;
                for measure = ["MI", "TC"]
                    floor_row = floors(floors.d == d & floors.T_over_tau == T ...
                                       & floors.samples_per_tau == s & floors.measure == measure, :);
                    if measure == "MI" && d > 2
                        verifyEmpty(testCase, floor_row);
                        continue
                    end
                    q = quantile(long.(measure + "_delta_hat")(own), [0.95, 0.99]);
                    verifyEqual(testCase, [floor_row.n, floor_row.n_ok], [T * s, reps]);
                    verifyEqual(testCase, [floor_row.q95, floor_row.q99], [q(1), q(2)], ...
                                'AbsTol', 0, sprintf('%s floor at d = %d, T/tau = %d, s = %d', ...
                                             measure, d, T, s));
                end
            end
        end
    end

    law = tables.spread_law_by_d_s;
    verifyEqual(testCase, height(law), 4, 'one spread law per (d, samples per tau)');
    T_values = [10 100];
    for d = [2 3]
        for s = [5 10]
            spread = zeros(1, 2);
            for t = 1:2
                q = quantile(long.p_hat(long.p_true == 0.5 & long.d == d ...
                                        & long.T_over_tau == T_values(t) & long.samples_per_tau == s), ...
                             [0.25, 0.75]);
                spread(t) = (q(2) - q(1)) / 0.5 * sqrt(T_values(t));
            end
            law_row = law(law.d == d & law.samples_per_tau == s, :);
            verifyEqual(testCase, law_row.n_cells, 2);
            verifyEqual(testCase, law_row.spread_constant, median(spread), ...
                        'RelTol', 1e-12, sprintf('spread law at d = %d, s = %d', d, s));
        end
    end
end

function R = localEqui(d, rho)
    R = (1 - rho) * eye(d) + rho * ones(d);
end
