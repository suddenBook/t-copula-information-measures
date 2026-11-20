classdef unitFitTest < matlab.unittest.TestCase
    % UNITFITTEST The unit row: one recipe, one field set, on every path.
    %
    % Phase 2, Phase 3 and Phase 7 all emit the same flat unit row, so its field set has to be the same whatever the dimension and whatever went wrong.
    % tFitChannelSet has to be exactly the composition it documents (pseudo-observations, Kendall tau, unitCorrelationFromTau, tFitUnit), because Phase 3 refits surrogates through the split entry points while Phase 2 goes through the driver, and the two must not drift apart.
    %
    % The suite uses synthetic inputs only, a t-copula sample treated as a [samples x channels] matrix, and reads no recordings.

    properties (Constant)
        NumRows = 3000
        NumChannels = 8
        EquiCorrelation = 0.4
    end

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), 'unitFitTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function driverEqualsTheCompositionItDocuments(testCase)
            X = localRecording(testCase, 0.25, 5150);
            grp = [2, 4, 5, 7, 8];

            [row, tau, R] = tFitChannelSet(X, grp);

            [U, mask] = computeCopulaPseudoObservations(X(:, grp), ...
                                                        'MinSamples', 1);
            expectedTau = kendallTauMatrix(X(mask, grp));
            [expectedR, corr] = unitCorrelationFromTau(expectedTau);
            expected = tFitUnit(U, expectedR, corr);
            expected.channels = [grp, nan(1, 20 - numel(grp))];

            % fit_seconds is wall clock and differs between two runs of the same arithmetic, so it is the one field excluded from the comparison.
            elapsed = row.fit_seconds;
            row.fit_seconds = NaN;
            expected.fit_seconds = NaN;
            testCase.verifyGreaterThan(elapsed, 0);
            testCase.verifyEqual(row, expected, ...
                                 ['tFitChannelSet must be exactly its four documented steps: a ' ...
                                  'phase that composes them by hand has to get the same row.'], ...
                                 AbsTol = 0);
            testCase.verifyEqual(tau, expectedTau, ...
                                 ['the driver must hand back the SAME Kendall matrix the phase ' ...
                                  'MAT file stores per unit.'], AbsTol = 0);
            testCase.verifyEqual(R, expectedR, AbsTol = 0);
            testCase.verifyTrue(row.numerical_ok, row.reason);
            testCase.verifyEqual(row.status, 'ok');
            testCase.verifyEqual(row.d, numel(grp));
            testCase.verifyEqual(row.n, testCase.NumRows);
        end

        function correlationAuditCarriesTheDeterminantTerms(testCase)
            % logdet_R and sum_logdet_minors are what Phase 4 turns into TC, DTC and O, so they are checked against their definitions rather than against stored values.
            X = localRecording(testCase, 0.25, 6150);
            grp = 1:5;
            tau = kendallTauMatrix(X(:, grp));
            [R, corr] = unitCorrelationFromTau(tau);

            testCase.verifyTrue(corr.numerical_valid);
            testCase.verifyEqual(corr.logdet_R, log(det(R)), RelTol = 1e-10);
            minors = 0;
            for j = 1:numel(grp)
                keep = true(1, numel(grp));
                keep(j) = false;
                minors = minors + log(det(R(keep, keep)));
            end
            testCase.verifyEqual(corr.sum_logdet_minors, minors, RelTol = 1e-10);
            testCase.verifyEqual(corr.min_eig, min(eig(R)), RelTol = 1e-10);
            testCase.verifyEqual(corr.cond_R, cond(R), RelTol = 1e-10);
            testCase.verifyEqual(corr.max_abs_rho, max(abs(R(~eye(5)))));
            testCase.verifyTrue(isnan(corr.rho), ...
                                'rho is a pair quantity and must stay NaN above d = 2.');

            % A pair carries rho and tau, and each leave-one-out margin is univariate, so the sum of leave-one-out log-determinants is exactly 0.
            pairTau = kendallTauMatrix(X(:, [1, 2]));
            [Rpair, pairCorr] = unitCorrelationFromTau(pairTau);
            testCase.verifyEqual(pairCorr.rho, Rpair(1, 2), AbsTol = 0);
            testCase.verifyEqual(pairCorr.tau, pairTau(1, 2), AbsTol = 0);
            testCase.verifyEqual(pairCorr.sum_logdet_minors, 0, AbsTol = 0);
            testCase.verifyFalse(pairCorr.repair_applied, ...
                                 'a pair has one free parameter and is never repaired.');
        end

        function rowFieldSetIsIdenticalAcrossDimensionsAndErrorPaths(testCase)
            X = localRecording(testCase, 0.25, 7150);

            pair = tFitChannelSet(X, [1, 3]);
            group = tFitChannelSet(X, 1:5);
            tooFewRows = tFitChannelSet(X(1:5, :), [1, 2]);
            duplicated = tFitChannelSet([X(:, 1), X(:, 1)], [1, 2]);
            constant = tFitChannelSet([X(:, 1), zeros(testCase.NumRows, 1)], ...
                                      [1, 2]);

            rows = [pair, group, tooFewRows, duplicated, constant];
            testCase.verifyNumElements(rows, 5, ...
                                       ['rows of any dimension and any status must concatenate into ' ...
                                        'one struct array, which they cannot do with unequal fields.']);
            for other = {group, tooFewRows, duplicated, constant}
                testCase.verifyEqual(fieldnames(other{1}), fieldnames(pair), ...
                                     'every path must emit the same names in the same order.');
            end

            testCase.verifyEqual(tooFewRows.status, 'numerical_failure');
            testCase.verifyEqual(tooFewRows.reason, 'too_few_rows');
            testCase.verifyTrue(isnan(tooFewRows.p_hat));
            testCase.verifyEqual(duplicated.reason, ...
                                 'degenerate_d2_duplicate_channel');
            testCase.verifyTrue(duplicated.degenerate_pair);
            testCase.verifyFalse(duplicated.numerical_ok);
            testCase.verifyEqual(constant.reason, 'nonfinite_kendall_tau');
            testCase.verifyFalse(constant.numerical_ok);

            testCase.verifyEqual(pair.channels, [1, 3, nan(1, 18)]);
            testCase.verifyEqual(group.channels, [1:5, nan(1, 15)]);
            testCase.verifyTrue(isnan(group.unit_index), ...
                                'the phase script fills unit_index, not the fitter.');
            testCase.verifyEmpty(group.dataset_key);
        end

        function reportableRowCarriesEveryDeclaredNumber(testCase)
            X = localRecording(testCase, 0.25, 8150);
            row = tFitChannelSet(X, [1, 2, 3]);

            testCase.assertTrue(row.numerical_ok, row.reason);
            testCase.verifyEqual(row.nu_hat, 1 / row.p_hat);
            testCase.verifyEqual(row.aic_diff, ...
                                 2 * (row.loglik_t - row.loglik_gaussian) - 2, RelTol = 1e-12);
            testCase.verifyGreaterThanOrEqual(row.loglik_t, row.loglik_gaussian);
            testCase.verifyEqual(numel(row.profile_p_grid), 41);
            testCase.verifyEqual(numel(row.profile_loglik_grid), 41);
            testCase.verifyTrue(all(isfinite(row.profile_loglik_grid)));
            testCase.verifyGreaterThan(row.fit_seconds, 0);
            testCase.verifyTrue(isfinite(row.logdet_R));
            testCase.verifyTrue(isfinite(row.sum_logdet_minors));
            testCase.verifyTrue(isfinite(row.cond_R));
            testCase.verifyGreaterThan(row.min_eig_R, 0);
            testCase.verifyFalse(row.repair_gross);
        end

        function channelSetIsGatedOnOrientationAndWidth(testCase)
            X = localRecording(testCase, 0.25, 9150);

            testCase.verifyError( ...
                                 @() tFitChannelSet(X, [1, testCase.NumChannels + 1]), ...
                                 'tFitChannelSet:ChannelOutOfRange', ...
                                 ['a channel index past the column count is the signature of a ' ...
                                  'transposed [channels x samples] matrix.']);
            testCase.verifyError(@() tFitChannelSet(X, 1), ...
                                 'tFitChannelSet:GroupWidthOutOfRange');
        end

    end
end

function X = localRecording(testCase, p_true, seed)
    % LOCALRECORDING A synthetic [samples x channels] recording from a t copula.
    d = testCase.NumChannels;
    R = (1 - testCase.EquiCorrelation) * eye(d) ...
        + testCase.EquiCorrelation * ones(d);
    X = tCopulaSample(testCase.NumRows, d, R, p_true, seed);
end
