classdef tFitDomainTest < matlab.unittest.TestCase
    % TFITDOMAINTEST The shape domain p in [0, 3] and its recorded flags.
    %
    % The study fits p on [0, 3] (nu >= 1/3) and excludes nothing inside the domain: the heavy end is a flag, not a verdict.
    % These tests check that the estimator reaches the heavy part of the domain (a nu = 0.5 and a nu = 1/3 truth are recovered, not clipped), that the heavy-end flag fires where it should and nowhere else, that a constrained maximum exactly on the upper bound is a reportable fit and not a numerical failure, and that a Gaussian truth comes back as a numerically sound fit at the other end of the same domain.
    %
    % The suite uses synthetic t-copula samples at d = 3 only: n = 20000 for the recovery tests and n = 1000 for the endpoint tests, where a heavy truth pins the profile maximum on the bound.
    % The tolerances are a smoke test of the estimator's reach, not the simulation study's bias measurement, which Phase 7 owns.

    properties (Constant)
        NumRows = 20000
        Dimension = 3
        EquiCorrelation = 0.4
    end

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), 'tFitDomainTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function heavyTruthsBeyondTheOldBoundAreRecovered(testCase)
            % The heavy truths nu = 0.5 (p = 2) and nu = 1/3 (p = 3) must both be reachable estimates.
            for nu = [0.5, 1 / 3]
                p_true = 1 / nu;
                entry = localFit(testCase, p_true, 6100 + round(1000 * nu));
                label = sprintf('nu = %.4g (p = %.4g)', nu, p_true);

                testCase.verifyTrue(entry.numerical_ok, ...
                                    sprintf('%s: %s', label, entry.exclusion_reason));
                testCase.verifyLessThanOrEqual(abs(entry.p_hat - p_true), 0.35, ...
                                               sprintf('%s: p_hat = %.4f', label, entry.p_hat));
                testCase.verifyGreaterThanOrEqual( ...
                                                  entry.loglik, entry.loglik_gaussian, label);
                testCase.verifyEqual(entry.nu_hat, 1 / entry.p_hat, label);
            end
        end

        function heavyBoundaryFlagFiresOnlyAtTheHeavyEnd(testCase)
            % A truth at p = 2.99 sits inside boundary_p_tolerance of the upper bound, so the flag must fire, and the fit must stay reportable.
            atEdge = localFit(testCase, 2.99, 6301);
            testCase.verifyTrue(atEdge.numerical_ok, atEdge.exclusion_reason);
            testCase.verifyTrue(atEdge.heavy_boundary, sprintf( ...
                                                               ['a p_hat of %.4f is within %.3f of the upper bound and must ' ...
                                                                'be flagged heavy_boundary.'], atEdge.p_hat, ...
                                                               atEdge.profile_boundary_tolerance));
            testCase.verifyEmpty(atEdge.exclusion_reason, ...
                                 'the heavy end is a flag, never an exclusion code');
            testCase.verifyEqual(atEdge.valid, atEdge.numerical_ok);

            % A moderate truth must not trip it.
            interior = localFit(testCase, 0.5, 6302);
            testCase.verifyTrue(interior.numerical_ok, ...
                                interior.exclusion_reason);
            testCase.verifyFalse(interior.heavy_boundary);
        end

        function constrainedMaximumExactlyAtTheUpperBoundIsAValidFit(testCase)
            % A fit merely near 3 does not exercise this; the case under test is a winner that is the upper node itself.
            % The constrained maximum of the closed domain is a successful fit flagged heavy_boundary, not a numerical failure, which would silently drop the heaviest channel sets and simulation replicates from every summary exactly where the study reads the heavy end as an observation about the recording.
            entry = localFitRows(testCase, 3, 6, 1000);

            testCase.verifyEqual(entry.p_hat, 3, ...
                                 'the fixture must land ON the upper bound, not near it.', ...
                                 AbsTol = 0);
            testCase.verifyEqual(entry.profile_selected_source, 'upper_endpoint');
            testCase.verifyTrue(entry.numerical_ok, entry.exclusion_reason);
            testCase.verifyEmpty(entry.exclusion_reason, ...
                                 'the heavy end is a flag, never an exclusion code');
            testCase.verifyTrue(entry.heavy_boundary);
            testCase.verifyEqual(entry.valid, entry.numerical_ok);
            testCase.verifyEqual(entry.nu_hat, 1 / 3, AbsTol = 0);
            testCase.verifyGreaterThan(entry.optimizer_exitflag, 0, ...
                                       ['the upper endpoint inherits its own guard refinement''s ' ...
                                        'exit flag, as the Gaussian node inherits the lower ' ...
                                        'guard''s.']);
            testCase.verifyTrue(entry.profile_neighbor_verified, ...
                                ['the neighbour check at the upper bound is one-sided: there ' ...
                                 'is no node above it to compare against.']);
            testCase.verifyTrue(isnan(entry.profile_upper_neighbor_loglik));
            testCase.verifyGreaterThanOrEqual(entry.profile_neighbor_gap, ...
                                              -entry.profile_compare_tolerance);
        end

        function theRestOfTheDomainIsUnchangedAtTheSameSampleSize(testCase)
            % Same estimator, R and sample size as the endpoint fixture: an interior truth must give a refined interior optimum, and a Gaussian truth must snap onto the exact p = 0 node.
            interior = localFitRows(testCase, 0.5, 6, 1000);
            testCase.verifyTrue(interior.numerical_ok, interior.exclusion_reason);
            testCase.verifyFalse(interior.heavy_boundary);
            testCase.verifyEqual(interior.profile_selected_source, 'refined');
            testCase.verifyLessThanOrEqual(abs(interior.p_hat - 0.5), 0.35, ...
                                           sprintf('p_hat = %.4f at an interior truth of 0.5', ...
                        interior.p_hat));

            gaussian = localFitRows(testCase, 0, 7003, 1000);
            testCase.verifyTrue(gaussian.numerical_ok, gaussian.exclusion_reason);
            testCase.verifyEqual(gaussian.p_hat, 0, ...
                                 'a Gaussian truth still snaps onto the exact p = 0 node.', ...
                                 AbsTol = 0);
            testCase.verifyTrue(gaussian.gaussian_boundary);
            testCase.verifyFalse(gaussian.heavy_boundary);
            testCase.verifyEqual(gaussian.nu_hat, Inf);
        end

        function gaussianTruthIsNumericallySoundOnTheWidenedDomain(testCase)
            % The heavy reach of the domain must cost the Gaussian end nothing: p = 0 is an exactly evaluated node with a converged lower guard, and the whole 41-node profile is finite.
            entry = localFit(testCase, 0, 6401);

            testCase.verifyTrue(entry.numerical_ok, entry.exclusion_reason);
            testCase.verifyLessThan(entry.p_hat, 0.05);
            testCase.verifyGreaterThanOrEqual(entry.p_hat, 0);
            testCase.verifyFalse(entry.heavy_boundary);
            testCase.verifyEqual(entry.profile_p_bounds, [0, 3], AbsTol = 0);
            testCase.verifyEqual(numel(entry.profile_p_grid), 41);
            testCase.verifyTrue(all(isfinite(entry.profile_loglik_grid)));
        end

    end
end

function entry = localFit(testCase, p_true, seed)
    % One d = 3 equicorrelated t-copula sample at the suite's default size.
    entry = localFitRows(testCase, p_true, seed, testCase.NumRows);
end

function entry = localFitRows(testCase, p_true, seed, n_rows)
    % One d = 3 equicorrelated t-copula sample of n_rows rows, fitted at its true R.
    d = testCase.Dimension;
    R = (1 - testCase.EquiCorrelation) * eye(d) ...
        + testCase.EquiCorrelation * ones(d);
    X = tCopulaSample(n_rows, d, R, p_true, seed);
    n = size(X, 1);
    U = zeros(n, d);
    for c = 1:d
        U(:, c) = tiedrank(X(:, c));
    end
    entry = fitTCopulaShape(U / (n + 1), R);
end
