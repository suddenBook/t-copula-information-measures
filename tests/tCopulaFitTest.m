classdef tCopulaFitTest < matlab.unittest.TestCase
    %TCOPULAFITTEST Tests of the t-copula shape fitter and its options.
    %
    % Covers fitTCopulaShape and tFitDefaultOptions: shape recovery from synthetic t-copula samples, the lower bound (p = 0 is the Gaussian copula and a valid fit), an interior heavy-tailed optimum that is not flagged heavy_boundary, the Gaussian snap rule, the ambiguity and scan-consistency bookkeeping, and the option validator.
    % tFitDomainTest covers fits at the heavy end of the domain, where heavy_boundary is a recorded flag and not an exclusion.
    % unitFitTest tests the unit row that tFitUnit and tFitChannelSet build from this fitter.
    %
    % The suite uses synthetic inputs only (tCopulaSample with fixed seeds) and reads no recordings.
    % The recovery tolerances are deliberately loose, because this is a smoke test of the estimator, not the study's bias measurement.

    properties (Constant)
        NumRows = 20000
        EquiCorrelation = 0.4
        GaussianSeeds = 4001:4006
    end

    methods (TestClassSetup)
        function addSourceTree(testCase)
            % The fixture folder comes from this file's own location, not from repoRoot, because it has to work with nothing on the path, the state a bare runtests on this file starts from, and repoRoot lives under src/.
            % The assert makes a relocation out of <repo>/tests/ fail loudly rather than silently applying a fixture over a folder that does not exist.
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), 'tCopulaFitTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function recoversShapeWithTheTrueCorrelation(testCase)
            % The plug-in estimate must land near the truth for light through moderate tails across the range of dimensions the study fits.
            for nu = [5, 10, 30]
                for d = [2, 5, 20]
                    R = localEquiCorrelation(d, testCase.EquiCorrelation);
                    X = tCopulaSample(testCase.NumRows, d, R, 1 / nu, ...
                        91000 + 100 * nu + d);
                    entry = fitTCopulaShape(localRanks(X), R);

                    label = sprintf('nu = %d, d = %d', nu, d);
                    testCase.verifyTrue(entry.numerical_ok, ...
                        sprintf('%s: %s', label, entry.exclusion_reason));
                    testCase.verifyEqual(entry.valid, entry.numerical_ok, label);
                    testCase.verifyFalse(entry.gaussian_boundary, label);
                    testCase.verifyFalse(entry.heavy_boundary, label);
                    testCase.verifyLessThanOrEqual( ...
                        abs(entry.p_hat - 1 / nu), 0.25 / nu + 0.02, label);
                    testCase.verifyTrue(isfinite(entry.loglik), label);
                    testCase.verifyGreaterThanOrEqual( ...
                        entry.loglik, entry.loglik_gaussian, label);
                    testCase.verifyEqual(entry.nu_hat, 1 / entry.p_hat, label);
                end
            end
        end

        function gaussianSamplesFitAtOrNearTheLowerBound(testCase)
            % p = 0 is a model on the lower bound, so a Gaussian sample must produce a reportable fit, never a not-a-max exclusion.
            seeds = testCase.GaussianSeeds;
            entries = cell(1, numel(seeds));
            for k = 1:numel(seeds)
                entries{k} = localGaussianFit(testCase, seeds(k), struct());
                label = sprintf('seed %d', seeds(k));
                testCase.verifyTrue(entries{k}.numerical_ok, ...
                    sprintf('%s: %s', label, entries{k}.exclusion_reason));
                testCase.verifyFalse(entries{k}.ambiguous_profile, label);
                testCase.verifyLessThan(entries{k}.p_hat, 0.05, label);
                testCase.verifyGreaterThanOrEqual(entries{k}.p_hat, 0, label);
            end

            atBound = cellfun(@(e) e.gaussian_boundary, entries);
            testCase.assertTrue(any(atBound), ...
                'no Gaussian sample optimised exactly at p = 0');
            for entry = entries(atBound)
                e = entry{1};
                testCase.verifyEqual(e.p_hat, 0);
                testCase.verifyEqual(e.nu_hat, Inf);
                testCase.verifyEqual(e.loglik, e.loglik_gaussian);
                testCase.verifyEqual(e.profile_selected_source, 'lower_endpoint');
                testCase.verifyTrue(e.profile_neighbor_verified);
                testCase.verifyTrue(e.profile_gaussian_anchor_evaluated);
                testCase.verifyTrue(isfinite(e.d2));
                testCase.verifyTrue(isfinite(e.profile_boundary_slope));
                testCase.verifyLessThanOrEqual(e.profile_boundary_slope, 0);
            end
        end

        function snapRuleReportsTheClosedFormGaussianNode(testCase)
            % With a wide snap window every Gaussian sample must be reported at the exactly evaluated node, and the snap flag must be set exactly for the samples whose default fit selects a refined fminbnd optimum.
            forced = struct( ...
                'gaussian_snap_tolerance', 0.045, ...
                'profile_ambiguity_min_p_separation', 0.048, ...
                'profile_ll_tol_per_row', 1e-3);
            snapped = false(1, numel(testCase.GaussianSeeds));
            for k = 1:numel(testCase.GaussianSeeds)
                seed = testCase.GaussianSeeds(k);
                label = sprintf('seed %d', seed);
                byDefault = localGaussianFit(testCase, seed, struct());
                entry = localGaussianFit(testCase, seed, forced);

                testCase.verifyEqual(entry.p_hat, 0, label);
                testCase.verifyEqual(entry.nu_hat, Inf, label);
                testCase.verifyEqual(entry.loglik, entry.loglik_gaussian, label);
                testCase.verifyEqual( ...
                    entry.profile_selected_source, 'lower_endpoint', label);
                testCase.verifyEqual(entry.profile_gaussian_snapped, ...
                    strcmp(byDefault.profile_selected_source, 'refined'), label);
                snapped(k) = entry.profile_gaussian_snapped;
            end
            testCase.verifyTrue(any(snapped), ...
                'the snap branch was never exercised');
        end

        function ambiguityIsRecordedAndNoLongerExcludes(testCase)
            % Basins within profile_ambiguity_min_p_separation of the optimum are one basin, so a near-zero optimum has exactly one separated competitor, the heavy-end endpoint of the [0, 3] domain.
            % With the tie threshold forced absurdly wide, the fit is flagged ambiguous and stays reportable.
            entry = localGaussianFit(testCase, testCase.GaussianSeeds(1), ...
                struct('profile_ambiguity_ll_tol_per_row', 1e6));

            testCase.verifyEqual(entry.profile_num_separated_competitors, 1);
            testCase.verifyEqual(entry.profile_second_best_p, 3);
            testCase.verifyEqual( ...
                entry.profile_second_best_source, 'upper_endpoint');
            testCase.verifyTrue(entry.ambiguous_profile);
            testCase.verifyTrue(entry.numerical_ok, entry.exclusion_reason);
            testCase.verifyEmpty(entry.exclusion_reason, ...
                'an ambiguous profile is a flag, never an exclusion code');
        end

        function heavyTailOptimumStaysInsideTheWidenedDomain(testCase)
            % On [0, 3], nu = 1 (p = 1) is an ordinary interior optimum: reportable, not flagged, and nowhere near the heavy end.
            R = [1, testCase.EquiCorrelation; testCase.EquiCorrelation, 1];
            X = tCopulaSample(testCase.NumRows, 2, R, 1, 777);
            entry = fitTCopulaShape(localRanks(X), R);

            testCase.verifyTrue(entry.numerical_ok, entry.exclusion_reason);
            testCase.verifyEmpty(entry.exclusion_reason);
            testCase.verifyFalse(entry.heavy_boundary);
            testCase.verifyLessThanOrEqual(abs(entry.p_hat - 1), 0.1);
            testCase.verifyGreaterThan( ...
                entry.profile_distance_to_upper_p_bound, ...
                entry.profile_boundary_tolerance);
        end

        function optionDefaultsAndValidatorsHoldTheProtocol(testCase)
            options = tFitDefaultOptions();

            testCase.verifyEqual(options.p_bounds, [0, 3], AbsTol=0);
            testCase.verifyEqual(options.tol_x, 1e-6);
            testCase.verifyEqual(options.gaussian_snap_tolerance, 1e-5);
            testCase.verifyEqual(options.curvature_step, 0.005);
            testCase.verifyEqual(options.profile_grid_size, 41);
            testCase.verifyEqual(options.profile_ll_tol_per_row, 2e-6);
            testCase.verifyEqual(options.profile_ambiguity_ll_tol_per_row, 3e-5);
            testCase.verifyEqual(options.profile_ambiguity_min_p_separation, 0.01);
            testCase.verifyEqual(options.boundary_p_tolerance, 0.05);
            testCase.verifyTrue(options.store_profile);
            testCase.verifyEqual(options.min_rows, 10);
            testCase.verifyEqual(options.r_min_eig, 1e-10);
            testCase.verifyEqual(options.r_sym_tol, 1e-10);
            testCase.verifyEqual(options.r_diag_tol, 1e-8);
            testCase.verifyEqual(options.max_projection_shift_rms, 0.02);

            % The distinct-peak separation must fit inside one grid step of the domain: 0.01 < 3/40 = 0.075.
            testCase.verifyLessThan(options.profile_ambiguity_min_p_separation, ...
                options.p_bounds(2) / (options.profile_grid_size - 1));

            % Another upper bound, here 5, is admissible, because the domain is a working choice and not a property of the family.
            widened = tFitDefaultOptions(struct('p_bounds', [0, 5]));
            testCase.verifyEqual(widened.p_bounds, [0, 5], AbsTol=0);

            testCase.verifyError( ...
                @() tFitDefaultOptions(struct('p_bounds', [0.01, 3])), ...
                'tFitDefaultOptions:GaussianBoundNotZero');
            testCase.verifyError( ...
                @() tFitDefaultOptions(struct('profile_grid_size', 40)), ...
                'tFitDefaultOptions:EvenProfileGrid');
            testCase.verifyError( ...
                @() tFitDefaultOptions(struct('tol_x', 1e-4)), ...
                'tFitDefaultOptions:ToleranceOrdering');
        end
    end
end


function R = localEquiCorrelation(d, rho)
    R = (1 - rho) * eye(d) + rho * ones(d);
end


function U = localRanks(X)
    % Pseudo-observations exactly as tFitChannelSet builds them.
    n = size(X, 1);
    U = zeros(size(X));
    for c = 1:size(X, 2)
        U(:, c) = tiedrank(X(:, c));
    end
    U = U / (n + 1);
end


function entry = localGaussianFit(testCase, seed, options)
    % One d = 5 Gaussian-copula sample (p = 0) fitted with the given overrides.
    d = 5;
    R = localEquiCorrelation(d, testCase.EquiCorrelation);
    X = tCopulaSample(testCase.NumRows, d, R, 0, seed);
    entry = fitTCopulaShape(localRanks(X), R, options);
end
