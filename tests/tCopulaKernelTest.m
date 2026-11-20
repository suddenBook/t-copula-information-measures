classdef tCopulaKernelTest < matlab.unittest.TestCase
    % TCOPULAKERNELTEST Tests of the Student-t copula kernels against independent references.
    %
    % Scope: the kernels tTailQuantile, tCopulaQuantilePlan, tCopulaLoglik, tCopulaLogDensity, tCopulaEntropy and tCopulaSample under src/t_copula/, plus informationMeasuresT.
    %
    % No check uses an earlier run of a kernel as its own reference:
    %   * the tail quantile against closed forms (Cauchy, nu = 2, the normal limit), against MATLAB's tinv, and against a round trip through tcdf;
    %   * the log-likelihood against localTCopulaLogDensity, a separate implementation in this file that shares no code with the kernel it checks, and against the Gaussian copula closed form at p = 0;
    %   * the d = 2 density against a 2-D quadrature of its own total mass;
    %   * kappa_t against the mpmath reference tests/oracles/t_copula_kappa_reference.csv, computed at 40 significant digits;
    %   * the sampler against Monte-Carlo entropy, the Kendall sine relation and marginal kurtosis;
    %   * informationMeasuresT against the identities O = TC - DTC, Delta(TC) = -kappa and MI = TC at d = 2.
    %
    % Some tolerances are far looser than double precision suggests.
    % MATLAB's betaincinv and tinv are each inaccurate in a different corner, and neither is uniformly the better reference.
    % Each level rests on a measurement, cross-checked against an mpmath evaluation of the exact incomplete beta at 40 digits and recorded at its assertion, so that no tolerance is loosened without a measurement behind it.

    properties (Constant)
        % Two-sided tail probabilities and degrees of freedom for the quantile rows.
        % q = 1 is included on purpose: it is the |z| = 0 boundary, the one cell where a relative tolerance has no meaning.
        TailProbabilities = logspace(-7, 0, 15)
        DegreesOfFreedom = [1, 3, 10, 100, 1e4, 1e6]
    end

    methods (TestClassSetup)

        function addSourceTree(testCase)
            % The fixture folder comes from this file's own location, not repoRoot(): it has to work with nothing on the path, the state a bare runtests on this single file starts from, and repoRoot lives under src/.
            % The assert makes a relocation out of <repo>/tests/ fail loudly rather than silently applying a fixture over a folder that does not exist.
            % One fixture over the whole src/ tree with IncludingSubfolders covers src/t_copula, src/shared/statistics and src/information_theory at once.
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'tCopulaKernelTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        % ---------------------------------------------------------------- (a)
        function tailQuantileMatchesTinv(testCase)
            testCase.assumeTrue(exist('tinv', 'file') == 2, ...
                                'tinv (Statistics and Machine Learning Toolbox) is required.');
            % Why 1e-8 and not 1e-11: MATLAB's betaincinv and tinv are each wrong somewhere, and an mpmath evaluation of the exact incomplete beta at 40 digits says which one.
            %   nu = 1, q = 1e-7: betaincinv gives 6366197.7236757725 and tinv gives 6366197.7140203984.
            %     The exact Cauchy value is cot(pi q/2) = 6366197.7236757611, so tinv is wrong by 1.5e-9 relative: it is handed the double 1 - q/2, and that rounding alone moves the quantile by eps(u)/2 / f_t(z).
            %   nu = 1e6, q = 0.1: betaincinv gives 1.6448551538645091 and tinv gives 1.6448551507220401.
            %     Feeding both back through the exact tail gives 0.0999999993518 and 0.1000000000000, so betaincinv is the wrong one here, by 1.9e-9 relative; its accuracy degrades as its first parameter nu/2 grows, which is why tTailQuantile uses the Cornish-Fisher expansion from nu = 1e5 on.
            % Library limits thus put the floor of this comparison at a few times 1e-9, and 1e-8 is the smallest round level above it.
            % That is still three orders of magnitude tighter than the error of a wrong degrees of freedom, a one-sided tail or a missing factor.
            q = testCase.TailProbabilities;
            for nu = testCase.DegreesOfFreedom
                actual = tTailQuantile(q, 1 / nu);
                expected = tinv(1 - q / 2, nu);
                worst = max(abs(actual - expected) ./ max(abs(expected), realmin));
                testCase.verifyEqual(actual, expected, sprintf( ...
                                                               ['|z| from tTailQuantile disagrees with tinv(1 - q/2, ' ...
                                                                '%g) beyond the 1e-8 library floor; worst relative gap ' ...
                                                                '%.3e.'], nu, worst), RelTol = 1e-8, AbsTol = 1e-14);
            end
        end

        function tailQuantileMatchesItsClosedForms(testCase)
            q = testCase.TailProbabilities;

            % nu = 1 (Cauchy): tinv(u, 1) = tan(pi(u - 1/2)), so with u = 1 - q/2 the magnitude is tan(pi(1 - q)/2) = cot(pi q/2).
            % The second spelling is the one asserted: writing it as tan(pi*(1 - q/2) - pi/2) puts the argument next to pi/2, where tan has unbounded derivative, and the double rounding of that argument costs 1.5e-9 relative at q = 1e-7 (measured).
            % cot(pi q/2) has no such cancellation and agrees to 8.4e-15.
            cauchy = 1 ./ tan(pi * q / 2);
            testCase.verifyEqual(tTailQuantile(q, 1), cauchy, ...
                                 ['|z| at nu = 1 must equal the Cauchy closed form ' ...
                                  'cot(pi q / 2).'], RelTol = 1e-13, AbsTol = 1e-15);

            % nu = 2: from tinv(u, 2) = (2u - 1)/sqrt(2u(1 - u)) with u = 1 - q/2, the numerator is 1 - q and 2u(1 - u) = 2(1 - q/2)(q/2) = q(2 - q)/2, so |z| = sqrt(2) (1 - q) / sqrt(q (2 - q)).
            % The sqrt(2) matters, since dropping it is a 41% error, and at q = 1/2 the form must reproduce tinv(0.75, 2) = 0.81649658.
            twoDegrees = sqrt(2) * (1 - q) ./ sqrt(q .* (2 - q));
            testCase.verifyEqual(tTailQuantile(q, 0.5), twoDegrees, ...
                                 ['|z| at nu = 2 must equal sqrt(2)(1 - q)/sqrt(q(2 - q)), ' ...
                                  'the closed form derived from tinv(u, 2).'], ...
                                 RelTol = 1e-13, AbsTol = 1e-15);
            testCase.verifyEqual(tTailQuantile(0.5, 0.5), tinv(0.75, 2), ...
                                 '|z| at nu = 2, q = 1/2 must equal tinv(0.75, 2).', ...
                                 RelTol = 1e-13);

            % p = 0 is the normal magnitude.
            % -norminv(q/2) is the same expression as sqrt(2) erfcinv(q) routed through a different function, so this pins the p = 0 branch and its interface, not the accuracy of erfcinv.
            testCase.verifyEqual(tTailQuantile(q, 0), -norminv(q / 2), ...
                                 '|z| at p = 0 must equal the standard normal magnitude.', ...
                                 RelTol = 1e-14, AbsTol = 1e-15);
        end

        function tailQuantileInvertsTheTwoSidedTail(testCase)
            % A consistency identity, not an accuracy oracle: below nu = 1e5, tcdf and tTailQuantile's betaincinv share MATLAB's incomplete beta, so both sides inherit the same error and can agree better than either is right.
            % It still catches the errors it exists to catch: a one-sided tail, a swapped degrees-of-freedom argument, a factor 2.
            %
            % The 'upper' spelling is required.
            % Writing the round trip as 2*(1 - tcdf(absz, nu)) subtracts a number within 5e-8 of 1 from 1 and pins the recovered q at 5.8e-10 relative for every nu (measured), the very cancellation tTailQuantile exists to avoid.
            q = testCase.TailProbabilities;
            for nu = testCase.DegreesOfFreedom
                recovered = 2 * tcdf(tTailQuantile(q, 1 / nu), nu, 'upper');
                % Measured worst: 4.3e-13 at nu = 1e4.
                % From nu = 1e5 on, the quantile uses the Cornish-Fisher expansion (accurate to 1e-20 relative on this grid at nu = 1e6), while tcdf's own accuracy at nu = 1e6 is only ~7e-9 relative, so the tolerance there measures the toolbox, not the kernel.
                tolerance = 1e-12 + 1e-8 * (nu > 1e4);
                testCase.verifyEqual(recovered, q, sprintf( ...
                                                           ['the round trip q -> |z| -> 2 P(T > |z|) failed at ' ...
                                                            'nu = %g; worst relative gap %.3e against tolerance ' ...
                                                            '%.1e.'], nu, max(abs(recovered - q) ./ q), tolerance), ...
                                     RelTol = tolerance);
            end
        end

        function tailQuantileRaisesNoWarnings(testCase)
            [previousMessage, previousIdentifier] = lastwarn;
            testCase.addTeardown(@() lastwarn(previousMessage, ...
                                              previousIdentifier));
            lastwarn('', 'tCopulaKernelTest:NoWarningYet');

            for nu = testCase.DegreesOfFreedom
                tTailQuantile(testCase.TailProbabilities, 1 / nu);
            end
            tTailQuantile(testCase.TailProbabilities, 0);

            [message, identifier] = lastwarn;
            testCase.verifyEmpty(message, sprintf( ...
                                                  ['tTailQuantile must evaluate the whole grid silently; it ' ...
                                                   'raised [%s] %s.'], identifier, message));
        end

        % ---------------------------------------------------------------- (b)
        function quantilePlanReproducesItsTailProbabilities(testCase)
            % Column 1 carries one tied pair and column 3 a tied triple, so the multiplicity bookkeeping is exercised rather than assumed.
            tied = [3, 1, 2; 3, 2, 2; 1, 3, 2; 4, 4, 5; 5, 5, 6; 2, 6, 7; ...
                    6, 7, 8; 7, 8, 9; 8, 9, 10; 9, 10, 11; 10, 11, 12; ...
                    11, 12, 13];
            [n, d] = size(tied);
            U = tiedrank(tied) / (n + 1);
            q = 2 * min(U, 1 - U);

            plan = tCopulaQuantilePlan(U);

            testCase.verifyEqual(plan.n, n, 'plan.n must be the row count.');
            testCase.verifyEqual(plan.d, d, 'plan.d must be the column count.');
            testCase.verifySize(plan.q_unique, [plan.num_distinct, 1], ...
                                'plan.q_unique must be a num_distinct x 1 column.');
            testCase.verifySize(plan.q_count, [plan.num_distinct, 1], ...
                                'plan.q_count must be a num_distinct x 1 column.');
            testCase.verifyTrue(all(diff(plan.q_unique) > 0), ...
                                'plan.q_unique must be strictly ascending and distinct.');
            testCase.verifyEqual(sum(plan.q_count), n * d, ...
                                 'plan.q_count must account for every one of the n*d entries.');
            testCase.verifyClass(plan.q_index, 'uint32', ...
                                 'plan.q_index must be uint32.');
            testCase.verifySize(plan.q_index, [n, d], ...
                                'plan.q_index must be n x d.');
            testCase.verifyClass(plan.negate, 'logical', ...
                                 'plan.negate must be logical.');
            % To a few ulps, not exactly: tCopulaQuantilePlan merges arguments that differ by less than 8 eps, because the mirror ranks r/(n+1) and 1 - (n+1-r)/(n+1) are the same tail probability in exact arithmetic and differ by rounding in double.
            % Merging them is what halves the number of quantile calls, so the requirement is reconstruction to the merge tolerance, and anything looser is a defect.
            testCase.verifyEqual(plan.q_unique(plan.q_index), q, ...
                                 'q_unique(q_index) must reconstruct 2*min(U, 1 - U).', ...
                                 RelTol = 1e-14);
            testCase.verifyEqual(plan.negate, U < 0.5, ...
                                 'plan.negate must mark exactly the entries with U < 0.5.');
            % The field is documented as the smallest one-sided tail probability min(u, 1-u), which is half the smallest q.
            testCase.verifyEqual(plan.min_tail_probability, ...
                                 min(min(U, 1 - U), [], 'all'), ...
                                 ['plan.min_tail_probability must be the smallest one-sided ' ...
                                  'tail probability min(u, 1 - u).'], RelTol = 1e-14);
            testCase.verifyEqual( ...
                                 accumarray(double(plan.q_index(:)), 1, ...
                           [plan.num_distinct, 1]), plan.q_count, ...
                                 'plan.q_count must be the multiplicity of each q_unique value.');

            % With no ties each column is a permutation of the same n values, so the distinct count is bounded well below n + 1; the bound is asserted, not the actual count.
            untied = reshape(1:(n * d), n, d);
            untiedPlan = tCopulaQuantilePlan(tiedrank(untied) / (n + 1));
            testCase.verifyLessThanOrEqual(untiedPlan.num_distinct, n + 1, ...
                                           'an untied plan cannot hold more than n + 1 distinct tail masses.');
        end

        % ---------------------------------------------------------------- (c)
        function loglikMatchesTheSensitivityKernel(testCase)
            n = 2000;
            d = 5;
            p = 0.2;
            R = localEquicorrelated(d, 0.4);
            X = tCopulaSample(n, d, R, p, 7);
            U = tiedrank(X) / (n + 1);
            Z = sign(U - 0.5) .* tTailQuantile(2 * min(U, 1 - U), p);

            Rinfo = localRinfo(R);
            reference = localTCopulaLogDensity(Z, R, 1 / p);

            testCase.verifyEqual(tCopulaLoglik(p, tCopulaQuantilePlan(U), Rinfo), ...
                                 sum(reference), ...
                                 ['the total t-copula log-likelihood must equal the summed ' ...
                                  'peSensitivityCopulaDraw kernel on the same latent points.'], ...
                                 RelTol = 1e-10);
            testCase.verifyEqual(tCopulaLogDensity(Z, R, p), reference, ...
                                 ['the per-row t-copula log density must equal the ' ...
                                  'peSensitivityCopulaDraw kernel row by row.'], RelTol = 1e-10);
        end

        function loglikMatchesTheGaussianCopulaAtZeroP(testCase)
            n = 2000;
            d = 5;
            R = localEquicorrelated(d, 0.4);
            U = tiedrank(tCopulaSample(n, d, R, 0, 7)) / (n + 1);
            Z = norminv(U);

            Rinfo = localRinfo(R);
            gaussian = -0.5 * n * Rinfo.logdetR ...
                - 0.5 * sum(sum((Z / Rinfo.Lr).^2, 2) - sum(Z.^2, 2));

            testCase.verifyEqual( ...
                                 tCopulaLoglik(0, tCopulaQuantilePlan(U), Rinfo), gaussian, ...
                                 ['at p = 0 the total log-likelihood must equal the Gaussian ' ...
                                  'copula closed form -n/2 logdet R - 1/2 sum(z''R^-1 z - ' ...
                                  'z''z).'], RelTol = 1e-10);
            testCase.verifyEqual(sum(tCopulaLogDensity(Z, R, 0)), gaussian, ...
                                 ['at p = 0 the per-row log density must sum to the Gaussian ' ...
                                  'copula closed form.'], RelTol = 1e-10);
        end

        % ---------------------------------------------------------------- (d)
        function loglikIsContinuousAndLinearApproachingZeroP(testCase)
            % The Gaussian point of the study is p = 0, so the likelihood has to reach it continuously: log c_t(z; p) = log c_G(z) + p A(z) + O(p^2), hence ll(p) - ll(0) is linear in p to leading order and vanishes with p.
            %
            % Two large-nu branches make this hold.
            % tTailQuantile uses the Cornish-Fisher expansion at p <= 1e-5, because MATLAB's betaincinv breaks down at nu >= 1e8, where it is 26% wrong.
            % tCopulaLoglik assembles its constant from exact log-gamma ratios, because the direct difference gammaln((nu+d)/2) - gammaln(nu/2) cancels badly at large nu and leaves a noise floor that swamps the O(p) signal.
            % The test asks for linear growth of the gap over p = 1e-7 to 1e-5 and a gap below 1e-2 nats at p = 1e-8.
            n = 20000;
            d = 20;
            R = localEquicorrelated(d, 0.4);
            U = tiedrank(tCopulaSample(n, d, R, 0, 4242)) / (n + 1);
            plan = tCopulaQuantilePlan(U);
            Rinfo = localRinfo(R);

            baseline = tCopulaLoglik(0, plan, Rinfo);
            pGrid = [1e-7, 1e-6, 1e-5];
            gaps = arrayfun(@(p) abs(tCopulaLoglik(p, plan, Rinfo) - baseline), ...
                            pGrid);
            scaling = (gaps(2:end) ./ gaps(1:end - 1)) ...
                ./ (pGrid(2:end) ./ pGrid(1:end - 1));

            testCase.verifyLessThanOrEqual(max(scaling), 2, sprintf( ...
                                                                    ['|ll(p) - ll(0)| must grow no faster than linearly in p; ' ...
                                                                     'gaps %s at p = %s give scaling %s.'], mat2str(gaps, 4), ...
                                                                    mat2str(pGrid), mat2str(scaling, 4)));
            testCase.verifyGreaterThanOrEqual(min(scaling), 0.5, sprintf( ...
                                                                         ['|ll(p) - ll(0)| must grow no slower than linearly in p; ' ...
                                                                          'gaps %s at p = %s give scaling %s.'], mat2str(gaps, 4), ...
                                                                         mat2str(pGrid), mat2str(scaling, 4)));

            tinyGap = abs(tCopulaLoglik(1e-8, plan, Rinfo) - baseline);
            testCase.verifyLessThanOrEqual(tinyGap, 1e-2, sprintf( ...
                                                                  ['at p = 1e-8 the total log-likelihood must sit within 1e-2 ' ...
                                                                   'nats of its p = 0 value over n = %d rows and d = %d ' ...
                                                                   'columns; the gap was %.4e nats.'], n, d, tinyGap));
        end

        % ---------------------------------------------------------------- (e)
        function loglikRefusesANonFiniteResult(testCase)
            % q = 0 means |z| = Inf, so the density is not finite.
            % The required behaviour is an error, never a quiet -Inf or NaN that an optimiser would then walk downhill into.
            n = 12;
            d = 3;
            U = tiedrank(reshape(1:(n * d), n, d)) / (n + 1);
            broken = tCopulaQuantilePlan(U);
            broken.q_unique(1) = 0;
            broken.min_tail_probability = 0;

            caught = MException.empty;
            try
                tCopulaLoglik(0.2, broken, ...
                              localRinfo(localEquicorrelated(d, 0.4)));
            catch caught
            end

            testCase.assertNotEmpty(caught, ...
                                    ['tCopulaLoglik must raise on a plan holding a zero tail ' ...
                                     'probability, not return a non-finite likelihood.']);
            % A zero tail probability is refused by the quantile guard before the likelihood can go non-finite, so an error from either function satisfies 'raise, never return -Inf'.
            testCase.verifyTrue(contains(caught.identifier, 'tCopulaLoglik') ...
                                || contains(caught.identifier, 'tTailQuantile'), ...
                                sprintf(['the non-finite-likelihood error must come from ' ...
                         'tCopulaLoglik or its quantile guard; got ''%s''.'], ...
                        caught.identifier));
        end

        % ---------------------------------------------------------------- (f)
        function pairCopulaDensityIntegratesToOne(testCase)
            % The density is the only object in this file whose normalisation is not implied by anything else checked here, and a missing Jacobian or a marginal term on the wrong scale shows up as a total mass away from 1.
            % The domain is clipped off the corners because the integrand is unbounded there and integral2 cannot place nodes at an infinite |z|; the excluded strip carries O(1e-9) of the mass.
            R = [1, 0.5; 0.5, 1];
            lower = 1e-9;
            upper = 1 - 1e-9;
            for p = [0.5, 0.1]
                mass = integral2( ...
                                 @(u, v) localPairCopulaDensity(u, v, R, p), ...
                                 lower, upper, lower, upper, 'AbsTol', 1e-9, 'RelTol', 1e-8);
                testCase.verifyEqual(mass, 1, sprintf( ...
                                                      ['the d = 2 copula density at p = %g, rho = 0.5 must ' ...
                                                       'integrate to 1 over the unit square; got %.12f.'], ...
                                                      p, mass), AbsTol = 1e-6);
            end
        end

        % ---------------------------------------------------------------- (g)
        function copulaEntropyMatchesTheMpmathOracle(testCase)
            reference = localReadKappaOracle(testCase);

            for d = unique(reference.d).'
                rows = reference(reference.d == d, :);
                ce = tCopulaEntropy(d, (1 ./ rows.nu).');
                kappa = reshape(ce.kappa, [], 1);
                kappaMargin = reshape(ce.kappa_margin, [], 1);

                testCase.assertEqual(numel(kappa), height(rows), sprintf( ...
                                                                         ['tCopulaEntropy must return one kappa per element of a ' ...
                                                                          'vector p; asked for %d at d = %d, got %d.'], ...
                                                                         height(rows), d, numel(kappa)));
                testCase.verifyTrue(all(ce.valid), sprintf( ...
                                                           'tCopulaEntropy reported an invalid cell at d = %d.', d));
                testCase.verifyTrue(all(isfinite(ce.H_joint)) ...
                                    && all(isfinite(ce.H_marginal)), sprintf( ...
                                                             ['tCopulaEntropy must report finite H_joint and ' ...
                                                              'H_marginal at d = %d.'], d));

                % One absolute tolerance of 1e-10 nats applies at every nu in the oracle, including nu = 1e6, where kappa_t itself lies between -5e-13 and -9.5e-11 and the closed form is a small difference of O(d log nu) terms.
                % It holds there because tLogGammaRatio and tDigammaDiff evaluate those differences without the cancellation of direct gammaln and psi differences.
                tolerance = 1e-10;
                worst = max(abs(kappa - rows.kappa_t) - tolerance);
                testCase.verifyLessThanOrEqual(worst, 0, sprintf( ...
                                                                 ['kappa at d = %d departs from the dps = 40 oracle ' ...
                                                                  'beyond tolerance; worst excess %.3e nats.'], d, worst));
                worstMargin = max( ...
                                  abs(kappaMargin - rows.kappa_t_dm1) - tolerance);
                testCase.verifyLessThanOrEqual(worstMargin, 0, sprintf( ...
                                                                       ['kappa_margin at d = %d departs from the oracle''s ' ...
                                                                        'kappa_t(d - 1, nu) beyond tolerance; worst excess ' ...
                                                                        '%.3e nats.'], d, worstMargin));
            end
        end

        % ---------------------------------------------------------------- (h)
        function copulaEntropyShapeTermHasItsStructuralProperties(testCase)
            reference = localReadKappaOracle(testCase);
            dims = unique(reference.d).';

            % Sign: the Gaussian copula maximises copula entropy at fixed R, so kappa <= 0 everywhere, with equality only in the Gaussian limit.
            for d = dims
                rows = reference(reference.d == d, :);
                kappa = tCopulaEntropy(d, (1 ./ rows.nu).').kappa;
                testCase.verifyLessThanOrEqual(max(kappa), 0, sprintf( ...
                                                                      ['kappa must be <= 0 at every nu (entropy maximum at the ' ...
                                                                       'Gaussian); worst value %.6e at d = %d.'], ...
                                                                      max(kappa), d));
            end

            % The Gaussian point is exact, not approached: p = 0 is a branch, not a limit, so no tolerance is allowed here.
            for d = dims
                testCase.verifyEqual(tCopulaEntropy(d, 0).kappa, 0, sprintf( ...
                                                                            'kappa(%d, 0) must be exactly 0 at the Gaussian point.', d));
            end
            testCase.verifyEqual(tCopulaEntropy(2, 0.3).kappa_margin, 0, ...
                                 ['kappa_margin at d = 2 must be exactly 0: it is ' ...
                                  'kappa_t(1, nu) = h_1 - h_1.']);

            % Approach and curvature: kappa is O(p^2) near p = 0, because the Gaussian is a stationary point of the copula entropy along the t path, so kappa/p^2 must settle rather than drift.
            % Measured relative change between p = 1e-3 and 2e-3: 6.7e-4 at d = 2, 1.2e-2 at d = 20.
            for d = dims
                kappa = reshape(tCopulaEntropy(d, [1e-3, 2e-3]).kappa, [], 1);
                testCase.verifyLessThanOrEqual(abs(kappa(1)), 1e-4, sprintf( ...
                                                                            ['|kappa| at p = 1e-3, d = %d must be below 1e-4 nats; ' ...
                                                                             'got %.4e.'], d, abs(kappa(1))));
                curvature = kappa ./ ([1e-3; 2e-3].^2);
                drift = abs(diff(curvature)) / abs(curvature(1));
                testCase.verifyLessThanOrEqual(drift, 0.05, sprintf( ...
                                                                    ['kappa/p^2 at d = %d must agree within 5%% between ' ...
                                                                     'p = 1e-3 and 2e-3; it drifted %.3f%%.'], d, 100 * drift));
            end

            % The margin term is the same code path at d - 1, so it must agree bit for bit and not merely to a tolerance.
            for d = dims(dims >= 3)
                testCase.verifyEqual(tCopulaEntropy(d, 0.2).kappa_margin, ...
                                     tCopulaEntropy(d - 1, 0.2).kappa, sprintf( ...
                                                              ['kappa_margin(%d, p) must equal kappa(%d, p) exactly ' ...
                                                               '(one code path, evaluated at d - 1).'], d, d - 1));
            end
        end

        % ---------------------------------------------------------------- (i)
        function monteCarloEntropyMatchesHalfLogDetPlusKappa(testCase)
            % H_c(R; d, nu) = 1/2 logdet R + kappa_t(d, nu), and -E log c under the copula is that same entropy.
            % This is the one row that ties the sampler, the density and the entropy closed form together: a mismatch between any two of the three shows up here.
            n = 200000;
            d = 5;
            p = 0.2;
            batches = 20;
            R = localEquicorrelated(d, 0.4);

            X = tCopulaSample(n, d, R, p, 11);
            batchMeans = mean(reshape(-tCopulaLogDensity(X, R, p), ...
                                      n / batches, batches), 1);
            estimate = mean(batchMeans);
            standardError = std(batchMeans) / sqrt(batches);
            expected = 0.5 * localRinfo(R).logdetR + tCopulaEntropy(d, p).kappa;

            testCase.verifyLessThan(standardError, 5e-3, sprintf( ...
                                                                 ['the batch-means standard error must stay below 5e-3 nats ' ...
                                                                  'for this row to be able to see a discrepancy; got %.3e.'], ...
                                                                 standardError));
            testCase.verifyLessThanOrEqual(abs(estimate - expected), ...
                                           4 * standardError, sprintf( ...
                                         ['the Monte-Carlo copula entropy %.6f must agree with ' ...
                                          '1/2 logdet R + kappa = %.6f within 4 standard errors ' ...
                                          '(%.3e nats); the gap was %.3e.'], estimate, expected, ...
                                         4 * standardError, abs(estimate - expected)));
        end

        % ---------------------------------------------------------------- (j)
        function sampledKendallTauFollowsTheSineRelation(testCase)
            % tau = (2/pi) asin(rho) for every elliptical copula with continuous margins, so sin(pi tau / 2) recovers the latent correlation without any reference to nu.
            % It checks that the sampler's dependence is the R it was handed, not a rescaled version of it.
            rho = 0.6;
            X = tCopulaSample(200000, 2, [1, rho; rho, 1], 0.25, 3);
            tau = kendallTauKnight(X(:, 1), X(:, 2));
            recovered = sin(pi * tau / 2);

            testCase.verifyLessThan(abs(recovered - rho), 0.01, sprintf( ...
                                                                        ['sin(pi tau / 2) must recover rho = %.2f from the sample; ' ...
                                                                         'tau = %.6f gives %.6f.'], rho, tau, recovered));
        end

        % ---------------------------------------------------------------- (k)
        function samplerIsSeededPrivateAndTailed(testCase)
            n = 200000;
            d = 3;
            R = localEquicorrelated(d, 0.3);

            stateBefore = rng;
            testCase.addTeardown(@() rng(stateBefore));
            first = tCopulaSample(n, d, R, 0.25, 11);
            stateAfter = rng;
            again = tCopulaSample(n, d, R, 0.25, 11);
            other = tCopulaSample(n, d, R, 0.25, 12);

            testCase.verifyEqual(stateAfter, stateBefore, ...
                                 ['tCopulaSample must draw on a private RandStream and leave ' ...
                                  'the global generator state untouched.']);
            testCase.verifyEqual(again, first, ...
                                 'one seed must reproduce a sample bit for bit.');
            testCase.verifyNotEqual(other, first, ...
                                    'two different seeds must not produce the same sample.');

            gaussianKurtosis = kurtosis(tCopulaSample(n, d, R, 0, 11));
            heavyKurtosis = kurtosis(first);
            testCase.verifyLessThan(max(abs(gaussianKurtosis - 3)), 0.1, ...
                                    sprintf(['p = 0 must give Gaussian margins; column kurtosis ' ...
                         'was %s.'], mat2str(gaussianKurtosis, 6)));
            % nu = 4 has infinite population kurtosis, so the sample value is large and unstable by nature; 5 asserts the tail is there without asserting a number the estimator does not have.
            testCase.verifyGreaterThan(min(heavyKurtosis), 5, sprintf( ...
                                                                      ['p = 0.25 must give visibly heavier-than-Gaussian margins; ' ...
                                                                       'column kurtosis was %s.'], mat2str(heavyKurtosis, 6)));
        end

        % ---------------------------------------------------------------- (m)
        function informationMeasureIdentitiesHoldAtDimensionTwo(testCase)
            testCase.assumeTrue(exist('informationMeasuresT', 'file') == 2, ...
                                'informationMeasuresT has not been written yet.');
            p = 0.2;
            R = [1, 0.5; 0.5, 1];
            out = informationMeasuresT(R, p);

            testCase.verifyEqual(out.O_delta_nats, ...
                                 out.TC_delta_nats - out.DTC_delta_nats, ...
                                 'O = TC - DTC must hold on the beyond-Gaussian differences.', ...
                                 AbsTol = 1e-12);
            testCase.verifyEqual(out.TC_delta_nats, ...
                                 -tCopulaEntropy(2, p).kappa, ...
                                 ['the beyond-Gaussian TC difference must be exactly -kappa: ' ...
                                  'the margins are unchanged, so the whole difference is the ' ...
                                  'copula entropy shape term.'], AbsTol = 1e-12);
            testCase.verifyEqual(out.MI_delta_nats, out.TC_delta_nats, ...
                                 'at d = 2 mutual information and total correlation coincide.', ...
                                 AbsTol = 1e-12);
        end

    end
end

% -------------------------------------------------------------------------
% Local helpers.
% localTCopulaLogDensity is a separate implementation rather than a call to tCopulaLogDensity: a test that checked the kernel's likelihood with the kernel's own density would only show that the two agree with each other.
% -------------------------------------------------------------------------

function logc = localTCopulaLogDensity(X, R, nu)
    % Per-row log density of the t copula at the latent points X on the t scale, with correlation matrix R and nu degrees of freedom.
    [n, d] = size(X);
    if d == 1
        logc = zeros(n, 1);
        return
    end
    L = chol(R, 'lower');
    whitened = X / L.';
    q = sum(whitened.^2, 2);
    logJoint = gammaln((nu + d) / 2) - gammaln(nu / 2) ...
        - d / 2 * log(nu * pi) - sum(log(diag(L))) ...
        - (nu + d) / 2 .* log1p(q / nu);
    logUnivariate = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
        - 0.5 * log(nu * pi) ...
        - (nu + 1) / 2 .* log1p(X.^2 / nu);
    logc = logJoint - sum(logUnivariate, 2);
end

function R = localEquicorrelated(d, rho)
    R = (1 - rho) * eye(d) + rho * ones(d);
end

function Rinfo = localRinfo(R)
    % The Rinfo struct tCopulaLoglik takes: an upper-triangular factor Lr with R = Lr' * Lr, which chol returns by default, plus log det R and d.
    Lr = chol(R);
    Rinfo = struct('Lr', Lr, 'logdetR', 2 * sum(log(diag(Lr))), 'd', size(R, 1));
end

function value = localPairCopulaDensity(u, v, R, p)
    % Copula density at (u, v) for integral2, which passes a pair of matrices and expects a matrix back.
    z = [localSignedQuantile(u(:), p), localSignedQuantile(v(:), p)];
    value = reshape(exp(tCopulaLogDensity(z, R, p)), size(u));
end

function z = localSignedQuantile(u, p)
    z = sign(u - 0.5) .* tTailQuantile(2 * min(u, 1 - u), p);
end

function reference = localReadKappaOracle(testCase)
    % The oracle lives under tests/oracles/, beside the suites that read it, so the path comes from this file's own location and needs nothing on the MATLAB path.
    oraclePath = fullfile(fileparts(mfilename('fullpath')), 'oracles', ...
                          't_copula_kappa_reference.csv');
    testCase.assertTrue(isfile(oraclePath), sprintf( ...
                                                    ['the mpmath kappa oracle is missing from %s; regenerate it with ' ...
                                                     '`uv run tests/oracles/t_copula_kappa_reference.py`.'], ...
                                                    oraclePath));
    reference = readtable(oraclePath);
    testCase.assertEqual(reference.Properties.VariableNames, ...
                         {'d', 'nu', 'p', 'kappa_t', 'kappa_t_dm1'}, ...
                         'the kappa oracle CSV does not carry the expected columns.');
    testCase.assertNotEmpty(reference, ...
                            'the kappa oracle CSV is empty, and an empty oracle proves nothing.');
end
