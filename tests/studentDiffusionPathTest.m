classdef studentDiffusionPathTest < matlab.unittest.TestCase
    %STUDENTDIFFUSIONPATHTEST The simulator keeps the Student law and its own interface.
    %
    % The Metropolis correction is needed because the frozen-multiplier proposal on its own has the wrong radial law, and at D = 2, nu = 1/3 no invariant law at all once h/tau > 0.0548.
    % The first check therefore runs exactly there, at h/tau = 0.1, on independent exact-start paths, where a one-time Kolmogorov-Smirnov calibration is valid.
    % The corrected chain must keep the radial beta law and the frozen chain, run on the same starts, must visibly lose it, which shows that the check can see the failure it guards against.
    % The Gaussian branch is checked as the exact AR(1) skeleton of the OU limit, drawn in blocks.
    % The rest pins the interface: seeding, the global generator, orientation, the substep rule, the nesting of rates, the frozen control and the input errors.

    properties (Constant)
        NumPaths = 4000
        KsCritical99 = 1.6276   % sqrt(n) times the asymptotic 99% KS critical value
    end

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'studentDiffusionPathTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function metropolisKeepsTheRadialLawWhereTheFrozenChainFails(testCase)
            % D = 2, nu = 1/3, one forced substep per observation at ten samples per tau: h/tau = 0.1, past the frozen chain's stability threshold.
            % Ten steps from exact Student starts, one value per independent path.
            R = [1, 0.4; 0.4, 1];
            L = chol(R, 'lower');
            nu = 1 / 3;
            n_obs = 11;
            n = testCase.NumPaths;
            pit_metropolis = zeros(n, 1);
            pit_frozen = zeros(n, 1);
            for p = 1:n
                stream = RandStream('twister', 'Seed', 50000 + p);
                [X, info] = studentDiffusionPath(R, nu, n_obs, 10, 1, stream, ...
                    struct('substeps', 1));
                pit_metropolis(p) = localRadialPit(L \ X(end, :).', nu);
                stream = RandStream('twister', 'Seed', 50000 + p);
                pit_frozen(p) = localRadialPit(localFrozenChain(stream, nu, n_obs, ...
                    info.h), nu);
            end
            critical = testCase.KsCritical99 / sqrt(n);
            testCase.verifyEqual(info.h, 0.1, AbsTol=1e-15);
            testCase.verifyLessThan(localKs(pit_metropolis), critical, sprintf( ...
                ['the Metropolis chain must keep Q/(nu+Q) ~ Beta(D/2, nu/2) at ' ...
                'h/tau = 0.1: KS %.4f against the 99%% critical value %.4f'], ...
                localKs(pit_metropolis), critical));
            testCase.verifyGreaterThan(localKs(pit_frozen), 3 * critical, sprintf( ...
                ['the frozen chain on the same starts must lose the radial law ' ...
                'here, or this check could not see the failure: KS %.4f'], ...
                localKs(pit_frozen)));
        end

        function gaussianBranchIsTheExactAR1Skeleton(testCase)
            % nu = Inf: y(k+1) = exp(-Delta) y(k) + sqrt(1 - exp(-2 Delta)) xi.
            % On a long path each whitened coordinate has lag-1 correlation exp(-Delta) and unit variance, and the observations have covariance R.
            % The tolerances are about five standard errors of each estimate at this length.
            R = [1, 0.5, 0.2; 0.5, 1, 0.3; 0.2, 0.3, 1];
            L = chol(R, 'lower');
            n_obs = 200000;
            X = studentDiffusionPath(R, Inf, n_obs, 10, 1, ...
                RandStream('twister', 'Seed', 424242));
            Y = (L \ X.').';
            a = exp(-0.1);
            for c = 1:3
                lag1 = corr(Y(1:end-1, c), Y(2:end, c));
                testCase.verifyEqual(lag1, a, sprintf( ...
                    'whitened coordinate %d: lag-1 correlation %.4f, expected %.4f', ...
                    c, lag1, a), AbsTol=0.005);
                testCase.verifyEqual(var(Y(:, c)), 1, AbsTol=0.05);
            end
            testCase.verifyEqual(cov(X), R, ...
                'the observed covariance of the OU limit is R', AbsTol=0.05);
        end

        function seededStreamReproducesAndGlobalGeneratorIsUntouched(testCase)
            R = 0.7 * eye(4) + 0.3 * ones(4);
            rng(20260925, 'twister');
            before = rng;
            for nu = [3, Inf]
                first = studentDiffusionPath(R, nu, 60, 10, 1, ...
                    RandStream('twister', 'Seed', 7));
                again = studentDiffusionPath(R, nu, 60, 10, 1, ...
                    RandStream('twister', 'Seed', 7));
                other = studentDiffusionPath(R, nu, 60, 10, 1, ...
                    RandStream('twister', 'Seed', 8));
                longer = studentDiffusionPath(R, nu, 150, 10, 1, ...
                    RandStream('twister', 'Seed', 7));
                testCase.verifyEqual(again, first, ...
                    sprintf('nu = %g: one seed must give one path', nu), AbsTol=0);
                testCase.verifyNotEqual(other, first, ...
                    sprintf('nu = %g: another seed must give another path', nu));
                if isfinite(nu)
                    % The draws come in fixed blocks, so a longer path with the same seed starts with the shorter one.
                    testCase.verifyEqual(longer(1:60, :), first, AbsTol=0);
                end
            end
            testCase.verifyEqual(rng, before, ...
                'the simulator must draw from its own stream only');
        end

        function outputIsSamplesByChannels(testCase)
            R = 0.6 * eye(5) + 0.4 * ones(5);
            for nu = [0.5, Inf]
                for n_obs = [1, 37]
                    [X, info] = studentDiffusionPath(R, nu, n_obs, 4, 2, ...
                        RandStream('twister', 'Seed', 3));
                    testCase.verifyEqual(size(X), [n_obs, 5], sprintf( ...
                        'nu = %g, n_obs = %d: X must be [n_obs x D]', nu, n_obs));
                    testCase.verifyEqual(info.n_obs, n_obs);
                    testCase.verifyEqual(info.D, 5);
                    testCase.verifyEqual(info.Delta, 0.25, AbsTol=0);
                end
            end
            % A fixed whitened start is the first observation, mapped through L.
            y0 = [0.5; -1; 2; 0; 1];
            X = studentDiffusionPath(R, 2, 3, 10, 1, ...
                RandStream('twister', 'Seed', 3), struct('y0', y0));
            testCase.verifyEqual(X(1, :), (chol(R, 'lower') * y0).', AbsTol=1e-15);
        end

        function substepRuleIsTheDeclaredOne(testCase)
            % ns = refine * max(1, ceil(Delta/delta0)) with delta0 = 0.005 min(1, m/(D+2), nu/(1+nu)) and h = Delta/ns.
            R20 = 0.6 * eye(20) + 0.4 * ones(20);
            cases = [ ...           % nu, samples per tau, refine, expected ns
                1/3, 10, 1, 80; ... % delta0 = 0.00125
                2,   10, 1, 30; ... % delta0 = 1/300: an exact quotient, no extra substep
                2,   10, 2, 60; ...
                10,  10, 1, 22; ... % delta0 = 0.005 * 10/11
                50,   1, 4, 4 * 204];   % delta0 = 0.005 * 50/51
            for k = 1:size(cases, 1)
                [~, info] = studentDiffusionPath(R20, cases(k, 1), 2, cases(k, 2), ...
                    cases(k, 3), RandStream('twister', 'Seed', k));
                testCase.verifyEqual(info.substeps, cases(k, 4), sprintf( ...
                    'nu = %g, %g samples per tau, refine %d', cases(k, 1:3)));
                testCase.verifyEqual(info.h, info.Delta / info.substeps, AbsTol=0);
            end
            [~, info] = studentDiffusionPath(R20, 1/3, 2, 10, 4, ...
                RandStream('twister', 'Seed', 1), struct('substeps', 3));
            testCase.verifyEqual(info.substeps, 3, ...
                'a forced substep count replaces the rule and the refine factor');
            [~, info] = studentDiffusionPath(R20, Inf, 2, 10, 4, ...
                RandStream('twister', 'Seed', 1));
            testCase.verifyEqual([info.substeps, info.h, info.acceptance], [1, 0.1, 1]);
        end

        function nestedRatesObserveOnePath(testCase)
            % On one substep grid a rate of s per tau reads the finest path every step/s observations.
            % R = I makes the observations the whitened states, so the nesting is pinned bit for bit at the study's D = 20 on its step rate of 160 per tau, in both branches: the Metropolis chain (3 tau crosses a block of draws at nu = 1/3) and the exact OU limit run at the fine interval.
            % A step rate of 15 read at 5 per tau (k = 3, not a power of two) pins that the substep is the fine grid's own double.
            R = eye(20);
            grids = {160, [5, 10, 20, 40, 80]; 15, 5};
            for g = 1:size(grids, 1)
                step = grids{g, 1};
                options = struct('step_samples_per_tau', step);
                for nu = [1/3, 2, 10, Inf]
                    [fine, info_fine] = studentDiffusionPath(R, nu, 3 * step + 1, ...
                        step, 2, RandStream('twister', 'Seed', 2027), options);
                    for s = grids{g, 2}
                        k = step / s;
                        [coarse, info_coarse] = studentDiffusionPath(R, nu, 3 * s + 1, ...
                            s, 2, RandStream('twister', 'Seed', 2027), options);
                        testCase.verifyEqual(coarse, fine(1:k:end, :), sprintf( ...
                            ['nu = %g: the %g-per-tau path must be the %g-per-tau one ' ...
                            'read every %g, bit for bit'], nu, s, step, k), AbsTol=0);
                        testCase.verifyEqual(info_coarse.h, info_fine.h, ...
                            'every rate of one grid shares the substep', AbsTol=0);
                        testCase.verifyEqual(info_coarse.substeps, k * info_fine.substeps);
                    end
                end
            end
            testCase.verifyError(@() studentDiffusionPath(R, 3, 5, 7, 2, ...
                RandStream('twister', 'Seed', 1), struct('step_samples_per_tau', 80)), ...
                'studentDiffusionPath:RateNotNested');
        end

        function ornsteinUhlenbeckBlocksAreOneSkeleton(testCase)
            % The OU limit draws and filters its steps 4096 observation intervals at a time.
            % The path must be, bit for bit, the one-shot skeleton of the whole window from the same stream, across two block boundaries, at the observation interval (k = 1) and nested on the 160-per-tau grid (k = 16 and 32).
            % R = I makes the observations the whitened states.
            D = 20;
            n_obs = 2 * 4096 + 10;
            for s = [160, 10, 5]
                k = 160 / s;
                X = studentDiffusionPath(eye(D), Inf, n_obs, s, 1, ...
                    RandStream('twister', 'Seed', 31), struct('step_samples_per_tau', 160));
                expected = localOneShotOrnsteinUhlenbeck( ...
                    RandStream('twister', 'Seed', 31), D, n_obs, 1 / 160, k);
                testCase.verifyEqual(X, expected.', sprintf( ...
                    'k = %d: the blocked OU must be the one-shot skeleton', k), AbsTol=0);
            end
        end

        function stepRateOfTheObservationIsTheDefaultRule(testCase)
            % step_samples_per_tau equal to the observation rate is the rule on the observation interval, so passing it changes nothing, bit for bit.
            R = 0.6 * eye(5) + 0.4 * ones(5);
            for nu = [1/3, 2, 10, Inf]
                plain = studentDiffusionPath(R, nu, 40, 10, 2, ...
                    RandStream('twister', 'Seed', 5));
                explicit = studentDiffusionPath(R, nu, 40, 10, 2, ...
                    RandStream('twister', 'Seed', 5), struct('step_samples_per_tau', 10));
                testCase.verifyEqual(explicit, plain, sprintf( ...
                    'nu = %g: the default rate must reproduce the plain call', nu), ...
                    AbsTol=0);
            end
            % At nu = 1/3 the substep of the 80-per-tau grid equals that of the 10-per-tau interval (delta0 divides both), so the nested path at 10 per tau is the plain one.
            R20 = 0.6 * eye(20) + 0.4 * ones(20);
            plain = studentDiffusionPath(R20, 1/3, 30, 10, 2, RandStream('twister', 'Seed', 6));
            nested = studentDiffusionPath(R20, 1/3, 30, 10, 2, ...
                RandStream('twister', 'Seed', 6), struct('step_samples_per_tau', 80));
            testCase.verifyEqual(nested, plain, AbsTol=0);
        end

        function frozenOptionIsTheUncorrectedChain(testCase)
            % The frozen chain accepts every proposal on the same draws, so it is the corrected chain until that one first rejects.
            % At D = 2, nu = 1/3 and h = 0.1 it loses the radial law the corrected one keeps.
            R = [1, 0.4; 0.4, 1];
            L = chol(R, 'lower');
            nu = 1 / 3;
            options = struct('substeps', 1, 'frozen', true);
            [frozen, info] = studentDiffusionPath(R, nu, 200, 10, 1, ...
                RandStream('twister', 'Seed', 77), options);
            [corrected, info_corrected] = studentDiffusionPath(R, nu, 200, 10, 1, ...
                RandStream('twister', 'Seed', 77), struct('substeps', 1));
            testCase.verifyEqual(info.acceptance, 1);
            testCase.verifyTrue(info.frozen);
            first_rejection = find(any(diff(corrected) == 0, 2), 1);
            testCase.verifyLessThan(info_corrected.acceptance, 1, ...
                'the heavy corner must reject some proposals at h = 0.1');
            testCase.verifyEqual(frozen(1:first_rejection, :), ...
                corrected(1:first_rejection, :), ...
                'until the first rejection the two chains are one chain', AbsTol=0);

            n = testCase.NumPaths;
            pit = zeros(n, 1);
            for p = 1:n
                X = studentDiffusionPath(R, nu, 11, 10, 1, ...
                    RandStream('twister', 'Seed', 50000 + p), options);
                pit(p) = localRadialPit(L \ X(end, :).', nu);
            end
            testCase.verifyGreaterThan(localKs(pit), ...
                3 * testCase.KsCritical99 / sqrt(n), ...
                'the frozen chain must lose the radial law here');
            testCase.verifyError(@() studentDiffusionPath(R, Inf, 5, 10, 1, ...
                RandStream('twister', 'Seed', 1), options), ...
                'studentDiffusionPath:FrozenGaussian');
        end

        function frozenChainReturnsItsDivergence(testCase)
            % At D = 2, nu = 1/3 and h = 0.5 tau the frozen chain has no invariant law, and its radius grows until it overflows (after 750 to 920 tau over 20 seeds).
            % The negative control returns that path, non-finite values and all, for the check to count, without raising; the corrected chain on the same draws stays finite.
            R = [1, 0.4; 0.4, 1];
            [frozen, info] = studentDiffusionPath(R, 1/3, 2001, 2, 1, ...
                RandStream('twister', 'Seed', 1), struct('substeps', 1, 'frozen', true));
            [corrected, info_corrected] = studentDiffusionPath(R, 1/3, 2001, 2, 1, ...
                RandStream('twister', 'Seed', 1), struct('substeps', 1));
            testCase.verifySize(frozen, [2001, 2]);
            testCase.verifyTrue(all(isfinite(frozen(1, :))) ...
                && any(~isfinite(frozen), 'all'), ...
                'the frozen chain must diverge here and return its non-finite path');
            testCase.verifyEqual(info.acceptance, 1);
            testCase.verifyTrue(all(isfinite(corrected), 'all'), ...
                'the corrected chain on the same draws must stay finite');
            testCase.verifyLessThan(info_corrected.acceptance, 1);
        end

        function badInputsAreRefused(testCase)
            stream = RandStream('twister', 'Seed', 1);
            R = [1, 0.3; 0.3, 1];
            testCase.verifyError(@() studentDiffusionPath(ones(2, 3), 2, 5, 10, 1, stream), ...
                'studentDiffusionPath:BadCorrelation');
            testCase.verifyError(@() studentDiffusionPath(1, 2, 5, 10, 1, stream), ...
                'studentDiffusionPath:BadCorrelation');
            testCase.verifyError(@() studentDiffusionPath([2, 0.3; 0.3, 1], 2, 5, 10, 1, stream), ...
                'studentDiffusionPath:BadCorrelation');
            testCase.verifyError(@() studentDiffusionPath([1, 1.2; 1.2, 1], 2, 5, 10, 1, stream), ...
                'studentDiffusionPath:BadCorrelation');
            testCase.verifyError(@() studentDiffusionPath([1, 0.3; 0.2, 1], 2, 5, 10, 1, stream), ...
                'studentDiffusionPath:BadCorrelation');
            testCase.verifyError(@() studentDiffusionPath(R, 0, 5, 10, 1, stream), ...
                'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() studentDiffusionPath(R, NaN, 5, 10, 1, stream), ...
                'MATLAB:validators:mustBeNonNan');
            testCase.verifyError(@() studentDiffusionPath(R, 2, 0, 10, 1, stream), ...
                'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() studentDiffusionPath(R, 2, 5.5, 10, 1, stream), ...
                'MATLAB:validators:mustBeInteger');
            testCase.verifyError(@() studentDiffusionPath(R, 2, 5, 0, 1, stream), ...
                'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() studentDiffusionPath(R, 2, 5, 10, 0, stream), ...
                'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() studentDiffusionPath(R, 2, 5, 10, 1, 17), ?MException);
            testCase.verifyError(@() studentDiffusionPath(R, 2, 5, 10, 1, stream, ...
                struct('substep', 2)), 'studentDiffusionPath:UnknownOption');
            testCase.verifyError(@() studentDiffusionPath(R, 2, 5, 10, 1, stream, ...
                struct('y0', [1; 2; 3])), ?MException);
            % An observation that overflows is an error, never a clipped number.
            testCase.verifyError(@() studentDiffusionPath(R, Inf, 3, 10, 1, stream, ...
                struct('y0', 0.9 * realmax * [1; 1])), ...
                'studentDiffusionPath:NonFiniteObservation');
        end
    end
end


function u = localRadialPit(y, nu)
    % P(Q/(nu+Q) <= b) under Beta(D/2, nu/2), through the stable complement nu/(nu+Q) = (sqrt(nu)/hypot(sqrt(nu), |y|))^2 and its Beta(nu/2, D/2) upper tail.
    w = (sqrt(nu) / hypot(sqrt(nu), norm(y)))^2;
    u = betainc(w, nu / 2, numel(y) / 2, 'upper');
end


function ks = localKs(u)
    % Kolmogorov-Smirnov distance of a sample from the uniform law.
    u = sort(u(:));
    n = numel(u);
    ks = max(max((1:n).' / n - u), max(u - (0:n - 1).' / n));
end


function Y = localOneShotOrnsteinUhlenbeck(stream, D, n_obs, step, k)
    % The OU skeleton of the whole window in one draw and one filter call, the reference the blocked simulator must reproduce: [D x n_obs] whitened states.
    y_start = randn(stream, D, 1);
    a = exp(-step);
    s = sqrt(-expm1(-2 * step));
    states = filter(s, [1, -a], randn(stream, D, (n_obs - 1) * k).', ...
        a * y_start.', 1).';
    Y = [y_start, states(:, k:k:end)];
end


function y = localFrozenChain(stream, nu, n_obs, h)
    % The uncorrected frozen-multiplier chain, whitened, from the same exact start the simulator draws (the same first draws of the same stream).
    % It is the Metropolis proposal with every step accepted, written out independently of the simulator as the negative control that shows what this test detects.
    D = 2;
    m = nu + D - 2;
    g = randn(stream, D, 1);
    u = min(max(rand(stream, 1, 1), realmin), 1 - eps);
    y = g * sqrt(nu / (2 * gammaincinv(u, nu / 2)));
    a = exp(-h);
    sv = sqrt(-expm1(-2 * h) / m);
    for step = 2:n_obs
        y = a * y + sv * hypot(sqrt(nu), norm(y)) * randn(stream, D, 1);
    end
end
