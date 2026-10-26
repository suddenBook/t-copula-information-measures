classdef tCopulaQuantileCacheTest < matlab.unittest.TestCase
    %TCOPULAQUANTILECACHETEST The quantile cache must change speed and nothing else.
    %
    % The cache exists so that one recording's 41-node coarse profile is not re-inverted for every one of its roughly 180,000 channel sets.
    % It is admissible only if it is exact: a cached and a direct likelihood must agree bit for bit, and a fit with a cache must return the same p_hat as a fit without one.
    % Anything looser would make the cache a silent second estimator.

    properties (Constant)
        NumRows = 4000
        Dimension = 4
        EquiCorrelation = 0.35
    end

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'tCopulaQuantileCacheTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function cacheCarriesThePlansOwnTailProbabilities(testCase)
            n = 501;
            plan = tCopulaQuantilePlan(localRanks( ...
                tCopulaSample(n, 3, eye(3), 0.2, 1234)));
            cache = tCopulaQuantileCache(n, [0, 0.5]);

            testCase.verifyEqual(cache.n, n);
            testCase.verifyEqual(cache.q_unique, plan.q_unique, ...
                ['the cache must key on the SAME distinct tail probabilities ' ...
                'the plan builds, or its column index means something else.'], ...
                AbsTol=0);
            testCase.verifyEqual(cache.num_distinct, plan.num_distinct);
            testCase.verifySize(cache.absz, [plan.num_distinct, 2]);
            testCase.verifyEqual(cache.absz(:, 1), ...
                tTailQuantile(plan.q_unique, 0), AbsTol=0);
        end

        function cachedLoglikEqualsTheDirectValueAtEveryGridNode(testCase)
            opts = tFitDefaultOptions();
            p_grid = linspace(opts.p_bounds(1), opts.p_bounds(2), ...
                opts.profile_grid_size);
            [U, R] = localSample(testCase, 0.3, 8801);
            plan = tCopulaQuantilePlan(U);
            Rinfo = localRinfo(R);
            cache = tCopulaQuantileCache(testCase.NumRows, p_grid);

            for k = 1:numel(p_grid)
                direct = tCopulaLoglik(p_grid(k), plan, Rinfo);
                cached = tCopulaLoglik(p_grid(k), plan, Rinfo, cache);
                testCase.verifyEqual(cached, direct, sprintf( ...
                    ['the cached log-likelihood must equal the direct one BIT ' ...
                    'FOR BIT at p = %.6g; the gap was %.3e nats.'], ...
                    p_grid(k), abs(cached - direct)), AbsTol=0);
            end

            % Between the nodes there is nothing to read, so the cache must fall through to the direct route rather than interpolate.
            offNode = mean(p_grid(2:3));
            testCase.verifyEqual( ...
                tCopulaLoglik(offNode, plan, Rinfo, cache), ...
                tCopulaLoglik(offNode, plan, Rinfo), AbsTol=0);
        end

        function aCacheBuiltForAnotherRowCountFallsBackToTheDirectRoute(testCase)
            % A channel set with fewer complete rows than the recording must still fit correctly: the cache is ignored, never consulted wrongly.
            [U, R] = localSample(testCase, 0.3, 8802);
            plan = tCopulaQuantilePlan(U);
            wrong = tCopulaQuantileCache(testCase.NumRows + 1, [0, 0.3]);

            testCase.verifyEqual( ...
                tCopulaLoglik(0.3, plan, localRinfo(R), wrong), ...
                tCopulaLoglik(0.3, plan, localRinfo(R)), ...
                'a cache for a different n is bypassed, not applied.', AbsTol=0);
        end

        function fitIsIdenticalWithAndWithoutTheCache(testCase)
            opts = tFitDefaultOptions();
            p_grid = linspace(opts.p_bounds(1), opts.p_bounds(2), ...
                opts.profile_grid_size);
            cache = tCopulaQuantileCache(testCase.NumRows, p_grid);

            for p_true = [0, 0.25, 1.5]
                [U, R] = localSample(testCase, p_true, 8900 + round(100 * p_true));
                plain = fitTCopulaShape(U, R);
                cached = fitTCopulaShape(U, R, struct(), cache);
                label = sprintf('p_true = %g', p_true);

                testCase.verifyEqual(cached.p_hat, plain.p_hat, label, AbsTol=0);
                testCase.verifyEqual(cached.loglik, plain.loglik, label, AbsTol=0);
                testCase.verifyEqual(cached.loglik_gaussian, ...
                    plain.loglik_gaussian, label, AbsTol=0);
                testCase.verifyEqual(cached.profile_loglik_grid, ...
                    plain.profile_loglik_grid, label, AbsTol=0);
                testCase.verifyEqual(cached.numerical_ok, plain.numerical_ok, ...
                    label);
            end
        end
    end
end


function [U, R] = localSample(testCase, p_true, seed)
    % Pseudo-observations of an equicorrelated t-copula sample of the suite's size at shape p_true, with its true R.
    d = testCase.Dimension;
    R = (1 - testCase.EquiCorrelation) * eye(d) ...
        + testCase.EquiCorrelation * ones(d);
    U = localRanks(tCopulaSample(testCase.NumRows, d, R, p_true, seed));
end


function U = localRanks(X)
    n = size(X, 1);
    U = zeros(size(X));
    for c = 1:size(X, 2)
        U(:, c) = tiedrank(X(:, c));
    end
    U = U / (n + 1);
end


function Rinfo = localRinfo(R)
    % The Rinfo struct tCopulaLoglik takes: an upper-triangular factor with R = Lr' * Lr, which chol returns by default.
    Lr = chol(R);
    Rinfo = struct('Lr', Lr, 'logdetR', 2 * sum(log(diag(Lr))), 'd', size(R, 1));
end
