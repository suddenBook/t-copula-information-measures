classdef distributionOverlapTest < matlab.unittest.TestCase
    % DISTRIBUTIONOVERLAPTEST Tests of distributionOverlap, the overlap coefficient the surrogate comparison reports.
    %
    % OVL = sum over cells of min(P_a, P_b), with the exact values p = 0 and p = 3 as cells of their own and equal-count bins of the pooled non-atom values.
    % The suite checks the two ends of the scale (identical samples give 1, separated samples give 0), the atoms as cells of their own on hand-checkable samples, the equal-count property of the bins, the invariance under a strictly increasing map of the non-atom values, the per-sample normalisation of unequal sample sizes, and NaN and empty inputs.
    % The suite uses synthetic inputs only and reads no recordings.
    % Every sample comes from a private stream, so the global generator is untouched.

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'distributionOverlapTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function identicalSamplesOverlapCompletely(testCase)
            a = localShapeSample(RandStream('twister', 'Seed', 1), 400);

            [ovl, info] = distributionOverlap(a, a, 20, [0, 3]);

            testCase.verifyEqual(ovl, 1, AbsTol = 1e-12);
            testCase.verifyEqual(info.status, 'ok');
            testCase.verifyEqual(info.mass_a, info.mass_b, AbsTol = 0);
            testCase.verifyEqual(sum(info.mass_a), 1, AbsTol = 1e-12);
        end

        function separatedSamplesDoNotOverlap(testCase)
            stream = RandStream('twister', 'Seed', 2);
            low = 0.1 + 0.4 * rand(stream, 100, 1);
            high = 1 + rand(stream, 100, 1);

            % Equal sizes put the pooled median between the two samples, so no bin holds values of both.
            testCase.verifyEqual(distributionOverlap(low, high, 20, [0, 3]), 0, ...
                                 AbsTol = 0);
            % The two atoms alone separate a Gaussian sample from a heavy one.
            testCase.verifyEqual(distributionOverlap(zeros(5, 1), 3 * ones(7, 1), ...
                                                     20, [0, 3]), 0, AbsTol = 0);
        end

        function atomsAreCellsOfTheirOwn(testCase)
            % Half of a at the Gaussian atom, none of b: the atom contributes min(1/2, 0) = 0 and the two bins (edge 1.5) contribute 1/4 each.
            [ovl, info] = distributionOverlap([0; 0; 1; 2], [1; 2; 1; 2], 2, [0, 3]);
            testCase.verifyEqual(ovl, 0.5, AbsTol = 1e-12);
            testCase.verifyEqual(info.edges, 1.5, AbsTol = 1e-12);
            testCase.verifyEqual(info.mass_a, [0.5, 0, 0.25, 0.25], AbsTol = 1e-12);
            testCase.verifyEqual(info.mass_b, [0, 0, 0.5, 0.5], AbsTol = 1e-12);

            % The heavy atom p = 3: min(2/4, 1/4) on the atom plus 1/4 + 1/4 on the bins either side of the pooled median 2.
            [ovl, info] = distributionOverlap([3; 3; 1; 2], [3; 1; 2; 2], 2, [0, 3]);
            testCase.verifyEqual(ovl, 0.75, AbsTol = 1e-12);
            testCase.verifyEqual(info.mass_a(1:2), [0, 0.5], ...
                                 'the atom masses come first, in the order of info.atoms.', ...
                                 AbsTol = 1e-12);

            % A value beside an atom is not on it.
            testCase.verifyEqual(distributionOverlap([0; 0], [1e-12; 1e-12], ...
                                                     20, [0, 3]), 0, AbsTol = 0);
        end

        function allAtomSamplesAreCompared(testCase)
            [ovl, info] = distributionOverlap([0; 0; 3], [0; 3; 3], 20, [0, 3]);

            testCase.verifyEqual(ovl, 2 / 3, AbsTol = 1e-12);
            testCase.verifyEmpty(info.edges, ...
                                 'with no non-atom value there is no quantile to bin on.');
            testCase.verifyEqual(info.mass_a, [2 / 3, 1 / 3, 0], AbsTol = 1e-12);
        end

        function binsHoldEqualCountsOfThePooledSample(testCase)
            stream = RandStream('twister', 'Seed', 3);
            a = 3 * rand(stream, 1000, 1);
            b = 3 * rand(stream, 1000, 1).^2;

            [~, info] = distributionOverlap(a, b, 20, [0, 3]);

            pooled_counts = info.mass_a * numel(a) + info.mass_b * numel(b);
            testCase.verifyNumElements(info.edges, 19);
            testCase.verifyEqual(pooled_counts(1:2), [0, 0], AbsTol = 1e-9);
            testCase.verifyEqual(pooled_counts(3:end), 100 * ones(1, 20), ...
                                 'each bin must hold one twentieth of the pooled values.', ...
                                 AbsTol = 1e-9);
        end

        function overlapIsInvariantUnderAStrictlyIncreasingMap(testCase)
            stream = RandStream('twister', 'Seed', 4);
            a = localShapeSample(stream, 300);
            b = 1.4 * localShapeSample(stream, 250);
            b(b > 3) = 3;
            maps = {@(p) 3 * (p / 3).^2, @(p) 3 * (p / 3).^0.25, ...
                    @(p) log(p) - 20};

            [ovl, info] = distributionOverlap(a, b, 20, [0, 3]);
            for k = 1:numel(maps)
                [ovl_mapped, info_mapped] = distributionOverlap( ...
                                                                localMapNonAtoms(a, maps{k}), localMapNonAtoms(b, maps{k}), ...
                                                                20, [0, 3]);
                testCase.verifyEqual(ovl_mapped, ovl, sprintf( ...
                                                              'map %d moved the overlap.', k), AbsTol = 1e-12);
                testCase.verifyEqual(info_mapped.mass_a, info.mass_a, ...
                                     sprintf('map %d moved a cell mass.', k), AbsTol = 1e-12);
                testCase.verifyEqual(info_mapped.mass_b, info.mass_b, ...
                                     sprintf('map %d moved a cell mass.', k), AbsTol = 1e-12);
            end
        end

        function unequalSampleSizesAreComparedAsDistributions(testCase)
            % a = {0, 1}, b = {0, 0, 0, 1}: min(1/2, 3/4) on the atom and min(1/2, 1/4) on the one occupied bin.
            [ovl, info] = distributionOverlap([0; 1], [0; 0; 0; 1], 20, [0, 3]);
            testCase.verifyEqual(ovl, 0.75, AbsTol = 1e-12);
            testCase.verifyEqual([info.n_a, info.n_b], [2, 4]);

            stream = RandStream('twister', 'Seed', 5);
            small = localShapeSample(stream, 40);
            large = localShapeSample(stream, 4000);
            [ovl, info] = distributionOverlap(small, large, 20, [0, 3]);
            testCase.verifyEqual(sum(info.mass_a), 1, AbsTol = 1e-12);
            testCase.verifyEqual(sum(info.mass_b), 1, AbsTol = 1e-12);
            testCase.verifyGreaterThan(ovl, 0);
            testCase.verifyLessThanOrEqual(ovl, 1);
        end

        function oneValueSamplesAreCounted(testCase)
            % A sample of one value, such as a half of a dimension with one to three channel sets, is a distribution like any other.
            % Logical indexing of a single value gives a 0 x 0 index, which the cell count has to handle.
            testCase.verifyEqual(distributionOverlap(0, 0, 20, [0, 3]), 1);
            testCase.verifyEqual(distributionOverlap(0.5, 0.5, 20, [0, 3]), 1);
            testCase.verifyEqual(distributionOverlap(3, 0.5, 20, [0, 3]), 0);
            [ovl, info] = distributionOverlap(0.5, [0; 1], 20, [0, 3]);
            testCase.verifyGreaterThanOrEqual(ovl, 0);
            testCase.verifyLessThanOrEqual(ovl, 1);
            testCase.verifyEqual(sum(info.mass_a), 1, AbsTol = 1e-12);
            testCase.verifyEqual(sum(info.mass_b), 1, AbsTol = 1e-12);
        end

        function nonFiniteValuesAreDroppedAndCounted(testCase)
            [ovl, info] = distributionOverlap([NaN; 0; 1; Inf], [0; 1], 20, [0, 3]);

            testCase.verifyEqual(ovl, 1, AbsTol = 1e-12);
            testCase.verifyEqual([info.n_a, info.n_dropped_a], [2, 2]);
            testCase.verifyEqual([info.n_b, info.n_dropped_b], [2, 0]);
        end

        function anEmptySampleHasNoOverlap(testCase)
            [ovl, info] = distributionOverlap(zeros(0, 1), [0; 1], 20, [0, 3]);
            testCase.verifyTrue(isnan(ovl));
            testCase.verifyEqual(info.status, 'empty_a');
            testCase.verifyTrue(all(isnan(info.mass_a)));

            [ovl, info] = distributionOverlap([0; 1], [NaN; NaN], 20, [0, 3]);
            testCase.verifyTrue(isnan(ovl));
            testCase.verifyEqual(info.status, 'empty_b', ...
                                 'a sample of NaN only is an empty sample.');

            [ovl, info] = distributionOverlap([], [], 20, [0, 3]);
            testCase.verifyTrue(isnan(ovl));
            testCase.verifyEqual(info.status, 'empty_both');
        end

        function invalidArgumentsAreRefused(testCase)
            testCase.verifyError(@() distributionOverlap(1, 1, 0, [0, 3]), ...
                                 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() distributionOverlap(1, 1, 20, NaN), ...
                                 'MATLAB:validators:mustBeFinite');
        end

    end
end

function p = localShapeSample(stream, n)
    % A sample shaped like p_hat on Gaussian data: about half exactly 0, a positive body, and a few fits exactly at the heavy end p = 3.
    p = -log(rand(stream, n, 1)) * 0.05;
    p(rand(stream, n, 1) < 0.45) = 0;
    p(rand(stream, n, 1) < 0.02) = 3;
end

function mapped = localMapNonAtoms(p, map)
    % The map applied to the values that are not atoms; the atoms stay where they are.
    mapped = p;
    inside = p ~= 0 & p ~= 3;
    mapped(inside) = map(p(inside));
end
