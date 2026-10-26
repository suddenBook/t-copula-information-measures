classdef tStudyHelpersTest < matlab.unittest.TestCase
    %TSTUDYHELPERSTEST Tests of the Student-t study's supporting helpers.
    %
    % Covers informationMeasuresT (the full measures with their percentage policy), tCopulaTheoryGrid (the data-free reference table) and studySeed (the collision-free seed map).

    methods (TestClassSetup)
        function addSourceTree(testCase)
            % The fixture folder comes from this file's own location, not from repoRoot(), because repoRoot lives under src/ and the test must work with nothing on the path, the state a bare runtests on this single file starts from.
            % The assert makes a relocation out of <repo>/tests/ fail loudly rather than silently applying a fixture over a folder that does not exist.
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'tStudyHelpersTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    % =====================================================================
    % informationMeasuresT
    % =====================================================================
    methods (Test)
        function deltaAlgebraMatchesTheEntropyDecomposition(testCase)
            R = localExchangeable(4, 0.35);
            p = 0.2;

            out = informationMeasuresT(R, p);
            ce = tCopulaEntropy(4, p);

            testCase.verifyTrue(out.valid);
            testCase.verifyEqual(out.kappa, ce.kappa, AbsTol=0);
            testCase.verifyEqual(out.kappa_margin, ce.kappa_margin, AbsTol=0);
            testCase.verifyEqual(out.TC_delta_nats, -ce.kappa, AbsTol=1e-12);
            testCase.verifyEqual(out.DTC_delta_nats, ...
                4 * ce.kappa_margin - 3 * ce.kappa, AbsTol=1e-12);
            testCase.verifyEqual(out.O_delta_nats, ...
                out.TC_delta_nats - out.DTC_delta_nats, AbsTol=1e-12);
            testCase.verifyGreaterThan(out.TC_delta_nats, 0, ...
                'kappa < 0 for p > 0, so the TC difference must be positive.');
        end

        function gaussianPointLeavesEveryMeasureUntouched(testCase)
            R = localExchangeable(5, -0.12);

            out = informationMeasuresT(R, 0);
            G = gaussianInformationMeasures(R);

            testCase.verifyTrue(out.valid);
            testCase.verifyEqual(out.kappa, 0, AbsTol=0);
            testCase.verifyEqual(out.kappa_margin, 0, AbsTol=0);
            testCase.verifyEqual(out.nu, Inf);
            % TC and DTC equal the Gaussian values bit for bit at p = 0, because every shape term is exactly zero, so their differences are exact zeros rather than small numbers.
            testCase.verifyEqual(out.TC_t, G.TC, AbsTol=0);
            testCase.verifyEqual(out.DTC_t, G.DTC, AbsTol=0);
            testCase.verifyEqual(out.TC_delta_nats, 0, AbsTol=0);
            testCase.verifyEqual(out.DTC_delta_nats, 0, AbsTol=0);
            % O is the one exception, and not because of the t leg: O_t is formed as TC_t - DTC_t while O_gauss is formed from the log-determinants directly, so the two agree only to roundoff even though they are algebraically identical.
            testCase.verifyEqual(out.O_delta_nats, 0, AbsTol=1e-14);
        end

        function fieldNamesAndOrderSurviveTheInvalidPath(testCase)
            validOut = informationMeasuresT(localExchangeable(3, 0.25), 0.3);
            % Not positive definite, so the Gaussian leg fails closed and the function returns on the invalid path.
            invalidOut = informationMeasuresT([1, 2; 2, 1], 0.3);

            testCase.verifyFalse(invalidOut.valid);
            testCase.verifyEqual(fieldnames(invalidOut), fieldnames(validOut), ...
                'the invalid path must emit the same names in the same order');
        end

        function mutualInformationIsPopulatedOnlyForPairs(testCase)
            pairOut = informationMeasuresT([1, 0.4; 0.4, 1], 0.2);
            groupOut = informationMeasuresT(localExchangeable(4, 0.4), 0.2);

            testCase.verifyEqual(pairOut.MI_delta_nats, ...
                pairOut.TC_delta_nats, ...
                'MI = TC at d = 2, so their differences coincide.', AbsTol=1e-15);
            testCase.verifyEqual(pairOut.MI_gauss, pairOut.TC_gauss, AbsTol=1e-14);
            % The names exist at every d, which keeps the schema constant, but they carry no value above d = 2.
            testCase.verifyTrue(isfield(groupOut, 'MI_delta_nats'));
            testCase.verifyTrue(isnan(groupOut.MI_gauss));
            testCase.verifyTrue(isnan(groupOut.MI_t));
            testCase.verifyTrue(isnan(groupOut.MI_delta_nats));
            testCase.verifyEqual(groupOut.MI_pct_status_code, 0);
        end

        function shapeParameterIsReportedWithItsDegreesOfFreedom(testCase)
            out = informationMeasuresT([1, 0.5; 0.5, 1], 0.25);

            testCase.verifyEqual(out.p, 0.25);
            testCase.verifyEqual(out.nu, 4, AbsTol=1e-14);
            names = fieldnames(out);
            testCase.verifyEqual(names(1:3), {'d'; 'p'; 'nu'}, ...
                'nu is documented as sitting next to p');
        end

        function negativeShapeParameterIsRejected(testCase)
            testCase.verifyError(@() informationMeasuresT(eye(2), -0.1), ...
                ?MException);
        end

        function percentagePolicyIsUnchangedFromThePowerExponentialEntryPoint(testCase)
            % A near-independent unit: the Gaussian denominators fall below the 1e-3 nats floor, so both percentage families must suppress with status code 2 rather than divide by a meaningless baseline.
            out = informationMeasuresT(localExchangeable(3, 1e-6), 1e-4);

            testCase.verifyTrue(out.valid);
            testCase.verifyEqual(out.pct_min_abs_denominator_nats, 1e-3);
            testCase.verifyEqual(out.O_pct_min_cancellation_ratio, 0.1);
            testCase.verifyFalse(out.TC_pct_reportable);
            testCase.verifyEqual(out.TC_pct_status_code, 2);
            testCase.verifyFalse(out.TC_pct_reportable_t);
            testCase.verifyEqual(out.TC_pct_status_code_t, 2);
            testCase.verifyTrue(isnan(out.TC_delta_pct));
            testCase.verifyTrue(isnan(out.TC_delta_pct_t));
        end
    end

    % =====================================================================
    % tCopulaTheoryGrid
    % =====================================================================
    methods (Test)
        function theoryGridHasOneRowPerDimensionAndShapeValue(testCase)
            dList = [2, 3, 6];
            pGrid = [0, 0.05, 0.4, 1];

            T = tCopulaTheoryGrid(dList, pGrid);

            testCase.verifyEqual(height(T), numel(dList) * numel(pGrid));
            testCase.verifyEqual(T.Properties.VariableNames, ...
                {'d', 'p', 'nu', 'kappa_t', 'kappa_t_dm1', 'MI_delta_nats', ...
                'TC_delta_nats', 'DTC_delta_nats', 'O_delta_nats', ...
                'kappa_over_p2'});
            % d-major nesting: the whole p grid is emitted for each d in turn.
            testCase.verifyEqual(T.d.', repelem(dList, numel(pGrid)));
            testCase.verifyEqual(T.p.', repmat(pGrid, 1, numel(dList)));
            testCase.verifyEqual(T.nu, 1 ./ T.p);
        end

        function theoryGridSatisfiesTheDeltaIdentities(testCase)
            T = tCopulaTheoryGrid(2:8, [0, 0.01, 0.1, 0.3, 0.75, 1.5]);

            testCase.verifyEqual(T.O_delta_nats, ...
                T.TC_delta_nats - T.DTC_delta_nats, AbsTol=1e-12);
            testCase.verifyEqual(T.TC_delta_nats, -T.kappa_t, AbsTol=0);
            testCase.verifyEqual(T.DTC_delta_nats, ...
                T.d .* T.kappa_t_dm1 - (T.d - 1) .* T.kappa_t, AbsTol=1e-12);
            testCase.verifyLessThanOrEqual(T.kappa_t, 0, ...
                'Gaussian extremality: the shape term is never positive.');
            testCase.verifyEqual(T.kappa_t(T.p == 0), ...
                zeros(nnz(T.p == 0), 1), AbsTol=0);
        end

        function theoryGridWithholdsInapplicableAndUndefinedColumns(testCase)
            T = tCopulaTheoryGrid([2, 4], [0, 0.2]);

            isPair = T.d == 2;
            testCase.verifyEqual(T.MI_delta_nats(isPair), ...
                T.TC_delta_nats(isPair), AbsTol=0);
            testCase.verifyTrue(all(isnan(T.MI_delta_nats(~isPair))), ...
                'MI is a bivariate measure and must not be invented at d >= 3.');
            testCase.verifyEqual(T.kappa_t_dm1(isPair), ...
                zeros(nnz(isPair), 1), ...
                'the margin of a pair is univariate, so its shape term is 0', ...
                AbsTol=0);
            testCase.verifyTrue(all(isnan(T.kappa_over_p2(T.p == 0))), ...
                'kappa/p^2 is a genuine 0/0 at p = 0.');
            positive = T.p > 0;
            testCase.verifyEqual(T.kappa_over_p2(positive), ...
                T.kappa_t(positive) ./ T.p(positive) .^ 2, AbsTol=1e-14);
        end

        function theoryGridAgreesWithTheMeasureEntryPoint(testCase)
            % The deltas are R-independent, so the grid and the per-unit entry point have to agree for any admissible R.
            % Disagreement here means one of the two implements the decomposition wrongly.
            d = 5;
            p = 0.45;
            T = tCopulaTheoryGrid(d, p);
            out = informationMeasuresT(localExchangeable(d, 0.28), p);

            testCase.verifyEqual(T.TC_delta_nats, out.TC_delta_nats, AbsTol=1e-13);
            testCase.verifyEqual(T.DTC_delta_nats, out.DTC_delta_nats, AbsTol=1e-13);
            testCase.verifyEqual(T.O_delta_nats, out.O_delta_nats, AbsTol=1e-13);
        end
    end

    % =====================================================================
    % studySeed
    % =====================================================================
    methods (Test)
        function seedIsInjectiveAndInsideTheStreamRange(testCase)
            % A seed collision would silently correlate two cells the design declares independent, so injectivity is checked across namespaces as well as within one.
            sizes = [2, 3, 4, 5];
            namespaces = {'recovery', 'factorial', 'surrogate', 'check'};
            seeds = zeros(1, numel(namespaces) * prod(sizes));
            next = 0;
            for a = 1:numel(namespaces)
                for i1 = 1:sizes(1)
                    for i2 = 1:sizes(2)
                        for i3 = 1:sizes(3)
                            for i4 = 1:sizes(4)
                                next = next + 1;
                                seeds(next) = studySeed(namespaces{a}, ...
                                    [i1, i2, i3, i4], sizes);
                            end
                        end
                    end
                end
            end

            testCase.verifyNumElements(unique(seeds), numel(seeds), ...
                'studySeed collided within or across its namespaces.');
            testCase.verifyTrue(all(seeds < 2^32), ...
                'studySeed produced a seed at or above 2^32.');
            testCase.verifyTrue(all(seeds == round(seeds) & seeds >= 0), ...
                'studySeed produced a non-integer or negative seed.');
            testCase.verifyError(@() studySeed('not_a_namespace', 1, 1), ...
                'studySeed:UnknownNamespace');
            testCase.verifyError(@() studySeed('recovery', [1, 2], [1, 1]), ...
                'studySeed:IndexOutOfRange');
        end

        function seedDependsOnlyOnTheDeclaredGrid(testCase)
            % The seed is a pure function of the subscripts and the declared sizes, so a reduced run reproduces the full run's replicates.
            full = studySeed('recovery', [2, 3], [9, 1000]);
            testCase.verifyEqual(studySeed('recovery', [2, 3], [9, 1000]), full);
            testCase.verifyNotEqual(studySeed('recovery', [2, 3], [9, 10]), full);
        end
    end
end


% ===========================================================================
% Local helpers
% ===========================================================================
function R = localExchangeable(d, rho)
    %LOCALEXCHANGEABLE Exchangeable correlation matrix, the shape the theory uses.
    R = rho * ones(d) + (1 - rho) * eye(d);
end
