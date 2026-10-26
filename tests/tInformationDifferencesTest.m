classdef tInformationDifferencesTest < matlab.unittest.TestCase
    %TINFORMATIONDIFFERENCESTEST Tests of the closed-form t-minus-Gaussian information differences.
    %
    % tInformationDifferences gives the R-free part of the study's main quantity: how far each information measure of a t copula sits from the Gaussian value at the same correlation matrix.
    % The tests check the exact zero at p = 0, the entropy shape term the differences are built from, informationMeasuresT on random R (which carries the Gaussian level and so takes a different arithmetic path), and the second-order approach to the Gaussian point.

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'tInformationDifferencesTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function gaussianPointIsExactlyZero(testCase)
            % p = 0 is a branch of the shape term, not a limit, so every difference is an exact zero and no tolerance is allowed here.
            for d = [2, 3, 7, 20]
                diff = tInformationDifferences(d, 0);
                label = sprintf('d = %d', d);
                testCase.verifyEqual(diff.kappa_d, 0, label, AbsTol=0);
                testCase.verifyEqual(diff.kappa_dm1, 0, label, AbsTol=0);
                testCase.verifyEqual(diff.TC, 0, label, AbsTol=0);
                testCase.verifyEqual(diff.DTC, 0, label, AbsTol=0);
                testCase.verifyEqual(diff.O, 0, label, AbsTol=0);
                testCase.verifyEqual(diff.nu, Inf, label);
                if d == 2
                    testCase.verifyEqual(diff.MI, 0, label, AbsTol=0);
                else
                    testCase.verifyTrue(isnan(diff.MI), label);
                end
            end
        end

        function differencesAreTheEntropyShapeTerms(testCase)
            p = [0, 0.01, 0.2, 1, 3];
            for d = [2, 3, 9]
                diff = tInformationDifferences(d, p);
                ce = tCopulaEntropy(d, p);
                label = sprintf('d = %d', d);

                testCase.verifySize(diff.TC, size(p), label);
                testCase.verifyEqual(diff.kappa_d, ce.kappa, label, AbsTol=0);
                testCase.verifyEqual(diff.kappa_dm1, ce.kappa_margin, label, ...
                    AbsTol=0);
                testCase.verifyEqual(diff.TC, -ce.kappa, label, AbsTol=0);
                testCase.verifyEqual(diff.DTC, ...
                    d * ce.kappa_margin - (d - 1) * ce.kappa, label, ...
                    AbsTol=1e-14);
                testCase.verifyEqual(diff.O, diff.TC - diff.DTC, label, ...
                    AbsTol=0);
                testCase.verifyGreaterThanOrEqual(diff.TC, 0, ...
                    'kappa <= 0, so the TC difference is never negative.');
            end
            % kappa_t(1, nu) = h_1 - h_1 = 0, so a pair's margin term is an exact zero and DTC must coincide with TC there.
            pair = tInformationDifferences(2, p);
            testCase.verifyEqual(pair.kappa_dm1, zeros(size(p)), AbsTol=0);
            testCase.verifyEqual(pair.DTC, pair.TC, AbsTol=0);
            testCase.verifyEqual(pair.MI, pair.TC, AbsTol=0);
            testCase.verifyEqual(pair.O, zeros(size(p)), AbsTol=0);
        end

        function differencesAgreeWithTheFullMeasuresOnRandomCorrelations(testCase)
            % The differences are R-independent by construction.
            % The full-measure entry point reaches the same numbers by a different route, adding the Gaussian log-determinant level and subtracting it again, so agreement on a random R is a real cross-check, not a tautology.
            stream = RandStream('twister', 'Seed', 90210);
            for d = [2, 4, 8]
                R = localRandomCorrelation(d, stream);
                for p = [0.05, 0.4, 2.5]
                    diff = tInformationDifferences(d, p);
                    out = informationMeasuresT(R, p);
                    label = sprintf('d = %d, p = %g', d, p);

                    testCase.assertTrue(out.valid, label);
                    testCase.verifyEqual(diff.TC, out.TC_delta_nats, label, ...
                        AbsTol=1e-12);
                    testCase.verifyEqual(diff.DTC, out.DTC_delta_nats, label, ...
                        AbsTol=1e-12);
                    testCase.verifyEqual(diff.O, out.O_delta_nats, label, ...
                        AbsTol=1e-12);
                    if d == 2
                        testCase.verifyEqual(diff.MI, out.MI_delta_nats, ...
                            label, AbsTol=1e-12);
                    end
                end
            end
        end

        function theShapeTermIsSecondOrderAtTheGaussianPoint(testCase)
            % The Gaussian copula is a stationary point of the copula entropy along the t path, so kappa is O(p^2) and halving p must quarter it.
            % This is what turns a small p-hat into a much smaller difference, the reading the study rests on.
            for d = [2, 5, 20]
                p = 1e-3;
                ratio = tInformationDifferences(d, 2 * p).kappa_d ...
                    / tInformationDifferences(d, p).kappa_d;
                testCase.verifyEqual(ratio, 4, sprintf( ...
                    ['kappa(2p)/kappa(p) must approach 4 at d = %d; got ' ...
                    '%.6f at p = %g.'], d, ratio, p), RelTol=0.05);
            end
        end
    end
end


function R = localRandomCorrelation(d, stream)
    %LOCALRANDOMCORRELATION A well-conditioned random correlation matrix.
    A = randn(stream, d + 2, d);
    C = cov(A);
    s = sqrt(diag(C));
    R = C ./ (s * s.');
    R(1:(d + 1):end) = 1;
    R = (R + R.') / 2;
end
