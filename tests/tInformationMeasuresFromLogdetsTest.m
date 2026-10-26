classdef tInformationMeasuresFromLogdetsTest < matlab.unittest.TestCase
    %TINFORMATIONMEASURESFROMLOGDETSTEST Tests that the log-determinant route equals the matrix route.
    %
    % Phase 4 reads the two log-determinants the Phase 2 unit row already carries instead of unpacking R and refactorising it once per unit, which is valid only if both routes are the same closed form.
    % The suite runs both entry points on fifty random (dimension, correlation matrix, shape) draws and requires every shared field to agree to 1e-10, seven orders of magnitude below the millinat at which the measures are reported.
    % The percentage policy has to agree as well: both denominators, their independent suppression state, the absolute floor and the O-information cancellation guard.
    % The suite uses synthetic inputs only and reads no recordings.

    properties (Constant)
        NumDraws = 50
        Tolerance = 1e-10
    end

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'tInformationMeasuresFromLogdetsTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function agreesWithTheMatrixRouteOnRandomUnits(testCase)
            stream = RandStream('twister', 'Seed', 20260922);
            dimensions = [2, 2, 3, 4, 7, 12, 20];
            shapes = [0, 1e-4, 0.01, 0.1, 0.5, 1, 2, 3];

            for draw = 1:testCase.NumDraws
                d = dimensions(randi(stream, numel(dimensions)));
                p = shapes(randi(stream, numel(shapes)));
                R = localRandomCorrelation(stream, d);
                G = gaussianInformationMeasures(R);
                testCase.assertTrue(G.valid, 'the draw must be a usable R.');

                actual = tInformationMeasuresFromLogdets(d, p, G.logdet_R, ...
                    sum(G.logdet_R_minor));
                expected = informationMeasuresT(R, p);

                localVerifySharedFields(testCase, actual, expected, ...
                    testCase.Tolerance, sprintf('draw %d (d = %d, p = %g)', ...
                    draw, d, p));
            end
        end

        function gaussianReferenceIsTheLogDeterminantForm(testCase)
            % The three Gaussian levels, read straight off the two stored log-determinants, against their definitions.
            R = localRandomCorrelation(RandStream('twister', 'Seed', 11), 6);
            G = gaussianInformationMeasures(R);

            out = tInformationMeasuresFromLogdets(6, 0.3, G.logdet_R, ...
                sum(G.logdet_R_minor));

            testCase.verifyEqual(out.TC_gauss, -0.5 * G.logdet_R, AbsTol=0);
            testCase.verifyEqual(out.DTC_gauss, ...
                0.5 * sum(G.logdet_R_minor) - 0.5 * 5 * G.logdet_R, AbsTol=1e-12);
            testCase.verifyEqual(out.O_gauss, out.TC_gauss - out.DTC_gauss, AbsTol=0);
            testCase.verifyTrue(isnan(out.MI_gauss), ...
                'mutual information is reported at d = 2 only.');
        end

        function pairIdentitiesHold(testCase)
            % At d = 2 the leave-one-out minors are 1 x 1, so their log-determinants are 0, DTC coincides with TC and O vanishes.
            % The Gaussian mutual information is the textbook -1/2 log(1 - rho^2).
            rho = 0.62;
            R = [1, rho; rho, 1];
            G = gaussianInformationMeasures(R);

            out = tInformationMeasuresFromLogdets(2, 0.4, G.logdet_R, ...
                sum(G.logdet_R_minor));

            testCase.verifyEqual(sum(G.logdet_R_minor), 0, AbsTol=1e-14);
            testCase.verifyEqual(out.MI_gauss, -0.5 * log(1 - rho^2), AbsTol=1e-12);
            testCase.verifyEqual(out.MI_t, out.TC_t, AbsTol=0);
            testCase.verifyEqual(out.DTC_gauss, out.TC_gauss, AbsTol=1e-12);
            testCase.verifyEqual(out.O_gauss, 0, AbsTol=1e-12);
            testCase.verifyEqual(out.O_t, 0, AbsTol=1e-12);
        end

        function gaussianShapeLeavesEveryDifferenceAtZero(testCase)
            R = localRandomCorrelation(RandStream('twister', 'Seed', 5), 9);
            G = gaussianInformationMeasures(R);

            out = tInformationMeasuresFromLogdets(9, 0, G.logdet_R, ...
                sum(G.logdet_R_minor));

            testCase.verifyEqual(out.TC_delta_nats, 0, AbsTol=0);
            testCase.verifyEqual(out.DTC_delta_nats, 0, AbsTol=0);
            testCase.verifyEqual(out.O_delta_nats, 0, AbsTol=0);
            testCase.verifyEqual(out.nu, Inf);
            testCase.verifyTrue(out.valid);
        end

        function aUnitWithNoFitFailsClosed(testCase)
            % Phase 4 passes a numerically failed unit in as p = NaN.
            % Every measure must come back NaN with status code 0, the code for "no percentage attempted", and the field set must stay the same, so rows from every dimension still concatenate.
            reference = tInformationMeasuresFromLogdets(5, 0.2, -1.4, -3.1);

            for out = [tInformationMeasuresFromLogdets(5, NaN, NaN, NaN), ...
                    tInformationMeasuresFromLogdets(5, 0.2, NaN, -3.1), ...
                    tInformationMeasuresFromLogdets(5, Inf, -1.4, -3.1)]
                testCase.verifyEqual(fieldnames(out), fieldnames(reference));
                testCase.verifyFalse(out.valid);
                for name = {'MI', 'TC', 'DTC', 'O'}
                    n = name{1};
                    testCase.verifyTrue(isnan(out.([n '_gauss'])));
                    testCase.verifyTrue(isnan(out.([n '_t'])));
                    testCase.verifyTrue(isnan(out.([n '_delta_nats'])));
                    testCase.verifyEqual(out.([n '_pct_status_code']), 0);
                    testCase.verifyEqual(out.([n '_pct_status_code_t']), 0);
                    testCase.verifyFalse(out.([n '_pct_reportable']));
                    testCase.verifyFalse(out.([n '_pct_reportable_t']));
                end
            end
        end

        function fieldSetIsConstantAcrossDimensions(testCase)
            names = fieldnames(tInformationMeasuresFromLogdets(2, 0.1, -0.3, 0));
            for d = [3, 8, 20]
                testCase.verifyEqual( ...
                    fieldnames(tInformationMeasuresFromLogdets(d, 0.1, -1.2, -2.0)), ...
                    names, sprintf('d = %d', d));
            end
        end

        function thresholdsAreConfigurable(testCase)
            % A tiny Gaussian denominator is suppressed with status 2 by default and reportable once the floor is lowered below it.
            weak = -2e-4;                     % logdet_R, so |TC_G| = 1e-4 nats
            default = tInformationMeasuresFromLogdets(2, 0.1, weak, 0);
            lowered = tInformationMeasuresFromLogdets(2, 0.1, weak, 0, ...
                struct('pct_min_abs_denominator', 1e-6));

            testCase.verifyEqual(default.TC_pct_status_code, 2);
            testCase.verifyTrue(isnan(default.TC_delta_pct));
            testCase.verifyEqual(lowered.TC_pct_status_code, 1);
            testCase.verifyEqual(lowered.TC_delta_pct, ...
                100 * lowered.TC_delta_nats / lowered.TC_gauss, AbsTol=1e-12);
        end
    end
end


function R = localRandomCorrelation(stream, d)
    % A random positive-definite correlation matrix: a Wishart-style Gram matrix rescaled to a unit diagonal, which keeps every leave-one-out minor usable.
    A = randn(stream, d, d + 5);
    S = A * A.';
    scale = sqrt(diag(S));
    R = S ./ (scale * scale.');
    R = (R + R.') / 2;
    R(1:d + 1:end) = 1;
end


function localVerifySharedFields(testCase, actual, expected, tolerance, label)
    % Every field of the log-determinant route against the same field of the matrix route.
    % informationMeasuresT also carries the model-free M_ksg fields, which this entry point leaves out because Phase 4 joins the model-free estimate itself, so those names are absent here rather than NaN.
    names = fieldnames(actual);
    for k = 1:numel(names)
        name = names{k};
        testCase.assertTrue(isfield(expected, name), sprintf( ...
            '%s: informationMeasuresT has no field %s.', label, name));
        left = actual.(name);
        right = expected.(name);
        if islogical(left) || ischar(left) || ~isfinite(left) || ~isfinite(right)
            % Flags, labels, NaN (no value) and Inf (nu at the Gaussian point) have to match exactly, because a tolerance on them means nothing.
            testCase.verifyEqual(left, right, sprintf('%s: %s', label, name));
        else
            testCase.verifyEqual(left, right, sprintf('%s: %s', label, name), ...
                AbsTol=tolerance, RelTol=tolerance);
        end
    end
end
