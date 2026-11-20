classdef coherentPhaseSurrogateTest < matlab.unittest.TestCase
    % COHERENTPHASESURROGATETEST Tests of the coherent phase-randomised surrogate generator.
    %
    % The surrogate defines the Gaussian-copula reference, so what it preserves is checked exactly: every column's periodogram, every cross-spectrum and every column mean.
    % That it never mixes rows across a trial boundary is checked by comparing a two-segment call with two one-segment calls on the same stream.
    % The suite uses synthetic inputs only and reads no recordings.

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'coherentPhaseSurrogateTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function surrogateIsRealForEvenAndOddLengths(testCase)
            % The conjugate mirror is what makes the inverse transform real.
            % An odd length has no Nyquist bin; an even length has one, and it must keep phase 0.
            % In the reference computed here, a broken mirror shows up as an imaginary part that is large relative to the signal, not as an error.
            for n = [256, 257]
                X = localProcess(n, 4, 11);
                stream = RandStream('twister', 'Seed', 5);
                spectrum = fft(X, [], 1);
                phase = localSharedPhase(n, RandStream('twister', 'Seed', 5));
                rotated = ifft(abs(spectrum) .* ...
                               exp(1i * (angle(spectrum) + phase)), [], 1);

                Xs = coherentPhaseSurrogate(X, n, stream);

                testCase.verifyEqual(Xs, real(rotated), sprintf( ...
                                                                'n = %d: the surrogate must be real(ifft(.)).', n), ...
                                     AbsTol = 1e-12);
                residual = max(abs(imag(rotated)), [], 'all') ...
                    / max(abs(X), [], 'all');
                testCase.verifyLessThan(residual, 1e-12, sprintf( ...
                                                                 ['n = %d: the rotated spectrum must be conjugate ' ...
                                                                  'symmetric, so its inverse transform is real; relative ' ...
                                                                  'imaginary residual %.3e.'], n, residual));
            end
        end

        function secondOrderStructureIsPreserved(testCase)
            % One shared phase per frequency leaves |X_a(f)| and X_a(f) conj(X_b(f)) untouched, so the periodogram, the cross-spectrum and, by Parseval's theorem, the covariance all survive.
            n = 512;
            d = 5;
            X = localProcess(n, d, 23);
            Xs = coherentPhaseSurrogate(X, n, RandStream('twister', 'Seed', 7));

            F = fft(X, [], 1);
            Fs = fft(Xs, [], 1);
            testCase.verifyEqual(abs(Fs), abs(F), ...
                                 'the per-channel periodogram must be preserved.', ...
                                 RelTol = 1e-10, AbsTol = 1e-9);

            for a = 1:d - 1
                for b = a + 1:d
                    cross = F(:, a) .* conj(F(:, b));
                    crossS = Fs(:, a) .* conj(Fs(:, b));
                    testCase.verifyEqual(crossS, cross, sprintf( ...
                                                                'the cross-spectrum of channels %d and %d must be preserved.', ...
                                                                a, b), RelTol = 1e-10, AbsTol = 1e-9);
                end
            end

            testCase.verifyEqual(mean(Xs, 1), mean(X, 1), ...
                                 'the DC bin keeps phase 0, so the column means are preserved.', ...
                                 AbsTol = 1e-12);
            testCase.verifyEqual(cov(Xs), cov(X), ...
                                 'Parseval then preserves the sample covariance.', ...
                                 AbsTol = 1e-9);
        end

        function noRowMixesAcrossASegmentBoundary(testCase)
            % Surrogating a concatenation must equal surrogating each segment in turn on the same stream.
            % If any row of segment 2 drew on segment 1, this equality would break, and so would the trial structure of the reference.
            n1 = 300;
            n2 = 201;
            X = localProcess(n1 + n2, 3, 31);

            joint = coherentPhaseSurrogate(X, [n1, n1 + n2], ...
                                           RandStream('twister', 'Seed', 19));

            stream = RandStream('twister', 'Seed', 19);
            first = coherentPhaseSurrogate(X(1:n1, :), n1, stream);
            second = coherentPhaseSurrogate(X(n1 + 1:end, :), n2, stream);

            testCase.verifyEqual(joint, [first; second], ...
                                 ['a segmented surrogate must equal the concatenation of the ' ...
                                  'per-segment surrogates drawn in the same stream order.'], ...
                                 AbsTol = 0);
            testCase.verifyEqual(mean(joint(1:n1, :), 1), ...
                                 mean(X(1:n1, :), 1), ...
                                 'each segment keeps its own column means.', AbsTol = 1e-12);
            testCase.verifyEqual(mean(joint(n1 + 1:end, :), 1), ...
                                 mean(X(n1 + 1:end, :), 1), ...
                                 'each segment keeps its own column means.', AbsTol = 1e-12);
        end

        function seedReproducesTheSurrogateAndTheGlobalStreamIsUntouched(testCase)
            n = 128;
            X = localProcess(n, 3, 41);
            stateBefore = rng;
            testCase.addTeardown(@() rng(stateBefore));

            first = coherentPhaseSurrogate(X, n, ...
                                           RandStream('twister', 'Seed', 2));
            again = coherentPhaseSurrogate(X, n, ...
                                           RandStream('twister', 'Seed', 2));
            other = coherentPhaseSurrogate(X, n, ...
                                           RandStream('twister', 'Seed', 3));

            testCase.verifyEqual(again, first, ...
                                 'one seed must reproduce a surrogate bit for bit.', AbsTol = 0);
            testCase.verifyNotEqual(other, first, ...
                                    'two seeds must not produce the same surrogate.');
            testCase.verifyEqual(rng, stateBefore, ...
                                 ['the surrogate must draw on the RandStream it is handed and ' ...
                                  'leave the global generator untouched.']);
        end

        function malformedInputIsRefusedWithANamedAssumption(testCase)
            n = 64;
            X = localProcess(n, 2, 5);
            stream = RandStream('twister', 'Seed', 1);

            testCase.verifyError( ...
                                 @() coherentPhaseSurrogate(X, [20, 50], stream), ...
                                 'coherentPhaseSurrogate:BoundaryRowCountMismatch', ...
                                 'the last boundary must be the row count of X.');
            testCase.verifyError( ...
                                 @() coherentPhaseSurrogate(X, [40, 20, n], stream), ...
                                 'coherentPhaseSurrogate:InvalidBoundaries');
            broken = X;
            broken(3, 1) = NaN;
            testCase.verifyError( ...
                                 @() coherentPhaseSurrogate(broken, n, stream), ...
                                 'coherentPhaseSurrogate:NonFiniteData');
        end

    end
end

function X = localProcess(n, d, seed)
    % LOCALPROCESS A serially and cross-correlated multivariate process.
    %
    % An AR(1) filter gives each column a non-flat spectrum, and a shared factor gives the columns a cross-spectrum, so preserving either is a real constraint rather than a property of white noise.
    stream = RandStream('twister', 'Seed', seed);
    innovations = randn(stream, n, d);
    factor = randn(stream, n, 1);
    X = filter(1, [1, -0.7], innovations + 0.8 * factor);
end

function phase = localSharedPhase(n, stream)
    % LOCALSHAREDPHASE The conjugate-symmetric phase column, written out here so the test does not call the implementation it checks.
    phase = zeros(n, 1);
    if mod(n, 2) == 0
        draw = 2 * pi * rand(stream, (n - 2) / 2, 1) - pi;
        phase(2:n) = [draw; 0; -flipud(draw)];
    else
        draw = 2 * pi * rand(stream, (n - 1) / 2, 1) - pi;
        phase(2:n) = [draw; -flipud(draw)];
    end
end
