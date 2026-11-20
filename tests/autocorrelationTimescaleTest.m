classdef autocorrelationTimescaleTest < matlab.unittest.TestCase
    % AUTOCORRELATIONTIMESCALETEST Tests of the time-scale estimators of Phase 6.
    %
    % averagedRankAutocorrelation and fitEnvelopeTimescale are the one estimator applied to the recordings (Phase 6) and to the simulated paths (Phase 7), so what they recover is pinned on processes whose time scale is known.
    % A Gaussian AR(1) has the rank-normal autocorrelation exp(-lag/tau), and its squared scores have the square of it, exp(-2 lag/tau).
    % A damped oscillation (an AR(2) with complex roots) has an envelope that decays as exp(-lag/tau) while the function itself changes sign.
    % Segment handling is checked against the definition written out here and by invariance to the order of the segments, on segments built so that a pair straddling a boundary would change the answer.
    % It is checked under both centrings: over the whole recording, and within each segment, which must equal the first on one segment and remove a level that differs between segments.
    % The integrated autocorrelation time is pinned on closed forms, and its amplitude excess on a Gaussian process, where it is 1.
    % The Gaussian reference calibrated for centring within segments is pinned on its brute-force expectation and on Gaussian trials of the LFP shape, whose within-trial excess it reads as 1.
    % Phase 6 itself, run on a synthetic one-segment recording in a temporary results root, must publish its within-trial columns equal to the headline.
    % Every draw comes from a private RandStream.
    % The suite uses synthetic inputs only and reads no recordings.

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'autocorrelationTimescaleTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function gaussianAr1TimeScalesAreRecovered(testCase)
            % The rank-normal scores of a Gaussian AR(1) are the process itself up to scale, so the score transform sees exp(-lag/tau).
            % For a Gaussian series the autocorrelation of Z.^2 is the square of that of Z, so the square transform sees exp(-2 lag/tau), a time of tau / 2.
            tau = 20;
            X = localAr1(4e5, 4, tau, 11);

            score = averagedRankAutocorrelation(X, [], 200, 'score');
            square = averagedRankAutocorrelation(X, [], 200, 'square');
            reading = envelopeReading(score.acf_mean, 0.05);
            signal = fitEnvelopeTimescale(score.acf_mean, score.lags, 0.05, reading);
            amplitude = fitEnvelopeTimescale(square.acf_mean, square.lags, 0.05, reading);
            reference = fitEnvelopeTimescale(score.acf_square_gaussian, score.lags, 0.05, ...
                                             reading);

            testCase.verifyEqual(reading.mode, 'monotone', ...
                                 'an AR(1) autocorrelation never dips below the negative floor.');
            testCase.verifyEqual(signal.tau, tau, sprintf( ...
                                                          'the score transform must recover tau = %g samples; got %.3f.', ...
                                                          tau, signal.tau), RelTol = 0.05);
            testCase.verifyEqual(amplitude.tau, tau / 2, sprintf( ...
                                                                 ['the square transform of a Gaussian AR(1) must recover tau / 2 = ' ...
                                                                  '%g samples; got %.3f.'], tau / 2, amplitude.tau), RelTol = 0.05);
            testCase.verifyEqual(amplitude.tau / reference.tau, 1, sprintf( ...
                                                                           ['a Gaussian series has no amplitude memory beyond its own ' ...
                                                                            'autocorrelation: excess 1, got %.3f.'], amplitude.tau / reference.tau), ...
                                 RelTol = 0.05);
            testCase.verifyEqual(score.n_valid_channels, 4, ...
                                 'every non-constant channel enters the average.');

            explicit = averagedRankAutocorrelation(X, size(X, 1), 200, 'score');
            testCase.verifyEqual(score.acf_by_channel, explicit.acf_by_channel, ...
                                 'segment ends [] must mean one segment of the whole recording.', ...
                                 AbsTol = 0);
        end

        function dampedOscillationEnvelopeIsRecovered(testCase)
            % A band-limited signal's autocorrelation oscillates, and its time scale is the decay of the envelope, not the first zero crossing.
            % The fit reads lag 0 and the peaks of |ACF|, which lie on the envelope (within 0.5% for this AR(2), whose phase offset is 84 degrees), so the 5% tolerance covers only the estimation noise of the autocorrelation.
            tau = 30;
            X = localDampedOscillation(4e5, 4, tau, 20, 13);

            score = averagedRankAutocorrelation(X, [], 300, 'score');
            fit = fitEnvelopeTimescale(score.acf_mean, score.lags, 0.05);

            testCase.verifyLessThan(min(score.acf_mean), -0.3, ...
                                    'the test process must oscillate through negative correlation.');
            testCase.verifyEqual(fit.mode, 'peaks', ...
                                 'an oscillating autocorrelation must be read at its peaks.');
            testCase.verifyEqual(fit.tau, tau, sprintf( ...
                                                       'the envelope decay tau = %g samples must be recovered; got %.3f.', ...
                                                       tau, fit.tau), RelTol = 0.05);
            testCase.verifyGreaterThanOrEqual(fit.n_points, 3);
            testCase.verifyTrue(fit.floor_reached);
            testCase.verifyEqual(fit.fit_values, abs(score.acf_mean(fit.fit_lags + 1)), ...
                                 'the fit must read |ACF| at its own lags.', AbsTol = 0);
            testCase.verifyTrue(all(isnan(fit.envelope(score.lags > fit.last_lag))), ...
                                'the envelope is only defined over the fit range.');
        end

        function lateExcursionIsLeftOutOfTheFit(testCase)
            % An excursion above the floor after the envelope has decayed (a noise bump, or a slow residual oscillation) must not extend the fit range.
            % The fit stops at the first peak (or value) below the floor, on a monotone decay and on an oscillation alike.
            lags = (0:399)';
            monotone = exp(-lags / 20);
            monotone(lags == 300) = 0.06;
            fit = fitEnvelopeTimescale(monotone, lags, 0.05);
            testCase.verifyEqual(fit.mode, 'monotone', ...
                                 'one isolated peak above the floor does not make an oscillation.');
            testCase.verifyEqual(fit.tau, 20, RelTol = 1e-6);
            testCase.verifyEqual(fit.last_lag, 59, ...
                                 'exp(-59/20) = 0.052 is the last value above the floor.');

            oscillating = exp(-lags / 30) .* cos(2 * pi * lags / 10);
            oscillating(lags == 350) = 0.08;
            fit = fitEnvelopeTimescale(oscillating, lags, 0.05);
            testCase.verifyEqual(fit.mode, 'peaks');
            testCase.verifyEqual(fit.tau, 30, ...
                                 ['the peaks of a damped cosine whose half period is a whole number of ' ...
                                  'lags (5) fall on the lag grid, on its envelope.'], RelTol = 1e-6);
            testCase.verifyLessThan(fit.last_lag, 350);
        end

        function twoLateSpikesDoNotMakeAMonotoneDecayOscillate(testCase)
            % A monotone decay with two excursions above the floor long after it has decayed.
            % Read as an envelope, lag 0 and the two spikes would give tau 102, five times the truth.
            % The mode comes from the score's sign, which never dips below -floor here, so no peak is read.
            lags = (0:399)';
            acf = exp(-lags / 20);
            acf(lags == 250) = 0.07;
            acf(lags == 320) = 0.07;

            reading = envelopeReading(acf, 0.05);
            fit = fitEnvelopeTimescale(acf, lags, 0.05, reading);
            testCase.verifyEqual(reading.mode, 'monotone');
            testCase.verifyEqual(reading.peak_index, 1, ...
                                 'a monotone reading consults no peak.');
            testCase.verifyEqual(fit.tau, 20, RelTol = 1e-6);
            testCase.verifyEqual(fit.last_lag, 59, ...
                                 'exp(-59/20) = 0.052 is the last value above the floor.');
        end

        function squareAndReferenceAreReadAtTheScorePeaks(testCase)
            % One reading for every curve of a series.
            % On an exact damped cosine the peaks of |score| lie on the envelope exp(-lag/30).
            % The square of the score, a Gaussian series' amplitude autocorrelation, read at the same lags lies on the squared envelope exp(-lag/15), so its time is exactly half, in any band.
            % With its own reading the square, which is never negative, would be read along itself, oscillation and all, in the other mode than the score.
            lags = (0:399)';
            score = exp(-lags / 30) .* cos(2 * pi * lags / 10);
            reading = envelopeReading(score, 0.05);
            testCase.verifyEqual(envelopeReading(score.^2, 0.05).mode, 'monotone', ...
                                 'the square alone never dips below the negative floor.');

            signal = fitEnvelopeTimescale(score, lags, 0.05, reading);
            amplitude = fitEnvelopeTimescale(score.^2, lags, 0.05, reading);
            testCase.verifyEqual(reading.mode, 'peaks');
            testCase.verifyEqual(lags(reading.peak_index(1:4)), [0; 5; 10; 15], ...
                                 'the peaks of |cos| at period 10 are every 5 lags.');
            testCase.verifyEqual(signal.tau, 30, RelTol = 1e-6);
            testCase.verifyEqual(amplitude.tau, 15, ...
                                 'the square read at the score peaks lies on exp(-2 lag / 30).', ...
                                 RelTol = 1e-6);
            testCase.verifyEqual(amplitude.fit_lags, ...
                                 lags(reading.peak_index(1:amplitude.n_points)), ...
                                 'the square is read at the score''s peak lags, in order.');
            testCase.verifyTrue(amplitude.floor_reached);
        end

        function gaussianOscillationHasNoAmplitudeExcess(testCase)
            % For a Gaussian band-limited process the autocorrelation of Z.^2 is the square of that of Z, channel by channel.
            % The square and its Gaussian reference therefore estimate the same curve, and their times agree to the estimation noise of the autocorrelations.
            X = localDampedOscillation(4e5, 4, 30, 20, 13);
            score = averagedRankAutocorrelation(X, [], 300, 'score');
            square = averagedRankAutocorrelation(X, [], 300, 'square');
            reading = envelopeReading(score.acf_mean, 0.05);

            amplitude = fitEnvelopeTimescale(square.acf_mean, square.lags, 0.05, reading);
            reference = fitEnvelopeTimescale(score.acf_square_gaussian, score.lags, ...
                                             0.05, reading);
            testCase.verifyEqual(reading.mode, 'peaks');
            testCase.verifyEqual(score.acf_square_gaussian, ...
                                 mean(score.acf_by_channel.^2, 2), ...
                                 'the reference is the channel mean of the squared autocorrelations.', ...
                                 AbsTol = 0);
            testCase.verifyEqual(amplitude.tau / reference.tau, 1, sprintf( ...
                                                                           'excess of a Gaussian oscillation must be 1; got %.3f.', ...
                                                                           amplitude.tau / reference.tau), RelTol = 0.05);
            testCase.verifyEqual(reference.tau, 15, ...
                                 'the reference time of this AR(2) is half its envelope time 30.', ...
                                 RelTol = 0.05);
        end

        function gaussianOscillationHasNoIntegratedAmplitudeExcess(testCase)
            % The integrated excess, the ratio of the integrated times of the square and of its Gaussian reference, is 1 for a Gaussian signal at any window, since the two curves estimate the same function.
            % It is checked over the first 150 lags and over all 300.
            X = localDampedOscillation(4e5, 4, 30, 20, 13);
            score = averagedRankAutocorrelation(X, [], 300, 'score');
            square = averagedRankAutocorrelation(X, [], 300, 'square');
            for window = [150, 300]
                span = 1:(window + 1);
                excess = integratedAutocorrelationTime(square.acf_mean(span), ...
                                                       square.lags(span)) / integratedAutocorrelationTime( ...
                                                                       score.acf_square_gaussian(span), score.lags(span));
                testCase.verifyEqual(excess, 1, sprintf( ...
                                                        ['the integrated excess of a Gaussian oscillation over %d lags ' ...
                                                         'must be 1; got %.4f.'], window, excess), RelTol = 0.05);
            end
        end

        function aSlowOscillationWithoutAPeakIsReadAlongItself(testCase)
            % A score that crosses below -floor but whose |ACF| has no local maximum inside the window has shown no peak, and is read along itself rather than refused.
            lags = (0:50)';
            reading = envelopeReading(cos(2 * pi * lags / 150), 0.05);
            testCase.verifyEqual(reading.mode, 'monotone');
            testCase.verifyEqual(reading.peak_index, 1);
        end

        function aMalformedReadingIsRefused(testCase)
            lags = (0:99)';
            acf = exp(-lags / 30) .* cos(2 * pi * lags / 10);
            reading = envelopeReading(acf, 0.05);
            testCase.verifyError(@() fitEnvelopeTimescale(acf, lags, 0.1, reading), ...
                                 'fitEnvelopeTimescale:BadReading', ...
                                 'a reading decided at another floor must be refused.');
            longer = envelopeReading([acf; acf(end) * ones(50, 1)], 0.05);
            longer.peak_index(end + 1) = 150;
            testCase.verifyError(@() fitEnvelopeTimescale(acf, lags, 0.05, longer), ...
                                 'fitEnvelopeTimescale:BadReading', ...
                                 'a reading of a longer window must be refused.');
            testCase.verifyEqual(fitEnvelopeTimescale(acf, lags, 0.05).tau, ...
                                 fitEnvelopeTimescale(acf, lags, 0.05, reading).tau, ...
                                 'the default reading is the curve''s own.');
        end

        function ouEquivalentTimeOfExponentialAndDampedCosine(testCase)
            % Twice the integral of the squared autocorrelation is tau for an exponential, up to the sampling: (1 + a^2) / (1 - a^2) with a = exp(-1/tau) at unit lag step.
            % For a damped cosine it is about half the envelope time, because the oscillation decorrelates products faster.
            lags = (0:2000)';
            a = exp(-1 / 20);
            testCase.verifyEqual(ouEquivalentTimescale(exp(-lags / 20), lags), ...
                                 (1 + a^2) / (1 - a^2), RelTol = 1e-9);
            testCase.verifyEqual(ouEquivalentTimescale(exp(-lags / 20), 0.004 * lags), ...
                                 0.004 * (1 + a^2) / (1 - a^2), ...
                                 'the time is returned in the units of the lags.', RelTol = 1e-9);
            damped = exp(-lags / 30) .* cos(2 * pi * lags / 10);
            testCase.verifyEqual(ouEquivalentTimescale(damped, lags), 15, RelTol = 0.02);
            testCase.verifyError(@() ouEquivalentTimescale(exp(-lags / 20), lags.^1.5), ...
                                 'ouEquivalentTimescale:InvalidLags');
        end

        function integratedTimeOfAnAr1CurveAndOfAPlateau(testCase)
            % The integrated time dt (1 + 2 sum over lags 1..L of the curve) has the closed form dt (1 + 2 a (1 - a^L) / (1 - a)) for a^lag.
            % A plateau of height c adds 2 c L dt, so the time grows with the window, by 2 c dt per lag once the exponential part has decayed.
            a = exp(-1 / 20);
            dt = 0.004;
            lags = (0:400)';
            testCase.verifyEqual(integratedAutocorrelationTime(a.^lags, dt * lags), ...
                                 dt * (1 + 2 * a * (1 - a^400) / (1 - a)), RelTol = 1e-12);

            c = 0.1;
            tau = zeros(1, 2);
            windows = [200, 400];
            for k = 1:2
                L = windows(k);
                plateau = (1 - c) * a.^(0:L)' + c;
                tau(k) = integratedAutocorrelationTime(plateau, dt * (0:L)');
                testCase.verifyEqual(tau(k), ...
                                     dt * (1 + 2 * (1 - c) * a * (1 - a^L) / (1 - a) + 2 * c * L), ...
                                     sprintf('the plateau closed form at a window of %d lags.', L), ...
                                     RelTol = 1e-12);
            end
            testCase.verifyEqual(tau(2) - tau(1), 2 * c * dt * 200, ...
                                 'the plateau adds 2 c dt per lag of window.', RelTol = 1e-3);
            testCase.verifyError( ...
                                 @() integratedAutocorrelationTime(a.^lags, lags.^1.5), ...
                                 'integratedAutocorrelationTime:InvalidLags');
        end

        function pairsNeverCrossASegmentBoundary(testCase)
            % Three segments at very different levels: after the whole-recording rank transform the first is high, the second low and the third in the middle, so a lagged product straddling a boundary is large and of the opposite sign to the products inside a segment.
            % The lag window reaches past the shortest segment, which then contributes no pairs at the longer lags.
            lengths = [300; 450; 250];
            ends = cumsum(lengths);
            starts = [1; ends(1:end - 1) + 1];
            levels = [10; -10; 0];
            stream = RandStream('twister', 'Seed', 17);
            X = zeros(ends(end), 2);
            for s = 1:numel(lengths)
                X(starts(s):ends(s), :) = levels(s) + ...
                    filter(1, [1, -0.8], randn(stream, lengths(s), 2));
            end
            max_lag = 280;
            order = [3; 1; 2];
            permuted = cell2mat(arrayfun(@(s) X(starts(s):ends(s), :), order, ...
                                         'UniformOutput', false));

            for transform = {'score', 'square'}
                for centring = {'recording', 'segment'}
                    name = transform{1};
                    label = [name, ', ', centring{1}, ' centring'];
                    out = averagedRankAutocorrelation(X, ends, max_lag, name, centring{1});

                    testCase.verifyEqual(out.pair_counts, ...
                                         sum(max(lengths.' - (0:max_lag).', 0), 2), sprintf( ...
                                                                           ['%s: the pairs at lag k must number the sum over segments ' ...
                                                                            'of max(L_s - k, 0).'], label));
                    testCase.verifyEqual(out.acf_by_channel, ...
                                         localDefinitionAcf(X, ends, max_lag, name, centring{1}), sprintf( ...
                                                                                         ['%s: the autocorrelation must equal the direct sum over the ' ...
                                                                                          'lagged pairs inside each segment.'], label), AbsTol = 1e-12);

                    reordered = averagedRankAutocorrelation(permuted, ...
                                                            cumsum(lengths(order)), max_lag, name, centring{1});
                    testCase.verifyEqual(reordered.acf_by_channel, out.acf_by_channel, ...
                                         sprintf(['%s: reordering the segments must not change the ' ...
                                 'autocorrelation, which it would if a pair crossed a ' ...
                                 'boundary.'], label), AbsTol = 1e-12);

                    joined = averagedRankAutocorrelation(X, [], max_lag, name);
                    testCase.verifyGreaterThan( ...
                                               max(abs(joined.acf_by_channel - out.acf_by_channel), [], 'all'), ...
                                               0.05, sprintf(['%s: the fixture must make the boundary ' ...
                                       'matter: treating the segments as one recording has to ' ...
                                       'change the answer.'], label));
                end
            end

            % The within-segment estimand: the segments' AR(1) has decayed by lag 50 (0.8^50 = 1e-5), but centred over the recording the three levels stay in every within-segment product as a constant, and centred within each segment they do not.
            long = (50:200)' + 1;
            recording = averagedRankAutocorrelation(X, ends, max_lag, 'score');
            segment = averagedRankAutocorrelation(X, ends, max_lag, 'score', 'segment');
            testCase.verifyGreaterThan(mean(recording.acf_mean(long)), 0.5, ...
                                       'centred over the recording, the segment levels read as memory.');
            testCase.verifyLessThan(abs(mean(segment.acf_mean(long))), 0.1, ...
                                    'centred within each segment, the levels are gone at lags 50-200.');
        end

        function segmentCentringIsRecordingCentringOnOneSegment(testCase)
            % With one segment, centring within it is centring over the recording, and the two must be the same computation, bit for bit, for both transforms and for the score's Gaussian reference.
            X = localAr1(5000, 3, 8, 21);
            for transform = {'score', 'square'}
                recording = averagedRankAutocorrelation(X, [], 150, transform{1});
                segment = averagedRankAutocorrelation(X, [], 150, transform{1}, 'segment');
                testCase.verifyEqual(segment, recording, sprintf( ...
                                                                 '%s: on one segment the two centrings must coincide.', transform{1}), ...
                                     AbsTol = 0);
            end
        end

        function segmentCentringRemovesTheBetweenTrialPower(testCase)
            % Trials whose power differs by a factor shared across channels (a scale mixture over trials, Gaussian within each) make the trial mean of Z.^2 differ between trials.
            % Centred over the recording, the square carries that between-trial variance as a constant at every within-trial lag, about its share of var(Z.^2).
            % Centred within each trial, it has decayed by lags 100-150 (20-30 tau) to the -1 / n_eff of a trial that averagedRankAutocorrelation describes, about -0.013 for 400 samples of a square whose integrated time is 5 samples.
            n_trials = 200;
            trial_length = 400;
            stream = RandStream('twister', 'Seed', 23);
            gain = repelem(exp(0.5 * randn(stream, n_trials, 1)), trial_length);
            X = gain .* filter(1, [1, -exp(-1 / 5)], randn(stream, n_trials * trial_length, 3));
            ends = (1:n_trials)' * trial_length;
            long = (100:150)' + 1;

            squared = norminv(tiedrank(X) ./ (size(X, 1) + 1)).^2;
            trial_means = squeeze(mean(reshape(squared, trial_length, n_trials, []), 1));
            share = mean(var(trial_means, 1, 1) ./ var(squared, 1, 1));
            testCase.assertGreaterThan(share, 0.1, ...
                                       'the fixture must give the trials clearly different powers.');

            recording = averagedRankAutocorrelation(X, ends, 150, 'square');
            segment = averagedRankAutocorrelation(X, ends, 150, 'square', 'segment');
            testCase.verifyEqual(mean(recording.acf_mean(long)), share, sprintf( ...
                                                                                ['centred over the recording, the square must carry the between-trial ' ...
                                                                                 'share of var(Z.^2), %.3f, at every within-trial lag; got %.3f.'], ...
                                                                                share, mean(recording.acf_mean(long))), RelTol = 0.2);
            testCase.verifyLessThan(abs(mean(segment.acf_mean(long))), 0.03, sprintf( ...
                                                                                     ['centred within each trial, the between-trial power must be gone; ' ...
                                                                                      'got %.3f at lags 100-150.'], mean(segment.acf_mean(long))));
        end

        function calibratedReferenceEqualsTheBruteForceExpectation(testCase)
            % The expected autocorrelation of the within-segment-centred square of a Gaussian series, written out from its definition.
            % Per segment of L samples, the covariance of y = Z.^2 - 1, 2 rho(u - v)^2, is centred on both sides by M = I - 1/L, and its k-th superdiagonal sums to the expected lag-k product sum, which is pooled over the segments longer than k, divided by their pairs, then by lag 0.
            % It is a direct matrix computation, so the test does not use the cumulative sums it audits.
            % The fixture has mixed and repeated lengths, a segment shorter than the window, and a constant (NaN) channel.
            lengths = [7; 9; 9; 12; 5];
            max_lag = 6;
            h = (0:max(lengths) - 1)';
            rho = [0.8.^h, exp(-h / 3) .* cos(2 * pi * h / 5), nan(size(h))];
            testCase.verifyEqual( ...
                                 segmentCentredGaussianReference(rho, cumsum(lengths), max_lag), ...
                                 localBruteForceReference(rho, lengths, max_lag), AbsTol = 1e-12);
            testCase.verifyError( ...
                                 @() segmentCentredGaussianReference(rho(1:11, :), cumsum(lengths), max_lag), ...
                                 'segmentCentredGaussianReference:AcfTooShort', ...
                                 'the score autocorrelation must reach one lag short of the longest segment.');
        end

        function calibratedReferenceReadsGaussianTrialsWithoutExcess(testCase)
            % Stationary Gaussian trials of the LFP shape (119 segments of 800-808 samples, a window of half the shortest, a narrow-band AR(2)) carry no amplitude memory beyond their own autocorrelation, so their within-trial fitted excess must be 1.
            % Centring within each trial pulls the square's long lags down to about -0.047.
            % The calibrated reference carries that pull and reads the excess as 0.996 (sd 0.011 over 24 seeds of this fixture), where the square of the likewise-centred score autocorrelation stays near 0 there and reads 0.885 (sd 0.011).
            % The tolerance 0.04 is about four of those standard deviations, and the tail tolerance 0.015 about four of the square tail's (0.004).
            lengths = 800 + mod((0:118)', 9);
            ends = cumsum(lengths);
            max_lag = floor(min(lengths) / 2);
            lags = (0:max_lag)';
            r = exp(-1 / 70);
            stream = RandStream('twister', 'Seed', 101);
            X = filter(1, [1, -2 * r * cos(2 * pi / 55), r^2], ...
                       randn(stream, 3000 + ends(end), 8));
            X = X(3001:end, :);

            score = averagedRankAutocorrelation(X, ends, max_lag, 'score');
            reading = envelopeReading(score.acf_mean, 0.05);
            square = averagedRankAutocorrelation(X, ends, max_lag, 'square', 'segment');
            headline = averagedRankAutocorrelation(X, ends, max(lengths) - 1, 'score');
            calibrated = mean(segmentCentredGaussianReference(headline.acf_by_channel, ...
                                                              ends, max_lag), 2);
            trial_score = averagedRankAutocorrelation(X, ends, max_lag, 'score', 'segment');

            amplitude = fitEnvelopeTimescale(square.acf_mean, lags, 0.05, reading);
            excess = amplitude.tau / fitEnvelopeTimescale(calibrated, lags, 0.05, reading).tau;
            uncalibrated = amplitude.tau / fitEnvelopeTimescale( ...
                                                                trial_score.acf_square_gaussian, lags, 0.05, reading).tau;
            testCase.verifyEqual(excess, 1, sprintf( ...
                                                    ['the within-trial excess of Gaussian trials against the calibrated ' ...
                                                     'reference must be 1; got %.4f.'], excess), RelTol = 0.04);
            testCase.verifyLessThan(uncalibrated, 0.96, sprintf( ...
                                                                ['the fixture must make the calibration matter: the uncalibrated ' ...
                                                                 'reference reads %.4f.'], uncalibrated));
            tail = (300:max_lag)' + 1;
            testCase.verifyEqual(mean(calibrated(tail)), mean(square.acf_mean(tail)), ...
                                 'the calibrated reference must carry the centring''s pull at long lags.', ...
                                 AbsTol = 0.015);
        end

        function oneSegmentRecordingPublishesItsHeadlineAsTheTrialSet(testCase)
            % Phase 6 on a synthetic one-segment recording in a temporary results root.
            % Centring within its one segment is the headline centring, so every _trialc column of the three tables must equal its headline column, with no calibrated reference and no withheld integrated time.
            % The phase opens a diary of its own and closes it, so a diary the caller had open is resumed afterwards.
            import matlab.unittest.fixtures.TemporaryFolderFixture
            import matlab.unittest.fixtures.EnvironmentVariableFixture
            root = testCase.applyFixture(TemporaryFolderFixture).Folder;
            base = fullfile(root, 'synthetic', 'S1', 'rest');
            mkdir(fullfile(base, 'Phase1', 'mat'));
            X_filtered = localDampedOscillation(20000, 3, 30, 20, 5);
            fs = 100;
            channel_ids = 1:3;
            save(fullfile(base, 'Phase1', 'mat', 'preprocessed_data.mat'), ...
                 'X_filtered', 'fs', 'channel_ids');
            writelines(jsonencode(struct('dataset_key', 'synthetic/S1', 'seed_base', 1, ...
                                         'phase1_mat', 'Phase1/mat/preprocessed_data.mat', 'fs', fs, ...
                                         'n_samples', 20000, 'n_channels_retained', 3, 'n_segments', 1, ...
                                         'modality', 'synthetic', 'settings_identity', 'synthetic', ...
                                         'phase1_schema_version', 'synthetic')), fullfile(base, 'recording.json'));
            environment = {'TC_RESULTS_ROOT', root; 'TC_RENDER_FIGURES', '0'; ...
                           'TC_OVERWRITE', '0'; 'TC_RECORDINGS', ''; 'TC_RECORDING_INDEX', ''};
            for k = 1:size(environment, 1)
                testCase.applyFixture(EnvironmentVariableFixture(environment{k, :}));
            end
            diary_was_on = strcmp(get(0, 'Diary'), 'on');
            diary_file = get(0, 'DiaryFile');
            testCase.addTeardown(@() localResumeDiary(diary_was_on, diary_file));

            localRunScript(fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                                    'scripts', 'Phase6_autocorrelation_timescale.m'));
            for name = {'timescale_summary.csv', 'acf_curves.csv', 'timescale_by_channel.csv'}
                published = readtable(fullfile(base, 'Phase6', 'text', name{1}), ...
                                      'TextType', 'string');
                columns = published.Properties.VariableNames;
                twins = columns(endsWith(columns, '_trialc'));
                testCase.verifyNotEmpty(twins, sprintf('%s carries no _trialc column.', name{1}));
                for twin = twins
                    testCase.verifyTrue(isequaln(published.(twin{1}), ...
                                                 published.(erase(twin{1}, '_trialc'))), sprintf( ...
                                                                        '%s: %s must equal its headline column on one segment.', ...
                                                                        name{1}, twin{1}));
                end
            end
        end

        function malformedInputIsRefusedWithANamedAssumption(testCase)
            X = localAr1(200, 2, 5, 3);

            testCase.verifyError( ...
                                 @() averagedRankAutocorrelation(X, [120; 80; 200], 10, 'score'), ...
                                 'averagedRankAutocorrelation:InvalidSegments', ...
                                 'segment ends must increase strictly.');
            testCase.verifyError( ...
                                 @() averagedRankAutocorrelation(X, [100; 199], 10, 'score'), ...
                                 'averagedRankAutocorrelation:InvalidSegments', ...
                                 'the last segment end must be the sample count.');
            testCase.verifyError( ...
                                 @() averagedRankAutocorrelation(X, [100.5; 200], 10, 'score'), ...
                                 'averagedRankAutocorrelation:InvalidSegments', ...
                                 'segment ends must be integers.');
            testCase.verifyError( ...
                                 @() averagedRankAutocorrelation(X, [0; 200], 10, 'score'), ...
                                 'averagedRankAutocorrelation:InvalidSegments', ...
                                 'a segment end must be a sample index, 1 or more.');
            testCase.verifyError( ...
                                 @() averagedRankAutocorrelation(X, [150; 200], 150, 'score'), ...
                                 'averagedRankAutocorrelation:LagTooLong', ...
                                 'the lag window must be shorter than the longest segment.');
            testCase.verifyError( ...
                                 @() averagedRankAutocorrelation(ones(200, 3), [], 10, 'score'), ...
                                 'averagedRankAutocorrelation:AllChannelsConstant', ...
                                 'with every channel constant there is no autocorrelation to average.');
        end

        function envelopeFitIsExactOnAnExactExponential(testCase)
            % A monotone exponential is its own envelope, so the least-squares fit must return its time constant to the optimiser's tolerance, in the units of the lags it is given.
            lags = (0:200)';
            tau = 12.5;
            acf = exp(-lags / tau);

            fit = fitEnvelopeTimescale(acf, lags, 0.05);
            testCase.verifyEqual(fit.tau, tau, RelTol = 1e-6);
            testCase.verifyEqual(fit.mode, 'monotone');
            testCase.verifyEqual(fit.envelope(1:38), acf(1:38), ...
                                 'a monotone decay is its own envelope over the fit range.', AbsTol = 0);
            testCase.verifyTrue(all(isnan(fit.envelope(39:end))), ...
                                'the envelope is only defined over the fit range.');
            testCase.verifyTrue(fit.floor_reached);
            testCase.verifyEqual(fit.last_lag, 37, ...
                                 ['the fit range ends at the last lag at or above the floor: ' ...
                                  'exp(-37/12.5) = 0.052 and exp(-38/12.5) = 0.048.']);
            testCase.verifyEqual(fit.n_points, 38);

            fs = 250;
            seconds = fitEnvelopeTimescale(acf, lags / fs, 0.05);
            testCase.verifyEqual(seconds.tau, tau / fs, ...
                                 'tau is returned in the units of the lags.', RelTol = 1e-6);

            slow_lags = (0:50)';
            slow = fitEnvelopeTimescale(exp(-slow_lags / 100), slow_lags, 0.05);
            testCase.verifyFalse(slow.floor_reached, ...
                                 'an envelope that stays above the floor must say so.');
            testCase.verifyEqual(slow.n_points, numel(slow_lags), ...
                                 'without a floor crossing every lag is in the fit range.');
            testCase.verifyEqual(slow.last_lag, 50);
            testCase.verifyEqual(slow.tau, 100, RelTol = 1e-6);

            testCase.verifyError( ...
                                 @() fitEnvelopeTimescale(ones(51, 1), slow_lags, 0.05), ...
                                 'fitEnvelopeTimescale:BoundaryFit', ...
                                 'a flat envelope has no finite decay time and must be refused.');
        end

    end
end

function X = localAr1(n, n_channels, tau, seed)
    % LOCALAR1 Independent Gaussian AR(1) channels with autocorrelation exp(-lag/tau).
    stream = RandStream('twister', 'Seed', seed);
    X = filter(1, [1, -exp(-1 / tau)], randn(stream, n, n_channels));
end

function X = localDampedOscillation(n, n_channels, tau, period, seed)
    % LOCALDAMPEDOSCILLATION Independent AR(2) channels with complex roots r e^(+-iw).
    %
    % The autocorrelation is r^lag sin(w lag + psi) / sin(psi), an oscillation of PERIOD samples under the envelope exp(-lag / tau), with r = exp(-1 / tau).
    r = exp(-1 / tau);
    w = 2 * pi / period;
    stream = RandStream('twister', 'Seed', seed);
    X = filter(1, [1, -2 * r * cos(w), r^2], randn(stream, n, n_channels));
end

function reference = localBruteForceReference(rho, lengths, max_lag)
    % LOCALBRUTEFORCEREFERENCE The calibrated Gaussian reference from its definition.
    %
    % Per segment, the covariance matrix of y = Z.^2 - 1, 2 rho(u - v)^2, is centred on both sides by M = I - 1/L, and the sum of its k-th superdiagonal is the expected lag-k product sum of the centred series.
    % The sums are pooled over the segments longer than k, divided by their pairs, then by the lag-0 value.
    n_channels = size(rho, 2);
    sums = zeros(max_lag + 1, n_channels);
    pairs = zeros(max_lag + 1, 1);
    for s = 1:numel(lengths)
        L = lengths(s);
        M = eye(L) - ones(L) / L;
        lag_index = abs((1:L)' - (1:L)) + 1;
        for c = 1:n_channels
            channel = rho(:, c);
            centred = M * (2 * channel(lag_index).^2) * M;
            for k = 0:min(max_lag, L - 1)
                sums(k + 1, c) = sums(k + 1, c) + sum(diag(centred, k));
            end
        end
        n_used = min(max_lag, L - 1) + 1;
        pairs(1:n_used) = pairs(1:n_used) + (L - (0:n_used - 1))';
    end
    covariance = sums ./ pairs;
    reference = covariance ./ covariance(1, :);
end

function localRunScript(path)
    % LOCALRUNSCRIPT Run a phase script in a workspace of its own.
    run(path);
end

function localResumeDiary(was_on, file)
    % LOCALRESUMEDIARY Resume the caller's diary, which a phase run switches off.
    if was_on
        diary(file);
    end
end

function acf = localDefinitionAcf(X, segment_ends, max_lag, transform, centring)
    % LOCALDEFINITIONACF The segmented autocorrelation written out from its definition.
    %
    % Rank-normal scores over the whole recording (squared for 'square') are centred by their whole-recording mean ('recording') or by their mean within each segment ('segment').
    % At each lag the products of the pairs lying inside one segment are summed over segments and divided by their number, then by the lag-0 value.
    % It is a direct time-domain sum, so the test does not call the FFT implementation it audits.
    n = size(X, 1);
    S = norminv(tiedrank(X) ./ (n + 1));
    if strcmp(transform, 'square')
        S = S.^2;
    end
    starts = [1; segment_ends(1:end - 1) + 1];
    if strcmp(centring, 'segment')
        for s = 1:numel(segment_ends)
            rows = starts(s):segment_ends(s);
            S(rows, :) = S(rows, :) - mean(S(rows, :), 1);
        end
    else
        S = S - mean(S, 1);
    end
    sums = zeros(max_lag + 1, size(X, 2));
    counts = zeros(max_lag + 1, 1);
    for s = 1:numel(segment_ends)
        for k = 0:max_lag
            last = segment_ends(s) - k;
            if last < starts(s)
                continue
            end
            sums(k + 1, :) = sums(k + 1, :) + ...
                sum(S(starts(s):last, :) .* S(starts(s) + k:last + k, :), 1);
            counts(k + 1) = counts(k + 1) + (last - starts(s) + 1);
        end
    end
    autocov = sums ./ counts;
    acf = autocov ./ autocov(1, :);
end
