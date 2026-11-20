function fit = fitEnvelopeTimescale(acf, lags, floor_level, reading)
    % FITENVELOPETIMESCALE Exponential decay time of the envelope of an autocorrelation function.
    %
    % Returns the time constant tau of the exponential exp(-lag / tau) that best matches, in least squares, the envelope of the autocorrelation ACF, read the way READING (envelopeReading) says.
    % READING is decided once, from the autocorrelation of the rank-normal scores, and applied to every curve of the same series (the score, its square, the square's Gaussian reference), so that all of them are read the same way:
    %   'peaks'    - the score oscillates, as a band-pass filtered recording does.
    %                ACF is read at lag 0 and at READING.peak_index, the lags where |score| has a local maximum, up to the first of them at which |ACF| falls below FLOOR_LEVEL, and never at fewer than lag 0 and the first peak.
    %   'monotone' - the score does not oscillate, as the model's reversible paths do not.
    %                ACF is read at every lag from 0 up to the last lag before |ACF| first falls below FLOOR_LEVEL, and never at fewer than lags 0 and 1; no peak is consulted.
    %
    % Reading the peaks makes the fit exact on a damped cosine exp(-lag/tau) cos(omega lag) whose half period P = pi/omega is a whole number of lags and for which exp(1/tau) cos(pi/P) <= 1 (tau and P in lags), since its peaks then fall on the lag-grid points where |cos| = 1 and so on its envelope.
    % Its square, read at the same lags, lies on the squared envelope.
    % Otherwise the peaks read lie below the envelope (the continuous local maxima lie where tan(omega lag) = -1/(omega tau), a factor omega tau / sqrt(1 + (omega tau)^2) below it), and tau comes out a few per cent low: 7% at omega tau = 2, 4% at 3, 1% at 5 and 0.5% at 12.
    % The Gaussian reference is read at the same lags, so a Gaussian signal's amplitude excess stays 1.
    % Stopping at the first point below the floor keeps a later excursion above it (a noise bump, or a slow residual oscillation after the envelope has decayed) out of the fit range.
    % Because the mode comes from the score's sign, excursions above the floor after a monotone decay cannot switch it to peaks mode.
    % The envelope is read off the autocorrelation itself, because a Hilbert envelope inflates a monotone exponential several-fold within a few time constants.
    %
    % The fitted exponential is pinned to 1 at lag 0.
    % An autocorrelation whose points after lag 0 lie on a slower envelope of smaller amplitude (a fast initial decay followed by a small slow oscillation) therefore gets a time constant between the two components', not either one; the rmse and the recorded envelope show when that happens.
    % ouEquivalentTimescale gives the memory that sets the effective sample size of second-order statistics instead.
    % An envelope that does not decay like an exponential over the fit range puts tau at a search bound, [LAGS(2) / 100, 100 LAGS(end)], and raises fitEnvelopeTimescale:BoundaryFit.
    %
    % Inputs:
    %   ACF         - [L+1 x 1] finite autocorrelation at LAGS, 1 at lag 0, at least 3 lags.
    %   LAGS        - [L+1 x 1] strictly increasing lags starting at 0; tau is returned in these units.
    %   FLOOR_LEVEL - scalar in (0, 1), the level at which the fit range ends (default 0.05).
    %   READING     - (optional) envelopeReading of the score autocorrelation of the same series, window and floor; by default the reading of ACF itself, which is the score's own reading when ACF is the score.
    %
    % Outputs:
    %   FIT - struct with fields
    %     tau           - fitted time constant, in the units of LAGS.
    %     mode          - 'peaks' or 'monotone', READING's.
    %     fit_lags      - [m x 1] the lags the fit reads.
    %     fit_values    - [m x 1] |ACF| at those lags.
    %     envelope      - [L+1 x 1] the envelope over the fit range (|ACF| in monotone mode, the straight lines joining the points read in peaks mode), NaN beyond it.
    %     fitted        - [L+1 x 1] exp(-LAGS / tau).
    %     last_lag      - the last lag the fit reads.
    %     n_points      - number of lags the fit reads.
    %     rmse          - root-mean-square residual at those lags.
    %     floor_reached - false when no point read (peaks mode) or value (monotone mode) falls below FLOOR_LEVEL inside LAGS, so the fit reads to the end of the window.
    %     method        - 'score_mode_peak_envelope'.

    arguments
        acf (:, 1) double {mustBeReal}
        lags (:, 1) double {mustBeReal, mustBeFinite}
        floor_level (1, 1) double {mustBeGreaterThan(floor_level, 0), mustBeLessThan(floor_level, 1)} = 0.05
        reading (1, 1) struct = struct()
    end

    n = numel(acf);
    assert(n == numel(lags) && n >= 3, 'fitEnvelopeTimescale:ShapeMismatch', ...
           'ACF and LAGS must have the same length of at least 3; got %d and %d.', n, numel(lags));
    assert(lags(1) == 0 && all(diff(lags) > 0), 'fitEnvelopeTimescale:InvalidLags', ...
           'LAGS must start at 0 and increase strictly.');
    assert(all(isfinite(acf)), 'fitEnvelopeTimescale:NonFiniteAcf', ...
           'The autocorrelation contains %d non-finite values.', nnz(~isfinite(acf)));
    if isempty(fieldnames(reading))
        reading = envelopeReading(acf, floor_level);
    end
    localCheckReading(reading, n, floor_level);

    magnitude = abs(acf);
    envelope = nan(n, 1);
    switch reading.mode
        case 'peaks'
            candidates = reading.peak_index(:);
            first_low = find(magnitude(candidates) < floor_level, 1, 'first');
            floor_reached = ~isempty(first_low);
            if floor_reached
                points = candidates(1:max(first_low - 1, 2));
            else
                points = candidates;
            end
            span = 1:points(end);
            envelope(span) = interp1(lags(points), magnitude(points), lags(span));
        case 'monotone'
            below = find(magnitude < floor_level, 1, 'first');
            floor_reached = ~isempty(below);
            if floor_reached
                last = max(below - 1, 2);
            else
                last = n;
            end
            points = (1:last)';
            envelope(points) = magnitude(points);
    end
    x = lags(points);
    y = magnitude(points);

    sse = @(log_tau) sum((y - exp(-x ./ exp(log_tau))).^2);
    lower = log(lags(2) / 100);
    upper = log(100 * lags(end));
    log_tau = fminbnd(sse, lower, upper, optimset('TolX', 1e-10));
    tau = exp(log_tau);
    assert(log_tau > lower + 1e-6 && log_tau < upper - 1e-6, 'fitEnvelopeTimescale:BoundaryFit', ...
           ['The fitted time constant %.4g sits at the search bound [%.4g, %.4g]; the envelope does ' ...
            'not decay like an exponential over the fit range.'], tau, exp(lower), exp(upper));

    fitted = exp(-lags ./ tau);
    fit = struct( ...
                 'tau', tau, ...
                 'mode', reading.mode, ...
                 'fit_lags', x, ...
                 'fit_values', y, ...
                 'envelope', envelope, ...
                 'fitted', fitted, ...
                 'last_lag', x(end), ...
                 'n_points', numel(points), ...
                 'rmse', sqrt(mean((y - fitted(points)).^2)), ...
                 'floor_reached', floor_reached, ...
                 'method', 'score_mode_peak_envelope');
end

function localCheckReading(reading, n, floor_level)
    % A reading must come from envelopeReading on a window of the same length and floor.
    assert(all(isfield(reading, {'mode', 'peak_index'})), 'fitEnvelopeTimescale:BadReading', ...
           'READING must carry mode and peak_index (envelopeReading).');
    assert(ismember(reading.mode, {'peaks', 'monotone'}), 'fitEnvelopeTimescale:BadReading', ...
           'READING.mode must be ''peaks'' or ''monotone''; got ''%s''.', char(string(reading.mode)));
    index = reading.peak_index(:);
    assert(~isempty(index) && index(1) == 1 && all(diff(index) > 0) && index(end) <= n && ...
           all(index == round(index)), 'fitEnvelopeTimescale:BadReading', ...
           ['READING.peak_index must be increasing indices from 1 (lag 0) inside the %d lags ' ...
            'of this autocorrelation.'], n);
    assert(~strcmp(reading.mode, 'peaks') || numel(index) >= 2, ...
           'fitEnvelopeTimescale:BadReading', ...
           'A peaks-mode READING needs at least one peak after lag 0.');
    if isfield(reading, 'floor_level')
        assert(reading.floor_level == floor_level, 'fitEnvelopeTimescale:BadReading', ...
               'READING was decided at the floor %g, but this fit uses the floor %g.', ...
               reading.floor_level, floor_level);
    end
end
