function reading = envelopeReading(score_acf, floor_level)
    % ENVELOPEREADING How the envelope of a series' autocorrelations is read, decided once from the score.
    %
    % The reading is decided from SCORE_ACF, the autocorrelation of the rank-normal scores, and fitEnvelopeTimescale applies it to every curve derived from the same series (the score itself, its square, and the square's Gaussian reference), so that all of them are read the same way:
    %   'peaks'    - the score dips below -FLOOR_LEVEL somewhere in its window, so the series oscillates (a band-pass filtered recording).
    %                Every curve is read at lag 0 and at the lags of the local maxima of |SCORE_ACF|, where the score sits on its envelope exactly when its half period P is a whole number of lags and exp(1/tau) cos(pi/P) <= 1, and a few per cent below it otherwise (fitEnvelopeTimescale).
    %                For a Gaussian series the autocorrelation of the square is the square of the score's, so at those lags it sits on the squared envelope, and the amplitude time of a narrow-band Gaussian signal is half its signal time in any band.
    %   'monotone' - the score never dips below -FLOOR_LEVEL (the reversible simulation model, or any decay without an oscillation).
    %                Every curve is read along itself and no peak is consulted, so a late excursion above the floor is never read as an envelope.
    %                A score that dips below -FLOOR_LEVEL without any local maximum of |SCORE_ACF| inside the window, an oscillation too slow to complete a half period in it, shows no peak to read and is read this way too.
    %
    % Deciding the mode per curve would read the square of a band-limited Gaussian signal in the other mode than the score, because the square's peaks are the squares of the score's and fewer of them clear the floor, and the ratio of the two times would then have no stable Gaussian value.
    %
    % The decision assumes that the estimation noise of the autocorrelation is well below FLOOR_LEVEL at every lag of the window.
    % That holds for the channel average, for every channel of the recordings analysed and for the model's pooled paths, but on a short, noisy, monotone series a noise dip below -FLOOR_LEVEL would read it as oscillating.
    %
    % Inputs:
    %   SCORE_ACF   - [L+1 x 1] autocorrelation of the rank-normal scores, 1 at lag 0, finite, at least 3 lags.
    %   FLOOR_LEVEL - scalar in (0, 1), the envelope floor of fitEnvelopeTimescale (default 0.05).
    %
    % Outputs:
    %   READING - struct with fields
    %     mode        - 'peaks' or 'monotone'.
    %     peak_index  - [m x 1] ascending indices into SCORE_ACF: 1 (lag 0) and, in peaks mode, every interior local maximum of |SCORE_ACF| (a flat top counts once, at its last sample); 1 alone in monotone mode.
    %     floor_level - FLOOR_LEVEL.

    arguments
        score_acf (:, 1) double {mustBeReal}
        floor_level (1, 1) double {mustBeGreaterThan(floor_level, 0), mustBeLessThan(floor_level, 1)} = 0.05
    end

    n = numel(score_acf);
    assert(n >= 3, 'envelopeReading:ShapeMismatch', ...
           'The score autocorrelation needs at least 3 lags; got %d.', n);
    assert(all(isfinite(score_acf)), 'envelopeReading:NonFiniteAcf', ...
           'The score autocorrelation contains %d non-finite values.', nnz(~isfinite(score_acf)));

    mode = 'monotone';
    peak_index = 1;
    if min(score_acf) < -floor_level
        magnitude = abs(score_acf);
        interior = (2:n - 1)';
        is_peak = false(n, 1);
        is_peak(interior) = magnitude(interior) >= magnitude(interior - 1) & ...
            magnitude(interior) > magnitude(interior + 1);
        if any(is_peak)
            mode = 'peaks';
            peak_index = [1; find(is_peak)];
        end
    end
    reading = struct('mode', mode, 'peak_index', peak_index, 'floor_level', floor_level);
end
