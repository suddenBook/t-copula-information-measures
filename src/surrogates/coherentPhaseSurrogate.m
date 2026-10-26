function Xs = coherentPhaseSurrogate(X, boundaries, stream)
    %COHERENTPHASESURROGATE Coherent multivariate phase-randomised surrogate.
    %
    % Xs = coherentPhaseSurrogate(X, boundaries, stream) returns a surrogate of the [samples x channels] matrix X that keeps every column's power spectrum (hence its autocorrelation) and every cross-spectrum between columns (hence the sample correlation matrix).
    % It adds one shared random, conjugate-symmetric phase per frequency to the phase spectrum of every column.
    % Because the added phase is shared across columns, X_a(f) conj(X_b(f)) is preserved exactly, so by Parseval's theorem the sample covariance, and therefore the correlation matrix, is preserved to rounding.
    % The DC bin gets no added phase, so the column means are preserved, and an even-length segment keeps its Nyquist bin real.
    % The surrogate is therefore a draw from a stationary Gaussian process with the full second-order structure of X.
    % Applied to the rank-normal scores norminv(tiedrank(x)/(n+1)) of a recording, which the caller computes, it is the Gaussian-copula reference with the same auto- and cross-spectra.
    %
    % The randomisation runs independently within each segment, so no surrogate row mixes data across a trial boundary.
    % boundaries holds the cumulative end index of each segment; a continuous recording is one segment, boundaries = n.
    % The stream is drawn from once per segment, in segment order, so surrogating a concatenation equals surrogating the segments one after another on the same stream.
    %
    % debt: a short trial randomises few phases (a 0.8 s trial at 1 kHz has about ten Fourier bins in the 12-25 Hz band, so surrogates of one trial are far from independent), and each segment is treated as periodic without end matching.
    % The upgrade path is an end-matched or windowed surrogate applied to longer segments.
    %
    % Inputs:
    %   X          - [n_samples x n_channels] real finite matrix, samples down the rows.
    %   boundaries - ascending cumulative end indices of the segments; the last must be n_samples.
    %   stream     - RandStream the phases are drawn from, so the surrogate is reproducible from its seed and the global generator is untouched.
    % Outputs:
    %   Xs - [n_samples x n_channels] surrogate, real by construction.

    arguments
        X (:, :) double {mustBeReal}
        boundaries (1, :) double
        stream (1, 1) RandStream
    end

    [n, d] = size(X);
    if n < 1 || d < 1
        error('coherentPhaseSurrogate:EmptyMatrix', ...
            ['X must be a non-empty [samples x channels] matrix; got ' ...
            '%d x %d.'], n, d);
    end
    if any(~isfinite(X), 'all')
        error('coherentPhaseSurrogate:NonFiniteData', ...
            ['X must be finite: the Fourier transform of a NaN or Inf row ' ...
            'contaminates every frequency of that column (X is %d x %d, ' ...
            '%d non-finite entries).'], n, d, nnz(~isfinite(X)));
    end
    if isempty(boundaries) || any(~isfinite(boundaries)) ...
            || any(boundaries ~= fix(boundaries)) || boundaries(1) < 1 ...
            || any(diff(boundaries) < 1)
        error('coherentPhaseSurrogate:InvalidBoundaries', ...
            ['boundaries must be strictly ascending positive integer ' ...
            'cumulative segment ends; got %s.'], mat2str(boundaries));
    end
    if boundaries(end) ~= n
        error('coherentPhaseSurrogate:BoundaryRowCountMismatch', ...
            ['The last segment boundary must be the row count of X: X is ' ...
            '%d x %d but boundaries end at %d.'], n, d, boundaries(end));
    end

    Xs = zeros(n, d);
    segment_start = 1;
    for s = 1:numel(boundaries)
        rows = segment_start:boundaries(s);
        Xs(rows, :) = localSurrogateSegment(X(rows, :), stream);
        segment_start = boundaries(s) + 1;
    end
end


function Ys = localSurrogateSegment(Y, stream)
    % One segment: one shared conjugate-symmetric random phase column added to every channel.
    % The DC bin gets phase 0 and, for an even length, so does the Nyquist bin, the only other bin that must stay real.
    m = size(Y, 1);
    Yf = fft(Y, [], 1);
    phase = zeros(m, 1);
    if mod(m, 2) == 0
        half = (m - 2) / 2;
        draw = 2 * pi * rand(stream, half, 1) - pi;
        phase(2:m) = [draw; 0; -flipud(draw)];
    else
        half = (m - 1) / 2;
        draw = 2 * pi * rand(stream, half, 1) - pi;
        phase(2:m) = [draw; -flipud(draw)];
    end
    Ws = ifft(abs(Yf) .* exp(1i * (angle(Yf) + phase)), [], 1);
    % With a conjugate-symmetric phase the inverse transform is real up to rounding.
    % A larger imaginary part means the phase vector is not conjugate symmetric, so it is an error rather than something to discard silently.
    imag_scale = max(abs(imag(Ws)), [], 'all');
    real_scale = max(abs(real(Ws)), [], 'all');
    if imag_scale > 1e-10 * max(real_scale, eps)
        error('coherentPhaseSurrogate:NonRealSurrogate', ...
            ['The surrogate of a %d-sample segment has an imaginary part of ' ...
            'size %.3g against a real part of size %.3g; the phase vector is ' ...
            'not conjugate symmetric.'], m, imag_scale, real_scale);
    end
    Ys = real(Ws);
end
