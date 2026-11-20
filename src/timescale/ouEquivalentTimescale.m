function tau = ouEquivalentTimescale(acf, lags)
    % OUEQUIVALENTTIMESCALE Time constant of the Ornstein-Uhlenbeck process with the same squared-ACF integral.
    %
    % Returns TAU = dt * (1 + 2 * sum(ACF(2:end) .^ 2)), with dt the lag step: twice the integral of the squared autocorrelation over the lags given, which is the time constant of an Ornstein-Uhlenbeck process whose squared autocorrelation has the same integral.
    % It is the memory that sets the effective sample size of second-order statistics such as Kendall's tau (the variance of a mean of products of the signal decays like 1 / (T / TAU)), and it needs no envelope.
    % An exponential exp(-lag / tau) returns tau up to the sampling of the lags, dt (1 + a^2) / (1 - a^2) with a = exp(-dt / tau).
    % A damped cosine returns about half its envelope time, because the oscillation decorrelates products faster than the envelope alone.
    % The sum is truncated at the last lag given.
    %
    % Inputs:
    %   ACF  - [L+1 x 1] finite autocorrelation at LAGS, 1 at lag 0.
    %   LAGS - [L+1 x 1] equally spaced lags starting at 0, at least 2 of them; TAU is returned in these units.

    arguments
        acf (:, 1) double {mustBeReal}
        lags (:, 1) double {mustBeReal, mustBeFinite}
    end

    assert(numel(acf) == numel(lags) && numel(lags) >= 2, 'ouEquivalentTimescale:ShapeMismatch', ...
           'ACF and LAGS must have the same length of at least 2; got %d and %d.', ...
           numel(acf), numel(lags));
    step = diff(lags);
    assert(lags(1) == 0 && all(step > 0) && max(abs(step - step(1))) <= 1e-9 * max(1, abs(step(1))), ...
           'ouEquivalentTimescale:InvalidLags', 'LAGS must start at 0 and be equally spaced.');
    assert(all(isfinite(acf)), 'ouEquivalentTimescale:NonFiniteAcf', ...
           'The autocorrelation contains %d non-finite values.', nnz(~isfinite(acf)));

    tau = step(1) * (1 + 2 * sum(acf(2:end).^2));
end
