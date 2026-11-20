function tau = integratedAutocorrelationTime(acf, lags)
    % INTEGRATEDAUTOCORRELATIONTIME Integrated autocorrelation time over a lag window.
    %
    % Returns TAU = dt * (1 + 2 * sum(ACF(2:end))), with dt the lag step: the integrated autocorrelation time of the series over the lags given.
    % TAU / dt is the factor by which serial dependence inflates the variance of a sample mean of the series, so the effective sample size of n samples is n dt / TAU.
    % Unlike an exponential fitted to the envelope, it counts the size and the duration of a slow component together: a component of height c that lasts over a whole window of length W adds about 2 c W to TAU, where a fitted time constant can come out the same whether the component lasts 100 tau or forever.
    % Applied to the autocorrelation of the squared rank-normal scores and to its Gaussian reference (the channel mean of the squared score autocorrelations, which it equals for Gaussian series), the ratio of the two is the integrated amplitude excess, 1 for a Gaussian signal at any window.
    % The sum is truncated at the last lag given, so for a curve still above its floor at the window's end TAU is a lower bound that grows with the window.
    %
    % Inputs:
    %   ACF  - [L+1 x 1] finite autocorrelation at LAGS, 1 at lag 0.
    %   LAGS - [L+1 x 1] equally spaced lags starting at 0, at least 2 of them; TAU is returned in these units.

    arguments
        acf (:, 1) double {mustBeReal}
        lags (:, 1) double {mustBeReal, mustBeFinite}
    end

    assert(numel(acf) == numel(lags) && numel(lags) >= 2, ...
           'integratedAutocorrelationTime:ShapeMismatch', ...
           'ACF and LAGS must have the same length of at least 2; got %d and %d.', ...
           numel(acf), numel(lags));
    step = diff(lags);
    assert(lags(1) == 0 && all(step > 0) && max(abs(step - step(1))) <= 1e-9 * max(1, abs(step(1))), ...
           'integratedAutocorrelationTime:InvalidLags', 'LAGS must start at 0 and be equally spaced.');
    assert(all(isfinite(acf)), 'integratedAutocorrelationTime:NonFiniteAcf', ...
           'The autocorrelation contains %d non-finite values.', nnz(~isfinite(acf)));

    tau = step(1) * (1 + 2 * sum(acf(2:end)));
end
