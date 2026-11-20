function time_tau = diffusionRadialTime(nu, D)
    % DIFFUSIONRADIALTIME Inverse radial L2 spectral gap of the Student-t diffusion, in tau.
    %
    % time_tau = diffusionRadialTime(nu, D) is 1/gamma_rad, the inverse radial spectral gap of the Student-t diffusion with the isotropic drift -X/tau at dimension D.
    % With m = nu + D - 2 it is
    %
    %   nu <= 4    4 m / nu^2          the continuous radial threshold
    %   nu >  4    m / (2 (nu - 2))    the first radial polynomial mode
    %   nu = Inf   1/2                 the OU squared-radius time
    %
    % It is the slowest correlation time of a bounded radial statistic and a property of the model.
    % At D = 20 it is 660 tau at nu = 1/3, 20 tau at nu = 2 and 7/4 tau at nu = 10.
    % An estimator's own precision depends on how much of its influence function sits on the slow radial modes, so this is context for a window T/tau, not a rule for it.
    %
    % Inputs:
    %   nu - degrees of freedom, positive or Inf, any shape.
    %   D  - dimension, integer >= 2.
    % Output:
    %   time_tau - same shape as nu.

    arguments
        nu double {mustBeReal, mustBeNonNan, mustBePositive}
        D (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(D, 2)}
    end

    m = nu + D - 2;
    time_tau = 4 * m ./ nu.^2;
    lighter = nu > 4;
    time_tau(lighter) = m(lighter) ./ (2 * (nu(lighter) - 2));
    time_tau(isinf(nu)) = 0.5;
end
