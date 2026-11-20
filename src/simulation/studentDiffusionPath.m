function [X, info] = studentDiffusionPath(R, nu, n_obs, samples_per_tau, refine, stream, options)
    % STUDENTDIFFUSIONPATH One observed path of the Student-t diffusion, in units of tau.
    %
    % [X, info] = studentDiffusionPath(R, nu, n_obs, samples_per_tau, refine, stream) simulates the stochastic differential equation below and returns it on an observation grid.
    %
    %   dX = -X dt + sqrt(2 (nu + X' R^-1 X) / (nu + D - 2)) R^(1/2) dW
    %
    % For D >= 2 and every nu > 0 its stationary law is t_nu(0, R), the multivariate Student t with scatter R (a Fokker-Planck calculation).
    % Time is in units of the drift time scale tau (tau = 1): the process is exactly self-similar in time, so a study at tau = 1 covers every tau through T/tau and the number of samples per tau.
    %
    % The scheme runs in whitened coordinates y = L^-1 X with L = chol(R, 'lower'), and every observation is X = (L y)'.
    %
    % At nu = Inf the process is the Ornstein-Uhlenbeck limit, simulated exactly at any interval.
    % Its AR(1) skeleton y(j+1) = a y(j) + sqrt(1 - a^2) xi with a = exp(-Delta_step) runs at Delta_step (below) and keeps every k-th state, so with k = 1 it runs at the observation interval Delta = 1/samples_per_tau itself.
    % It starts from N(0, I), and refine is not used.
    %
    % At finite nu each observation interval is ns substeps of h = Delta/ns, computed as Delta_step/(ns/k) (nested rates, below).
    % Each substep proposes the exponential-Euler update with the diffusion multiplier frozen over the step:
    %
    %   z = a_h y + sv r0 zeta,   a_h = exp(-h),   sv = sqrt(-expm1(-2h)/m),
    %   r0 = sqrt(nu + |y|^2),    m = nu + D - 2,  zeta ~ N(0, I).
    %
    % The proposal is accepted when log U < min(0, log alpha), with U uniform on (0, 1) and alpha the Student target times the reverse proposal density over their forward counterparts:
    %
    %   log alpha = -(nu+D)(log r1 - log r0) - D log r1 - |rev|^2/2 + D log r0 + |zeta|^2/2,
    %   rev = (y - a_h z)/(sv r1),   r1 = sqrt(nu + |z|^2).
    %
    % A rejection keeps y and still consumes the substep.
    % Detailed balance keeps t_nu(0, I) exactly invariant at every h, which the proposal on its own does not (the frozen option, below).
    % The dynamics are still an approximation that runs slightly slow at a coarse step, which the refine factor controls through the substep count:
    %
    %   ns = refine * max(1, ceil(Delta_step/delta0)) * k,
    %   delta0 = 0.005 min(1, m/(D + 2), nu/(1 + nu)),
    %   Delta_step = 1/step_samples_per_tau,
    %   k = step_samples_per_tau / samples_per_tau (1 unless options.step_samples_per_tau is given).
    %
    % The ceiling is taken with a 1e-9 slack, so a quotient that is an exact integer in exact arithmetic does not gain a substep from rounding.
    % With k = 1 this is the rule on the observation interval itself.
    % The start is an exact stationary draw, y0 = g sqrt(nu/V) with g ~ N(0, I) and one V = 2 gammaincinv(u, nu/2) ~ chi-square(nu) for the whole vector (tCopulaSample's construction, with u clipped as it clips it), so no burn-in is needed and every observation is marginally t_nu(0, R).
    %
    % Nested rates: options.step_samples_per_tau fixes the substep on a grid finer than the observations, of which the observation interval is k whole intervals.
    % The substep h = Delta_step / (refine * max(1, ceil(Delta_step/delta0))) is then the same at every rate that divides step_samples_per_tau, and because it is computed from Delta_step it is the same double (Delta/ns can differ from it by one ulp when k is not a power of two).
    % The OU limit then runs its exact AR(1) at Delta_step and keeps every k-th state, with substeps = k and h = Delta_step.
    % The substep draws come from the stream in the same order whatever the rate, so two calls with one seed, one step_samples_per_tau and rates s and s' = s/k' observe one path.
    % The coarse path is exactly the fine one read every k' observations: the whitened states bit for bit, and the coloured observations to rounding (below).
    %
    % Every random number comes from stream, and the global generator is never touched.
    % The substep draws are taken in fixed blocks of randn(stream, D, 4096) and rand(stream, 1, 4096), and the OU limit draws randn(stream, D, 4096 k), the steps of 4096 observation intervals, which are the same draws in the same order as one call over the window.
    % So a seed determines the path, and two calls that differ only in n_obs share their common prefix of whitened states.
    % The coloured observations X = (L y)' then agree to rounding (about 1e-14 relative), not bit for bit, because the matrix product may be evaluated in a different order for a different count.
    %
    % Nothing is clipped: a non-finite start, acceptance ratio or observation is an error naming where it happened, because a clipped radius is a silently wrong radial law.
    %
    % With options.frozen = true every proposal is accepted, which gives the uncorrected frozen-multiplier chain.
    % That chain has the wrong radial law, and at D = 2, nu = 1/3 no invariant law at all for h > 0.0548.
    % It serves only as the negative control of the simulator check: it runs on the same draws as the corrected chain (the acceptance uniforms are drawn and ignored), and a non-finite state is returned as it is instead of raising, so a diverging control is counted.
    %
    % Inputs:
    %   R               - [D x D] correlation matrix (symmetric, unit diagonal, positive definite), D >= 2: the scatter of the stationary law.
    %   nu              - degrees of freedom, positive, or Inf for the Gaussian limit.
    %   n_obs           - number of observations, >= 1; row k of X is time (k-1)/samples_per_tau.
    %   samples_per_tau - observations per unit of tau (Delta = 1/samples_per_tau).
    %   refine          - positive integer multiplying the substep count (finite nu only).
    %   stream          - RandStream that supplies every draw.
    %   options         - (optional) struct with the fields
    %                       substeps              positive integer that forces ns (finite nu only; the tests and the coarse part of the simulator check), with refine and step_samples_per_tau then ignored;
    %                       y0                    [D x 1] finite whitened start in place of the stationary draw (the check's fixed-start relaxation);
    %                       step_samples_per_tau  the rate whose interval carries the substep rule, an integer multiple of samples_per_tau (nested rates, above); default samples_per_tau;
    %                       frozen                true for the frozen chain, the check's negative control (finite nu only); default false.
    %
    % Outputs:
    %   X    - [n_obs x D] observations, samples down the rows, channels across.
    %   info - struct with the fields
    %            nu, D, n_obs, samples_per_tau, step_samples_per_tau, Delta, refine  the call's settings;
    %            substeps    ns, or k at nu = Inf (the exact steps of one observation interval);
    %            h           the substep, Delta_step = Delta/k at nu = Inf;
    %            delta0      the step scale of the substep rule, NaN at nu = Inf;
    %            acceptance  accepted over attempted substeps, 1 at nu = Inf and for the frozen chain (0 at finite nu when n_obs = 1, since no substep is attempted);
    %            repeated_observation_fraction  the fraction of observation intervals in which no substep is accepted, so that the observation repeats the previous one exactly; 0 at nu = Inf;
    %            start       'stationary' or 'fixed';
    %            frozen      the frozen option.
    %
    % debt: the substep loop is interpreted MATLAB (tens of millions of substeps for a path of T/tau = 3e4 at nu = 1/3 and D = 20), and the whole path is held in memory as [D x n_obs] whitened states and observations, with a third copy while one is coloured into the other.
    % A MEX kernel of the same step, or streaming the observations out, is the upgrade path if the observation window or the refinement grows by an order of magnitude.

    arguments
        R (:, :) double {mustBeReal, mustBeFinite}
        nu (1, 1) double {mustBeReal, mustBeNonNan, mustBePositive}
        n_obs (1, 1) double {mustBeInteger, mustBePositive}
        samples_per_tau (1, 1) double {mustBeReal, mustBeFinite, mustBePositive}
        refine (1, 1) double {mustBeInteger, mustBePositive}
        stream (1, 1) RandStream
        options (1, 1) struct = struct()
    end

    L = localWhitener(R);
    D = size(R, 1);
    Delta = 1 / samples_per_tau;
    [forced_substeps, y_start, step_samples_per_tau, frozen] = localOptions(options, ...
                                                                            D, samples_per_tau);
    % Substep-rule intervals per observation interval, an integer by localOptions.
    k = round(step_samples_per_tau / samples_per_tau);

    info = struct( ...
                  'nu', nu, ...
                  'D', D, ...
                  'n_obs', n_obs, ...
                  'samples_per_tau', samples_per_tau, ...
                  'step_samples_per_tau', step_samples_per_tau, ...
                  'Delta', Delta, ...
                  'refine', refine, ...
                  'substeps', 1, ...
                  'h', Delta, ...
                  'delta0', NaN, ...
                  'acceptance', 1, ...
                  'repeated_observation_fraction', 0, ...
                  'start', 'stationary', ...
                  'frozen', frozen);
    if ~isempty(y_start)
        info.start = 'fixed';
    end

    if isinf(nu)
        assert(~frozen, 'studentDiffusionPath:FrozenGaussian', ...
               'The frozen chain is the finite-nu negative control; nu = Inf is exact.');
        Y = localOrnsteinUhlenbeck(y_start, stream, D, n_obs, ...
                                   1 / step_samples_per_tau, k);
        info.substeps = k;
        info.h = 1 / step_samples_per_tau;
    else
        m = nu + D - 2;
        delta0 = 0.005 * min([1, m / (D + 2), nu / (1 + nu)]);
        if isempty(forced_substeps)
            % The quotient is an exact integer for several nu (30 at nu = 2 and 22 at nu = 10, for D = 20 and ten samples per tau), and its floating-point value can land a hair above it.
            % The 1e-9 slack keeps that from adding a substep.
            ns = refine * max(1, ceil(1 / step_samples_per_tau / delta0 - 1e-9)) * k;
            % The fine grid's own substep, the same double at every rate that divides step_samples_per_tau; with k = 1 it is Delta / ns, operation for operation.
            info.h = (1 / step_samples_per_tau) / (ns / k);
        else
            ns = forced_substeps;
            info.h = Delta / ns;
        end
        info.substeps = ns;
        info.delta0 = delta0;
        if isempty(y_start)
            y_start = localStationaryStart(stream, D, nu);
        end
        [Y, accepted, repeated] = localMetropolis(y_start, stream, D, nu, m, ...
                                                  n_obs, ns, info.h, frozen);
        info.acceptance = accepted / max(1, ns * (n_obs - 1));
        info.repeated_observation_fraction = repeated / max(1, n_obs - 1);
    end

    X = (L * Y).';
    if ~frozen && ~all(isfinite(X), 'all')
        [row, col] = find(~isfinite(X), 1);
        error('studentDiffusionPath:NonFiniteObservation', ...
              ['Observation %d, channel %d of a %d x %d path (nu = %g) is not ' ...
               'finite; nothing is clipped.'], row, col, n_obs, D, nu);
    end
end

function L = localWhitener(R)
    % chol(R, 'lower') after the checks that make R a correlation matrix.
    D = size(R, 1);
    if size(R, 2) ~= D || D < 2
        error('studentDiffusionPath:BadCorrelation', ...
              'R must be a square [D x D] correlation matrix with D >= 2; got %d x %d.', ...
              size(R, 1), size(R, 2));
    end
    if norm(R - R.', 'fro') > 1e-10 * max(1, norm(R, 'fro'))
        error('studentDiffusionPath:BadCorrelation', ...
              'R (%d x %d) is not symmetric to 1e-10.', D, D);
    end
    if max(abs(diag(R) - 1)) > 1e-10
        error('studentDiffusionPath:BadCorrelation', ...
              'R (%d x %d) must have a unit diagonal; the largest deviation is %.3g.', ...
              D, D, max(abs(diag(R) - 1)));
    end
    [L, flag] = chol((R + R.') / 2, 'lower');
    if flag ~= 0
        error('studentDiffusionPath:BadCorrelation', ...
              'R (%d x %d) is not positive definite (Cholesky failed at column %d).', ...
              D, D, flag);
    end
end

function [forced_substeps, y_start, step_samples_per_tau, frozen] = localOptions( ...
                                                                                 options, D, samples_per_tau)
    % Reads the test, check and nesting hooks from OPTIONS; any other field is a typo and raises rather than passing as a no-op.
    known = {'substeps', 'y0', 'step_samples_per_tau', 'frozen'};
    unknown = setdiff(fieldnames(options), known);
    if ~isempty(unknown)
        error('studentDiffusionPath:UnknownOption', ...
              'Unknown option(s) %s; the options are %s.', strjoin(unknown, ', '), ...
              strjoin(known, ', '));
    end
    forced_substeps = [];
    if isfield(options, 'substeps') && ~isempty(options.substeps)
        validateattributes(options.substeps, {'numeric'}, ...
                           {'scalar', 'integer', 'positive'}, mfilename, 'options.substeps');
        forced_substeps = double(options.substeps);
    end
    y_start = [];
    if isfield(options, 'y0') && ~isempty(options.y0)
        validateattributes(options.y0, {'numeric'}, ...
                           {'real', 'finite', 'numel', D}, mfilename, 'options.y0');
        y_start = double(reshape(options.y0, D, 1));
    end
    step_samples_per_tau = samples_per_tau;
    if isfield(options, 'step_samples_per_tau') && ~isempty(options.step_samples_per_tau)
        validateattributes(options.step_samples_per_tau, {'numeric'}, ...
                           {'scalar', 'real', 'finite', 'positive'}, mfilename, ...
                           'options.step_samples_per_tau');
        step_samples_per_tau = double(options.step_samples_per_tau);
        ratio = step_samples_per_tau / samples_per_tau;
        if abs(ratio - round(ratio)) >= 1e-9 || round(ratio) < 1
            error('studentDiffusionPath:RateNotNested', ...
                  ['step_samples_per_tau = %g is not an integer multiple of ' ...
                   'samples_per_tau = %g, so the observations would not sit on the ' ...
                   'substep grid.'], step_samples_per_tau, samples_per_tau);
        end
    end
    frozen = false;
    if isfield(options, 'frozen') && ~isempty(options.frozen)
        validateattributes(options.frozen, {'logical', 'numeric'}, {'scalar'}, ...
                           mfilename, 'options.frozen');
        frozen = logical(options.frozen);
    end
end

function y = localStationaryStart(stream, D, nu)
    % An exact t_nu(0, I) draw: one Gaussian vector over a single chi-square scale.
    g = randn(stream, D, 1);
    u = rand(stream, 1, 1);
    u = min(max(u, realmin), 1 - eps);
    V = 2 * gammaincinv(u, nu / 2);
    if ~(isfinite(V) && V > 0)
        error('studentDiffusionPath:StartUnderflow', ...
              ['The stationary start drew chi-square(%g) = %g from u = %.17g; ' ...
               'nothing is clipped.'], nu, V, u);
    end
    y = g * sqrt(nu / V);
    if ~all(isfinite(y))
        error('studentDiffusionPath:StartOverflow', ...
              'The stationary start overflowed (chi-square(%g) = %g).', nu, V);
    end
end

function Y = localOrnsteinUhlenbeck(y_start, stream, D, n_obs, step, k)
    % The exact AR(1) skeleton of the OU limit, run at the interval STEP and read every k-th state (k = 1: STEP is the observation interval itself).
    % The innovations are drawn and filtered one block of observation intervals at a time, with the filter state carried across, so the branch holds [D x n_obs] and one block instead of the whole fine grid and still gives the draws and the states of one call over the window, bit for bit.
    if isempty(y_start)
        y_start = randn(stream, D, 1);
    end
    block = 4096;                       % observation intervals per block of draws
    a = exp(-step);
    s = sqrt(-expm1(-2 * step));
    Y = zeros(D, n_obs);
    Y(:, 1) = y_start;
    % filter runs y(j) = s xi(j) + a y(j-1) down each column, with dimension 1 named so that a single innovation row is not filtered along its channels.
    % Its initial condition a*y_start makes the first output s xi(1) + a y_start, and its final condition continues the recursion in the next block.
    state = a * y_start.';
    for first = 2:block:n_obs
        last = min(n_obs, first + block - 1);
        [states, state] = filter(s, [1, -a], ...
                                 randn(stream, D, (last - first + 1) * k).', state, 1);
        Y(:, first:last) = states(k:k:end, :).';
    end
end

function [Y, accepted, repeated] = localMetropolis(y, stream, D, nu, m, n_obs, ns, h, ...
                                                   frozen)
    % ns Metropolis-corrected substeps per observation interval, with the draws taken in blocks of 4096 substeps.
    % With FROZEN set every proposal is accepted on the same draws, which gives the uncorrected chain of the check's negative control.
    block = 4096;
    a_h = exp(-h);
    sv = sqrt(-expm1(-2 * h) / m);
    sqrt_nu = sqrt(nu);

    Y = zeros(D, n_obs);
    Y(:, 1) = y;
    accepted = 0;
    repeated = 0;
    k = block;
    Z = [];
    log_u = [];
    for i = 2:n_obs
        moved = false;
        for j = 1:ns
            k = k + 1;
            if k > block
                Z = randn(stream, D, block);
                log_u = log(rand(stream, 1, block));
                k = 1;
            end
            r0 = hypot(sqrt_nu, norm(y));
            zeta = Z(:, k);
            z = a_h * y + (sv * r0) * zeta;
            if frozen
                y = z;
                accepted = accepted + 1;
                moved = true;
                continue
            end
            r1 = hypot(sqrt_nu, norm(z));
            rev = (y - a_h * z) / (sv * r1);
            log_alpha = -(nu + D) * (log(r1) - log(r0)) ...
                + (-D * log(r1) - 0.5 * (rev.' * rev)) ...
                - (-D * log(r0) - 0.5 * (zeta.' * zeta));
            if ~isfinite(log_alpha)
                error('studentDiffusionPath:NonFiniteAcceptance', ...
                      ['The acceptance ratio is %g at substep %d of observation %d ' ...
                       '(nu = %g, D = %d, |y| = %g, |z| = %g); nothing is clipped.'], ...
                      log_alpha, j, i, nu, D, norm(y), norm(z));
            end
            if log_u(k) < min(0, log_alpha)
                y = z;
                accepted = accepted + 1;
                moved = true;
            end
        end
        repeated = repeated + ~moved;
        Y(:, i) = y;
    end
end
