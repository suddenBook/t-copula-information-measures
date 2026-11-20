function out = tInformationMeasuresFromLogdets(d, p, logdet_R, ...
                                               sum_logdet_minors, opts)
    % TINFORMATIONMEASURESFROMLOGDETS Gaussian and Student-t information measures of one unit from its two log-determinants.
    %
    % informationMeasuresT(R, p) computes the Gaussian reference from R with one Cholesky factorisation of R and one of each of its d leave-one-out minors.
    % The Phase 2 unit row already carries the two resulting log-determinants, logdet_R and sum_logdet_minors, which unitCorrelationFromTau takes from the same gaussianInformationMeasures call.
    % Phase 4 therefore reads them instead of unpacking and refactorising R for every unit.
    % The arithmetic is the same closed form, not an approximation of it:
    %
    %   TC_G  = -1/2 logdet_R
    %   DTC_G =  1/2 sum_logdet_minors - (d-1)/2 logdet_R
    %   O_G   =  TC_G - DTC_G
    %   MI_G  =  TC_G                                       (d = 2 only)
    %
    % The Student-t values add the closed-form, R-independent differences of tInformationDifferences, the same kappa_t algebra informationMeasuresT applies:
    %
    %   TC_t = TC_G + Delta(TC),  DTC_t = DTC_G + Delta(DTC),  O_t = TC_t - DTC_t.
    %
    % The percentage policy is that of informationMeasuresT, applied through the same two helpers, percentageStatus and cancellationRatio.
    % Both denominators, the Gaussian value (unsuffixed) and the fitted Student-t value (suffix _t), carry independent suppression state, an absolute floor each and their own O-information cancellation ratio.
    % Status codes are 0 unavailable, 1 reportable, 2 denominator below the floor, 3 cancellation unstable.
    %
    % The function fails closed.
    % Phase 4 passes a unit with no usable fit in as p = NaN.
    % A p that is not finite and non-negative, or a non-finite log-determinant, returns every measure NaN, every reportable flag false and every status code 0, the code for "no percentage attempted".
    % The field set is the same on that path and at every d, so rows from all dimensions concatenate into one table.
    %
    % Inputs:
    %   d                 - dimension, integer >= 2.
    %   p                 - fitted shape parameter p = 1/nu, or NaN for no fit.
    %   logdet_R          - log det R of the unit (NaN when R is not positive definite).
    %   sum_logdet_minors - sum over j of log det of R with coordinate j removed; 0 at d = 2.
    %   opts              - optional struct with pct_min_abs_denominator (default 1e-3 nats) and o_pct_min_cancellation_ratio (default 0.1).
    % Outputs:
    %   out - flat struct with d, p, nu, kappa, kappa_margin, the two thresholds, the two O cancellation ratios, valid, and per measure M in {MI, TC, DTC, O} the fields M_gauss, M_t, M_delta_nats and the two percentage families (M_pct_denominator, M_pct_reportable, M_pct_status_code, M_delta_pct and their _t twins).
    %         The names are those of informationMeasuresT without its M_ksg fields, because Phase 4 joins the model-free estimate itself.

    arguments
        d (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(d, 2)}
        p (1, 1) double {mustBeReal}
        logdet_R (1, 1) double {mustBeReal}
        sum_logdet_minors (1, 1) double {mustBeReal}
        opts (1, 1) struct = struct()
    end

    pct_min_denominator = localOption(opts, 'pct_min_abs_denominator', 1e-3);
    o_min_cancellation_ratio = localOption(opts, 'o_pct_min_cancellation_ratio', 0.1);
    validateattributes(pct_min_denominator, {'numeric'}, ...
                       {'scalar', 'real', 'finite', 'nonnegative'}, mfilename, ...
                       'opts.pct_min_abs_denominator');
    validateattributes(o_min_cancellation_ratio, {'numeric'}, ...
                       {'scalar', 'real', 'finite', '>=', 0, '<=', 1}, mfilename, ...
                       'opts.o_pct_min_cancellation_ratio');

    out = localNanOut(d, p, pct_min_denominator, o_min_cancellation_ratio);
    if ~(isfinite(p) && p >= 0 && isfinite(logdet_R) && isfinite(sum_logdet_minors))
        return
    end

    TC_gauss  = -0.5 * logdet_R;
    DTC_gauss =  0.5 * sum_logdet_minors - 0.5 * (d - 1) * logdet_R;
    O_gauss   = TC_gauss - DTC_gauss;

    differences = tInformationDifferences(d, p);
    out.kappa = differences.kappa_d;
    out.kappa_margin = differences.kappa_dm1;

    TC_t  = TC_gauss + differences.TC;
    DTC_t = DTC_gauss + differences.DTC;
    O_t   = TC_t - DTC_t;

    % MI, TC and DTC are not sign-indefinite differences of larger quantities, so only the absolute floor applies to them, on both denominators.
    if d == 2
        out = localPack(out, 'MI', TC_gauss, TC_t, pct_min_denominator, true, true);
    end
    out = localPack(out, 'TC', TC_gauss, TC_t, pct_min_denominator, true, true);
    out = localPack(out, 'DTC', DTC_gauss, DTC_t, pct_min_denominator, true, true);

    % O is TC - DTC on both denominators, so both can cancel; the Gaussian ratio does not bound the fitted one, and each guards its own percentage.
    ratio_gauss = cancellationRatio(O_gauss, TC_gauss, DTC_gauss);
    ratio_t = cancellationRatio(O_t, TC_t, DTC_t);
    out.O_pct_cancellation_ratio = ratio_gauss;
    out.O_pct_cancellation_ratio_t = ratio_t;
    out = localPack(out, 'O', O_gauss, O_t, pct_min_denominator, ...
                    ratio_gauss >= o_min_cancellation_ratio, ...
                    ratio_t >= o_min_cancellation_ratio);
    out.valid = true;
end

function value = localOption(opts, name, default)
    % The value of opts.(name) when that field is present and non-empty, else the default, as in tFitDefaultOptions.
    value = default;
    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    end
end

function out = localPack(out, name, gauss_value, t_value, min_denominator, ...
                         extra_stable_gauss, extra_stable_t)
    % One measure against both percentage denominators, in the field order of informationMeasuresT, so a row from either entry point reads the same.
    out.([name '_gauss']) = gauss_value;
    out.([name '_t']) = t_value;
    delta = t_value - gauss_value;
    out.([name '_delta_nats']) = delta;

    [reportable, status_code] = percentageStatus( ...
                                                 gauss_value, min_denominator, extra_stable_gauss);
    out.([name '_pct_denominator']) = gauss_value;
    out.([name '_pct_reportable']) = reportable;
    out.([name '_pct_status_code']) = status_code;
    if reportable
        out.([name '_delta_pct']) = 100 * delta / gauss_value;
    else
        out.([name '_delta_pct']) = NaN;
    end

    [reportable_t, status_code_t] = percentageStatus( ...
                                                     t_value, min_denominator, extra_stable_t);
    out.([name '_pct_denominator_t']) = t_value;
    out.([name '_pct_reportable_t']) = reportable_t;
    out.([name '_pct_status_code_t']) = status_code_t;
    if reportable_t
        out.([name '_delta_pct_t']) = 100 * delta / t_value;
    else
        out.([name '_delta_pct_t']) = NaN;
    end
end

function out = localNanOut(d, p, min_denominator, o_min_cancellation_ratio)
    % Every field, in the order of localPack, at its "nothing computed" value.
    % The MI fields exist at every d and stay NaN above d = 2, so the field names are the same at every dimension and for valid and invalid units alike.
    out = struct('d', d, 'p', p, 'nu', 1 / p, 'kappa', NaN, 'kappa_margin', NaN, ...
                 'pct_min_abs_denominator_nats', min_denominator, ...
                 'O_pct_min_cancellation_ratio', o_min_cancellation_ratio, ...
                 'O_pct_cancellation_ratio', NaN, ...
                 'O_pct_cancellation_ratio_t', NaN, 'valid', false);
    for name = {'MI', 'TC', 'DTC', 'O'}
        n = name{1};
        out.([n '_gauss']) = NaN;
        out.([n '_t']) = NaN;
        out.([n '_delta_nats']) = NaN;
        out.([n '_pct_denominator']) = NaN;
        out.([n '_pct_reportable']) = false;
        out.([n '_pct_status_code']) = 0;
        out.([n '_delta_pct']) = NaN;
        out.([n '_pct_denominator_t']) = NaN;
        out.([n '_pct_reportable_t']) = false;
        out.([n '_pct_status_code_t']) = 0;
        out.([n '_delta_pct_t']) = NaN;
    end
end
