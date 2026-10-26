function out = informationMeasuresT(R, p, varargin)
    %INFORMATIONMEASUREST Information measures of one channel set under the fitted Student-t copula.
    %
    % out = informationMeasuresT(R, p, ...) returns, for MI, TC, DTC and O, the Gaussian closed form, the Student-t value from the fitted shape (through the copula entropy H_c = 1/2 log det R + kappa_t(d, nu)), the optional model-free KSG reference, and the t-minus-Gaussian difference in nats with a denominator-audited descriptive percentage.
    % The t values come from tCopulaEntropy, which is closed form in log-gamma and digamma terms, so nothing is integrated and there is no quadrature tolerance.
    %
    % The margin term is exact.
    % The t family is closed under marginalisation with the same nu, so the leave-one-out shape term is kappa_t evaluated at d-1: kappa_margin = kappa_t(d-1, nu) is an identity, not an approximation.
    % At d = 2 the margin is univariate and kappa_margin = 0.
    %
    % Given kappa (full set) and kappa_margin (leave-one-out margin):
    %   TC  = TC_G  - kappa
    %   DTC = DTC_G + (d kappa_margin - (d-1) kappa)
    %   O   = TC - DTC
    %   MI  = TC        (d = 2 only)
    % Writing r = kappa_margin / kappa (both negative, so r > 0) gives the sign structure: Delta(O) = kappa*((d-2) - d r) is positive iff r > 1 - 2/d, and Delta(DTC) = kappa*(d r - (d-1)) is negative iff r > 1 - 1/d.
    % No general theorem here excludes either condition, so the code reports the DTC and O differences with their signs and asserts neither.
    % Delta(MI) = Delta(TC) = -kappa >= 0 does hold, because kappa <= 0 with equality only at p = 0 (the Gaussian copula maximizes copula entropy at fixed R; tests/oracles/ellipticalEntropyMaxCheck.py checks this independently of this file).
    %
    % The Gaussian references TC_G, DTC_G, O_G and MI_G come from gaussianInformationMeasures(R), and the model-free reference is estimateMIKSG (k = 6) when a sample X is supplied (MI at d = 2, TC at d >= 3).
    %
    % Inputs:
    %   R - d x d copula correlation matrix (kendallToCorrelation output).
    %   p - fitted Student-t shape parameter p = 1/nu for this channel set, scalar, finite and >= 0; p = 0 is the Gaussian copula, where every shape term and therefore every difference is exactly zero.
    % Optional name-value:
    %   'X'      - [n x d] sample for the model-free KSG reference (default []).
    %   'KSGk'   - KSG neighbour count (default 6).
    % Output: out struct, flat schema.
    %   Per measure M in {MI, TC, DTC, O}: M_gauss, M_t, M_ksg, M_delta_nats (= M_t - M_gauss), and two percentage families, one per denominator, each with its own denominator value, reportable flag and numeric status code:
    %
    %     Gaussian denominator (unsuffixed): M_delta_pct, M_pct_denominator, M_pct_reportable, M_pct_status_code; the error a practitioner incurs by using the Gaussian closed form.
    %     Fitted Student-t denominator (_t suffix): M_delta_pct_t, M_pct_denominator_t, M_pct_reportable_t, M_pct_status_code_t; the share of the modelled dependence the Gaussian calculation misses.
    %
    %   Status codes, identical in meaning on both denominators: 0 unavailable, 1 reportable, 2 denominator below the declared absolute floor, 3 O-information is cancellation-unstable.
    % Nats are always primary, and the DTC and O differences are not to be read as losses.
    % When gaussianInformationMeasures rejects R or the shape term is not finite, every measure is NaN, every reportable flag false, every status code 0 and valid is false.
    %
    % Each difference is expressed against both denominators, and they carry independent suppression state because they fail in different places: the fitted total for O-information can vanish where the Gaussian value does not.
    % Neither reportable flag may be inferred from the other.
    %
    % The _t percentage lies between 0 and 100 whenever the difference and the Gaussian value share a sign; the Gaussian-denominator percentage has no bound and reaches four figures on nearly independent channel sets.
    %
    % Percentage policy (prespecified defaults, applied to both denominators):
    %   PctMinAbsDenominator = 1e-3 nats.
    %       A baseline smaller than the displayed millinat reporting unit gets no relative percentage.
    %       The same floor applies to both denominators, because it asks whether a nats-valued quantity is meaningfully non-zero, which does not depend on which denominator it is.
    %   OPctMinCancellationRatio = 0.1, applied per denominator to its own cancellation ratio: |O_G|/(|TC_G|+|DTC_G|) for the Gaussian family and |O_t|/(|TC_t|+|DTC_t|) for the _t family.
    %       This keeps a small difference of two much larger quantities from producing a headline percentage.
    %       The Gaussian ratio does not bound the fitted one.
    %
    % Other output fields: d, p, nu, kappa, kappa_margin, valid, the two thresholds and the two O cancellation ratios; nu = 1/p is Inf at p = 0.

    validateattributes(p, {'double'}, ...
        {'scalar', 'real', 'finite', 'nonnegative'}, ...
        'informationMeasuresT', 'p');

    parser = inputParser;
    parser.addParameter('X', [], @(x) isempty(x) || isnumeric(x));
    parser.addParameter('KSGk', 6, @(x) isnumeric(x) && isreal(x) && isscalar(x));
    parser.addParameter('PctMinAbsDenominator', 1e-3, ...
        @(x) isscalar(x) && isfinite(x) && x >= 0);
    parser.addParameter('OPctMinCancellationRatio', 0.1, ...
        @(x) isscalar(x) && isfinite(x) && x >= 0 && x <= 1);
    parser.parse(varargin{:});
    X = parser.Results.X;
    ksgK = parser.Results.KSGk;
    if ~(isfinite(ksgK) && ksgK >= 1 && ksgK == fix(ksgK))
        error('informationMeasuresT:InvalidKSGk', ...
            'KSGk must be a finite positive integer.');
    end

    d = size(R, 1);
    pctMinDenominator = parser.Results.PctMinAbsDenominator;
    oMinCancellationRatio = parser.Results.OPctMinCancellationRatio;
    out = local_nan_out(d, p, pctMinDenominator, oMinCancellationRatio);

    G = gaussianInformationMeasures(R);
    if ~G.valid
        return;
    end
    ce = tCopulaEntropy(d, p);
    if ~ce.valid
        return;
    end
    out.kappa        = ce.kappa;
    out.kappa_margin = ce.kappa_margin;

    % --- Student-t measures from the copula entropy decomposition ---------
    TC_t  = G.TC  - ce.kappa;
    DTC_t = G.DTC + (d * ce.kappa_margin - (d - 1) * ce.kappa);
    O_t   = TC_t - DTC_t;
    MI_t  = TC_t;                                    % d = 2: MI = TC

    % --- model-free (KSG) reference (MI at d=2, TC at d>=3) --------------
    ksg_MI = NaN; ksg_TC = NaN;
    if ~isempty(X) && size(X, 2) == d
        mf = estimateMIKSG(X, ksgK);
        if d == 2, ksg_MI = mf; else, ksg_TC = mf; end
    end

    % --- pack per-measure (gauss / t / ksg / delta nats / delta %) -------
    % MI, TC and DTC are not sign-indefinite differences of larger quantities, so neither denominator needs a cancellation guard for them; only the absolute floor applies, on both.
    if d == 2
        out = local_pack(out, 'MI', G.MI, MI_t, ksg_MI, ...
            pctMinDenominator, true, true);
    end
    out = local_pack(out, 'TC',  G.TC,  TC_t,  ksg_TC, ...
        pctMinDenominator, true, true);
    out = local_pack(out, 'DTC', G.DTC, DTC_t, NaN, ...
        pctMinDenominator, true, true);
    % O is TC - DTC on both denominators, so both can be a small difference of two much larger quantities.
    % The guard is therefore evaluated separately per denominator: the Gaussian ratio does not bound the fitted one.
    oCancellationRatio = local_cancellation_ratio(G.O, G.TC, G.DTC);
    oCancellationRatioT = local_cancellation_ratio(O_t, TC_t, DTC_t);
    out.O_pct_cancellation_ratio = oCancellationRatio;
    out.O_pct_cancellation_ratio_t = oCancellationRatioT;
    out = local_pack(out, 'O', G.O, O_t, NaN, pctMinDenominator, ...
        oCancellationRatio >= oMinCancellationRatio, ...
        oCancellationRatioT >= oMinCancellationRatio);
    out.valid = true;
end

% ---------------------------------------------------------------------------
function out = local_pack(out, name, gval, tval, ksgval, minDenominator, ...
        extraStableGauss, extraStableT)
    %LOCAL_PACK Emit one measure against both percentage denominators.
    %
    % The unsuffixed family is the Gaussian denominator and the _t family the fitted Student-t denominator.
    % The two carry independent suppression state because they fail in different places, so neither reportable flag may be inferred from the other.
    %
    % Field order matters: local_nan_out must declare every field written here, in the same order, so that the valid and invalid paths return identical fieldnames().
    out.([name '_gauss'])      = gval;
    out.([name '_t'])          = tval;
    out.([name '_ksg'])        = ksgval;
    delta = tval - gval;
    out.([name '_delta_nats']) = delta;

    % --- Gaussian denominator: the error a practitioner incurs by using the Gaussian closed form.
    [reportable, statusCode] = local_percentage_status( ...
        gval, minDenominator, extraStableGauss);
    out.([name '_pct_denominator']) = gval;
    out.([name '_pct_reportable']) = reportable;
    out.([name '_pct_status_code']) = statusCode;
    if reportable
        out.([name '_delta_pct']) = 100 * delta / gval;
    else
        out.([name '_delta_pct']) = NaN;
    end

    % --- Fitted Student-t denominator: the share of the modelled dependence the Gaussian calculation misses.
    % It lies between 0 and 100 whenever the difference and the Gaussian value share a sign.
    [reportableT, statusCodeT] = local_percentage_status( ...
        tval, minDenominator, extraStableT);
    out.([name '_pct_denominator_t']) = tval;
    out.([name '_pct_reportable_t']) = reportableT;
    out.([name '_pct_status_code_t']) = statusCodeT;
    if reportableT
        out.([name '_delta_pct_t']) = 100 * delta / tval;
    else
        out.([name '_delta_pct_t']) = NaN;
    end
end


function [reportable, statusCode] = local_percentage_status( ...
        denominator, minDenominator, extraStable)
    %LOCAL_PERCENTAGE_STATUS Shared status rule, applied per denominator.
    %
    % Codes are identical in meaning on both denominators (see the file header): 1 reportable, 2 denominator below the declared absolute floor, 3 cancellation unstable.
    % The floor check runs first, so 2 pre-empts 3.
    % The floor is the same on both (1e-3 nats by default), because it asks whether a nats-valued denominator is meaningfully non-zero, and that question does not depend on which denominator is used.
    %
    % The rule lives in percentageStatus, which tInformationMeasuresFromLogdets also calls when Phase 4 rebuilds these percentages from the Phase 2 log-determinants, so one rule has one implementation.
    [reportable, statusCode] = percentageStatus( ...
        denominator, minDenominator, extraStable);
end


function ratio = local_cancellation_ratio(oValue, tcValue, dtcValue)
    %LOCAL_CANCELLATION_RATIO |O| / (|TC| + |DTC|) on one denominator's scale.
    %
    % O is a difference of two much larger positives, so |O| can be numerically tiny while clearing the absolute floor; this ratio detects that, and the floor alone does not.
    % It returns 0 for a zero scale, which suppresses rather than divides by zero.
    %
    % The rule lives in cancellationRatio, which tInformationMeasuresFromLogdets also calls.
    ratio = cancellationRatio(oValue, tcValue, dtcValue);
end

function out = local_nan_out(d, p, minDenominator, oMinCancellationRatio)
    % The invalid-path output: every measure NaN, every reportable flag false and every status code 0.
    % The valid path starts from this struct too, and the field order here mirrors local_pack exactly, so both paths agree on the names and their order.
    %
    % nu sits next to p so that a reader of one record never has to invert the shape parameter by hand; nu = 1/p is Inf at the Gaussian point p = 0.
    %
    % The MI fields are emitted at every d and stay NaN for d >= 3, where mutual information is not reported, which keeps the field set constant across d as well as across the valid and invalid paths.
    out = struct('d', d, 'p', p, 'nu', 1 / p, 'kappa', NaN, 'kappa_margin', NaN, ...
        'pct_min_abs_denominator_nats', minDenominator, ...
        'O_pct_min_cancellation_ratio', oMinCancellationRatio, ...
        'O_pct_cancellation_ratio', NaN, ...
        'O_pct_cancellation_ratio_t', NaN, 'valid', false);
    for name = {'MI', 'TC', 'DTC', 'O'}
        n = name{1};
        out.([n '_gauss'])      = NaN;
        out.([n '_t'])          = NaN;
        out.([n '_ksg'])        = NaN;
        out.([n '_delta_nats']) = NaN;
        out.([n '_pct_denominator']) = NaN;
        out.([n '_pct_reportable']) = false;
        out.([n '_pct_status_code']) = 0;
        out.([n '_delta_pct'])  = NaN;
        out.([n '_pct_denominator_t']) = NaN;
        out.([n '_pct_reportable_t']) = false;
        out.([n '_pct_status_code_t']) = 0;
        out.([n '_delta_pct_t'])  = NaN;
    end
end
