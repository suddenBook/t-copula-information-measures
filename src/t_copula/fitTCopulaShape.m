% FITTCOPULASHAPE Profile maximum-likelihood fit of the t-copula shape p = 1/nu at a fixed correlation matrix.
%
% Fits the generator shape parameter p = 1/nu of the Student-t elliptical copula to pseudo-observations U (n x d), with the correlation matrix R held fixed at its Kendall's tau estimate.
% Given R, it maximises the t-copula log-likelihood (tCopulaLoglik) over p, a plug-in estimate of a dimension-specific working-model parameter.
% p = 0 is the Gaussian member, but p_hat itself is not a distance, a goodness-of-fit verdict or a two-sided test of a Gaussian null.
% No standard error, confidence interval or likelihood-ratio statistic is produced.
%
% Estimator: a deterministic coarse profile over p on [0, p_bounds(2)], then fminbnd refinement of every sampled candidate basin.
% The winner is checked against the complete coarse profile, both endpoints and its local neighbours.
% This is an auditable multi-basin search on the bounded working domain, not a certificate of the continuous global maximum, because no finite scan can exclude an arbitrarily narrow unsampled peak.
%
% Neither bound excludes a fit.
% p = 0 (nu = Inf) is the Gaussian copula on the lower bound; it is a model, so an optimum there is a valid fit, reported with entry.gaussian_boundary = true and nu_hat = Inf.
% The heavy end p = p_bounds(2) is an edge of the working domain, not a defect: an optimum within boundary_p_tolerance of it is recorded in entry.heavy_boundary and stays reportable, because the study reads the heavy end as an observation about the recording rather than as a failed fit.
% A constrained maximum exactly at p_bounds(2) is certified the way the Gaussian node is: it takes the exit flag of its own guard refinement, wins as an ordinary candidate, and has a one-sided neighbour check against the point below it.
% A near-tied remote basin is likewise recorded, in entry.ambiguous_profile, and not excluded.
% A unit is dropped only on a numerical failure, which entry.numerical_ok carries and entry.valid repeats.
% fminbnd approaches 0 without reaching it, so a refined optimum within gaussian_snap_tolerance of 0 that does not beat the exactly evaluated Gaussian node by more than the comparison tolerance is snapped onto p = 0 (entry.profile_gaussian_snapped), making the Gaussian estimate the closed-form node rather than an optimiser residue.
%
% The joint and the marginal densities of the t family are closed form, so the objective needs no numerical marginal grid and no normalising-constant lookup; tCopulaLoglik evaluates p = 0 through the exact Gaussian expression.
%
% Inputs:
%   U     - [n x d] pseudo-observations strictly in (0, 1).
%   R     - [d x d] correlation matrix (unit diagonal, symmetric positive definite), from kendallToCorrelation.
%   opts  - optional struct of tFitDefaultOptions fields; that file gives the defaults and the required ordering of the p-scale tolerances.
%   cache - optional tCopulaQuantileCache for this recording's rank plan; it is read only at the coarse-profile nodes it tabulates and changes no number.
%
% Outputs (a fit is reportable only when entry.numerical_ok is true):
%   entry.numerical_ok      - finite loglik at p_hat, exitflag > 0 and a consistent scan
%   entry.valid             - the same verdict as numerical_ok, under the field name callers read
%   entry.heavy_boundary    - flag: p_bounds(2) - p_hat < boundary_p_tolerance
%   entry.ambiguous_profile - flag: a separated basin is within the near-tie slack
%   entry.exclusion_reason  - '' or one of 'not_a_max', 'R_singular', 'invalid_dimension', 'too_few_rows', 'nonfinite_data', 'invalid_pseudo_observations', 'constant_margin', 'invalid_data_type', 'invalid_R', 'nonfinite_R', 'R_not_symmetric' and 'R_not_correlation'
%                             These are numerical and data codes only; the heavy end and the near tie are the flags above, never exclusions.
%   entry.Sigma             - R, the fixed correlation matrix
%   entry.rho               - R(1,2) when d = 2, else NaN
%   entry.d                 - dimension
%   entry.p_hat             - profile MLE of p = 1/nu
%   entry.nu_hat            - 1/p_hat (Inf at the Gaussian bound)
%   entry.gaussian_boundary - true if and only if p_hat is exactly 0 (a valid Gaussian fit)
%   entry.loglik            - copula log-likelihood at p_hat
%   entry.loglik_gaussian   - copula log-likelihood at p = 0, evaluated exactly
%   entry.n_used            - number of rows used (n; all rows are used)
%   entry.d2                - diagnostic d^2 loglik / dp^2 at p_hat, never used for a Wald interval or as an adequacy or validity threshold
%   entry.d2_step           - p step used for entry.d2
%   entry.profile_boundary_slope - diagnostic forward slope dloglik/dp at p_hat = 0
%   entry.optimizer_exitflag - fminbnd exit flag of the winning basin, or of its guard refinement for an endpoint (> 0 means convergence)
%   entry.profile_*         - scalar scan and ambiguity diagnostics, the coarse profile (profile_p_grid, profile_loglik_grid) always, and the remaining vectors only when opts.store_profile is true

function entry = fitTCopulaShape(U, R, opts, cache)

    if nargin < 3 || isempty(opts)
        opts = tFitDefaultOptions();
    else
        opts = tFitDefaultOptions(opts);
    end
    if nargin < 4
        cache = [];
    end

    [n, d] = size(U);
    entry = local_blank_entry(R, d);

    % ---- Input checks: reject malformed input rather than let it become a likelihood artefact ----
    if d < 2
        entry.exclusion_reason = 'invalid_dimension';
        return
    end
    if n < opts.min_rows
        entry.exclusion_reason = 'too_few_rows';
        return
    end
    if ~isnumeric(U) || ~isreal(U)
        entry.exclusion_reason = 'invalid_data_type';
        return
    end
    if any(~isfinite(U(:)))
        entry.exclusion_reason = 'nonfinite_data';
        return
    end
    if any(U(:) <= 0 | U(:) >= 1)
        entry.exclusion_reason = 'invalid_pseudo_observations';
        return
    end
    if any(max(U, [], 1) - min(U, [], 1) <= 0)
        entry.exclusion_reason = 'constant_margin';
        return
    end

    % ---- Fixed R: dimension, correlation-form and conditioning checks ----
    if ~isnumeric(R) || ~isreal(R) || ~isequal(size(R), [d d])
        entry.exclusion_reason = 'invalid_R';
        return
    end
    if any(~isfinite(R(:)))
        entry.exclusion_reason = 'nonfinite_R';
        return
    end
    if norm(R - R.', 'fro') > opts.r_sym_tol
        entry.exclusion_reason = 'R_not_symmetric';
        return
    end
    if max(abs(diag(R) - 1)) > opts.r_diag_tol
        entry.exclusion_reason = 'R_not_correlation';
        return
    end
    R = (R + R.') / 2;
    min_eig = min(eig(R));
    entry.r_min_eig = min_eig;
    if ~isfinite(min_eig) || min_eig < opts.r_min_eig
        entry.exclusion_reason = 'R_singular';
        return
    end
    [Lr, chol_flag] = chol(R);
    if chol_flag ~= 0
        entry.exclusion_reason = 'R_singular';
        return
    end
    entry.Sigma = R;
    entry.r_condition = cond(R);
    logdetR = 2 * sum(log(diag(Lr)));
    Rinfo = struct('Lr', Lr, 'logdetR', logdetR, 'd', d);

    % ---- Deterministic multi-basin profile over p on all rows ---------------
    % The shape-independent preparation of U is done once, so every likelihood evaluation costs one quantile call per distinct tail probability.
    plan = tCopulaQuantilePlan(U);
    entry.profile_num_distinct_quantile_args = plan.num_distinct;
    negll = @(p) -safe_loglik(p, plan, Rinfo, cache);
    p_lo = opts.p_bounds(1);
    p_hi = opts.p_bounds(2);
    p_grid = linspace(p_lo, p_hi, opts.profile_grid_size);
    ll_grid = nan(size(p_grid));
    for k = 1:numel(p_grid)
        ll_grid(k) = -negll(p_grid(k));
    end
    entry.profile_p_grid = p_grid;
    entry.profile_loglik_grid = ll_grid;
    entry.profile_grid_size = numel(p_grid);
    entry.profile_p_bounds = opts.p_bounds;
    entry.profile_min_p_grid_step = min(diff(p_grid));
    entry.profile_max_p_grid_step = max(diff(p_grid));
    entry.profile_gaussian_anchor_evaluated = (p_grid(1) == 0);
    entry.loglik_gaussian = ll_grid(1);
    entry.profile_best_grid_loglik = max(ll_grid);
    entry.profile_compare_tolerance = max(1e-6, n * opts.profile_ll_tol_per_row);
    entry.profile_ambiguity_tolerance = max( ...
                                            1e-6, n * opts.profile_ambiguity_ll_tol_per_row);
    entry.profile_ambiguity_min_p_separation = ...
        opts.profile_ambiguity_min_p_separation;

    candidate_idx = local_candidate_indices(ll_grid);
    candidate_lower_idx = candidate_idx - 1;
    candidate_upper_idx = candidate_idx + 1;
    candidate_kind = repmat("interior", size(candidate_idx));
    % A coarse endpoint that is no lower than its only neighbour does not prove the optimum is on the bound, because a narrow interior peak can hide inside that first interval.
    % Such boundary-adjacent intervals are therefore refined explicitly.
    % At the lower end this also certifies the exactly evaluated Gaussian node against a hidden interior peak and supplies that node's convergence flag.
    if ll_grid(1) >= ll_grid(2)
        candidate_idx = [1, candidate_idx];
        candidate_lower_idx = [1, candidate_lower_idx];
        candidate_upper_idx = [2, candidate_upper_idx];
        candidate_kind = ["lower_guard", candidate_kind];
    end
    if ll_grid(end) >= ll_grid(end - 1)
        candidate_idx = [candidate_idx, numel(ll_grid)];
        candidate_lower_idx = [candidate_lower_idx, numel(ll_grid) - 1];
        candidate_upper_idx = [candidate_upper_idx, numel(ll_grid)];
        candidate_kind = [candidate_kind, "upper_guard"];
    end
    n_candidates = numel(candidate_idx);
    refined_p = nan(n_candidates, 1);
    refined_ll = -inf(n_candidates, 1);
    refined_exitflag = zeros(n_candidates, 1);
    for k = 1:n_candidates
        [refined_p(k), nll, refined_exitflag(k)] = fminbnd( ...
                                                           negll, p_grid(candidate_lower_idx(k)), ...
                                                           p_grid(candidate_upper_idx(k)), ...
                                                           optimset('TolX', opts.tol_x, 'Display', 'off'));
        refined_ll(k) = -nll;
    end

    entry.profile_num_candidates = n_candidates;
    entry.profile_num_refine_failures = nnz( ...
                                            refined_exitflag <= 0 | ~isfinite(refined_ll));
    if opts.store_profile
        entry.profile_candidate_indices = candidate_idx;
        entry.profile_candidate_kind = candidate_kind;
        entry.profile_candidate_lower_p = p_grid(candidate_lower_idx);
        entry.profile_candidate_upper_p = p_grid(candidate_upper_idx);
        entry.profile_refined_p = refined_p;
        entry.profile_refined_loglik = refined_ll;
        entry.profile_refined_exitflag = refined_exitflag;
    end

    finite_refined = refined_exitflag > 0 & isfinite(refined_ll) ...
        & isfinite(refined_p);
    endpoint_ll = [ll_grid(1); ll_grid(end)];
    if any(finite_refined)
        basin_p = [refined_p(finite_refined); p_lo; p_hi];
        basin_ll = [refined_ll(finite_refined); endpoint_ll];
        basin_exitflag = [refined_exitflag(finite_refined); 0; 0];
        basin_source = [ ...
                        repmat("refined", nnz(finite_refined), 1); ...
                        "lower_endpoint"; "upper_endpoint"];
    else
        [loglik, best_grid_idx] = max(ll_grid);
        basin_p = [p_grid(best_grid_idx); p_lo; p_hi];
        basin_ll = [loglik; endpoint_ll];
        basin_exitflag = zeros(3, 1);
        basin_source = ["grid"; "lower_endpoint"; "upper_endpoint"];
    end

    [loglik, selected_idx] = max(basin_ll);
    p_hat = basin_p(selected_idx);
    selected_source = char(basin_source(selected_idx));

    % fminbnd converges toward the Gaussian node but never reaches it, so a refined winner inside the snap tolerance that does not beat the exactly evaluated p = 0 node by more than the comparison tolerance is that node, and the closed-form estimate is reported instead of the residue.
    lower_idx = find(basin_source == "lower_endpoint", 1);
    if strcmp(selected_source, 'refined') ...
            && p_hat <= opts.gaussian_snap_tolerance ...
            && basin_ll(lower_idx) >= loglik - entry.profile_compare_tolerance
        selected_idx = lower_idx;
        loglik = basin_ll(lower_idx);
        p_hat = 0;
        selected_source = 'lower_endpoint';
        entry.profile_gaussian_snapped = true;
    end
    entry.profile_selected_source = selected_source;

    % An endpoint is a grid evaluation with no exit flag of its own, so its convergence evidence is the guard refinement of its boundary interval: the lower guard for the Gaussian node, the upper guard for the heavy end.
    % Without such a refinement the flag is 0, and the verdict below fails the fit rather than assume convergence.
    exitflag = basin_exitflag(selected_idx);
    guard_kind = "";
    if strcmp(selected_source, 'lower_endpoint')
        guard_kind = "lower_guard";
    elseif strcmp(selected_source, 'upper_endpoint')
        guard_kind = "upper_guard";
    end
    if strlength(guard_kind) > 0
        guard_slot = find(candidate_kind == guard_kind, 1);
        if isempty(guard_slot)
            exitflag = 0;
        else
            exitflag = refined_exitflag(guard_slot);
        end
    end

    % Only meaningfully separated basins are compared.
    % Numerical maxima closer than profile_ambiguity_min_p_separation (0.01 in p by default) describe one basin, whereas a near-tied remote peak makes the working-model estimate genuinely ambiguous.
    separation = abs(basin_p - p_hat);
    separated = isfinite(basin_ll) ...
        & separation >= opts.profile_ambiguity_min_p_separation;
    entry.profile_num_separated_competitors = nnz(separated);
    if any(separated)
        separated_idx = find(separated);
        [second_ll, relative_idx] = max(basin_ll(separated_idx));
        second_idx = separated_idx(relative_idx);
        entry.profile_second_best_loglik = second_ll;
        entry.profile_second_best_p = basin_p(second_idx);
        entry.profile_second_best_source = char(basin_source(second_idx));
        entry.profile_second_best_separation = separation(second_idx);
        entry.profile_separated_gap = loglik - second_ll;
    else
        entry.profile_separated_gap = Inf;
    end
    entry.profile_ambiguous = isfinite(entry.profile_separated_gap) ...
        && entry.profile_separated_gap <= entry.profile_ambiguity_tolerance;

    all_refined = n_candidates > 0 && entry.profile_num_refine_failures == 0;
    % Both endpoints are members of the closed domain [0, p_hi], so a winning endpoint node is as admissible as a refined interior optimum: it won the comparison against every refined candidate, and its guard refinement above supplies the convergence evidence a grid node cannot carry.
    winner_admissible = any(strcmp(selected_source, ...
                                   {'refined', 'lower_endpoint', 'upper_endpoint'}));
    winner_beats_grid = winner_admissible ...
        && loglik >= entry.profile_best_grid_loglik - entry.profile_compare_tolerance;
    endpoints_beaten = all(endpoint_ll <= loglik + entry.profile_compare_tolerance);

    nu_hat = 1 / p_hat;
    entry.gaussian_boundary = (p_hat == 0);

    % ---- Curvature and neighbour diagnostics on p ---------------------------
    % The curvature d2 builds no Wald interval and does not decide validity; it is kept as a coarse, stable number for numerical auditing.
    % The neighbour check feeds profile_scan_consistent, and at either bound it is one-sided because the domain holds no point beyond it: no p < 0 at the Gaussian end, no p > p_hi at the heavy end.
    entry.profile_distance_to_lower_p_bound = p_hat - p_lo;
    entry.profile_distance_to_upper_p_bound = p_hi - p_hat;
    if p_hat > p_lo && p_hat < p_hi
        h = min(opts.curvature_step, max(min(p_hat, p_hi - p_hat) / 2, eps));
        lp = safe_loglik(p_hat + h, plan, Rinfo, cache);
        lm = safe_loglik(p_hat - h, plan, Rinfo, cache);
        d2 = (lp - 2 * loglik + lm) / h^2;          % d2 / dp^2
        entry.profile_lower_neighbor_loglik = lm;
        entry.profile_upper_neighbor_loglik = lp;
        entry.profile_neighbor_gap = loglik - max(lp, lm);
        entry.profile_neighbor_verified = entry.profile_neighbor_gap ...
            >= -entry.profile_compare_tolerance;
    elseif p_hat == p_lo
        h = min(opts.curvature_step, p_hi / 2);
        lp = safe_loglik(h, plan, Rinfo, cache);
        lpp = safe_loglik(2 * h, plan, Rinfo, cache);
        d2 = (lpp - 2 * lp + loglik) / h^2;         % one-sided second difference
        entry.profile_lower_neighbor_loglik = NaN;
        entry.profile_upper_neighbor_loglik = lp;
        entry.profile_neighbor_gap = loglik - lp;
        entry.profile_neighbor_verified = entry.profile_neighbor_gap ...
            >= -entry.profile_compare_tolerance;
        entry.profile_boundary_slope = (lp - loglik) / h;
    elseif p_hat == p_hi
        % The heavy end mirrors the Gaussian node: its only neighbour inside the domain is below it, so the constrained maximum is certified against that neighbour alone.
        h = min(opts.curvature_step, (p_hi - p_lo) / 2);
        lm = safe_loglik(p_hat - h, plan, Rinfo, cache);
        lmm = safe_loglik(p_hat - 2 * h, plan, Rinfo, cache);
        d2 = (lmm - 2 * lm + loglik) / h^2;         % one-sided second difference
        entry.profile_lower_neighbor_loglik = lm;
        entry.profile_upper_neighbor_loglik = NaN;
        entry.profile_neighbor_gap = loglik - lm;
        entry.profile_neighbor_verified = entry.profile_neighbor_gap ...
            >= -entry.profile_compare_tolerance;
    else
        h = NaN;
        d2 = NaN;
        entry.profile_neighbor_verified = false;
    end
    entry.profile_scan_consistent = all_refined && winner_beats_grid ...
        && endpoints_beaten && entry.profile_neighbor_verified;

    if opts.store_profile
        entry.profile_basin_p = basin_p;
        entry.profile_basin_loglik = basin_ll;
        entry.profile_basin_source = basin_source;
    end

    % ---- Verdict: numerical only ----
    % The heavy end and the near tie are recorded conditions of the estimate, not defects of the arithmetic, so they leave numerical_ok untouched and are reported as their own flags.
    % p_hat = 0 is the Gaussian member and is kept.
    entry.profile_boundary_tolerance = opts.boundary_p_tolerance;
    entry.heavy_boundary = (p_hi - p_hat) < opts.boundary_p_tolerance;
    entry.ambiguous_profile = entry.profile_ambiguous;

    if d == 2
        entry.rho = R(1, 2);
    end
    entry.p_hat = p_hat;
    entry.nu_hat = nu_hat;
    entry.loglik = loglik;
    entry.n_used = n;
    entry.d2 = d2;
    entry.d2_step = h;
    entry.optimizer_exitflag = exitflag;

    entry.numerical_ok = isfinite(loglik) && exitflag > 0 ...
        && entry.profile_scan_consistent;
    entry.valid = entry.numerical_ok;
    if entry.numerical_ok
        entry.exclusion_reason = '';
    else
        entry.exclusion_reason = 'not_a_max';
    end
end

function idx = local_candidate_indices(ll)
    % Interior coarse-grid local maxima, consolidated so one plateau is one basin.
    raw = find(isfinite(ll(2:end - 1)) ...
               & ll(2:end - 1) >= ll(1:end - 2) ...
               & ll(2:end - 1) >= ll(3:end) ...
               & (ll(2:end - 1) > ll(1:end - 2) | ll(2:end - 1) > ll(3:end))) + 1;
    if isempty(raw)
        idx = zeros(1, 0);
        return
    end

    starts = [1, find(diff(raw) > 1) + 1];
    stops = [starts(2:end) - 1, numel(raw)];
    idx = zeros(1, numel(starts));
    for k = 1:numel(starts)
        block = raw(starts(k):stops(k));
        [~, rel] = max(ll(block));
        idx(k) = block(rel);
    end
end

function ll = safe_loglik(p, plan, Rinfo, cache)
    % Every likelihood evaluation of the fitter goes through this call, and tCopulaLoglik raises on a non-finite value instead of returning one.
    % Valid inputs on the supported p domain must be numerically evaluable, so a failure is a programming error, and turning it into an apparent low-likelihood basin could silently move the MLE.
    ll = tCopulaLoglik(p, plan, Rinfo, cache);
end

function entry = local_blank_entry(R, d)
    % The entry with every field present and nothing fitted, which a failed input check returns with its exclusion reason set.
    entry = struct( ...
                   'valid', false, 'numerical_ok', false, ...
                   'heavy_boundary', false, 'ambiguous_profile', false, ...
                   'exclusion_reason', 'unfit', ...
                   'Sigma', R, 'rho', NaN, 'd', d, ...
                   'p_hat', NaN, 'nu_hat', NaN, 'gaussian_boundary', false, ...
                   'loglik', NaN, 'loglik_gaussian', NaN, 'n_used', 0, 'd2', NaN, ...
                   'd2_step', NaN, 'optimizer_exitflag', NaN, ...
                   'r_min_eig', NaN, 'r_condition', NaN, ...
                   'profile_scan_consistent', false, ...
                   'profile_num_candidates', 0, ...
                   'profile_num_refine_failures', 0, 'profile_grid_size', 0, ...
                   'profile_p_bounds', [NaN NaN], ...
                   'profile_min_p_grid_step', NaN, ...
                   'profile_max_p_grid_step', NaN, ...
                   'profile_gaussian_anchor_evaluated', false, ...
                   'profile_gaussian_snapped', false, ...
                   'profile_num_distinct_quantile_args', 0, ...
                   'profile_best_grid_loglik', NaN, ...
                   'profile_separated_gap', NaN, 'profile_selected_source', '', ...
                   'profile_compare_tolerance', NaN, 'profile_ambiguity_tolerance', NaN, ...
                   'profile_ambiguity_min_p_separation', NaN, ...
                   'profile_ambiguous', false, 'profile_num_separated_competitors', 0, ...
                   'profile_second_best_loglik', NaN, ...
                   'profile_second_best_p', NaN, ...
                   'profile_second_best_source', '', ...
                   'profile_second_best_separation', NaN, ...
                   'profile_distance_to_lower_p_bound', NaN, ...
                   'profile_distance_to_upper_p_bound', NaN, ...
                   'profile_boundary_tolerance', NaN, ...
                   'profile_lower_neighbor_loglik', NaN, ...
                   'profile_upper_neighbor_loglik', NaN, ...
                   'profile_neighbor_gap', NaN, 'profile_neighbor_verified', false, ...
                   'profile_boundary_slope', NaN, ...
                   'profile_p_grid', [], 'profile_loglik_grid', [], ...
                   'profile_candidate_indices', [], ...
                   'profile_candidate_kind', strings(1, 0), ...
                   'profile_candidate_lower_p', [], ...
                   'profile_candidate_upper_p', [], ...
                   'profile_refined_p', [], ...
                   'profile_refined_loglik', [], 'profile_refined_exitflag', [], ...
                   'profile_basin_p', [], 'profile_basin_loglik', [], ...
                   'profile_basin_source', strings(0, 1));
end
