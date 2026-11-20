function row = tFitUnit(U_sub, R, corr, fit_opts, cache)
    % TFITUNIT One flat unit row from prepared pseudo-observations and correlation.
    %
    % row = tFitUnit(U_sub, R, corr, fit_opts, cache) runs the profile fit of the Student-t shape parameter p = 1/nu at fixed R and packs the result into the study's unit row.
    % The field set is the same on every path, with NaN or false where a quantity does not apply, so rows from any mixture of dimensions, statuses and phases concatenate into one struct array and one table.
    %
    % A unit is dropped only on a numerical failure (row.numerical_ok false).
    % The heavy end of the p domain, a near-tied remote basin and a gross correlation repair are recorded conditions, each with its own boolean column, and their units stay reportable.
    %
    % Inputs:
    %   U_sub    - [n x d] pseudo-observations strictly in (0, 1), complete rows.
    %   R        - [d x d] correlation matrix from unitCorrelationFromTau.
    %   corr     - the audit struct unitCorrelationFromTau returns with R.
    %   fit_opts - (optional) tFitDefaultOptions overrides.
    %   cache    - (optional) tCopulaQuantileCache for this recording's rank plan.
    %
    % Outputs:
    %   row - struct whose fields, in this order, are
    %     identity     dataset_key, d, unit_index, channels (1 x 20, NaN-padded), n
    %     fit          p_hat, nu_hat, loglik_t, loglik_gaussian, aic_diff
    %     correlation  rho, tau, max_abs_rho, mean_abs_rho, logdet_R, sum_logdet_minors, cond_R, min_eig_R, repair_applied, repair_shift_rms, repair_gross, degenerate_pair
    %     outcome      gaussian_snapped, ambiguous_profile, heavy_boundary, numerical_ok, status, reason, optimizer_exitflag
    %     profile      profile_p_grid, profile_loglik_grid, fit_seconds
    %   aic_diff is the Gaussian AIC minus the t AIC, 2 (loglik_t - loglik_gaussian) - 2.
    %
    % dataset_key, unit_index and channels are the caller's to fill, because this function sees one channel set's numbers, not which recording or unit they came from.
    % tFitChannelSet, the Phase 3 chunk fit and simulationNestedFits fill channels, and the phase scripts and simulationPathJob fill the other two where they need them.
    %
    % status is the first that applies of degenerate_pair, numerical_failure, heavy_boundary, ambiguous_profile and ok.
    % The flags stay as separate columns, so a unit that is both heavy-ended and ambiguous still reads as both.

    arguments
        U_sub (:, :) double {mustBeReal}
        R (:, :) double {mustBeReal}
        corr (1, 1) struct
        fit_opts (1, 1) struct = struct()
        cache = []
    end

    fit_opts = tFitDefaultOptions(fit_opts);
    [n, d] = size(U_sub);
    if size(R, 1) ~= d || size(R, 2) ~= d
        error('tFitUnit:CorrelationShapeMismatch', ...
              ['R must be %d x %d for a %d-column pseudo-observation matrix; ' ...
               'got %d x %d.'], d, d, d, size(R, 1), size(R, 2));
    end

    row = local_blank_row(d, n, corr, fit_opts.profile_grid_size);
    if n < fit_opts.min_rows
        % Checked before the correlation reference, as in fitTCopulaShape, so a channel set with too few complete rows is reported as such and not by whatever condition its few-row Kendall matrix shows.
        row.reason = 'too_few_rows';
        row.status = local_status(row);
        return
    end
    if ~corr.numerical_valid
        % No usable correlation reference, so there is nothing to profile.
        % The audit's own code is the reason, and the flags already say which condition holds.
        row.reason = local_reason(corr.reason, 'invalid_correlation_reference');
        row.status = local_status(row);
        return
    end

    % At d = 2 there is one free parameter and no positive-definite repair, so 1 - |rho| is the minimum eigenvalue, and the fitter's singularity gate is raised to the duplicate-channel threshold of kendallToCorrelation.
    if d == 2
        fit_opts.r_min_eig = max(fit_opts.r_min_eig, 1e-6);
    end

    fit_timer = tic;
    entry = fitTCopulaShape(U_sub, R, fit_opts, cache);
    row.fit_seconds = toc(fit_timer);

    row.p_hat = entry.p_hat;
    row.nu_hat = entry.nu_hat;
    row.loglik_t = entry.loglik;
    row.loglik_gaussian = entry.loglik_gaussian;
    row.aic_diff = 2 * (entry.loglik - entry.loglik_gaussian) - 2;
    row.gaussian_snapped = entry.profile_gaussian_snapped;
    row.ambiguous_profile = entry.ambiguous_profile;
    row.heavy_boundary = entry.heavy_boundary;
    row.numerical_ok = entry.numerical_ok;
    row.optimizer_exitflag = entry.optimizer_exitflag;
    row.profile_p_grid = reshape(entry.profile_p_grid, 1, []);
    row.profile_loglik_grid = reshape(entry.profile_loglik_grid, 1, []);
    row.reason = local_reason(entry.exclusion_reason, '');
    row.status = local_status(row);
end

function row = local_blank_row(d, n, corr, grid_size)
    % Every field of the unit row in its declared order, with the correlation audit merged in; the fit is the only part that can still be missing, and it leaves NaN rather than a sentinel.
    row = struct( ...
                 'dataset_key', '', ...
                 'd', d, ...
                 'unit_index', NaN, ...
                 'channels', nan(1, 20), ...
                 'n', n, ...
                 'p_hat', NaN, ...
                 'nu_hat', NaN, ...
                 'loglik_t', NaN, ...
                 'loglik_gaussian', NaN, ...
                 'aic_diff', NaN, ...
                 'rho', corr.rho, ...
                 'tau', corr.tau, ...
                 'max_abs_rho', corr.max_abs_rho, ...
                 'mean_abs_rho', corr.mean_abs_rho, ...
                 'logdet_R', corr.logdet_R, ...
                 'sum_logdet_minors', corr.sum_logdet_minors, ...
                 'cond_R', corr.cond_R, ...
                 'min_eig_R', corr.min_eig, ...
                 'repair_applied', logical(corr.repair_applied), ...
                 'repair_shift_rms', corr.repair_shift_rms, ...
                 'repair_gross', logical(corr.repair_gross), ...
                 'degenerate_pair', logical(corr.degenerate_pair), ...
                 'gaussian_snapped', false, ...
                 'ambiguous_profile', false, ...
                 'heavy_boundary', false, ...
                 'numerical_ok', false, ...
                 'status', '', ...
                 'reason', '', ...
                 'optimizer_exitflag', NaN, ...
                 'profile_p_grid', nan(1, grid_size), ...
                 'profile_loglik_grid', nan(1, grid_size), ...
                 'fit_seconds', NaN);
end

function status = local_status(row)
    % The single most severe label, in the declared precedence.
    % A degenerate pair is itself a numerical failure of R, so its named label comes first; otherwise it could never appear in the status column.
    if row.degenerate_pair
        status = 'degenerate_pair';
    elseif ~row.numerical_ok
        status = 'numerical_failure';
    elseif row.heavy_boundary
        status = 'heavy_boundary';
    elseif row.ambiguous_profile
        status = 'ambiguous_profile';
    else
        status = 'ok';
    end
end

function reason = local_reason(candidate, fallback)
    % Reason strings travel as char so the row packs into a table column; an empty candidate takes the fallback.
    reason = char(candidate);
    if isempty(reason)
        reason = fallback;
    end
end
