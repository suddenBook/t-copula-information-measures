function tables = simulationSummaries(recovery_long, factorial_long)
    % SIMULATIONSUMMARIES The simulation study's summary, floor and spread-law tables.
    %
    % Turns the two per-replicate long tables into the four tables the renderers read.
    % Every statistic of the estimate is taken over the numerically ok replicates of a cell only, and the n_ok column beside it counts them, so a thin cell is visible rather than merely quiet.
    %
    % The replicates are paths of the Student-t diffusion in units of tau, so the amount of data is the observation window T/tau (with n = T/tau x samples per tau fitted rows).
    % A cell is (p_true, d, T_over_tau, samples_per_tau): the sampling rate is an axis of its own, whose cells observe the same paths at nested rates (simulationJobs).
    %
    % Tables produced:
    %   recovery_summary  One row per recovery cell: the recovery of p (median and mean bias, sd, quartiles, relative bias, the fraction estimated exactly at the Gaussian bound p = 0 and the fraction flagged at the heavy end) and, per information measure, the error of the correction in nats (the study's estimand) and, separately, the error of the full measure with its p-only, R-only and interaction attributions.
    %                     Beside them are the window (T_over_tau, samples_per_tau, n, refine, h), the model's closed-form radial time of the parent path (diffusionRadialTime) and the median Metropolis acceptance.
    %   floor_by_d_T_s    The Gaussian-truth floor.
    %                     At p_true = 0 the truth's correction is exactly zero, so the fitted correction Delta(d, p_hat) is the positive finite-sample departure a perfectly Gaussian path still produces, and its 95th and 99th percentiles per (d, T_over_tau, samples_per_tau, measure) are the smallest correction a recording of that window could be claimed to show.
    %                     The full-measure error is reported beside it at the same percentiles, as a separate pair of columns, because it also carries the error of the Gaussian correlation term.
    %   spread_law_by_d_s Per (d, samples_per_tau), the median over the p_true > 0 cells of iqr(p_hat) / p_true * sqrt(T_over_tau); a root-T law makes this flat in the window, which is what the column checks.
    %   r_factorial_summary  The same statistics per (design, d), with the design's condition number and smallest eigenvalue.
    %
    % Inputs:
    %   recovery_long  - table of recovery replicates (the columns Phase 7 writes to text/recovery_long.csv).
    %   factorial_long - (optional) table of factorial replicates; an empty table yields an empty r_factorial_summary.
    %
    % Output:
    %   tables - struct with recovery_summary, floor_by_d_T_s, spread_law_by_d_s and r_factorial_summary.

    arguments
        recovery_long table
        factorial_long table = table()
    end

    measures = {'MI', 'TC', 'DTC', 'O'};

    recovery_summary = localSummaryByCell(recovery_long, measures, {});
    tables = struct( ...
                    'recovery_summary', recovery_summary, ...
                    'floor_by_d_T_s', localFloorByDTS(recovery_long, measures), ...
                    'spread_law_by_d_s', localSpreadLawByDS(recovery_summary), ...
                    'r_factorial_summary', localSummaryByCell(factorial_long, measures, ...
                                                  {'design', 'cond_R', 'min_eig_R'}));
end

function summary = localSummaryByCell(long, measures, extra_keys)
    % One row per cell.
    % EXTRA_KEYS are columns that are constant within a cell and belong in the summary (the factorial design's name and conditioning); they are carried from the cell's first replicate.
    if isempty(long)
        summary = table();
        return
    end
    localRequireColumns(long, [{'cell', 'p_true', 'd', 'T_over_tau', ...
                                'samples_per_tau', 'n', 'refine', 'h', 'parent_dim', 'acceptance', 'rep', ...
                                'p_hat', 'numerical_ok', 'heavy_boundary'}, extra_keys]);

    [group, cell_ids] = findgroups(long.cell);
    rows = cell(1, numel(cell_ids));
    for g = 1:numel(cell_ids)
        rows{g} = localCellRow(long(group == g, :), measures, extra_keys);
    end
    summary = vertcat(rows{:});
end

function row = localCellRow(cell_long, measures, extra_keys)
    % One cell's summary row: the recovery statistics of p over the numerically ok replicates, then one block of columns per information measure.
    ok = logical(cell_long.numerical_ok);
    p_true = cell_long.p_true(1);
    T_over_tau = cell_long.T_over_tau(1);
    p_hat = cell_long.p_hat(ok);

    quartiles = localQuantile(p_hat, [0.25, 0.75]);
    iqr_p = quartiles(2) - quartiles(1);
    median_bias = median(p_hat - p_true, 'omitnan');
    if p_true > 0
        relative_bias = median_bias / p_true;
        spread_law = iqr_p / p_true * sqrt(T_over_tau);
    else
        % A relative bias and a root-T spread law have no meaning at the Gaussian truth; the floor table reports that cell.
        relative_bias = NaN;
        spread_law = NaN;
    end

    row = table( ...
                cell_long.cell(1), p_true, 1 / p_true, cell_long.d(1), T_over_tau, ...
                cell_long.samples_per_tau(1), cell_long.n(1), cell_long.refine(1), ...
                cell_long.h(1), height(cell_long), nnz(ok), ...
                localFraction(p_hat == 0), ...
                localFraction(logical(cell_long.heavy_boundary(ok))), ...
                median_bias, mean(p_hat - p_true, 'omitnan'), std(p_hat, 'omitnan'), ...
                quartiles(1), quartiles(2), iqr_p, relative_bias, spread_law, ...
                diffusionRadialTime(1 / p_true, cell_long.parent_dim(1)), ...
                median(cell_long.acceptance, 'omitnan'), ...
                'VariableNames', {'cell', 'p_true', 'nu_true', 'd', 'T_over_tau', ...
                          'samples_per_tau', 'n', 'refine', 'h', 'reps', ...
                          'n_ok', 'fraction_p0', 'fraction_heavy', 'median_bias', ...
                          'mean_bias', 'sd', 'q25', 'q75', 'iqr', ...
                          'relative_bias', 'spread_law', 'radial_time_tau', ...
                          'acceptance_median'});

    for k = 1:numel(extra_keys)
        row.(extra_keys{k}) = cell_long.(extra_keys{k})(1);
    end
    for k = 1:numel(measures)
        row = [row, localMeasureRow(cell_long, ok, measures{k})]; %#ok<AGROW>
    end
end

function row = localMeasureRow(cell_long, ok, measure)
    % One information measure in this cell, reported as two quantities.
    % The _delta_ columns are the error of the correction in nats, which is the study's estimand and a function of the fitted shape alone; the _full_ columns are the error of the full measure, which the p-only, R-only and interaction columns attribute.
    % Each relative error divides by the magnitude of its own truth, which is constant within the cell, so it is a scale and not a per-replicate ratio.
    delta_err = cell_long.([measure '_delta_err'])(ok);
    full_err = cell_long.([measure '_err'])(ok);
    p_only = cell_long.([measure '_err_p_only'])(ok);
    r_only = cell_long.([measure '_err_R_only'])(ok);
    interaction = cell_long.([measure '_err_interaction'])(ok);
    delta_true = cell_long.([measure '_delta_true'])(1);
    full_true = cell_long.([measure '_full_true'])(1);

    delta_median_error = median(delta_err, 'omitnan');
    full_median_error = median(full_err, 'omitnan');

    row = table(delta_true, delta_median_error, mean(delta_err, 'omitnan'), ...
                std(delta_err, 'omitnan'), ...
                localRelative(delta_median_error, delta_true), ...
                full_median_error, mean(full_err, 'omitnan'), ...
                std(full_err, 'omitnan'), ...
                localRelative(full_median_error, full_true), ...
                median(p_only, 'omitnan'), median(r_only, 'omitnan'), ...
                localMaxAbs(r_only), median(interaction, 'omitnan'), ...
                'VariableNames', strcat(measure, {'_delta_true', ...
                                          '_delta_median_error', '_delta_mean_error', '_delta_sd', ...
                                          '_delta_relative_error', '_full_median_error', ...
                                          '_full_mean_error', '_full_sd', '_full_relative_error', ...
                                          '_p_only_median', '_R_only_median', '_R_only_max_abs', ...
                                          '_interaction_median'}));
end

function floor_table = localFloorByDTS(recovery_long, measures)
    % The Gaussian-truth floor: the correction the estimator fits on Gaussian data, Delta(d, p_hat), pooled over the replicates of each (d, T_over_tau, samples_per_tau) cell and read at its upper percentiles.
    % At p_true = 0 the true correction is exactly zero, so the fitted one is the whole departure, and it is the quantity an observed correction in nats is compared against.
    % The full-measure error is a different number, since it also carries the error of the Gaussian correlation term, and is reported beside it rather than in its place.
    if isempty(recovery_long)
        floor_table = table();
        return
    end
    keep = recovery_long.p_true == 0 & logical(recovery_long.numerical_ok);
    gaussian_truth = recovery_long(keep, :);
    if isempty(gaussian_truth)
        floor_table = table();
        return
    end

    [group, d_values, T_values, s_values] = findgroups(gaussian_truth.d, ...
                                                       gaussian_truth.T_over_tau, gaussian_truth.samples_per_tau);
    rows = cell(1, numel(d_values) * numel(measures));
    next = 0;
    for g = 1:numel(d_values)
        block = gaussian_truth(group == g, :);
        for k = 1:numel(measures)
            delta_hat = block.([measures{k} '_delta_hat']);
            full_err = block.([measures{k} '_err']);
            n_ok = nnz(isfinite(delta_hat));
            if n_ok == 0
                % MI exists at d = 2 only, so above it there is no floor to report; an all-NaN row would draw as an empty panel.
                continue
            end
            q = localQuantile(delta_hat, [0.95, 0.99]);
            q_full = localQuantile(full_err, [0.95, 0.99]);
            next = next + 1;
            rows{next} = table(d_values(g), T_values(g), ...
                               s_values(g), block.n(1), ...
                               string(measures{k}), n_ok, q(1), q(2), q_full(1), q_full(2), ...
                               'VariableNames', {'d', 'T_over_tau', 'samples_per_tau', 'n', ...
                                  'measure', 'n_ok', 'q95', 'q99', 'q95_full_err', ...
                                  'q99_full_err'});
        end
    end
    floor_table = vertcat(rows{1:next});
end

function law = localSpreadLawByDS(recovery_summary)
    % One number per (d, samples_per_tau): the median over every p_true > 0 cell (all windows) of iqr(p_hat) / p_true * sqrt(T_over_tau), with the quartiles and the cell count of that sample.
    if isempty(recovery_summary)
        law = table();
        return
    end
    usable = recovery_summary(isfinite(recovery_summary.spread_law), :);
    if isempty(usable)
        law = table();
        return
    end
    [group, d_values, s_values] = findgroups(usable.d, usable.samples_per_tau);
    quartiles = splitapply(@(x) localQuantile(x, [0.25, 0.75]), ...
                           usable.spread_law, group);
    law = table(d_values, s_values, ...
                splitapply(@(x) median(x, 'omitnan'), usable.spread_law, group), ...
                quartiles(:, 1), quartiles(:, 2), ...
                splitapply(@numel, usable.spread_law, group), ...
                'VariableNames', {'d', 'samples_per_tau', 'spread_constant', 'q25', 'q75', ...
                          'n_cells'});
end

function fraction = localFraction(flags)
    if isempty(flags)
        fraction = NaN;
    else
        fraction = mean(double(flags));
    end
end

function value = localRelative(numerator, truth)
    % A ratio against a truth of zero, or against one the dimension does not define, is not a number the study can use, so it is NaN; the nats column beside it carries the error.
    if isfinite(truth) && truth ~= 0
        value = numerator / abs(truth);
    else
        value = NaN;
    end
end

function value = localMaxAbs(x)
    x = abs(x(isfinite(x)));
    if isempty(x)
        value = NaN;
    else
        value = max(x);
    end
end

function q = localQuantile(x, probabilities)
    % Quantiles of the finite values of x, or NaN when there are none, so a cell with no usable replicate reports NaN rather than stopping the finalize of every other cell.
    x = x(isfinite(x));
    if isempty(x)
        q = nan(size(probabilities));
    else
        q = quantile(x, probabilities);
    end
end

function localRequireColumns(T, names)
    missing = names(~ismember(names, T.Properties.VariableNames));
    assert(isempty(missing), 'simulationSummaries:MissingColumn', ...
           'The long table has no %s column; it has %d rows and %d columns.', ...
           strjoin(missing, ', '), height(T), width(T));
end
