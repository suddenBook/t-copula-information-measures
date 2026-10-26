function [summary_by_d, status_counts_by_d] = summarizeUnitRowsByDimension(rows)
    %SUMMARIZEUNITROWSBYDIMENSION The two per-dimension tables of a unit sweep.
    %
    % [summary_by_d, status_counts_by_d] = summarizeUnitRowsByDimension(rows) reduces a sweep's flat unit rows (the field set tFitUnit produces) to two tables:
    %
    %   summary_by_d        one row per dimension: how many units the sweep holds (n_units) and how many are reportable (n_ok), where the shape parameter and the degrees of freedom sit, the fraction at p_hat = 0, how often each recorded flag (heavy_boundary, ambiguous_profile, repair_gross) occurs, and the median fit time.
    %   status_counts_by_d  long form (d, status, count) over every status observed in the sweep, so a dimension that never produced a status still carries its zero and the table pivots rectangularly.
    %
    % The denominator is the reportable set.
    % Every location, spread and fraction in summary_by_d is computed over the units with numerical_ok true, which is the set the study reports; n_units, median_fit_seconds and the status counts cover every unit.
    % With no reportable unit at a dimension those statistics are NaN rather than 0, because "no estimate" is not "an estimate of zero".
    %
    % nu_hat = 1/p_hat is Inf at the Gaussian estimate p_hat = 0, which is a valid estimate and not a failure.
    % Its median and quantiles therefore stay meaningful while its mean and standard deviation become Inf and NaN as soon as one unit lands on the bound; that is the honest reading of the quantity, and fraction_p0 is the column that says how much mass sits there.
    %
    % Input:
    %   rows - 1 x n struct array of unit rows, with at least d, p_hat, nu_hat, numerical_ok, heavy_boundary, ambiguous_profile, repair_gross, status and fit_seconds.
    % Outputs:
    %   summary_by_d       - table, one row per dimension, ascending.
    %   status_counts_by_d - table with columns d, status, count.

    arguments
        rows (1, :) struct
    end

    assert(~isempty(rows), 'summarizeUnitRowsByDimension:NoRows', ...
        'A per-dimension summary needs at least one unit row.');
    required = {'d', 'p_hat', 'nu_hat', 'numerical_ok', 'heavy_boundary', ...
        'ambiguous_profile', 'repair_gross', 'status', 'fit_seconds'};
    missing = required(~isfield(rows, required));
    assert(isempty(missing), 'summarizeUnitRowsByDimension:MissingField', ...
        'The unit rows carry no %s field.', strjoin(missing, ', '));

    d_all = [rows.d];
    dims = unique(d_all);
    statuses = unique(string({rows.status}), 'stable');

    n_dims = numel(dims);
    blank = nan(n_dims, 1);
    summary_by_d = table(dims(:), blank, blank, blank, blank, blank, blank, ...
        blank, blank, blank, blank, blank, blank, blank, blank, blank, blank, ...
        blank, 'VariableNames', {'d', 'n_units', 'n_ok', ...
        'median_p_hat', 'q25_p_hat', 'q75_p_hat', 'mean_p_hat', 'sd_p_hat', ...
        'median_nu_hat', 'q25_nu_hat', 'q75_nu_hat', 'mean_nu_hat', 'sd_nu_hat', ...
        'fraction_p0', 'fraction_heavy', 'fraction_ambiguous', ...
        'fraction_repair_gross', 'median_fit_seconds'});

    count_d = zeros(n_dims * numel(statuses), 1);
    count_status = strings(size(count_d));
    count_value = zeros(size(count_d));

    for k = 1:n_dims
        at_d = (d_all == dims(k));
        set = rows(at_d);
        ok = set(logical([set.numerical_ok]));

        summary_by_d.n_units(k) = numel(set);
        summary_by_d.n_ok(k) = numel(ok);
        summary_by_d.median_fit_seconds(k) = median([set.fit_seconds], 'omitnan');
        if ~isempty(ok)
            summary_by_d = localFillStatistics(summary_by_d, k, 'p_hat', [ok.p_hat]);
            summary_by_d = localFillStatistics(summary_by_d, k, 'nu_hat', [ok.nu_hat]);
            summary_by_d.fraction_p0(k) = mean([ok.p_hat] == 0);
            summary_by_d.fraction_heavy(k) = mean(logical([ok.heavy_boundary]));
            summary_by_d.fraction_ambiguous(k) = mean(logical([ok.ambiguous_profile]));
            summary_by_d.fraction_repair_gross(k) = mean(logical([ok.repair_gross]));
        end

        for s = 1:numel(statuses)
            slot = (k - 1) * numel(statuses) + s;
            count_d(slot) = dims(k);
            count_status(slot) = statuses(s);
            count_value(slot) = nnz(string({set.status}) == statuses(s));
        end
    end

    status_counts_by_d = table(count_d, cellstr(count_status), count_value, ...
        'VariableNames', {'d', 'status', 'count'});
end


function summary_by_d = localFillStatistics(summary_by_d, k, name, values)
    % The five location and spread columns of one quantity at one dimension.
    quartiles = quantile(values, [0.25, 0.75]);
    summary_by_d.(['median_' name])(k) = median(values);
    summary_by_d.(['q25_' name])(k) = quartiles(1);
    summary_by_d.(['q75_' name])(k) = quartiles(2);
    summary_by_d.(['mean_' name])(k) = mean(values);
    summary_by_d.(['sd_' name])(k) = std(values);
end
