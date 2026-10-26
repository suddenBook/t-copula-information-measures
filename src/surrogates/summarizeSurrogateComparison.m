function [comparison, floors, half] = summarizeSurrogateComparison(observed, surrogate, ...
        seed_base, opts)
    %SUMMARIZESURROGATECOMPARISON Observed against surrogate 1/nu_hat, per dimension.
    %
    % [comparison, floors, half] = summarizeSurrogateComparison(observed, surrogate, seed_base, opts) reduces one recording's Phase 3 reference to its two tables.
    % Phase 3 refits every Phase 2 channel set end to end on two coherent surrogate realizations of the whole recording: b = 1 is the reported reference, and b = 2 serves only as a yardstick.
    % At each dimension the pooled distribution of p_hat = 1/nu_hat over the channel sets is compared between the observed fits and the surrogate fits by the overlap coefficient OVL of distributionOverlap (mass outside = 1 - OVL).
    % The comparison is descriptive and has no p-value.
    %
    % Two yardsticks show how far OVL falls below 1 for a distribution that is a property of the recording rather than of one random draw:
    %   ovl_sur_sur2   - OVL between the two surrogate realizations, how far two draws of the same Gaussian-copula reference differ;
    %   ovl_obs_halves - OVL between two disjoint halves of the observed channel sets, how far two halves of the same observed population differ.
    % The halves yardstick compares samples half the size of the observed one, so it is expected to sit somewhat below a same-size comparison.
    %
    % The halves: at each dimension the channel sets, ordered by unit_index, are split by randperm on RandStream('twister', 'Seed', channelGroupSeed(seed_base, d) + opts.halves_seed_offset).
    % The first ceil(m/2) entries of the permutation form half 1 and the rest half 2.
    % The split depends only on the recording's own seed_base, the dimension and the unit indices, never on a fit outcome or the row order, so it is reproducible and the two halves differ in size by at most one.
    % The offset keeps the stream apart from the Phase 2 channel-set sample, which draws on channelGroupSeed(seed_base, d) itself.
    %
    % The population is the numerically successful fits.
    % A failed fit is not an estimate, so it enters no overlap, quantile, fraction or floor; the n_*_ok columns count the fits that do, and the status columns count every realization fit by its status.
    % The heavy-end fraction is taken over the successful fits, as in the Phase 2 summary_by_d table.
    %
    % The floors are the information differences the reference itself produces: tInformationDifferences(d, p_hat) at every successful surrogate fit, pooled per dimension, realization and measure.
    % Each floor row holds n and the 50th, 90th, 95th and 99th percentiles of the difference values themselves, not differences evaluated at percentiles of p.
    % Mutual information exists at d = 2 only, so its rows are written there only; a realization with no successful fit at a dimension keeps its rows, with n = 0.
    %
    % Inputs:
    %   observed  - struct of [m x 1] columns, one row per Phase 2 channel set: d, unit_index, p_hat, numerical_ok, heavy_boundary.
    %   surrogate - struct of [m x 2] columns, the same channel sets on realizations 1 and 2: p_hat, numerical_ok, heavy_boundary, and status, whose codes index the cellstr status_labels.
    %               status_labels must hold only the five unit statuses of tFitUnit.
    %   seed_base - the recording's stable seed_base from recording.json.
    %   opts      - struct with overlap_bins (equal-count bins of distributionOverlap), atoms (the exact values with a cell of their own: the domain endpoints of p), quantile_levels (per cent) and halves_seed_offset.
    % Outputs:
    %   comparison - table, one row per dimension, with columns
    %       d, n_units, n_obs_ok, n_sur1_ok, n_sur2_ok,
    %       ovl_obs_sur (observed against realization 1), mass_outside (1 - ovl_obs_sur), ovl_sur_sur2, ovl_obs_halves,
    %       obs_qLL, sur_qLL, sur2_qLL at every quantile level LL (2.5 -> q025, 5 -> q05, 97.5 -> q975; sur is realization 1, sur2 realization 2),
    %       obs_ / sur_ / sur2_fraction_p0 (exactly Gaussian),
    %       obs_ / sur_ / sur2_fraction_heavy (the heavy_boundary flag),
    %       sur_status_<status> and sur2_status_<status> for ok, heavy_boundary, ambiguous_profile, numerical_failure and degenerate_pair.
    %   floors     - table (d, realization, measure, n, q50, q90, q95, q99).
    %   half       - [m x 1] double, the observed half (1 or 2) of every channel set.

    arguments
        observed (1, 1) struct
        surrogate (1, 1) struct
        seed_base (1, 1) double {mustBeInteger, mustBeNonnegative}
        opts (1, 1) struct
    end

    required_opts = {'overlap_bins', 'atoms', 'quantile_levels', 'halves_seed_offset'};
    missing = required_opts(~isfield(opts, required_opts));
    assert(isempty(missing), 'summarizeSurrogateComparison:MissingOption', ...
        'opts carries no %s field.', strjoin(missing, ', '));
    [d_all, unit_index, p_obs, ok_obs, heavy_obs] = localObservedColumns(observed);
    m = numel(d_all);
    [p_sur, ok_sur, heavy_sur, status_sur] = localSurrogateColumns(surrogate, m);

    statuses = {'ok', 'heavy_boundary', 'ambiguous_profile', 'numerical_failure', ...
        'degenerate_pair'};
    measures = {'MI', 'TC', 'DTC', 'O'};
    levels = reshape(opts.quantile_levels, 1, []);
    level_names = arrayfun(@localLevelName, levels, 'UniformOutput', false);
    samples = {'obs', 'sur', 'sur2'};

    dims = reshape(unique(d_all), [], 1);
    names = [{'d', 'n_units', 'n_obs_ok', 'n_sur1_ok', 'n_sur2_ok', ...
        'ovl_obs_sur', 'mass_outside', 'ovl_sur_sur2', 'ovl_obs_halves'}, ...
        localPrefixed(samples, level_names), ...
        localPrefixed(samples, {'fraction_p0'}), ...
        localPrefixed(samples, {'fraction_heavy'}), ...
        localPrefixed({'sur', 'sur2'}, strcat('status_', statuses))];
    comparison = array2table(nan(numel(dims), numel(names)), 'VariableNames', names);
    comparison.d = dims;

    half = localHalves(d_all, unit_index, seed_base, opts.halves_seed_offset);
    floor_rows = cell(0, 1);
    for k = 1:numel(dims)
        at_d = d_all == dims(k);
        % The three samples of this dimension, each over its successful fits only.
        values = {p_obs(at_d & ok_obs), p_sur(at_d & ok_sur(:, 1), 1), ...
            p_sur(at_d & ok_sur(:, 2), 2)};
        heavy = {heavy_obs(at_d & ok_obs), heavy_sur(at_d & ok_sur(:, 1), 1), ...
            heavy_sur(at_d & ok_sur(:, 2), 2)};

        comparison.n_units(k) = nnz(at_d);
        comparison.n_obs_ok(k) = numel(values{1});
        comparison.n_sur1_ok(k) = numel(values{2});
        comparison.n_sur2_ok(k) = numel(values{3});
        comparison.ovl_obs_sur(k) = distributionOverlap(values{1}, values{2}, ...
            opts.overlap_bins, opts.atoms);
        comparison.mass_outside(k) = 1 - comparison.ovl_obs_sur(k);
        comparison.ovl_sur_sur2(k) = distributionOverlap(values{2}, values{3}, ...
            opts.overlap_bins, opts.atoms);
        comparison.ovl_obs_halves(k) = distributionOverlap( ...
            p_obs(at_d & ok_obs & half == 1), p_obs(at_d & ok_obs & half == 2), ...
            opts.overlap_bins, opts.atoms);

        for s = 1:numel(samples)
            if isempty(values{s})
                continue;      % no successful fit: the statistics stay NaN
            end
            quantiles = prctile(values{s}, levels);
            for q = 1:numel(levels)
                comparison.([samples{s} '_' level_names{q}])(k) = quantiles(q);
            end
            comparison.([samples{s} '_fraction_p0'])(k) = mean(values{s} == 0);
            comparison.([samples{s} '_fraction_heavy'])(k) = mean(heavy{s});
        end
        for b = 1:2
            for s = 1:numel(statuses)
                comparison.([samples{b + 1} '_status_' statuses{s}])(k) = ...
                    nnz(status_sur(at_d, b) == string(statuses{s}));
            end
            floor_rows{end + 1, 1} = localFloorRows(dims(k), b, ...
                p_sur(at_d & ok_sur(:, b), b), measures); %#ok<AGROW>
        end
    end
    floors = vertcat(floor_rows{:});
end


function [d_all, unit_index, p_obs, ok_obs, heavy_obs] = localObservedColumns(observed)
    % The observed columns, checked for presence and a common length.
    required = {'d', 'unit_index', 'p_hat', 'numerical_ok', 'heavy_boundary'};
    missing = required(~isfield(observed, required));
    assert(isempty(missing), 'summarizeSurrogateComparison:MissingField', ...
        'The observed columns carry no %s field.', strjoin(missing, ', '));
    d_all = double(observed.d(:));
    m = numel(d_all);
    assert(m > 0, 'summarizeSurrogateComparison:NoUnits', ...
        'The comparison needs at least one channel set; observed.d is empty.');
    for name = required
        assert(numel(observed.(name{1})) == m, ...
            'summarizeSurrogateComparison:ObservedLengthMismatch', ...
            ['observed.%s holds %d values but observed.d holds %d; every observed ' ...
            'column has one row per channel set.'], name{1}, ...
            numel(observed.(name{1})), m);
    end
    unit_index = double(observed.unit_index(:));
    p_obs = double(observed.p_hat(:));
    ok_obs = logical(observed.numerical_ok(:));
    heavy_obs = logical(observed.heavy_boundary(:));
end


function [p_sur, ok_sur, heavy_sur, status_sur] = localSurrogateColumns(surrogate, m)
    % The two realizations' columns, checked for shape and for known statuses, with the status codes decoded into their labels.
    required = {'p_hat', 'numerical_ok', 'heavy_boundary', 'status', 'status_labels'};
    missing = required(~isfield(surrogate, required));
    assert(isempty(missing), 'summarizeSurrogateComparison:MissingField', ...
        'The surrogate columns carry no %s field.', strjoin(missing, ', '));
    for name = required(1:4)
        assert(isequal(size(surrogate.(name{1})), [m, 2]), ...
            'summarizeSurrogateComparison:RealizationShapeMismatch', ...
            ['surrogate.%s must be [%d x 2] -- one row per channel set, realization ' ...
            '1 (reported) and realization 2 (the yardstick); got %s.'], ...
            name{1}, m, mat2str(size(surrogate.(name{1}))));
    end
    labels = string(surrogate.status_labels(:));
    known = ["ok", "heavy_boundary", "ambiguous_profile", "numerical_failure", ...
        "degenerate_pair"];
    unknown = setdiff(labels, known);
    assert(isempty(unknown), 'summarizeSurrogateComparison:UnknownStatus', ...
        'surrogate.status_labels holds the unknown unit status(es) %s.', ...
        strjoin(unknown, ', '));
    codes = double(surrogate.status);
    assert(all(codes >= 1 & codes <= numel(labels), 'all'), ...
        'summarizeSurrogateComparison:StatusCodeOutOfRange', ...
        'surrogate.status holds codes outside 1..%d, the length of status_labels.', ...
        numel(labels));
    status_sur = reshape(labels(codes), m, 2);
    p_sur = double(surrogate.p_hat);
    ok_sur = logical(surrogate.numerical_ok);
    heavy_sur = logical(surrogate.heavy_boundary);
end


function half = localHalves(d_all, unit_index, seed_base, offset)
    % The seeded 50/50 split of every dimension's channel sets, keyed on unit_index.
    half = nan(numel(d_all), 1);
    for d = reshape(unique(d_all), 1, [])
        at_d = find(d_all == d);
        [~, order] = sort(unit_index(at_d));
        at_d = at_d(order);
        stream = RandStream('twister', 'Seed', channelGroupSeed(seed_base, d) + offset);
        permutation = randperm(stream, numel(at_d));
        first = permutation(1:ceil(numel(at_d) / 2));
        half(at_d) = 2;
        half(at_d(first)) = 1;
    end
end


function rows = localFloorRows(d, b, p, measures)
    % The floor rows of one dimension and realization: every measure defined at d, with n = 0 and NaN percentiles when no fit succeeded.
    if isempty(p)
        differences = struct('MI', zeros(0, 1), 'TC', zeros(0, 1), ...
            'DTC', zeros(0, 1), 'O', zeros(0, 1));
    else
        differences = tInformationDifferences(d, p);
    end
    kept = measures(~strcmp(measures, 'MI') | d == 2);
    rows = cell(numel(kept), 1);
    for k = 1:numel(kept)
        values = differences.(kept{k});
        percentiles = nan(1, 4);
        if ~isempty(values)
            percentiles = reshape(prctile(values, [50, 90, 95, 99]), 1, []);
        end
        rows{k} = table(d, b, kept(k), numel(values), percentiles(1), ...
            percentiles(2), percentiles(3), percentiles(4), 'VariableNames', ...
            {'d', 'realization', 'measure', 'n', 'q50', 'q90', 'q95', 'q99'});
    end
    rows = vertcat(rows{:});
end


function name = localLevelName(level)
    % A percentile level as a column suffix: 2.5 -> q025, 5 -> q05, 97.5 -> q975.
    if level == round(level)
        name = sprintf('q%02d', level);
    else
        name = ['q' strrep(sprintf('%04.1f', level), '.', '')];
    end
end


function names = localPrefixed(prefixes, stems)
    % Every prefix joined to every stem, prefix by prefix: {'obs_q025', ...}.
    names = cell(1, 0);
    for p = 1:numel(prefixes)
        names = [names, strcat(prefixes{p}, '_', stems)]; %#ok<AGROW>
    end
end
