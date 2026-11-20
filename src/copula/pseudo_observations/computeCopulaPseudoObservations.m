function [U, row_mask, diag] = computeCopulaPseudoObservations(X, varargin)
    % COMPUTECOPULAPSEUDOOBSERVATIONS Copula pseudo-observations of the complete-case rows, ranked on that same sample.
    %
    % Complete-case rows are selected first, then each column is ranked on that sample (average ranks for ties), giving U = rank / (n + 1).
    % Selecting rows before ranking makes U the pseudo-observations of exactly the sample that is fitted.
    % The t-copula fit, the KSG reference and the empirical pair density all use this one definition, so their rank policies cannot differ.
    %
    %   [U, row_mask, diag] = computeCopulaPseudoObservations(X, ...)
    %
    % Inputs:
    %   X - [samples x channels] numeric matrix; a row with any non-finite entry is not a complete case.
    %
    % Name-value pairs:
    %   'Rows'        - 'complete' (the default and only supported value; any other is an error)
    %   'MinSamples'  - minimum complete rows for non-NaN U (default 10)
    %
    % Outputs:
    %   U        - [n_complete x d] pseudo-observations in (0, 1), all NaN when fewer than MinSamples rows are complete
    %   row_mask - [n_input x 1] logical, true for the complete-case rows kept
    %   diag     - struct of rank, tie and dropped-row diagnostics

    p = inputParser;
    addRequired(p, 'X', @(M) isnumeric(M) && ismatrix(M));
    addParameter(p, 'Rows', 'complete', @(s) ischar(s) || isstring(s));
    addParameter(p, 'MinSamples', 10, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    parse(p, X, varargin{:});

    rows_mode = lower(char(p.Results.Rows));
    min_samples = p.Results.MinSamples;

    switch rows_mode
        case 'complete'
            row_mask = all(isfinite(X), 2);
        otherwise
            error('computeCopulaPseudoObservations:BadRows', ...
                  'Rows="%s" is not supported. Use Rows="complete".', rows_mode);
    end

    X_fit = X(row_mask, :);
    [n, d] = size(X_fit);
    if n < min_samples
        U = nan(n, d);
        diag = makeDiagnostics(X, row_mask, nan(1, d), nan(1, d), true);
        diag.note = sprintf('Too few complete rows for copula pseudo-observations: %d < %d.', ...
                            n, min_samples);
        return
    end

    U = nan(n, d);
    tie_fraction = zeros(1, d);
    max_tie_count = zeros(1, d);
    n_unique = zeros(1, d);

    for j = 1:d
        x = X_fit(:, j);
        [r, tie_fraction(j), max_tie_count(j), n_unique(j)] = averageRanksWithTieStats(x);
        U(:, j) = r / (n + 1);
    end

    diag = makeDiagnostics(X, row_mask, tie_fraction, max_tie_count, false);
    diag.n_unique = n_unique;
    diag.constant_column = n_unique <= 1;
    diag.has_constant_column = any(diag.constant_column);
    diag.has_ties = any(max_tie_count > 1);
    diag.max_tie_fraction = max(tie_fraction);
    diag.rows_mode = rows_mode;
end

function diag = makeDiagnostics(X, row_mask, tie_fraction, max_tie_count, too_few)
    % The row-selection and tie diagnostics shared by both return paths.
    diag = struct();
    diag.n_input = size(X, 1);
    diag.n_complete = sum(row_mask);
    diag.n_dropped = diag.n_input - diag.n_complete;
    diag.drop_fraction = diag.n_dropped / max(diag.n_input, 1);
    diag.tie_fraction = tie_fraction;
    diag.max_tie_count = max_tie_count;
    diag.too_few_samples = too_few;
    diag.note = '';
end

function [ranks, tie_fraction, max_tie_count, n_unique] = averageRanksWithTieStats(x)
    % Average ranks of x (tied values share the mean of their positions), with the fraction of observations in tied blocks, the largest tied block and the number of distinct values.
    n = numel(x);
    [xs, order] = sort(x);

    ranks_sorted = nan(n, 1);
    max_tie_count = 1;
    tied_observations = 0;
    n_unique = 0;

    start_idx = 1;
    while start_idx <= n
        stop_idx = start_idx;
        while stop_idx < n && xs(stop_idx + 1) == xs(start_idx)
            stop_idx = stop_idx + 1;
        end

        n_unique = n_unique + 1;
        block_count = stop_idx - start_idx + 1;
        if block_count > 1
            tied_observations = tied_observations + block_count;
            max_tie_count = max(max_tie_count, block_count);
        end

        avg_rank = (start_idx + stop_idx) / 2;
        ranks_sorted(start_idx:stop_idx) = avg_rank;
        start_idx = stop_idx + 1;
    end

    ranks = nan(n, 1);
    ranks(order) = ranks_sorted;
    tie_fraction = tied_observations / max(n, 1);
end
