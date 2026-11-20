% ZSCORECOLUMNSSAFE Z-score each column, setting near-constant columns to zero.
%
%   [X_z, info] = zscoreColumnsSafe(X)
%
% Each column is centred by its mean and divided by its sample standard deviation, both over its finite entries.
% A column whose standard deviation is at most max(1e-12, 1e-10 * scale), with scale the larger of its absolute mean and its mean absolute value, is set to zero.
% Non-finite entries become NaN, and a column without any finite entry is returned as zeros.
%
% Inputs:
%   X - [samples x channels] numeric matrix.
%
% Outputs:
%   X_z  - standardized matrix, the same size and class as X.
%   info - struct with method, and per column constant_tol, valid_counts, means and scales, plus constant_channels (the near-constant or all-non-finite columns).

function [X_z, info] = zscoreColumnsSafe(X)

    orig_class = class(X);
    X_work = double(X);
    [n_samples, n_channels] = size(X_work);
    X_z = zeros(size(X_work));

    means = nan(1, n_channels);
    scales = nan(1, n_channels);
    constant_channels = [];
    valid_counts = zeros(1, n_channels);
    tolerances = nan(1, n_channels);

    for j = 1:n_channels
        x_col = X_work(:, j);
        valid_idx = isfinite(x_col);
        valid_x = x_col(valid_idx);

        valid_counts(j) = nnz(valid_idx);

        if isempty(valid_x)
            constant_channels(end + 1) = j; %#ok<AGROW>
            continue
        end

        mu = mean(valid_x);
        centered = valid_x - mu;

        if numel(valid_x) >= 2
            sigma = std(valid_x, 0);
        else
            sigma = 0;
        end

        means(j) = mu;
        scales(j) = sigma;
        scale_ref = max(abs(mu), mean(abs(valid_x)));
        tol = max(1e-12, 1e-10 * scale_ref);
        tolerances(j) = tol;

        if ~isfinite(sigma) || sigma <= tol
            z_valid = zeros(size(valid_x));
            constant_channels(end + 1) = j; %#ok<AGROW>
        else
            z_valid = centered ./ sigma;
        end

        z_col = nan(size(x_col));
        z_col(valid_idx) = z_valid;
        X_z(:, j) = z_col;
    end

    X_z = cast(X_z, orig_class);

    info = struct();
    info.method = 'columnwise_zscore';
    info.constant_tol = tolerances;
    info.constant_channels = constant_channels;
    info.valid_counts = valid_counts;
    info.means = means;
    info.scales = scales;

end
