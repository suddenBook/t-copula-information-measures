function [row, tau, R] = tFitChannelSet(X, grp, fit_opts, cache)
    % TFITCHANNELSET Canonical t-copula unit fit for one channel set (d = 2..20).
    %
    % The one pair and group driver for the Student-t shape parameter p = 1/nu: it runs the four steps of the study's unit recipe on one channel set of a recording and returns one flat unit row (the row tFitUnit defines).
    %
    %   1. pseudo-observations  computeCopulaPseudoObservations (complete rows first, then average ranks, U = rank / (n + 1))
    %   2. Kendall tau-b        kendallTauMatrix over those same complete rows
    %   3. correlation          unitCorrelationFromTau (sine relation plus repair)
    %   4. profile fit          tFitUnit (fitTCopulaShape at fixed R)
    %
    % Invalid data and unusable correlation references come back as rows with numerical_ok = false and a reason, while unexpected numerical or programming exceptions propagate and fail the enclosing checkpoint rather than being disguised as a failed fit.
    %
    % Inputs:
    %   X        - [n_samples x n_channels] signal matrix (samples down the rows).
    %   grp      - [1 x d] 1-based channel indices, 2 <= d <= 20.
    %   fit_opts - (optional) tFitDefaultOptions overrides.
    %   cache    - (optional) tCopulaQuantileCache built for this recording's rank plan; it changes no number, only the cost of the coarse profile.
    %
    % Outputs:
    %   row - the flat unit row, with channels filled from grp and NaN-padded to 20.
    %         dataset_key and unit_index keep their blank values for the phase script to fill.
    %   tau - [d x d] Kendall tau-b matrix of the channel set.
    %   R   - [d x d] correlation matrix.
    %         tau and R are returned because the phase MAT files keep them per unit, while the flat row (and every CSV built from it) carries no variable-length matrix.

    arguments
        X (:, :) double {mustBeReal}
        grp (1, :) double {mustBeInteger, mustBePositive}
        fit_opts (1, 1) struct = struct()
        cache = []
    end

    fit_opts = tFitDefaultOptions(fit_opts);
    d = numel(grp);
    if d < 2 || d > 20
        error('tFitChannelSet:GroupWidthOutOfRange', ...
              ['The row pads channels to width 20, so a channel set holds ' ...
               'between 2 and 20 channels; got d = %d.'], d);
    end
    if max(grp) > size(X, 2)
        error('tFitChannelSet:ChannelOutOfRange', ...
              ['X is [samples x channels] and has %d columns, so channel %d does ' ...
               'not exist (X is %d x %d).'], size(X, 2), max(grp), ...
              size(X, 1), size(X, 2));
    end

    Xg = X(:, grp);
    % MinSamples 1 keeps the row-count decision in one place, fit_opts.min_rows, which tFitUnit enforces and reports as too_few_rows.
    [U, row_mask] = computeCopulaPseudoObservations(Xg, 'MinSamples', 1);
    tau = kendallTauMatrix(Xg(row_mask, :));
    [R, corr] = unitCorrelationFromTau(tau, fit_opts);

    row = tFitUnit(U, R, corr, fit_opts, cache);
    row.channels = [grp, nan(1, 20 - d)];
end
