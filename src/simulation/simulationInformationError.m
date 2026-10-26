function err = simulationInformationError(d, p_true, p_hat, G_true, G_hat)
    %SIMULATIONINFORMATIONERROR Split one replicate's information error into p and R.
    %
    % For a Student-t copula every information measure is the sum of a term that depends only on the correlation matrix and a term that depends only on the shape parameter:
    %
    %   M(p, R) = M_G(R) + Delta_M(d, p)
    %
    % Here M_G comes from gaussianInformationMeasures and Delta_M from tInformationDifferences, for M in {MI, TC, DTC, O} (MI at d = 2 only, NaN above).
    % Two different quantities follow from that identity, and this function keeps them apart because they are not the same estimand.
    %
    % The correction, Delta_M(d, p) in nats, is what the study reports for a channel set; it is a function of the fitted shape and the dimension only and carries no correlation at all:
    %
    %   delta_true = Delta(d, p_true)    the truth's correction
    %   delta_hat  = Delta(d, p_hat)     the fitted correction
    %   delta_err  = delta_hat - delta_true
    %
    % For the full measure, M(p, R), the estimator misses both arguments, p_hat for p_true and R_hat for R_true, and the four substitutions those two generate attribute its error:
    %
    %   full_true    = M(p_true, R_true)    the truth
    %   full_hat     = M(p_hat,  R_hat)     what the estimator reports
    %   full_p_only  = M(p_hat,  R_true)    only the shape estimate is used
    %   full_R_only  = M(p_true, R_hat)     only the correlation estimate is used
    %
    % The error terms are:
    %
    %   err             = full_hat    - full_true
    %   err_p_only      = full_p_only - full_true   ( = Delta(p_hat) - Delta(p_true) )
    %   err_R_only      = full_R_only - full_true   ( = M_G(R_hat)   - M_G(R_true)  )
    %   err_interaction = err - err_p_only - err_R_only
    %
    % By the same additivity err_p_only equals delta_err: the shape part of the full-measure error is exactly the error of the correction.
    % Both are reported, because one is a component of an attribution and the other is the error of the quantity the study publishes, so a reader of either column need not derive it from the other.
    %
    % The interaction is identically zero, to floating point, because the additive split leaves no term in which the two errors multiply.
    % It is computed and reported anyway as the arithmetic check that the p-only and R-only attributions are complete; a nonzero value in a summary would mean the identity is broken.
    %
    % Inputs:
    %   d      - dimension, integer >= 2.
    %   p_true - true shape parameter p = 1/nu, finite and >= 0.
    %   p_hat  - estimated shape parameter, finite and >= 0, or NaN when the fit failed (a non-finite or negative value counts as a failed fit).
    %            A failed fit makes every output that uses p_hat NaN, while the truth's columns, full_R_only and err_R_only keep their values.
    %   G_true - gaussianInformationMeasures(R_true).
    %   G_hat  - gaussianInformationMeasures(R_hat).
    %
    % Output:
    %   err - flat struct of values in nats, for each measure M in MI, TC, DTC, O:
    %         M_delta_true, M_delta_hat, M_delta_err,
    %         M_full_true, M_full_hat, M_full_p_only, M_full_R_only,
    %         M_err, M_err_p_only, M_err_R_only, M_err_interaction.

    arguments
        d (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(d, 2)}
        p_true (1, 1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
        p_hat (1, 1) double {mustBeReal}
        G_true (1, 1) struct
        G_hat (1, 1) struct
    end

    measures = {'MI', 'TC', 'DTC', 'O'};
    required = [measures, {'valid'}];
    missing = required(~isfield(G_true, required));
    assert(isempty(missing), 'simulationInformationError:BadGaussianTruth', ...
        'G_true has no %s field; gaussianInformationMeasures supplies it.', ...
        strjoin(missing, ', '));
    missing = required(~isfield(G_hat, required));
    assert(isempty(missing), 'simulationInformationError:BadGaussianEstimate', ...
        'G_hat has no %s field; gaussianInformationMeasures supplies it.', ...
        strjoin(missing, ', '));

    shape_true = tInformationDifferences(d, p_true);
    if isfinite(p_hat) && p_hat >= 0
        shape_hat = tInformationDifferences(d, p_hat);
    else
        % A failed fit has no shape estimate.
        % NaN travels through every column that uses it rather than a fabricated Gaussian substitute.
        shape_hat = struct('MI', NaN, 'TC', NaN, 'DTC', NaN, 'O', NaN);
    end

    err = struct();
    for k = 1:numel(measures)
        name = measures{k};
        delta_true = shape_true.(name);
        delta_hat = shape_hat.(name);
        full_true = G_true.(name) + delta_true;
        full_hat = G_hat.(name) + delta_hat;
        full_p_only = G_true.(name) + delta_hat;
        full_R_only = G_hat.(name) + delta_true;

        err.([name '_delta_true']) = delta_true;
        err.([name '_delta_hat']) = delta_hat;
        err.([name '_delta_err']) = delta_hat - delta_true;
        err.([name '_full_true']) = full_true;
        err.([name '_full_hat']) = full_hat;
        err.([name '_full_p_only']) = full_p_only;
        err.([name '_full_R_only']) = full_R_only;
        err.([name '_err']) = full_hat - full_true;
        err.([name '_err_p_only']) = full_p_only - full_true;
        err.([name '_err_R_only']) = full_R_only - full_true;
        err.([name '_err_interaction']) = (full_hat - full_true) ...
            - (full_p_only - full_true) - (full_R_only - full_true);
    end
end
