function [correlation, audit] = kendallCorrelationReference( ...
        kendallTau, fitOptions)
    %KENDALLCORRELATIONREFERENCE Build and audit a Kendall-derived correlation reference.
    %
    % The positive-definite repair performed by kendallToCorrelation is a numerical construction, not evidence that the empirical rank correlations are mutually compatible.
    % Every fit consumer therefore shares this audit: a repaired d >= 3 reference is reported as gross when the RMS off-diagonal change exceeds max_projection_shift_rms.
    %
    % Inputs:
    %   kendallTau - [d x d] real finite Kendall tau matrix, square.
    %   fitOptions - option overrides carrying max_projection_shift_rms (default 0.02).
    %
    % Outputs:
    %   correlation - [d x d] correlation matrix from kendallToCorrelation.
    %   audit       - kendallToCorrelation's info plus numerical_valid, off_diag_shift_rms, projection_threshold, projection_gross, valid, reason and fit_exclusion_reason.
    %                 valid is false when kendallToCorrelation rejects the matrix, the shift is not finite or the repair is gross, and reason and fit_exclusion_reason then name the cause.

    arguments
        kendallTau (:, :) double {mustBeReal, mustBeFinite}
        fitOptions (1, 1) struct = struct()
    end

    assert(size(kendallTau, 1) == size(kendallTau, 2), ...
        'kendallCorrelationReference:NonSquareTau', ...
        'The Kendall-tau matrix must be square.');

    % The declared bound on the root-mean-square entry shift of the repair, which fitOptions may override.
    threshold = 0.02;
    if isfield(fitOptions, 'max_projection_shift_rms') ...
            && ~isempty(fitOptions.max_projection_shift_rms)
        threshold = fitOptions.max_projection_shift_rms;
    end
    validateattributes(threshold, {'double'}, {'scalar', 'nonnegative', 'finite'}, ...
        mfilename, 'max_projection_shift_rms');
    [correlation, ~, numericalAudit] = ...
        kendallToCorrelation(kendallTau);

    dimension = size(kendallTau, 1);
    if dimension < 2
        shiftRms = 0;
    else
        shiftRms = numericalAudit.off_diag_shift_l2 ...
            / sqrt(dimension * (dimension - 1));
    end

    numericalValid = logical(numericalAudit.valid);
    projectionFinite = isfinite(shiftRms);
    projectionGross = numericalAudit.projection_applied ...
        && projectionFinite && shiftRms > threshold;
    valid = numericalValid && projectionFinite && ~projectionGross;

    if ~numericalValid
        reason = string(numericalAudit.reason);
        exclusionReason = "R_invalid";
    elseif ~projectionFinite
        reason = "nonfinite_correlation_projection";
        exclusionReason = "R_invalid";
    elseif projectionGross
        reason = "gross_correlation_projection";
        exclusionReason = "R_projection_gross";
    else
        reason = "";
        exclusionReason = "";
    end

    audit = numericalAudit;
    audit.numerical_valid = numericalValid;
    audit.off_diag_shift_rms = shiftRms;
    audit.projection_threshold = threshold;
    audit.projection_gross = projectionGross;
    audit.valid = valid;
    audit.reason = char(reason);
    audit.fit_exclusion_reason = char(exclusionReason);
end
