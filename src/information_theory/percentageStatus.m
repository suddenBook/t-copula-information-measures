function [reportable, statusCode] = percentageStatus( ...
                                                     denominator, minDenominator, extraStable)
    % PERCENTAGESTATUS Suppression rule for one percentage denominator.
    %
    % Each information difference is expressed against two denominators, the Gaussian value of the same measure and the fitted value.
    % Each denominator carries its own suppression rule, because they fail in different places: the fitted value of the O-information can vanish where the Gaussian value does not.
    % This function applies the rule to one denominator at a time, whichever denominator it is given.
    %
    % Inputs:
    %   denominator    - the value the percentage divides by, in nats.
    %   minDenominator - absolute floor in nats below which a ratio carries no meaning.
    %   extraStable    - false when a measure-specific instability applies (the O-information cancellation ratio), true otherwise.
    % Outputs:
    %   reportable - true only when the percentage may be shown.
    %   statusCode - 1 reportable, 2 denominator below the absolute floor or not finite, 3 cancellation unstable.
    %
    % The floor check runs first, so status 2 pre-empts status 3.
    % The floor is the same on both denominators, because it asks whether a nats-valued denominator is meaningfully non-zero, whichever denominator it is.
    % Status code 0 marks a unit whose fit is invalid, so no percentage is attempted; the caller assigns it, and this function never returns it.

    arguments
        denominator (1, 1) double
        minDenominator (1, 1) double {mustBeNonnegative}
        extraStable (1, 1) logical
    end

    if ~(isfinite(denominator) && abs(denominator) >= minDenominator)
        reportable = false;
        statusCode = 2;
    elseif ~extraStable
        reportable = false;
        statusCode = 3;
    else
        reportable = true;
        statusCode = 1;
    end
end
