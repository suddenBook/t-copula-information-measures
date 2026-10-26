function ratio = cancellationRatio(oValue, tcValue, dtcValue)
    %CANCELLATIONRATIO Cancellation ratio |O| / (|TC| + |DTC|) of the O-information on one denominator's scale.
    %
    % The O-information is the difference TC - DTC of two much larger positive quantities, so |O| can be numerically tiny while still clearing the absolute denominator floor.
    % This ratio detects that case, which the floor alone does not.
    % The three arguments come from one denominator's scale, all Gaussian or all fitted.
    % The Gaussian ratio does not bound the fitted one, so callers evaluate the ratio separately per denominator.
    % A zero scale returns 0, which suppresses the percentage instead of dividing by zero.
    %
    % Inputs:
    %   oValue, tcValue, dtcValue - O, TC and DTC in nats, all on the same scale.
    % Outputs:
    %   ratio - |O| / (|TC| + |DTC|), or 0 when that scale is zero.

    arguments
        oValue (1, 1) double
        tcValue (1, 1) double
        dtcValue (1, 1) double
    end

    scale = abs(tcValue) + abs(dtcValue);
    if scale > 0
        ratio = abs(oValue) / scale;
    else
        ratio = 0;
    end
end
