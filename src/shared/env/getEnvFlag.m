function value = getEnvFlag(name, defaultValue)
    % GETENVFLAG Strict boolean environment-variable parser.
    %
    % Inputs:
    %   name         - environment variable name.
    %   defaultValue - value returned when the variable is unset or empty.
    % Output:
    %   value - logical scalar; any value other than 1/0, true/false, yes/no or on/off (in any case) is an error, so a typo cannot silently select the default.

    arguments
        name (1, 1) string
        defaultValue (1, 1) logical
    end
    raw = strtrim(string(getenv(name)));
    if strlength(raw) == 0
        value = defaultValue;
        return
    end
    switch lower(raw)
        case {"1", "true", "yes", "on"}
            value = true;
        case {"0", "false", "no", "off"}
            value = false;
        otherwise
            error('getEnvFlag:InvalidValue', ...
                  '%s must be one of 1/0, true/false, yes/no, on/off; got "%s".', ...
                  name, raw);
    end
end
