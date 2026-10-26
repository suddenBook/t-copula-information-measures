function value = getEnvInteger(name, defaultValue, limits)
    %GETENVINTEGER Strict integer environment-variable parser.
    %
    % Inputs:
    %   name         - environment variable name.
    %   defaultValue - value returned when the variable is unset or empty; it is passed through unchecked, so [] is a valid "not requested".
    %   limits       - [lo hi] inclusive bounds, default [-Inf Inf].
    % Output:
    %   value - the parsed integer, or defaultValue; a non-integer or out-of-range value is an error rather than a silent clamp.

    arguments
        name (1, 1) string
        defaultValue
        limits (1, 2) double = [-Inf, Inf]
    end
    raw = strtrim(string(getenv(name)));
    if strlength(raw) == 0
        value = defaultValue;
        return;
    end
    value = str2double(raw);
    assert(isscalar(value) && isfinite(value) && value == fix(value) && ...
        value >= limits(1) && value <= limits(2), ...
        'getEnvInteger:InvalidValue', ...
        '%s must be an integer in [%g,%g]; got "%s".', ...
        name, limits(1), limits(2), raw);
end
