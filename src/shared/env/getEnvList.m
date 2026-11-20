function values = getEnvList(name)
    % GETENVLIST Comma-separated environment variable as a cell array of character vectors.
    %
    % values = getEnvList(name) splits the variable on commas, trims each element and drops the empty ones, so "a, b, ,c," yields {'a', 'b', 'c'}.
    % An unset, empty or all-separator value yields a 1x0 empty cell, which the callers read as "no selection" and replace with their full or production list.
    %
    % Input:
    %   name - environment variable name (for example TC_RECORDINGS).
    % Output:
    %   values - 1 x k cell array of char rows, k >= 0.

    arguments
        name (1, 1) string
    end
    raw = string(getenv(name));
    parts = strtrim(split(raw, ","));
    parts = parts(strlength(parts) > 0);
    if isempty(parts)
        values = cell(1, 0);
        return
    end
    values = cellstr(reshape(parts, 1, []));
end
