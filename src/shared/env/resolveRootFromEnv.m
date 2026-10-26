function root = resolveRootFromEnv(name, defaultRoot)
    %RESOLVEROOTFROMENV Root folder from an environment variable, or the default when it is unset.
    %
    % No folder is created here, so the callers decide whether an input root must already exist or an output root may be created.
    %
    % Inputs:
    %   name        - environment variable name (the pipeline uses TC_DATA_ROOT and TC_RESULTS_ROOT).
    %   defaultRoot - root used when the variable is unset or empty.
    % Output:
    %   root - char row; a value starting with "/" is taken as absolute, and any other value is resolved against the current working folder at call time.

    arguments
        name (1, 1) string
        defaultRoot (1, 1) string
    end
    raw = strtrim(string(getenv(name)));
    if strlength(raw) == 0
        root = char(defaultRoot);
        return;
    end
    if startsWith(raw, "/")
        root = char(raw);
    else
        root = fullfile(pwd, char(raw));
    end
end
