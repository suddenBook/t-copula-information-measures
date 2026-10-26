function shard = getEnvShard(name)
    %GETENVSHARD Strict "k/N" shard-selector parser for an environment variable.
    %
    % shard = getEnvShard(name) returns [k N] when the variable holds "k/N" with integers 1 <= k <= N, and [1 1] (the whole sweep) when it is unset or empty.
    % Any other value is an error, so a typo cannot silently run the whole sweep in one job.
    % A phase deals its items to the N shards by a deterministic rule (the item number modulo N, or a fixed load balance), so a later [1 1] pass covers every shard's items.
    %
    % Input:
    %   name - environment variable name (the pipeline uses TC_SHARD).
    % Output:
    %   shard - [k N] double row.

    arguments
        name (1, 1) string
    end
    raw = strtrim(string(getenv(name)));
    if strlength(raw) == 0
        shard = [1 1];
        return;
    end
    tokens = regexp(raw, '^(\d+)/(\d+)$', 'tokens', 'once');
    assert(~isempty(tokens), 'getEnvShard:InvalidValue', ...
        '%s must be "k/N" with integers 1 <= k <= N; got "%s".', name, raw);
    shard = [str2double(tokens{1}), str2double(tokens{2})];
    assert(shard(1) >= 1 && shard(1) <= shard(2), 'getEnvShard:InvalidValue', ...
        '%s must be "k/N" with integers 1 <= k <= N; got "%s".', name, raw);
end
