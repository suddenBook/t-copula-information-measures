function groups = sampleChannelGroups(n_channels, d, n_groups, seed)
    % SAMPLECHANNELGROUPS The d-dimensional unit set: a seeded, de-duplicated group sample.
    %
    % When the channels have at most n_groups combinations, all of them are returned in nchoosek order; otherwise distinct groups are rejection-sampled from a dedicated RandStream.
    %
    % The seed fixes the sample.
    % Callers pass channelGroupSeed(seed_base, d), where seed_base is the recording's stable id, so a rerun samples the same groups whatever the run list (see channelGroupSeed).
    %
    % Inputs:
    %   n_channels - number of channels in the signal matrix.
    %   d          - group size (production: 3..20; Phase 2 takes its pairs from selectAllPairs).
    %   n_groups   - number of groups requested.
    %   seed       - RandStream seed, channelGroupSeed(seed_base, d).
    %
    % Outputs:
    %   groups - [k x d] 1-based channel indices, ascending within a row, with k = n_groups, or fewer when the channels have fewer combinations or rejection sampling exhausts max_tries = 50*n_groups.

    total = nchoosek(n_channels, d);
    if total <= n_groups
        groups = nchoosek(1:n_channels, d);
        return
    end
    rs = RandStream('twister', 'Seed', seed);
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    groups = zeros(n_groups, d);
    k = 0;
    tries = 0;
    max_tries = 50 * n_groups;
    while k < n_groups && tries < max_tries
        tries = tries + 1;
        cand = sort(randperm(rs, n_channels, d));
        key = sprintf('%d_', cand);
        if ~isKey(seen, key)
            seen(key) = true;
            k = k + 1;
            groups(k, :) = cand;
        end
    end
    groups = groups(1:k, :);
end
