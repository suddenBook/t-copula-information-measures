function channel_pairs = selectAllPairs(n_channels, num_pairs, seed_base)
    %SELECTALLPAIRS The d = 2 unit set: every channel pair, in nchoosek order.
    %
    % With num_pairs = [], the production setting, it returns every pair without drawing a random number.
    % A reduced set requires seed_base and draws from a private stream seeded by channelGroupSeed(seed_base, 2), the seed namespace of the higher-dimensional group sampler, so it is an exactly reproducible random subset, kept in nchoosek order, rather than a truncation.
    %
    % Inputs:
    %   n_channels - number of channels in the signal matrix.
    %   num_pairs  - [] (default) for all pairs; a positive integer to subsample.
    %   seed_base  - the recording's stable seed_base; required for a reduced set.
    %
    % Outputs:
    %   channel_pairs - [n_pairs x 2] 1-based channel indices, ascending within a row.

    if nargin < 2
        num_pairs = [];
    end
    channel_pairs = nchoosek(1:n_channels, 2);
    if ~isempty(num_pairs)
        assert(nargin >= 3 && isscalar(seed_base) && isfinite(seed_base) ...
            && seed_base == fix(seed_base) && seed_base >= 0, ...
            'selectAllPairs:MissingSeedBase', ...
            'A reduced pair set requires a nonnegative integer seed_base.');
        stream = RandStream('twister', 'Seed', channelGroupSeed(seed_base, 2));
        selectionCount = min(num_pairs, size(channel_pairs, 1));
        sel = sort(randperm(stream, size(channel_pairs, 1), selectionCount));
        channel_pairs = channel_pairs(sel, :);
    end
end
