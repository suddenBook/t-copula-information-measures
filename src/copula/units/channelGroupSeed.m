function seed = channelGroupSeed(seed_base, d)
    %CHANNELGROUPSEED Deterministic RandStream seed for a d-dimensional channel-group sample.
    %
    % This is the only source of the channel-group seed.
    % Phase 2 draws a recording's channel groups at d >= 3 with sampleChannelGroups(n_channels, d, n_groups, channelGroupSeed(seed_base, d)), and later phases read those groups from its output.
    % Phase 3 splits each dimension's channel sets into halves on a separate stream seeded at channelGroupSeed(seed_base, d) plus an offset.
    %
    % seed_base is an arbitrary but stable integer carried with the recording (declared in Phase 1's recording table and persisted in recording.json).
    % It is not a row index or a position in any list, so reordering or adding recordings does not change any other recording's groups.
    % Renumbering a seed_base silently changes every sampled group while the code still looks correct.
    %
    % The 31*d term separates the streams of different dimensions, so the d = 3 and d = 4 samples of one recording are independent draws rather than nested prefixes of one stream.
    %
    % Inputs:
    %   seed_base - stable non-negative integer id of the recording.
    %   d         - group dimension (positive integer).
    % Outputs:
    %   seed      - non-negative integer seed for RandStream('twister', 'Seed', seed), well within the twister's [0, 2^32-1] range for any realistic (seed_base, d).

    seed = 7919 * seed_base + 31 * d;
end
