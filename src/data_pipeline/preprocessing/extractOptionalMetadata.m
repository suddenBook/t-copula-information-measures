function meta = extractOptionalMetadata(input_path)
    % EXTRACTOPTIONALMETADATA Load the optional top-level metadata variables of an input MAT file.
    %
    % Returns a struct with whichever of channel_ids, channel_labels, is_soz_channel, pair_info, seizure_sample and segment_boundaries the file holds.
    % When channel_ids or channel_labels is missing and the file also holds X and a struct named metadata, two fallbacks apply.
    % metadata.good_channels becomes channel_ids when it has one entry per column of X, and missing labels become 'ch<id>' built from channel_ids.
    meta = struct();
    info = whos('-file', input_path);
    names = {info.name};
    recognised = {'channel_ids', 'channel_labels', 'is_soz_channel', ...
                  'pair_info', 'seizure_sample', 'segment_boundaries'};
    present = intersect(recognised, names);
    if ~isempty(present)
        d = load(input_path, present{:});
        for k = 1:numel(present)
            meta.(present{k}) = d.(present{k});
        end
    end

    if isfield(meta, 'channel_ids') && isfield(meta, 'channel_labels')
        return
    end

    x_idx = find(strcmp(names, 'X'), 1);
    metadata_idx = find(strcmp(names, 'metadata'), 1);
    if isempty(x_idx) || isempty(metadata_idx) || numel(info(x_idx).size) < 2
        return
    end

    n_channels = info(x_idx).size(2);
    d = load(input_path, 'metadata');
    if ~isfield(d, 'metadata') || ~isstruct(d.metadata)
        return
    end

    source_metadata = d.metadata;
    if ~isfield(meta, 'channel_ids') && ...
            isfield(source_metadata, 'good_channels') && ...
            numel(source_metadata.good_channels) == n_channels
        meta.channel_ids = double(source_metadata.good_channels(:)).';
    end

    if ~isfield(meta, 'channel_labels') && isfield(meta, 'channel_ids')
        meta.channel_labels = arrayfun(@(id) sprintf('ch%d', id), ...
                                       meta.channel_ids, 'UniformOutput', false);
    end
end
