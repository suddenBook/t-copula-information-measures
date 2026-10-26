function info = buildTrialTypeCountStruct(trial_types)
    %BUILDTRIALTYPECOUNTSTRUCT Count the trials of each trial-type label.
    %
    % trial_types is a cell array, string array or char matrix of labels; empty labels are ignored.
    % info.labels holds the sorted unique labels as a 1 x k cell and info.counts the 1 x k count of each, both empty when no label is present.
    info = struct('labels', {{}}, 'counts', []);
    if isempty(trial_types)
        return;
    end

    trial_types = normalizeTrialTypes(trial_types);
    keep_mask = ~cellfun(@isempty, trial_types);
    trial_types = trial_types(keep_mask);
    if isempty(trial_types)
        return;
    end

    [labels, ~, label_idx] = unique(trial_types);
    counts = accumarray(label_idx, 1);
    info.labels = reshape(labels, 1, []);
    info.counts = reshape(counts, 1, []);
end
