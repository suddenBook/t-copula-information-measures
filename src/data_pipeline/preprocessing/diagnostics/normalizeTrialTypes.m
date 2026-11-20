function trial_types = normalizeTrialTypes(raw_trial_types)
    % NORMALIZETRIALTYPES Return trial-type labels as a column cell array.
    %
    % Accepts a cell array, a string array or a char matrix with one label per row.
    if iscell(raw_trial_types)
        trial_types = raw_trial_types(:);
    elseif isstring(raw_trial_types)
        trial_types = cellstr(raw_trial_types(:));
    else
        trial_types = cellstr(raw_trial_types);
        trial_types = trial_types(:);
    end
end
