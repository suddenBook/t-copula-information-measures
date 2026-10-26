function id = settingsIdentity(s)
    %SETTINGSIDENTITY The SHA-256 identity of the settings that define a phase's work.
    %
    % id = settingsIdentity(s) hashes one phase's scientific configuration, so that two runs doing the same work share one identity and a run doing different work does not.
    % Every resume compares the identity of its own settings with the one the artifact on disk carries, and refuses to reuse or merge an artifact whose settings are not this run's: a checkpoint keyed by a chunk, a replicate or a cell number describes other numbers under the same index as soon as a grid, a size or a fit option moves.
    %
    % What goes in is the caller's decision and belongs at the top of the phase script beside the settings themselves: every field the numbers depend on (grids, subset sizes, seed radices, fit options, the identity of the upstream phase), and nothing that only schedules them (chunk size, worker count, shard, the figure and overwrite switches, paths, timestamps).
    % Defaults are materialised by the caller: an empty "use the default" placeholder and the value it stands for must hash alike, which they only do once the placeholder is resolved (tFitDefaultOptions is the study's materialiser).
    %
    % The encoding is canonical, so nothing but the values can move the identity: every struct is ordered by field name before it is encoded, recursively, and a logical is encoded as its 0/1 value rather than as JSON true/false, so the class a caller happens to use cannot change the digest either.
    %
    % Input:
    %   s  - scalar struct of settings; nested structs, struct arrays, cells and numeric, logical, char or string values are supported.
    % Output:
    %   id - lowercase 64-character SHA-256 hex digest.
    %
    % See also sha256Json, tFitDefaultOptions.

    arguments
        s (1, 1) struct
    end

    id = sha256Json(localCanonical(s));
end


function value = localCanonical(value)
    % Order every struct by field name and normalise the classes whose JSON spelling depends on the class rather than on the value.
    if isstruct(value)
        names = fieldnames(value);
        for k = 1:numel(value)
            for f = 1:numel(names)
                value(k).(names{f}) = localCanonical(value(k).(names{f}));
            end
        end
        value = orderfields(value);
    elseif iscell(value)
        value = cellfun(@localCanonical, value, 'UniformOutput', false);
    elseif islogical(value)
        value = double(value);
    end
end
