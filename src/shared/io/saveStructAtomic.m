function saveStructAtomic(path, varName, value)
    %SAVESTRUCTATOMIC Save one variable to a -v7.3 MAT file through a temporary file.
    %
    % saveStructAtomic(path, varName, value) saves value under the name varName, writing to a temporary file and renaming it into place, so an interrupted save cannot leave a half-written checkpoint that a resume would trust.
    %
    % The staging file belongs to this writer alone.
    % Two shards that reach the same destination at the same time must not share it: with one common '<path>.tmp.mat', each writer deletes, overwrites and renames the other's half-written file, and the loser either corrupts the MAT file or publishes a truncated one.
    % The staging name therefore carries the host, the process and a per-call token, and it stays in the destination's own folder so the rename is a rename and not a copy across filesystems.
    % Which writer's file ends up at the destination is still a race, which only a single publication owner makes deterministic; this helper guarantees that neither writer can damage the other's data on the way there.

    arguments
        path (1, :) char
        varName (1, :) char
        value
    end
    S = struct();
    S.(varName) = value;
    % '.mat' has to be the last extension: save() appends it to anything else.
    tmp = localWriterTempName(path, '.tmp.mat');
    try
        save(tmp, '-struct', 'S', '-v7.3');
    catch err
        % A unique staging name is never reused, so a failed save would leave its part-file behind for good if it were not removed here.
        if exist(tmp, 'file') == 2
            delete(tmp);
        end
        rethrow(err);
    end
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'saveStructAtomic:MoveFailed', ...
        'Could not move %s into place: %s', tmp, msg);
end


function name = localWriterTempName(path, suffix)
    %LOCALWRITERTEMPNAME Staging name '<path>.<host>.<pid>.<token>' plus SUFFIX, unique to one writer.
    % The token comes from tempname, which is unique without drawing on the global random stream the study's reproducibility depends on, and the host and process id make a leftover attributable to the job that wrote it.
    host = getenv('HOSTNAME');
    if isempty(host)
        host = getenv('COMPUTERNAME');
    end
    if isempty(host)
        host = 'host';
    end
    [~, token] = fileparts(tempname);
    name = sprintf('%s.%s.%d.%s%s', path, regexprep(host, '[^\w-]', '_'), ...
        feature('getpid'), token, suffix);
end
