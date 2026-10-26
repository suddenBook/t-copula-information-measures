function writeTableAtomic(T, path)
    %WRITETABLEATOMIC Write a table to CSV through a temporary file and a rename.
    %
    % writeTableAtomic(T, path) writes T with writetable to a temporary file beside path and then moves it into place, so an interrupted write can never leave a half-written CSV that a later step would read as complete.
    %
    % The staging file belongs to this writer alone, for the reason given in saveStructAtomic: two shards publishing the same table must not share one '<path>.tmp', or each deletes and overwrites the other's half-written file.
    % The staging name carries the host, the process and a per-call token, and it stays in the destination's own folder so the rename stays a rename.

    arguments
        T table
        path (1, :) char
    end
    tmp = localWriterTempName(path, '.tmp');
    try
        writetable(T, tmp, 'FileType', 'text', 'Delimiter', ',');
    catch err
        % A unique staging name is never reused, so a failed write would leave its part-file behind for good if it were not removed here.
        if exist(tmp, 'file') == 2
            delete(tmp);
        end
        rethrow(err);
    end
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'writeTableAtomic:MoveFailed', ...
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
