function hex = sha256File(path)
    %SHA256FILE Stream a file through SHA-256 without loading it into memory.
    %
    % Reads the file in 8 MiB blocks and returns the digest as 64 lowercase hexadecimal characters; a missing or unreadable file raises an error.

    arguments
        path (1, 1) string
    end
    assert(isfile(path), 'sha256File:MissingFile', ...
        'Cannot hash missing file: %s', path);

    fid = fopen(path, 'rb');
    assert(fid ~= -1, 'sha256File:OpenFailed', ...
        'Cannot open file for hashing: %s', path);
    closer = onCleanup(@() fclose(fid));

    digest = java.security.MessageDigest.getInstance('SHA-256');
    chunk_bytes = 8 * 1024 * 1024;
    while true
        block = fread(fid, chunk_bytes, '*uint8');
        if isempty(block)
            break;
        end
        digest.update(typecast(uint8(block(:)), 'int8'));
    end
    raw = typecast(digest.digest(), 'uint8');
    hex = lower(reshape(dec2hex(raw, 2).', 1, []));
end
