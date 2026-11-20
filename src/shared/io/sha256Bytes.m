function hex = sha256Bytes(bytes)
    % SHA256BYTES SHA-256 digest of an in-memory byte vector.
    %
    % BYTES is a uint8 vector, hashed with Java's standard MessageDigest, which both the MATLAB desktop and -batch runtimes provide.
    % An int8 vector is accepted but converted with uint8(), which saturates its negative values to 0, so it is hashed as its own bytes only when no value is negative.
    % The digest is returned as a lowercase 64-character hex string, the form the checkpoint identities use.

    validateattributes(bytes, {'uint8', 'int8'}, {'vector'}, mfilename, 'bytes');
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(typecast(uint8(bytes(:)), 'int8'));
    raw = typecast(digest.digest(), 'uint8');
    hex = lower(reshape(dec2hex(raw, 2).', 1, []));
end
