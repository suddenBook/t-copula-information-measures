function hex = sha256Json(value)
    %SHA256JSON SHA-256 of MATLAB's deterministic JSON encoding of a value.
    %
    % Callers must construct structs with a canonical field order, as settingsIdentity does.
    % The hash covers the values, not workspace serialization metadata, so the same configuration has the same identity on any machine that runs the same MATLAB release.

    payload = jsonencode(value, 'PrettyPrint', false);
    hex = sha256Bytes(unicode2native(payload, 'UTF-8'));
end
