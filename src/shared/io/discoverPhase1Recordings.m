function recordings = discoverPhase1Recordings(results_root, keys)
    % DISCOVERPHASE1RECORDINGS List the recordings Phase 1 has written under a results root.
    %
    % Phases 2 and later carry no recording table: they discover what Phase 1 produced.
    % Every recording base holds a recording.json sidecar written by Phase 1, so the discovery is the glob <results_root>/*/*/*/recording.json.
    %
    % Inputs:
    %   results_root - the output tree root that Phase 1 wrote into.
    %   keys         - optional cell array of character vectors or string array of dataset keys to keep; [] or omitted keeps every recording.
    %                  A key that matches no discovered recording is an error, so a typo cannot silently shrink a run.
    %
    % Output:
    %   recordings - struct array, sorted by seed_base ascending, with every field of recording.json plus
    %                  rel_base       - '<family>/<subject>/<condition>'
    %                  phase1_mat_abs - absolute path of the Phase 1 MAT file
    %
    % Assumptions, each of which fails loudly rather than silently:
    %   - recording.json carries at least dataset_key, seed_base and phase1_mat;
    %   - phase1_mat is relative to the recording base <results_root>/<rel_base>;
    %   - that MAT file exists;
    %   - every recording.json under one root carries the same field set.

    arguments
        results_root {mustBeTextScalar}
        keys = {}
    end

    results_root = char(results_root);
    if isempty(keys)
        keys = cell(1, 0);
    end
    assert(iscellstr(keys) || isstring(keys), ...
           'discoverPhase1Recordings:BadKeys', ...
           'keys must be a cell array of character vectors or a string array.');
    keys = cellstr(reshape(string(keys), 1, []));

    pattern = fullfile(results_root, '*', '*', '*', 'recording.json');
    entries = dir(pattern);
    assert(~isempty(entries), 'discoverPhase1Recordings:NoRecordings', ...
           ['No Phase 1 sidecar matches %s. Run Phase 1, or point the results ' ...
            'root at the tree that holds its output.'], pattern);

    loaded = cell(1, numel(entries));
    for k = 1:numel(entries)
        json_path = fullfile(entries(k).folder, entries(k).name);
        loaded{k} = localReadRecording(json_path, results_root);
    end

    recordings = localConcatenate(loaded, entries);

    if ~isempty(keys)
        present = string({recordings.dataset_key});
        unknown = setdiff(string(keys), present, 'stable');
        assert(isempty(unknown), 'discoverPhase1Recordings:UnknownKey', ...
               'Requested dataset key(s) %s are not among the %d recordings under %s (%s).', ...
               strjoin(unknown, ', '), numel(recordings), ...
               results_root, strjoin(present, ', '));
        recordings = recordings(ismember(present, string(keys)));
    end

    [~, order] = sort([recordings.seed_base]);
    recordings = recordings(order);
end

function record = localReadRecording(json_path, results_root)
    % One sidecar: decode it, resolve rel_base from its own location and check that the Phase 1 MAT it names is actually there.
    raw = fileread(json_path);
    try
        record = jsondecode(raw);
    catch cause
        error('discoverPhase1Recordings:BadJson', ...
              'Could not decode %s as JSON: %s', json_path, cause.message);
    end
    assert(isstruct(record) && isscalar(record), ...
           'discoverPhase1Recordings:BadJson', ...
           'A recording sidecar must decode to a scalar struct: %s', json_path);

    required = {'dataset_key', 'seed_base', 'phase1_mat'};
    missing = required(~isfield(record, required));
    assert(isempty(missing), 'discoverPhase1Recordings:MissingField', ...
           'Recording sidecar %s has no %s field.', json_path, strjoin(missing, ', '));
    assert(isnumeric(record.seed_base) && isscalar(record.seed_base) ...
           && isfinite(record.seed_base), ...
           'discoverPhase1Recordings:BadSeedBase', ...
           'seed_base must be a finite numeric scalar in %s.', json_path);

    parts = split(string(strrep(fileparts(json_path), filesep, '/')), "/");
    rel_base = strjoin(parts(end - 2:end), "/");
    record.rel_base = rel_base;
    record.phase1_mat_abs = fullfile(results_root, ...
                                     char(parts(end - 2)), char(parts(end - 1)), char(parts(end)), ...
                                     char(string(record.phase1_mat)));
    assert(isfile(record.phase1_mat_abs), ...
           'discoverPhase1Recordings:MissingPhase1Mat', ...
           ['Recording %s declares phase1_mat "%s" in %s, but that file does ' ...
            'not exist: %s'], string(record.dataset_key), ...
           string(record.phase1_mat), json_path, record.phase1_mat_abs);
end

function recordings = localConcatenate(loaded, entries)
    % A struct array needs one field set.
    % Sidecars with different field sets come from different Phase 1 code writing into one tree, which is reported rather than silently merged.
    reference = fieldnames(loaded{1});
    for k = 2:numel(loaded)
        if ~isequal(sort(fieldnames(loaded{k})), sort(reference))
            error('discoverPhase1Recordings:SchemaMismatch', ...
                  ['Recording sidecars carry different field sets: %s and %s. ' ...
                   'Regenerate Phase 1 for the whole tree.'], ...
                  fullfile(entries(1).folder, entries(1).name), ...
                  fullfile(entries(k).folder, entries(k).name));
        end
        loaded{k} = orderfields(loaded{k}, reference);
    end
    recordings = [loaded{:}];
end
