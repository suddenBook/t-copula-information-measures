function config_path = writeRunConfig(ctx, settings, provenance)
    %WRITERUNCONFIG Write one phase's settings, environment and provenance stamp.
    %
    % Writes <ctx.text_dir>/run_config.json holding {settings, env, provenance}, where ENV holds the verbatim values of a fixed list of TC_* run variables (localEnvironment) and PROVENANCE is the caller's run facts merged with the git sha, MATLAB release and host name of the machine that runs it.
    % It is a record, not a gate: no phase checks its contents.
    % Phase 1 copies its provenance into the Phase 1 MAT file, and the later phases treat the file's presence as a completion marker.
    %
    % Inputs:
    %   ctx        - context from openPhaseOutput (needs text_dir).
    %   settings   - the phase's settings struct, serialised as given.
    %   provenance - struct carrying n_workers (nonnegative integer), started_utc and finished_utc (datetime, or text already in ISO 8601).
    %                Any other field is kept.
    %                git_sha, matlab_release and hostname are filled in here and overwrite anything passed under those names.
    %
    % Output:
    %   config_path - absolute path of the file written.

    arguments
        ctx (1, 1) struct
        settings (1, 1) struct
        provenance (1, 1) struct
    end

    assert(isfield(ctx, 'text_dir') && ~isempty(ctx.text_dir), ...
        'writeRunConfig:BadContext', ...
        'ctx must carry a non-empty text_dir; openPhaseOutput supplies it.');
    required = {'n_workers', 'started_utc', 'finished_utc'};
    missing = required(~isfield(provenance, required));
    assert(isempty(missing), 'writeRunConfig:MissingProvenance', ...
        'provenance has no %s field; the phase must supply it.', ...
        strjoin(missing, ', '));
    validateattributes(provenance.n_workers, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'}, mfilename, 'provenance.n_workers');

    provenance.started_utc = localIsoUtc(provenance.started_utc);
    provenance.finished_utc = localIsoUtc(provenance.finished_utc);
    provenance.git_sha = localGitSha();
    provenance.matlab_release = version('-release');
    provenance.hostname = localHostname();

    payload = struct( ...
        'settings',   settings, ...
        'env',        localEnvironment(), ...
        'provenance', provenance);

    config_path = fullfile(char(ctx.text_dir), 'run_config.json');
    fid = fopen(config_path, 'w');
    assert(fid ~= -1, 'writeRunConfig:OpenFailed', ...
        'Could not open %s for writing.', config_path);
    closer = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(payload, 'PrettyPrint', true));
end


function env = localEnvironment()
    % The values of a fixed list of 19 TC_* run variables, verbatim, an unset one as ""; a variable outside the list, such as the Phase 2 pilot overrides TC_GROUPS_PER_D and TC_MAX_PAIRS, is not recorded here.
    names = {'TC_DATA_ROOT', 'TC_RESULTS_ROOT', 'TC_WORKERS', 'TC_RECORDINGS', ...
        'TC_RECORDING_INDEX', 'TC_SHARD', 'TC_OVERWRITE', ...
        'TC_RENDER_FIGURES', 'TC_FINALIZE', ...
        'TC_SIM_CLASS', 'TC_SIM_STEPS', 'TC_SIM_REPS', 'TC_SIM_P_LIST', ...
        'TC_SIM_T_LIST', 'TC_SIM_S_LIST', 'TC_SIM_D_LIST', ...
        'TC_SIM_CHECK_PARTS', 'TC_SIM_CHECK_PATHS', 'TC_SIM_CHECK_REPS'};
    env = struct();
    for k = 1:numel(names)
        env.(names{k}) = getenv(names{k});
    end
end


function iso = localIsoUtc(value)
    % A datetime is converted to UTC (an unzoned one is taken as UTC); text is trusted as already formatted.
    if isdatetime(value)
        value.TimeZone = 'UTC';
        value.Format = 'uuuu-MM-dd''T''HH:mm:ss''Z''';
        iso = char(string(value));
    else
        iso = char(strtrim(string(value)));
    end
end


function sha = localGitSha()
    % The commit of the working tree, or 'unknown'.
    % It is never fatal: provenance is recorded, never enforced.
    sha = 'unknown';
    try
        root = repoRoot();
    catch
        return;
    end
    [status, out] = system(sprintf('git -C "%s" rev-parse HEAD', root));
    out = strtrim(out);
    if status == 0 && ~isempty(regexp(out, '^[0-9a-f]{40}$', 'once'))
        sha = out;
        return;
    end
    % A deployed tree carries no .git (and the MATLAB container may carry no git), so scripts/hpc/deploy.sh ships the commit as a GIT_SHA file instead.
    shaFile = fullfile(root, 'GIT_SHA');
    if exist(shaFile, 'file') == 2
        out = strtrim(fileread(shaFile));
        if ~isempty(regexp(out, '^[0-9a-f]{40}$', 'once'))
            sha = out;
        end
    end
end


function host = localHostname()
    % The machine's host name, or 'unknown' when the hostname command fails.
    [status, out] = system('hostname');
    if status == 0 && ~isempty(strtrim(out))
        host = strtrim(out);
    else
        host = 'unknown';
    end
end
