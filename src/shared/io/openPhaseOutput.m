function ctx = openPhaseOutput(rel_base, phase, results_root, options)
    % OPENPHASEOUTPUT Create the uniform per-phase output tree and open its diary.
    %
    % Every phase writes to the same layout, <results_root>/<rel_base>/Phase<N>/{images,text,mat}/, where REL_BASE is a three-component relative path supplied by the caller and validated here and nowhere else.
    % The caller supplies the path because it knows which recording or study it is writing, and a path is not a scientific fact.
    %   recordings : '<family>/<results_subject>/<results_condition>' (for example 'human_SEEG/001/sleep')
    %   simulation : 'simulation/t_copula/all'
    %
    % The diary on text/log.txt captures the whole run log, not a hand-picked subset.
    % It is closed automatically when the returned CTX goes out of scope (an onCleanup guard runs `diary off`), so an error mid-phase still flushes the log.
    % Opening it prints the line '[PhaseN] <rel_base>', by which scripts/hpc/pull.sh files each cluster log beside the outputs it records.
    %
    % Inputs:
    %   rel_base     - three-component relative path, '/'-separated, no empty part.
    %   phase        - phase number (a positive integer) or a 'PhaseN' / 'N' string.
    %   results_root - results root; '' or omitted means repoRoot()/results.
    % Name-value:
    %   'Diary' (default true) - open the text/log.txt diary.
    %
    % Output:
    %   ctx - struct with rel_base, family, subject, condition, phase (='PhaseN'), base, images_dir, text_dir, mat_dir, results_root, diary_path, diary_guard.
    %
    % Typical use in a phase (an explicit save list, never a whole-workspace dump):
    %   ctx = openPhaseOutput(rel_base, 2, results_root);
    %   save(fullfile(ctx.mat_dir, 'fit_all.mat'), '-struct', 'out', '-v7.3');
    %   writetable(T, fullfile(ctx.text_dir, 'summary_by_d.csv'));

    arguments
        rel_base {mustBeTextScalar}
        phase
        results_root {mustBeTextScalar} = ''
        options.Diary (1, 1) logical = true
    end

    if strlength(string(results_root)) == 0
        results_root = fullfile(repoRoot(), 'results');
    end
    results_root = char(results_root);

    parts = split(string(rel_base), "/");
    assert(numel(parts) == 3 && all(strlength(parts) > 0), ...
           'openPhaseOutput:MalformedRelBase', ...
           ['The output base must be exactly three non-empty path components ' ...
            '(<family>/<subject>/<condition>); got "%s" with %d component(s).'], ...
           string(rel_base), numel(parts));

    if isnumeric(phase)
        validateattributes(phase, {'numeric'}, ...
                           {'scalar', 'integer', 'positive'}, mfilename, 'phase');
        phase_id = sprintf('Phase%d', phase);
    else
        phase_id = char(phase);
        if ~startsWith(phase_id, 'Phase')
            phase_id = ['Phase' phase_id];
        end
    end

    base       = fullfile(results_root, char(parts(1)), char(parts(2)), ...
                          char(parts(3)), phase_id);
    images_dir = fullfile(base, 'images');
    text_dir   = fullfile(base, 'text');
    mat_dir    = fullfile(base, 'mat');
    for dd = {base, images_dir, text_dir, mat_dir}
        if ~exist(dd{1}, 'dir')
            [ok, msg] = mkdir(dd{1});
            if ~ok
                error('openPhaseOutput:MkdirFailed', ...
                      'Could not create %s: %s', dd{1}, msg);
            end
        end
    end

    ctx = struct( ...
                 'rel_base',     string(rel_base), ...
                 'family',       parts(1), ...
                 'subject',      parts(2), ...
                 'condition',    parts(3), ...
                 'phase',        string(phase_id), ...
                 'base',         base, ...
                 'images_dir',   images_dir, ...
                 'text_dir',     text_dir, ...
                 'mat_dir',      mat_dir, ...
                 'results_root', results_root, ...
                 'diary_path',   '', ...
                 'diary_guard',  []);

    if options.Diary
        diary_path = fullfile(text_dir, 'log.txt');
        diary(diary_path);
        ctx.diary_path  = diary_path;
        ctx.diary_guard = onCleanup(@() diary('off'));
        fprintf('[%s] %s\n        %s\n', phase_id, string(rel_base), base);
    end
end
