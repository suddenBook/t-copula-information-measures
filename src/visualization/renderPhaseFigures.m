function status = renderPhaseFigures(driver_name, mat_path, images_dir, text_dir)
    % RENDERPHASEFIGURES Render a phase's figures by running its Python driver.
    %
    % Each phase script, after it has saved its outputs, calls
    %
    %   renderPhaseFigures('render_phaseN_figures', source, ctx.images_dir, ctx.text_dir)
    %
    % which runs `uv run --quiet src/visualization/python_figures/render_phaseN_figures.py <source> <images_dir>`.
    % The driver imports that phase's renderers and calls render(source, images_dir) on each, so a pipeline run leaves the figures beside the data and a new phase needs only a new driver.
    %
    % Figures are diagnostics, so a rendering failure never fails a phase: every failure is reported and returned, never thrown.
    %
    % MATLAB runs system() through the shell named by SHELL, which it reads once at startup and starts without a PATH search.
    % When SHELL holds a bare name such as "zsh" instead of an absolute path, every system() call exits with status 127, and setting SHELL from inside MATLAB does not help.
    % Launching MATLAB as env SHELL=/bin/bash matlab -batch "..." avoids this.
    % When the shell is broken this function warns, writes images_dir/render_figures.sh so the figures stay one command away, and returns status.shell_broken = true.
    %
    % Inputs:
    %   driver_name - stem of the driver in python_figures/ without .py, such as 'render_phase1_figures'.
    %   mat_path    - the driver's data source: Phase 1's saved mat, or the phase directory for Phases 2 to 7; any existing file or folder is accepted.
    %   images_dir  - folder the figures are written to (ctx.images_dir), created when missing.
    %   text_dir    - folder figures_manifest.json is written to (ctx.text_dir); no manifest is written when it does not exist.
    %
    % Outputs:
    %   status - struct with fields ok, shell_broken, uv, driver, command, exit_code, output, n_figures, figures (cellstr) and manifest_path.

    status = struct('ok', false, 'shell_broken', false, 'uv', '', ...
                    'driver', char(driver_name), 'command', '', 'exit_code', NaN, ...
                    'output', '', 'n_figures', 0, 'figures', {{}}, 'manifest_path', '');

    % A rendering failure never fails a phase, so nothing in the figure layer throws into the caller.
    % A phase that has saved its outputs then finishes its remaining steps even when a renderer, the shell or this function itself is broken.
    try
        status = localRender(status, driver_name, mat_path, images_dir, text_dir);
    catch figure_error
        fprintf(2, '[figures] figure layer failed for %s: %s\n', ...
                char(driver_name), figure_error.message);
        fprintf(2, '[figures] %s\n', getReport(figure_error, 'basic', 'hyperlinks', 'off'));
        fprintf(2, '[figures] The DATA for this phase is unaffected.\n');
    end
end

function status = localRender(status, driver_name, mat_path, images_dir, text_dir)
    repo = localRepoRoot();
    driver_path = fullfile(repo, 'src', 'visualization', 'python_figures', ...
                           [char(driver_name) '.py']);
    if exist(driver_path, 'file') ~= 2
        fprintf(2, '[figures] driver not found: %s (skipping figures)\n', driver_path);
        return
    end
    % The source is a file (Phase 1's mat) or a folder (a phase directory, or a results root for a cross-recording driver).
    if exist(char(mat_path), 'file') ~= 2 && ~isfolder(char(mat_path))
        fprintf(2, '[figures] source not found: %s (skipping figures)\n', char(mat_path));
        return
    end
    if ~exist(char(images_dir), 'dir')
        mkdir(char(images_dir));
    end

    uv = localResolveUv();
    status.uv = uv;

    cmd = sprintf('%s run --quiet %s %s %s', localQuote(uv), localQuote(driver_path), ...
                  localQuote(char(mat_path)), localQuote(char(images_dir)));
    status.command = cmd;

    if ~localShellWorks()
        status.shell_broken = true;
        localWriteFallbackScript(images_dir, cmd);
        fprintf(2, [ ...
                    '\n[figures] ==================================================================\n' ...
                    '[figures] MATLAB cannot start a shell: system(''echo'') failed.\n' ...
                    '[figures] Cause: SHELL=%s is not an executable path, and MATLAB execs it\n' ...
                    '[figures]        directly. Every system() call returns 127.\n' ...
                    '[figures] Fix  : relaunch MATLAB with  env SHELL=/bin/bash matlab -batch "<statements>"\n' ...
                    '[figures]        (it exports SHELL=/bin/bash). Setting SHELL inside MATLAB\n' ...
                    '[figures]        does not work -- the shell is captured at startup.\n' ...
                    '[figures] The DATA for this phase is complete and correct; only the figures\n' ...
                    '[figures] were skipped. Re-create them with:\n' ...
                    '[figures]        bash %s\n' ...
                    '[figures] ==================================================================\n\n'], ...
                getenv('SHELL'), fullfile(char(images_dir), 'render_figures.sh'));
        return
    end

    fprintf('[figures] %s -> %s\n', driver_name, char(images_dir));
    t0 = tic;
    [rc, out] = system(cmd);
    status.exit_code = rc;
    status.output = out;

    % Echo the driver's output into the phase diary (text/log.txt), so the log records which figures this run produced.
    if ~isempty(out)
        fprintf('%s', out);
        if out(end) ~= newline
            fprintf('\n');
        end
    end

    status.figures = localListFigures(images_dir);
    status.n_figures = numel(status.figures);
    status.ok = (rc == 0);

    if status.ok
        fprintf('[figures] %d figure files in %.1f s\n', status.n_figures, toc(t0));
    else
        fprintf(2, '[figures] driver %s exited %d (%d figures written). Data is unaffected.\n', ...
                driver_name, rc, status.n_figures);
    end

    status.manifest_path = localWriteManifest(text_dir, driver_name, mat_path, status);
end

% ===========================================================================
% Local helpers
% ===========================================================================
function tf = localShellWorks()
    % Probe the shell with echo, which needs nothing on PATH beyond the shell itself.
    [rc, ~] = system('echo __probe__');
    tf = (rc == 0);
end

function uv = localResolveUv()
    % The first uv found at a usual install location, else the bare name 'uv' for a PATH lookup, because MATLAB's PATH need not be a login shell's PATH.
    candidates = {'uv', fullfile(getenv('HOME'), '.local', 'bin', 'uv'), ...
                  '/usr/local/bin/uv', '/usr/bin/uv'};
    for i = 2:numel(candidates)
        if exist(candidates{i}, 'file') == 2
            uv = candidates{i};
            return
        end
    end
    uv = 'uv';  % resolved through PATH
end

function q = localQuote(s)
    % Single-quote a string for /bin/sh, escaping any embedded single quote, because a results root is user-supplied.
    s = char(s);
    q = ['''' strrep(s, '''', '''\''''') ''''];
end

function repo = localRepoRoot()
    % The repository root: the first folder above this file that holds src/ and scripts/.
    repo = fileparts(mfilename('fullpath'));
    while ~(isfolder(fullfile(repo, 'src')) && isfolder(fullfile(repo, 'scripts')))
        parent = fileparts(repo);
        if strcmp(parent, repo)
            error('renderPhaseFigures:RepoRootNotFound', ...
                  'repository root (a dir with src/ and scripts/) not found');
        end
        repo = parent;
    end
end

function figs = localListFigures(images_dir)
    % The sorted names of the PDF, PNG and SVG files in images_dir.
    figs = {};
    for ext = {'*.pdf', '*.png', '*.svg'}
        d = dir(fullfile(char(images_dir), ext{1}));
        for i = 1:numel(d)
            figs{end + 1} = d(i).name; %#ok<AGROW>
        end
    end
    figs = sort(figs);
end

function localWriteFallbackScript(images_dir, cmd)
    % Write images_dir/render_figures.sh, a script that reruns the driver command from a working shell.
    p = fullfile(char(images_dir), 'render_figures.sh');
    fid = fopen(p, 'w');
    if fid == -1
        return
    end
    fprintf(fid, '#!/usr/bin/env bash\n');
    fprintf(fid, '# Written by renderPhaseFigures.m because MATLAB could not start a shell.\n');
    fprintf(fid, '# The phase data is complete; this reproduces only its figures.\n');
    fprintf(fid, 'set -euo pipefail\n%s\n', cmd);
    fclose(fid);
end

function manifest_path = localWriteManifest(text_dir, driver_name, mat_path, status)
    % Write text_dir/figures_manifest.json, the record of which driver drew which figures from which source.
    % Returns its path, or '' when text_dir does not exist or the write raises an error.
    manifest_path = '';
    if isempty(text_dir) || ~exist(char(text_dir), 'dir')
        return
    end
    payload = struct( ...
                     'driver',       char(driver_name), ...
                     'source_mat',   char(mat_path), ...
                     'exit_code',    status.exit_code, ...
                     'ok',           status.ok, ...
                     'n_figures',    status.n_figures, ...
                     'figures',      {status.figures}, ...
                     'timestamp',    char(datetime('now', 'TimeZone', 'UTC', ...
                                      'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')));
    manifest_path = fullfile(char(text_dir), 'figures_manifest.json');
    try
        fid = fopen(manifest_path, 'w');
        if fid ~= -1
            fwrite(fid, jsonencode(payload, 'PrettyPrint', true));
            fclose(fid);
        end
    catch manifest_err
        fprintf(2, '[figures] failed to write %s: %s\n', manifest_path, manifest_err.message);
        manifest_path = '';
    end
end
