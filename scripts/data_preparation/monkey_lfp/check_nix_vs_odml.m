% CHECK_NIX_VS_ODML LFP extraction check: compare the flat files of extract_lfp_from_nix.py with the dataset's odML rejection lists and MATLAB convenience files.
%
% Inputs (read-only), for each monkey:
%   1. the flat file written by extract_lfp_from_nix.py, data/monkey_LFP/raw/lfp_extracted/<sid>_lfp.mat
%   2. the dataset's Neo-style MATLAB convenience file, data/monkey_LFP/multielectrode_grasp/datasets_matlab/<sid>_lfp-spikes.mat
%
% Outputs: none on disk; the script prints one PASS or FAIL line per check and a summary.
%
% Checks:
%   (a) the per-band electrode-rejection sets (HFC, LFC, IFC) of the flat file and their union match the odML lists hard-coded below
%   (b) the per-band trial-rejection sets of the flat file and their union match the hard-coded odML lists
%   (c) for both monkeys, the behaviourally correct trial IDs of the flat file (performance_str == 'correct_trial') equal those of the convenience file (an_performance_in_trial == 255)
%   (d) inside each flat file, performance_str == 'correct_trial' agrees with performance_code == 255 at the trial level, repeating on the MATLAB side the checks extract_lfp_from_nix.py makes
%
% A missing input file raises an error at once; otherwise every check runs, and the script raises an error at the end when any check failed, so a batch run exits with a non-zero status.
% Run it before regenerating data/monkey_LFP/preprocessed/monkey_{N,L}/*.mat from new flat files.
%
% Usage, from the repository root:
%     >> run(fullfile('scripts', 'data_preparation', 'monkey_lfp', 'check_nix_vs_odml.m'))
% The data tree is resolved from the repository root, not from the working directory.
% Set TC_DATA_ROOT to check a data tree held outside the repository.

clearvars;
close all;
clc;

% Find the repository root (the first folder above this script that holds src/ and scripts/) and put src/ on the path.
script_dir = fileparts(mfilename('fullpath'));
worktree_root = script_dir;
while ~(isfolder(fullfile(worktree_root, 'src')) && isfolder(fullfile(worktree_root, 'scripts')))
    parent = fileparts(worktree_root);
    if strcmp(parent, worktree_root)
        error('bootstrap:RepoRootNotFound', ...
              'repository root (a dir with src/ and scripts/) not found above %s', script_dir);
    end
    worktree_root = parent;
end
addpath(genpath(fullfile(worktree_root, 'src')));
data_root = resolveRootFromEnv("TC_DATA_ROOT", string(fullfile(worktree_root, 'data')));

extracted_dir = fullfile(data_root, 'monkey_LFP', 'raw', 'lfp_extracted');
convenience_dir = fullfile(data_root, 'monkey_LFP', 'multielectrode_grasp', 'datasets_matlab');

fprintf('=== check_nix_vs_odml ===\n');
fprintf('Extracted dir:    %s\n', extracted_dir);
fprintf('Convenience dir:  %s\n\n', convenience_dir);

%% Expected rejection lists from the dataset's odML metadata (electrode indices and trial IDs per quality band)

expected = struct();

expected.N.session_id = 'i140703-001';
expected.N.monkey_label = 'N (Nikos)';
expected.N.el_HFC   = [73];
expected.N.el_LFC   = [15, 23, 38, 45, 46, 53, 67];
expected.N.el_IFC   = [23, 45];
expected.N.el_union = [15, 23, 38, 45, 46, 53, 67, 73];
expected.N.tr_HFC   = [10, 19, 21, 23, 29, 46, 47, 74, 75, 82, 89, 101, 124, 126, 157, 158];
expected.N.tr_LFC   = [83, 139, 152];
expected.N.tr_IFC   = [1, 4, 10, 19, 23, 46, 47, 74, 75, 84, 87, 89, 93, 124, 126, 127, 139, 146, 157, 158];
expected.N.tr_union = [1, 4, 10, 19, 21, 23, 29, 46, 47, 74, 75, 82, 83, 84, 87, 89, 93, 101, 124, 126, 127, 139, 146, 152, 157, 158];

expected.L.session_id = 'l101210-001';
expected.L.monkey_label = 'L (Lilou)';
expected.L.el_HFC   = [];
expected.L.el_LFC   = [];
expected.L.el_IFC   = [2, 4, 29];
expected.L.el_union = [2, 4, 29];
expected.L.tr_HFC   = [];
expected.L.tr_LFC   = [];
expected.L.tr_IFC   = [8, 45, 66, 80, 102, 106, 117, 136, 139, 140, 146, 173, 177, 178, 195, 200];
expected.L.tr_union = [8, 45, 66, 80, 102, 106, 117, 136, 139, 140, 146, 173, 177, 178, 195, 200];

monkeys = {'N', 'L'};

pass_count = 0;
fail_count = 0;

for mi = 1:length(monkeys)
    mk = monkeys{mi};
    fprintf('\n------ Monkey %s (%s) ------\n', mk, expected.(mk).monkey_label);
    extracted_path = fullfile(extracted_dir, [expected.(mk).session_id '_lfp.mat']);
    convenience_path = fullfile(convenience_dir, [expected.(mk).session_id '_lfp-spikes.mat']);

    if ~exist(extracted_path, 'file')
        error('Extracted file missing: %s', extracted_path);
    end
    if ~exist(convenience_path, 'file')
        error('Convenience file missing: %s', convenience_path);
    end

    fprintf('Loading extracted: %s\n', extracted_path);
    ex = load(extracted_path);

    % ------- (a) electrode rejection unions -------
    el_HFC = rowvec(find(ex.electrode_reject_HFC));
    el_LFC = rowvec(find(ex.electrode_reject_LFC));
    el_IFC = rowvec(find(ex.electrode_reject_IFC));
    el_union = rowvec(find(ex.electrode_reject_HFC | ex.electrode_reject_LFC | ex.electrode_reject_IFC));

    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] electrode_reject_HFC', mk), el_HFC, expected.(mk).el_HFC);
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] electrode_reject_LFC', mk), el_LFC, expected.(mk).el_LFC);
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] electrode_reject_IFC', mk), el_IFC, expected.(mk).el_IFC);
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] electrode_reject_union', mk), el_union, expected.(mk).el_union);

    % ------- (b) trial rejection unions -------
    tid = double(ex.trial_events.trial_id(:));
    mask_valid = tid > 0;

    tr_HFC = rowvec(unique(tid(logical(ex.trial_events.reject_HFC(:)) & mask_valid)));
    tr_LFC = rowvec(unique(tid(logical(ex.trial_events.reject_LFC(:)) & mask_valid)));
    tr_IFC = rowvec(unique(tid(logical(ex.trial_events.reject_IFC(:)) & mask_valid)));
    tr_union = rowvec(unique([tr_HFC, tr_LFC, tr_IFC]));

    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] trial_reject_HFC', mk), tr_HFC, expected.(mk).tr_HFC);
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] trial_reject_LFC', mk), tr_LFC, expected.(mk).tr_LFC);
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] trial_reject_IFC', mk), tr_IFC, expected.(mk).tr_IFC);
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] trial_reject_union', mk), tr_union, expected.(mk).tr_union);

    % ------- (c) correct-trial set equivalence (extracted vs convenience) -------
    % Extracted: trials where any event has performance_str == 'correct_trial'.
    correct_from_extracted = rowvec(unique( ...
                                           tid(strcmp(ex.trial_events.performance_str, 'correct_trial') & mask_valid)));

    % Convenience: trials where any event has numeric an_performance_in_trial == 255.
    fprintf('Loading convenience: %s\n', convenience_path);
    conv = load(convenience_path);
    seg = conv.block.segments{1};
    conv_tev = [];
    for j = 1:length(seg.events)
        ev = seg.events{j};
        if isfield(ev, 'name') && strcmp(strtrim(ev.name), 'TrialEvents')
            conv_tev = ev;
            break
        end
    end
    if isempty(conv_tev)
        error('[%s] convenience file has no event object named TrialEvents', mk);
    end
    conv_trial_ids = double(conv_tev.an_trial_id);
    conv_perf      = double(conv_tev.an_performance_in_trial);
    conv_valid     = conv_trial_ids > 0;
    correct_from_convenience = rowvec(unique( ...
                                             conv_trial_ids(conv_valid & conv_perf == 255)));

    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] correct-trial set (extracted vs convenience)', mk), ...
                                          correct_from_extracted, correct_from_convenience);

    % ------- (d) internal performance_str <-> performance_code consistency -------
    % The three conditions extract_lfp_from_nix.py enforces:
    %   1. forward implication: perf_str=='correct_trial' -> perf_code==255
    %   2. every trial has uniform perf_code across its events
    %   3. correct-trial sets via perf_str and perf_code agree
    perf_str  = ex.trial_events.performance_str;
    perf_code = double(ex.trial_events.performance_code(:));
    is_correct_str = strcmp(perf_str, 'correct_trial');
    is_correct_code = perf_code == 255;

    n_bad_forward = sum(is_correct_str(:) & ~is_correct_code(:));
    if n_bad_forward > 0
        report_fail(sprintf('[%s] (d.1) perf_str==correct_trial -> perf_code==255', mk), ...
                    sprintf('%d events violate the forward implication', n_bad_forward));
        fail_count = fail_count + 1;
    else
        report_pass(sprintf('[%s] (d.1) perf_str==correct_trial -> perf_code==255', mk));
        pass_count = pass_count + 1;
    end

    mixed_trials = [];
    uniq_tids = unique(tid(mask_valid));
    for k = 1:length(uniq_tids)
        t = uniq_tids(k);
        codes = unique(perf_code(tid == t));
        if length(codes) > 1
            mixed_trials(end + 1) = t; %#ok<AGROW>
        end
    end
    if ~isempty(mixed_trials)
        report_fail(sprintf('[%s] (d.2) uniform perf_code per trial', mk), ...
                    sprintf('%d trials have mixed perf_code; first few = %s', ...
                    length(mixed_trials), mat2str(mixed_trials(1:min(10, end)))));
        fail_count = fail_count + 1;
    else
        report_pass(sprintf('[%s] (d.2) uniform perf_code per trial', mk));
        pass_count = pass_count + 1;
    end

    correct_by_code = rowvec(unique(tid(is_correct_code(:) & mask_valid)));
    [pass_count, fail_count] = assert_set(pass_count, fail_count, ...
                                          sprintf('[%s] (d.3) correct-trial set via perf_str == via perf_code', mk), ...
                                          correct_from_extracted, correct_by_code);

    % Summary of per-monkey key numbers
    fprintf('\n  [%s] summary:\n', mk);
    fprintf('    X shape:           %s\n', mat2str(size(ex.X)));
    fprintf('    fs:                %g Hz\n', ex.fs);
    fprintf('    t_start:           %g s (units=%s)\n', ex.t_start, ex.t_start_units);
    fprintf('    correct trials:    %d (extracted), %d (convenience)\n', ...
            length(correct_from_extracted), length(correct_from_convenience));
    fprintf('    electrode union:   %s\n', mat2str(el_union));
    fprintf('    trial union:       %s\n', mat2str(tr_union));

    clear ex conv seg conv_tev conv_trial_ids conv_perf conv_valid;
end

fprintf('\n==========================================\n');
fprintf('Summary: %d checks passed, %d failed\n', pass_count, fail_count);
if fail_count > 0
    error('check_nix_vs_odml: %d check(s) failed. See log above. Do not proceed.', fail_count);
end
fprintf('All equivalence checks PASSED. Safe to regenerate preprocessed outputs.\n');

%% Local helpers

function v = rowvec(x)
    % Sorted double row vector of the elements of x.
    x = double(x(:)).';
    v = sort(x);
end

function [pass_count, fail_count] = assert_set(pass_count, fail_count, label, got, want)
    % Compare two sets, print PASS or FAIL with the extra and missing elements, and update the counters.
    got = rowvec(got);
    want = rowvec(want);
    if isequal(got, want)
        report_pass(label);
        pass_count = pass_count + 1;
    else
        extra   = setdiff(got,  want);
        missing = setdiff(want, got);
        detail = '';
        if ~isempty(extra)
            detail = sprintf('extra=%s', mat2str(extra));
        end
        if ~isempty(missing)
            if ~isempty(detail)
                detail = [detail ' '];
            end
            detail = [detail sprintf('missing=%s', mat2str(missing))];
        end
        report_fail(label, detail);
        fail_count = fail_count + 1;
    end
end

function report_pass(label)
    fprintf('  PASS  %s\n', label);
end

function report_fail(label, detail)
    fprintf('  FAIL  %s\n', label);
    if ~isempty(detail)
        fprintf('        %s\n', detail);
    end
end
