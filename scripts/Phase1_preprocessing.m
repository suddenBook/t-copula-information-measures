%PHASE1_PREPROCESSING Phase 1: preprocess each recording into the analysis input of the t-copula study.
%
% Pipeline per recording:
%   load -> stationarity QC -> artifact detection and cleaning -> marginal QC -> band-selection PSD -> optional band-pass filter -> save -> CSV tables -> figures
%
% Each recording becomes the cleaned, band-limited X_filtered, together with the QC results, the CSV tables and the recording.json sidecar that the later phases and the Python figures read.
% Every output except recording.json goes to <results_root>/<family>/<results_subject>/<results_condition>/Phase1/{images,text,mat} through openPhaseOutput, which also opens the run log (diary).
% recording.json sits one level above Phase1/, in the recording's own folder.
% The mat is written from an explicit list of scientific variables, never a whole-workspace dump, and every non-interactive figure is rendered on the Python path.
%
% The recording table below is the only one in the pipeline: phases 2 and later find the recordings this phase wrote by listing their recording.json files (discoverPhase1Recordings).
% A recording that fails is reported at the end, and the others still run.
%
% After the mat is saved, the phase calls renderPhaseFigures('render_phase1_figures', ...) once, as every phase does.
% A renderer failure never removes saved data and never fails the phase: nothing downstream reads a figure, and the renderers read only the mat and the artifact_clean_decision.json sidecar, so a missing panel can be redrawn from a results tree without recomputing anything.
% Launch MATLAB with SHELL=/bin/bash: the renderers run through system(), and without a usable shell renderPhaseFigures only warns and writes images/render_figures.sh.
%
% Two interactive figures stay in MATLAB as human input surfaces: the log-log PSD band-pass picker (runInteractiveBandpassSelection) and the artifact channel review (artifact_selection_mode = 'interactive').
% The band-choice PSD is always saved (psd_band_selection) and rendered by render_phase1_band_selection_psd.py.
%
% No rank-uniform matrix is saved: the copula phases compute their own pseudo-observations from X_filtered, and the QC figures that show uniforms recompute them.
%
% Input (read by src/shared/io/loadData.m): a .mat file with a top-level real [samples x channels] X without NaN or Inf, in double or single precision (Phase 1 converts it to double), and a top-level scalar fs in Hz.
% The optional top-level variables channel_labels, channel_ids, is_soz_channel, pair_info, seizure_sample and segment_boundaries are carried forward when present.
%
% Environment:
%   TC_DATA_ROOT, TC_RESULTS_ROOT - input and output roots (default data/ and results/ under the repository).
%   TC_RECORDINGS                 - comma-separated dataset keys to process; empty processes the whole table.
%   TC_RECORDING_INDEX            - one 1-based row of the table; it cannot be combined with TC_RECORDINGS.
%   TC_OVERWRITE                  - 1 recomputes recordings that are already finished.
%   TC_RENDER_FIGURES             - 0 skips the figures.
% TC_WORKERS has no effect, because Phase 1 runs serially.

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

%% ============================================================ settings
% The recording table: ten non-ictal recordings from five biological subjects, one localRecordingRow call per recording with these arguments in order.
%   dataset            - dataset key; its first path component is the family, which is also the first folder of the output path.
%   results_subject    - subject folder of the output path.
%   results_condition  - condition folder of the output path.
%   seed_base          - fixed per-recording integer in 1..10, the recording axis of the Phase 3 seed_grid, not a row index; every later phase seeds its channel-set sampling from it (channelGroupSeed), so changing it changes that recording's channel sets everywhere.
%   input_relpath      - input MAT file, relative to the data root.
%   modality           - 'SEEG', 'LFP' or 'ECoG'.
%   biological_subject - patient or animal identifier.
%   recording          - recording label.
%   state              - recording state, such as sleep, wake, baseline, eyes_closed or propofol.
%   analysis_role      - role of the recording in the study.
%   qc_adapter         - artifact-detection adapter; 'seeg_auto' is the only one implemented.
%   temporal_adapter   - 'continuous', or 'trial_segments' for concatenated trials, which must carry segment_boundaries.
%   filter_signal      - true when Phase 1 band-pass filters the recording.
%   bandpass           - [low high] in Hz that Phase 1 applies when filter_signal is true, otherwise [].
%   native_band        - [low high] in Hz applied upstream, or [] when the input is not band-limited.
%   expected_fs        - sampling rate in Hz the input must have.
%   expected_channels  - channel count the input must have.
%   note               - free-text provenance note.
% localRecordingRow derives artifact_profile from the modality: a z-scored ECoG input has no absolute voltage scale and needs a scale-relative flatline threshold, and a voltage-domain input uses an absolute one.
recordings = [ ...
    localRecordingRow('human_SEEG/001_sleep', '001', 'sleep', 1, ...
    'human_SEEG/preprocessed/001_sleep.mat', ...
    'SEEG', 'human-001', '001_sleep', 'sleep', 'primary', ...
    'seeg_auto', 'continuous', true, [4 14], [], 512, 136, ...
    'Raw SEEG; Phase 1 applies [4,14] Hz after SEEG QC.'); ...
    localRecordingRow('human_SEEG/001_wake', '001', 'wake', 2, ...
    'human_SEEG/preprocessed/001_wake.mat', ...
    'SEEG', 'human-001', '001_wake', 'wake', 'primary', ...
    'seeg_auto', 'continuous', true, [4 14], [], 512, 136, ...
    'Raw SEEG; Phase 1 applies [4,14] Hz after SEEG QC.'); ...
    localRecordingRow('human_SEEG/002_sleep', '002', 'sleep', 3, ...
    'human_SEEG/preprocessed/002_sleep.mat', ...
    'SEEG', 'human-002', '002_sleep', 'sleep', 'primary', ...
    'seeg_auto', 'continuous', true, [4 14], [], 512, 115, ...
    'Raw SEEG; Phase 1 applies [4,14] Hz after SEEG QC.'); ...
    localRecordingRow('human_SEEG/002_wake', '002', 'wake', 4, ...
    'human_SEEG/preprocessed/002_wake.mat', ...
    'SEEG', 'human-002', '002_wake', 'wake', 'primary', ...
    'seeg_auto', 'continuous', true, [4 14], [], 512, 115, ...
    'Raw SEEG; Phase 1 applies [4,14] Hz after SEEG QC.'); ...
    localRecordingRow('monkey_LFP/monkey_L/baseline_correct_epoch', ...
    'monkey_L', 'baseline', 5, ...
    'monkey_LFP/preprocessed/monkey_L/baseline_correct_epoch.mat', ...
    'LFP', 'monkey-L', 'monkey_L_baseline', 'baseline', 'primary', ...
    'seeg_auto', 'trial_segments', false, [], [12 25], 1000, 93, ...
    ['Upstream channel-QC and per-trial [12,25] Hz filtering; Phase 1 still ' ...
    'runs the common artifact-cleaning pipeline and preserves trial joins.']); ...
    localRecordingRow('monkey_LFP/monkey_N/baseline_correct_epoch', ...
    'monkey_N', 'baseline', 6, ...
    'monkey_LFP/preprocessed/monkey_N/baseline_correct_epoch.mat', ...
    'LFP', 'monkey-N', 'monkey_N_baseline', 'baseline', 'primary', ...
    'seeg_auto', 'trial_segments', false, [], [12 25], 1000, 94, ...
    ['Upstream channel-QC and per-trial [12,25] Hz filtering; Phase 1 still ' ...
    'runs the common artifact-cleaning pipeline and preserves trial joins.']); ...
    localRecordingRow('monkey_ECoG/EC1', 'EC1', 'eyes_closed', 7, ...
    'monkey_ECoG/preprocessed/EC1.mat', ...
    'ECoG', 'monkey-George', 'EC1', 'eyes_closed', 'primary', ...
    'seeg_auto', 'continuous', false, [], [1 8], 100, 49, ...
    ['Upstream good-channel intersection, [1,8] Hz, decimated and z-scored; ' ...
    'Phase 1 still runs the common artifact-cleaning pipeline.']); ...
    localRecordingRow('monkey_ECoG/EC2', 'EC2', 'eyes_closed', 8, ...
    'monkey_ECoG/preprocessed/EC2.mat', ...
    'ECoG', 'monkey-George', 'EC2', 'eyes_closed', 'primary', ...
    'seeg_auto', 'continuous', false, [], [1 8], 100, 49, ...
    ['Upstream good-channel intersection, [1,8] Hz, decimated and z-scored; ' ...
    'Phase 1 still runs the common artifact-cleaning pipeline.']); ...
    localRecordingRow('monkey_ECoG/PF1', 'PF1', 'propofol', 9, ...
    'monkey_ECoG/preprocessed/PF1.mat', ...
    'ECoG', 'monkey-George', 'PF1', 'propofol', 'primary', ...
    'seeg_auto', 'continuous', false, [], [1 8], 100, 128, ...
    ['Upstream [1,8] Hz, decimated and z-scored; Phase 1 still runs the ' ...
    'common artifact-cleaning pipeline.']); ...
    localRecordingRow('monkey_ECoG/PF2', 'PF2', 'propofol', 10, ...
    'monkey_ECoG/preprocessed/PF2.mat', ...
    'ECoG', 'monkey-George', 'PF2', 'propofol', 'primary', ...
    'seeg_auto', 'continuous', false, [], [1 8], 100, 128, ...
    ['Upstream [1,8] Hz, decimated and z-scored; Phase 1 still runs the ' ...
    'common artifact-cleaning pipeline.']) ...
    ];

% The preprocessing policy, set here and nowhere else.
% The other detector settings come from peArtifactDetectionDefaults; localArtifactParams sets only the flatline threshold and the private detector seed per recording.
settings = struct( ...
    'filter_order',                   4, ...          % designfilt order of the Butterworth band-pass (filterData)
    'severe_threshold',               10, ...         % percent; a detector score above it makes a channel a candidate
    'cleaning_strategy',              'channel_only', ...  % remove channels only, never time windows
    'apply_powerline_filter',         true, ...       % notch 50 and 60 Hz on the channels with detected line noise
    'artifact_selection_mode',        'auto', ...     % 'auto' removes abnormal-kurtosis channels; 'interactive' opens the channel review
    'run_stationarity_check',         true, ...       % sliding-window amplitude QC
    'run_artifact_detection',         true, ...       % artifact detection and channel cleaning
    'run_marginal_check',             true, ...       % per-channel kurtosis and skewness QC
    'seed_offset',                    100000, ...     % detector seed = seed_offset + seed_base
    'flatline_std_raw_voltage',       1.0, ...        % absolute flatline threshold in input units, voltage-domain input
    'flatline_relative_std_standardized', 0.01, ...   % flatline threshold as a fraction of robust std, z-scored input
    'autocorrelation_max_lag',        200, ...        % samples; lag window of the integrated-autocorrelation diagnostic
    'replay_saved_bandpass_decision', false, ...      % reuse a saved interactive band choice instead of prompting
    'phase1_schema_version',          '2.3.0');       % version stamp of the saved mat and recording.json
% autocorrelation_max_lag is a diagnostic window, not a band assumption.
% The sum stops at the first non-positive autocorrelation, which for every band in this study comes within tens of samples, so 200 lags never cut it short while the O(max_lag * samples * channels) scan stays affordable on the longest recording.

steps = struct( ...
    'save_results',   true, ...                                         % false stops before the provenance, the mat, the tables and the figures
    'render_figures', getEnvFlag("TC_RENDER_FIGURES", true), ...         % TC_RENDER_FIGURES=0 skips the figures
    'overwrite',      getEnvFlag("TC_OVERWRITE", false));               % TC_OVERWRITE=1 recomputes finished recordings

data_root = resolveRootFromEnv("TC_DATA_ROOT", string(fullfile(worktree_root, 'data')));
results_root = resolveRootFromEnv("TC_RESULTS_ROOT", string(fullfile(worktree_root, 'results')));
selected_keys = getEnvList("TC_RECORDINGS");
selected_index = getEnvInteger("TC_RECORDING_INDEX", [], [1, numel(recordings)]);

%% ============================================================ run
enableFigureAutoMaximize();
selected = localSelectRecordings(recordings, selected_keys, selected_index);
fprintf('[Phase1] %d of %d recording(s) selected | data %s | results %s\n', ...
    numel(selected), numel(recordings), data_root, results_root);

failures = strings(0, 1);
for i = 1:numel(selected)
    rec = selected(i);
    try
        localProcessRecording(rec, settings, steps, data_root, results_root);
    catch recording_error
        failures(end + 1, 1) = string(rec.dataset) + " [" + ...
            string(recording_error.identifier) + "] " + ...
            string(recording_error.message); %#ok<SAGROW>
        fprintf(2, '\n[Phase1] %s failed (%s)\n%s\n', rec.dataset, ...
            recording_error.identifier, ...
            getReport(recording_error, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\n[Phase1] finished: %d of %d recording(s) failed\n', ...
    numel(failures), numel(selected));
if ~isempty(failures)
    % Every other recording is already processed and saved; this error only sets the run's exit status.
    error('Phase1:RecordingFailures', ...
        'Phase 1 failed for %d recording(s):\n%s', numel(failures), ...
        strjoin(failures, newline));
end


% ===========================================================================
% Local functions
% ===========================================================================
function row = localRecordingRow(dataset, results_subject, results_condition, ...
        seed_base, input_relpath, modality, biological_subject, recording, ...
        state, analysis_role, qc_adapter, temporal_adapter, filter_signal, ...
        bandpass, native_band, expected_fs, expected_channels, note)
    % One row of the recording table, with the derived family and artifact_profile fields.
    if strcmp(modality, 'ECoG')
        artifact_profile = 'standardized_relative';
    else
        artifact_profile = 'raw_voltage';
    end
    row = struct( ...
        'dataset',            dataset, ...
        'family',             char(extractBefore([dataset '/'], '/')), ...
        'results_subject',    results_subject, ...
        'results_condition',  results_condition, ...
        'seed_base',          seed_base, ...
        'input_relpath',      input_relpath, ...
        'modality',           modality, ...
        'biological_subject', biological_subject, ...
        'recording',          recording, ...
        'state',              state, ...
        'analysis_role',      analysis_role, ...
        'qc_adapter',         qc_adapter, ...
        'artifact_profile',   artifact_profile, ...
        'temporal_adapter',   temporal_adapter, ...
        'filter_signal',      logical(filter_signal), ...
        'bandpass',           double(bandpass(:).'), ...
        'native_band',        double(native_band(:).'), ...
        'expected_fs',        expected_fs, ...
        'expected_channels',  expected_channels, ...
        'note',               note);
end


function selected = localSelectRecordings(recordings, keys, index)
    % The rows chosen by TC_RECORDINGS (dataset keys) or TC_RECORDING_INDEX (a table row), never both; neither selects the whole table.
    assert(isempty(keys) || isempty(index), 'Phase1:AmbiguousSelection', ...
        ['TC_RECORDINGS and TC_RECORDING_INDEX are mutually exclusive; got ' ...
        'keys "%s" and index %d.'], strjoin(keys, ', '), index);
    if ~isempty(index)
        selected = recordings(index);
        return;
    end
    if isempty(keys)
        selected = recordings;
        return;
    end
    present = string({recordings.dataset});
    unknown = setdiff(string(keys), present, 'stable');
    assert(isempty(unknown), 'Phase1:UnknownRecording', ...
        'TC_RECORDINGS names %s, which the recording table does not hold (%s).', ...
        strjoin(unknown, ', '), strjoin(present, ', '));
    selected = recordings(ismember(present, string(keys)));
end


function localProcessRecording(rec, settings, steps, data_root, results_root)
    % Preprocess one recording end to end and write its Phase1 output tree.
    %
    % ctx owns the diary through an onCleanup guard, so the diary flushes and closes when this function returns, also on an error.
    % An error propagates to the recording loop, which records it and continues with the next recording.
    rel_base = strjoin({rec.family, rec.results_subject, rec.results_condition}, '/');
    input_path = fullfile(data_root, rec.input_relpath);
    mat_relpath = fullfile('Phase1', 'mat', 'preprocessed_data.mat');

    probe = openPhaseOutput(rel_base, 1, results_root, 'Diary', false);
    save_path = fullfile(probe.mat_dir, 'preprocessed_data.mat');
    json_path = fullfile(fileparts(probe.base), 'recording.json');
    config_path = fullfile(probe.text_dir, 'run_config.json');
    identity = localSettingsIdentity(settings, rec);
    % A recording is done only when the analysis matrix, the provenance stamp and the sidecar that makes it discoverable all exist.
    % recording.json is written last, after the CSVs, so a run cut off between the matrix and the tables leaves the recording unfinished instead of finished with missing tables.
    % A finished recording is reused only under the settings identity its sidecar carries; after a change to a policy setting or to a row field in that identity (localSettingsIdentity) the recording fails with an error instead of reusing the stored output.
    % A change to a row field outside the identity, such as input_relpath or a label, leaves the stored output in use.
    if ~steps.overwrite && isfile(save_path) && isfile(json_path) && isfile(config_path)
        localAssertIdentity(json_path, identity);
        fprintf('[Phase1] %s: already done, skipping (%s)\n', rec.dataset, save_path);
        return;
    end

    ctx = openPhaseOutput(rel_base, 1, results_root);
    if steps.overwrite
        localWithdrawPublication(ctx, save_path, json_path);
        fprintf('[Phase1] %s: TC_OVERWRITE=1, the previous output was withdrawn\n', ...
            rec.dataset);
    end
    started_utc = datetime('now', 'TimeZone', 'UTC');
    fprintf('\n========================================================\n');
    fprintf('Phase 1 recording: %s\n', rec.dataset);
    fprintf('  input : %s\n', input_path);
    fprintf('  output: %s\n', ctx.base);
    fprintf('========================================================\n');

    %% 1.1 Load the input and its optional metadata
    if ~isfile(input_path)
        error('Phase1:MissingInput', ...
            'Input of %s does not exist: %s', rec.dataset, input_path);
    end
    [X, metadata] = loadData(input_path);
    fs = metadata.fs;
    X = double(X);
    localValidateInputContract(X, fs, rec, input_path);

    optional_meta = extractOptionalMetadata(input_path);
    % A recording without labels saves channel_labels = [], which the figure readers treat as absent; an empty cell would read as a zero-size cell.
    channel_labels = [];
    if isfield(optional_meta, 'channel_labels')
        channel_labels = optional_meta.channel_labels;
    end
    channel_labels_raw = channel_labels;
    if isfield(optional_meta, 'channel_ids')
        channel_ids = double(optional_meta.channel_ids(:)).';
    else
        channel_ids = 1:size(X, 2);
    end
    assert(numel(channel_ids) == size(X, 2) && isreal(channel_ids) && ...
        all(isfinite(channel_ids)) && numel(unique(channel_ids)) == numel(channel_ids), ...
        'Phase1:ChannelIdContract', ...
        ['Input %s must carry one finite, unique numeric channel ID per data ' ...
        'column (observed %d IDs for %d columns).'], ...
        input_path, numel(channel_ids), size(X, 2));
    channel_ids_raw = channel_ids;

    %% 1.2 Stationarity check (compute only; figure by render_phase1_stationarity.py)
    if settings.run_stationarity_check
        stationarity_info = checkStationarity(X, fs);
    else
        stationarity_info = struct('note', 'Stationarity check disabled.');
    end

    fprintf('Loaded %d samples x %d channels (%.1f sec at %g Hz)\n', ...
        size(X, 1), size(X, 2), size(X, 1) / fs, fs);

    %% 1.3 Artifact detection and cleaning
    X_raw = X;
    artifact_params = localArtifactParams(settings, rec);
    decision_sidecar_path = fullfile(ctx.text_dir, 'artifact_clean_decision.json');
    if settings.run_artifact_detection
        assert(strcmp(rec.qc_adapter, 'seeg_auto'), 'Phase1:UnknownQcAdapter', ...
            'Recording %s declares qc_adapter "%s"; only seeg_auto is implemented.', ...
            rec.dataset, rec.qc_adapter);
        [artifact_regions, artifact_info] = detectSEEGArtifacts(X, fs, artifact_params);

        clean_params = struct();
        clean_params.severe_threshold       = settings.severe_threshold;
        clean_params.cleaning_strategy      = settings.cleaning_strategy;
        clean_params.apply_powerline_filter = settings.apply_powerline_filter;
        clean_params.artifact_selection_mode = settings.artifact_selection_mode;
        clean_params.visualize = strcmp(settings.artifact_selection_mode, 'interactive');
        if ~isempty(channel_labels_raw)
            clean_params.channel_labels = channel_labels_raw;
        end
        clean_params.decision_sidecar_path = decision_sidecar_path;

        [X, clean_info] = cleanSEEGArtifacts(X, fs, artifact_regions, ...
            artifact_info, clean_params);

        % Adapter and policy provenance, saved beside the matrices they produced.
        artifact_info.artifact_schema_version = '1.2.0';
        artifact_info.qc_adapter = char(rec.qc_adapter);
        artifact_info.artifact_profile = char(rec.artifact_profile);
        artifact_info.temporal_adapter = char(rec.temporal_adapter);
        clean_info.clean_schema_version = '1.0.0';
        clean_info.qc_adapter = char(rec.qc_adapter);
        clean_info.artifact_profile = char(rec.artifact_profile);
        clean_info.temporal_adapter = char(rec.temporal_adapter);
        clean_info.automatic_exclusion_rule = 'abnormal_kurtosis_only';

        if ~isempty(channel_labels)
            channel_labels = channel_labels(clean_info.retained_channels);
        end
        channel_ids = channel_ids(clean_info.retained_channels);
    else
        fprintf(['Artifact detection disabled by settings; every channel is ' ...
            'retained and X is unchanged.\n']);
        artifact_info = struct('note', 'Artifact detection disabled.');
        clean_info = struct('note', 'Artifact cleaning disabled.', ...
            'retained_channels', 1:size(X, 2), 'excluded_channels', []);
    end

    %% 1.4 Marginal distribution check (compute only; figure on the Python path)
    if settings.run_marginal_check
        marginal_stats = checkMarginalDistributions(X, 'Artifact-cleaned pre-band-pass');
    else
        marginal_stats = struct('note', 'Marginal check disabled.');
    end

    %% 1.5 Band-selection PSD evidence (always saved; rendered on a linear frequency axis)
    % The across-channel |PSD|^0.5 summary that the interactive picker plots, computed on the cleaned signal before filtering, from which the band is chosen.
    psd_band_selection = computeBandSelectionPsd(X, fs); %#ok<NASGU>

    %% 1.6 Band-pass filter (per recording: rec.filter_signal)
    [X_filtered, filter_range, filter_applied, native_band, filter_params] = ...
        localApplyBandpass(X, fs, rec, settings, ctx.text_dir); %#ok<ASGLU>

    %% 1.7 Trial boundaries, as sample indices of X_filtered
    has_segments = isfield(optional_meta, 'segment_boundaries');
    if has_segments
        segment_boundaries = localValidateSegmentBoundaries( ...
            optional_meta.segment_boundaries, size(X_raw, 1), size(X_filtered, 1), rec);
        segment_boundaries_coordinate = ...
            'cumulative_trial_end_indices_in_X_filtered'; %#ok<NASGU>
        segment_boundaries_schema_version = '1.0.0'; %#ok<NASGU>
        segment_boundaries_source_sha256 = ...
            sha256Json(double(optional_meta.segment_boundaries(:))); %#ok<NASGU>
    elseif strcmp(rec.temporal_adapter, 'trial_segments')
        error('Phase1:MissingSegmentBoundaries', ...
            'Trial-segment recording %s carries no segment_boundaries metadata: %s', ...
            rec.dataset, input_path);
    else
        segment_boundaries = size(X_filtered, 1);
    end
    n_segments = numel(segment_boundaries);

    %% 1.8 Effective sample size of the analysis input
    autocorrelation = localAutocorrelationTable(X_filtered, channel_ids, ...
        segment_boundaries, settings.autocorrelation_max_lag);
    fprintf(['Integrated autocorrelation time over %d segment(s): median %.2f ' ...
        'samples, median effective n %.0f of %d\n'], n_segments, ...
        median(autocorrelation.tau_int_samples), ...
        median(autocorrelation.n_eff), size(X_filtered, 1));

    if ~steps.save_results
        fprintf('[Phase1] save_results is off; nothing written for %s\n', rec.dataset);
        return;
    end

    %% 1.9 Provenance, written once and reused by the mat
    run_settings = settings;
    run_settings.steps = steps;
    run_settings.recording = rec;
    run_settings.artifact_params = artifact_params;
    run_settings.settings_identity = identity;
    % run_config.json is written before the mat here, unlike the later phases, because the mat embeds the provenance stamp this call produces.
    % recording.json is the completion marker instead: it is written after the CSVs, and nothing discovers this recording without it.
    config_path = writeRunConfig(ctx, run_settings, struct( ...
        'n_workers',    0, ...
        'started_utc',  started_utc, ...
        'finished_utc', datetime('now', 'TimeZone', 'UTC'), ...
        'input_path',   input_path, ...
        'input_sha256', sha256File(input_path)));
    written_config = jsondecode(fileread(config_path));
    provenance = written_config.provenance; %#ok<NASGU>

    %% 1.10 Save the explicit list of scientific variables (no rank-uniform, no workspace dump)
    % Drop artifact_info's large per-sample diagnostics first: the QC figures read only channel_artifact_pct, and the [samples x channels] cell artifact_types stores under -v7.3 as millions of object references that inflate the file about tenfold.
    % final_artifact_mask is first reduced to a [channels x 1 s windows] occupancy matrix, well under 1 MB, which keeps the timing of the artifacts for its figure.
    if isstruct(artifact_info) && isfield(artifact_info, 'final_artifact_mask')
        artifact_info.time_profile = computeArtifactTimeProfile( ...
            artifact_info.final_artifact_mask, fs, 1);
    end
    if isstruct(artifact_info)
        for fld = {'artifact_scores', 'artifact_types', 'final_artifact_mask'}
            if isfield(artifact_info, fld{1})
                artifact_info = rmfield(artifact_info, fld{1});
            end
        end
    end

    % The next five exist only for the explicit save-list below.
    dataset_key = rec.dataset; %#ok<NASGU>
    study_record = rec; %#ok<NASGU>
    phase1_seed = artifact_params.random_seed; %#ok<NASGU>
    phase1_schema_version = settings.phase1_schema_version;
    settings_identity = identity; %#ok<NASGU>
    save_names = {'X_filtered', 'channel_ids', 'channel_ids_raw', ...
        'X', 'X_raw', 'fs', ...
        'filter_range', 'filter_applied', 'native_band', 'filter_params', ...
        'marginal_stats', 'stationarity_info', 'artifact_info', 'clean_info', ...
        'channel_labels', 'psd_band_selection', 'study_record', ...
        'phase1_seed', 'phase1_schema_version', 'dataset_key', 'provenance', ...
        'settings_identity'};

    % Save the optional metadata only when the input carries it.
    if isfield(optional_meta, 'is_soz_channel')
        is_soz_channel = optional_meta.is_soz_channel; %#ok<NASGU>
        save_names{end + 1} = 'is_soz_channel';
    end
    if isfield(optional_meta, 'pair_info')
        pair_info = optional_meta.pair_info; %#ok<NASGU>
        save_names{end + 1} = 'pair_info';
    end
    if isfield(optional_meta, 'seizure_sample')
        seizure_sample = optional_meta.seizure_sample; %#ok<NASGU>
        save_names{end + 1} = 'seizure_sample';
    end
    if has_segments
        save_names = [save_names, {'segment_boundaries', ...
            'segment_boundaries_coordinate', ...
            'segment_boundaries_schema_version', ...
            'segment_boundaries_source_sha256'}];
    end

    % Staged under a name no other writer can share, then renamed into place.
    temp_save_path = localStagingName(save_path, '.tmp.mat');
    try
        save(temp_save_path, save_names{:}, '-v7.3');
    catch save_error
        localDeleteFiles({temp_save_path});
        rethrow(save_error);
    end
    movefile(temp_save_path, save_path, 'f');
    fprintf('Saved Phase 1 output (explicit save-list, %d vars): %s\n', ...
        numel(save_names), save_path);

    %% 1.11 CSV tables -- every number of this phase also lands in a text file
    writeTableAtomic(localChannelTable(channel_ids_raw, channel_labels_raw, ...
        clean_info, artifact_info, decision_sidecar_path), ...
        fullfile(ctx.text_dir, 'channels.csv'));
    writeTableAtomic(localBandProvenanceTable(rec, settings, fs, filter_params), ...
        fullfile(ctx.text_dir, 'band_provenance.csv'));
    writeTableAtomic(autocorrelation, fullfile(ctx.text_dir, 'autocorrelation.csv'));
    writeTableAtomic(localStationarityTable(stationarity_info), ...
        fullfile(ctx.text_dir, 'stationarity.csv'));
    writeTableAtomic(localMarginalTable(marginal_stats, channel_ids, channel_labels), ...
        fullfile(ctx.text_dir, 'marginals.csv'));
    fprintf('Wrote 5 CSV tables: %s\n', ctx.text_dir);

    %% 1.12 recording.json: the sidecar through which later phases discover this recording, and its completion marker
    % It is written last, after the mat and the tables, so an interrupted run is resumed rather than skipped.
    localWriteRecordingJson(json_path, struct( ...
        'dataset_key',           rec.dataset, ...
        'family',                rec.family, ...
        'results_subject',       rec.results_subject, ...
        'results_condition',     rec.results_condition, ...
        'seed_base',             rec.seed_base, ...
        'modality',              rec.modality, ...
        'biological_subject',    rec.biological_subject, ...
        'recording',             rec.recording, ...
        'state',                 rec.state, ...
        'temporal_adapter',      rec.temporal_adapter, ...
        'artifact_profile',      rec.artifact_profile, ...
        'input_relpath',         rec.input_relpath, ...
        'fs',                    double(fs), ...
        'filter_signal',         rec.filter_signal, ...
        'bandpass',              rec.bandpass, ...
        'native_band',           rec.native_band, ...
        'filter_range',          filter_range, ...
        'expected_channels',     rec.expected_channels, ...
        'n_channels_raw',        numel(channel_ids_raw), ...
        'n_channels_retained',   numel(channel_ids), ...
        'n_samples',             size(X_filtered, 1), ...
        'n_segments',            n_segments, ...
        'phase1_schema_version', phase1_schema_version, ...
        'phase1_mat',            strrep(mat_relpath, filesep, '/'), ...
        'settings_identity',     identity, ...
        'written_utc',           localUtcTimestamp()));

    %% 1.13 Figures, rendered from the mat just written and never fatal
    if steps.render_figures
        localReportFigureStatus(renderPhaseFigures('render_phase1_figures', ...
            save_path, ctx.images_dir, ctx.text_dir), rec.dataset);
    end
    % ctx goes out of scope here, and its onCleanup guard turns the diary off.
end


function identity = localSettingsIdentity(settings, rec)
    % The settings identity of one recording's preprocessing: the whole policy struct, this recording's detector configuration (so a change to the detector defaults changes it too), and the row fields below, which set how the input is checked, cleaned and filtered.
    % Row fields that only label a recording (subject, state, role, note) change no number and stay out, and so do input_relpath, modality and qc_adapter.
    % Phase 2 folds this identity into its own, so every later phase depends on it.
    identity = settingsIdentity(struct( ...
        'phase',             'Phase1', ...
        'settings',          settings, ...
        'artifact_params',   localArtifactParams(settings, rec), ...
        'dataset_key',       rec.dataset, ...
        'seed_base',         rec.seed_base, ...
        'filter_signal',     rec.filter_signal, ...
        'bandpass',          rec.bandpass, ...
        'native_band',       rec.native_band, ...
        'expected_fs',       rec.expected_fs, ...
        'expected_channels', rec.expected_channels, ...
        'artifact_profile',  rec.artifact_profile, ...
        'temporal_adapter',  rec.temporal_adapter));
end


function localAssertIdentity(json_path, identity)
    % Refuse a finished recording whose sidecar carries another settings identity, because its matrix comes from a different row or policy and every later phase would read it as this one.
    record = jsondecode(fileread(json_path));
    stored = '';
    if isfield(record, 'settings_identity')
        stored = char(string(record.settings_identity));
    end
    if strcmp(stored, identity)
        return;
    end
    if isempty(stored)
        stored = '<none recorded>';
    end
    error('Phase1:SettingsIdentityMismatch', ...
        ['%s was written under the settings identity %s, but this run''s ' ...
        'settings are %s. Rerun with TC_OVERWRITE=1 to preprocess the ' ...
        'recording again, or restore the recording table row and the policy ' ...
        'settings that produced it.'], json_path, stored, identity);
end


function localWithdrawPublication(ctx, save_path, json_path)
    % Under TC_OVERWRITE=1, delete the existing output before anything is recomputed: the sidecar that makes the recording discoverable, the matrix, the five tables, run_config.json and the figure manifests.
    % An interrupted overwrite then leaves no recording rather than an outdated one that still looks finished.
    % The two decision sidecars (bandpass_decision.json, artifact_clean_decision.json) stay, because they record decisions, which a replay reads, rather than results.
    text_files = {'channels.csv', 'band_provenance.csv', 'autocorrelation.csv', ...
        'stationarity.csv', 'marginals.csv', 'run_config.json', ...
        'figures_manifest.json'};
    localDeleteFiles([{json_path, save_path}, ...
        fullfile(ctx.text_dir, text_files), ...
        {fullfile(ctx.images_dir, 'figures_manifest.json')}]);
end


function localDeleteFiles(paths)
    % Delete each listed file that exists.
    for k = 1:numel(paths)
        if isfile(paths{k})
            delete(paths{k});
        end
    end
end


function tmp = localStagingName(path, suffix)
    % A staging name no concurrent writer can share: the destination plus a per-call token, in the destination's folder so the final move is a rename.
    [~, token] = fileparts(tempname);
    tmp = [path, '.', token, suffix];
end


function params = localArtifactParams(settings, rec)
    % The detector configuration of one recording: the defaults, this recording's private random seed, and the flatline threshold its amplitude domain needs.
    % A z-scored ECoG input has no absolute voltage scale, so its flatline threshold is a fraction of the channel's own robust spread; a voltage-domain input uses the absolute threshold.
    params = peArtifactDetectionDefaults();
    params.random_seed = settings.seed_offset + double(rec.seed_base);
    switch rec.artifact_profile
        case 'raw_voltage'
            params.flatline_std = settings.flatline_std_raw_voltage;
            params.flatline_relative_std = [];
        case 'standardized_relative'
            params.flatline_std = [];
            params.flatline_relative_std = settings.flatline_relative_std_standardized;
        otherwise
            error('Phase1:UnknownArtifactProfile', ...
                'Recording %s declares artifact_profile "%s", which has no flatline rule.', ...
                rec.dataset, rec.artifact_profile);
    end
end


function localValidateInputContract(X, fs, rec, input_path)
    % Fail before any analysis unless the input matches its table row: [samples x expected_channels] with at least two samples, real and finite, at expected_fs.
    assert(isnumeric(X) && ismatrix(X) && size(X, 1) >= 2 && ...
        size(X, 2) == rec.expected_channels, ...
        'Phase1:UnexpectedInputShape', ...
        'Input %s must be [samples x %d channels]; observed %d x %d.', ...
        input_path, rec.expected_channels, size(X, 1), size(X, 2));
    assert(isreal(X) && all(isfinite(X), 'all'), 'Phase1:NonfiniteInput', ...
        'Input %s holds non-finite or complex values.', input_path);
    assert(isscalar(fs) && isfinite(fs) && ...
        abs(double(fs) - rec.expected_fs) <= ...
        64 * eps(max(double(fs), rec.expected_fs)), ...
        'Phase1:UnexpectedSamplingRate', ...
        'Input %s was expected at fs = %g Hz, observed fs = %g Hz.', ...
        input_path, rec.expected_fs, fs);
end


function boundaries = localValidateSegmentBoundaries(raw_boundaries, n_raw, ...
        n_filtered, rec)
    % The trial ends as strictly increasing cumulative sample indices of X_filtered, the last equal to its length.
    % Only a trial-segment recording may carry them, and cleaning and filtering must keep every sample.
    assert(strcmp(rec.temporal_adapter, 'trial_segments'), ...
        'Phase1:UnexpectedSegmentBoundaries', ...
        'Continuous recording %s unexpectedly supplied segment boundaries.', ...
        rec.dataset);
    assert(n_filtered == n_raw, 'Phase1:TrialBoundaryRemapRequired', ...
        ['Artifact cleaning or filtering changed the sample count of trial ' ...
        'recording %s (%d raw, %d filtered). A validated boundary remap is ' ...
        'required before this output can be used.'], rec.dataset, n_raw, n_filtered);
    boundaries = double(raw_boundaries(:));
    assert(~isempty(boundaries) && all(isfinite(boundaries)) && ...
        all(boundaries == fix(boundaries)) && all(diff(boundaries) > 0) && ...
        boundaries(1) >= 1 && boundaries(end) == n_filtered, ...
        'Phase1:BadSegmentBoundaries', ...
        ['segment_boundaries of %s must be strictly increasing cumulative integer ' ...
        'trial ends whose last value equals size(X_filtered, 1) = %d.'], ...
        rec.dataset, n_filtered);
end


function [X_filtered, filter_range, filter_applied, native_band, filter_params] = ...
        localApplyBandpass(X, fs, rec, settings, text_dir)
    % Band-pass dispatch: no filter (filter_signal false), the interactive picker (bandpass is the word 'interactive'), or a numeric [low high] preset.
    %
    % Returns X_filtered, the filter_range actually applied ([] when none), the filter_applied flag, the recording's upstream native_band, and a filter_params provenance struct.
    %
    % filter_range never holds a band the run did not apply, because the PSD renderers shade it and would show an invented band as real.
    % Empty means no filter; native_band records the band the recording already carries.
    bandpass_decision = struct('bandpass_mode', 'skipped', ...
        'bandpass_range', [], 'fs', fs, 'channel_count', size(X, 2), ...
        'dataset_name', rec.dataset);
    bandpass_sidecar_path = fullfile(text_dir, 'bandpass_decision.json');

    native_band = double(rec.native_band(:)).';
    bandpass_cfg = rec.bandpass;
    is_interactive_cfg = ischar(bandpass_cfg) || isstring(bandpass_cfg);
    is_preset_cfg = isnumeric(bandpass_cfg) && ~isempty(bandpass_cfg);

    if ~rec.filter_signal
        filter_range = [];              % no filter applied
        filter_applied = false;
        X_filtered = X;
        if isempty(native_band)
            fprintf('Filtering disabled (filter_signal = false); X_filtered = X.\n');
        else
            fprintf(['Filtering disabled (filter_signal = false); X_filtered = X. ' ...
                'Upstream band [%g %g] Hz recorded as native_band.\n'], ...
                native_band(1), native_band(2));
        end
        bandpass_decision.native_band = native_band;
        localWriteJsonSidecar(bandpass_sidecar_path, bandpass_decision);

    elseif is_interactive_cfg
        assert(strcmp(char(bandpass_cfg), 'interactive'), 'Phase1:BadBandpass', ...
            ['The bandpass of %s must be a numeric [lo hi] preset or the word ' ...
            '''interactive''; got %s.'], rec.dataset, char(bandpass_cfg));
        [filter_range, bandpass_decision] = localInteractiveBandpassDecision( ...
            X, fs, rec.dataset, bandpass_sidecar_path, ...
            settings.replay_saved_bandpass_decision);
        X_filtered = filterData(X, fs, filter_range, settings.filter_order);
        filter_applied = true;

    elseif is_preset_cfg
        assert(numel(bandpass_cfg) == 2 && bandpass_cfg(1) > 0 && ...
            bandpass_cfg(2) > bandpass_cfg(1) && bandpass_cfg(2) < fs / 2, ...
            'Phase1:BadBandpass', ...
            ['The bandpass preset of %s must satisfy 0 < low < high < fs/2 = %g; ' ...
            'got [%g %g].'], rec.dataset, fs / 2, bandpass_cfg(1), bandpass_cfg(2));
        filter_range = double(bandpass_cfg(:)).';
        fprintf('Applying preset bandpass = [%.3f %.3f] Hz.\n', ...
            filter_range(1), filter_range(2));
        X_filtered = filterData(X, fs, filter_range, settings.filter_order);
        filter_applied = true;
        bandpass_decision = struct( ...
            'bandpass_mode',  'preset', ...
            'bandpass_range', filter_range, ...
            'fs',             fs, ...
            'channel_count',  size(X, 2), ...
            'dataset_name',   rec.dataset, ...
            'timestamp',      localUtcTimestamp());
        localWriteJsonSidecar(bandpass_sidecar_path, bandpass_decision);

    else
        error('Phase1:BadBandpass', ...
            ['Recording %s sets filter_signal = true but carries no band; a ' ...
            'numeric [lo hi] preset or ''interactive'' is required.'], rec.dataset);
    end

    if filter_applied
        method = 'Butterworth bandpass via filterData.m';
    else
        method = 'none (recording band-limited upstream; X_filtered = X)';
    end
    filter_params = struct( ...
        'filter_range', filter_range, ...
        'filter_applied', filter_applied, ...
        'native_band', native_band, ...
        'filter_order', settings.filter_order, ...
        'method', method, ...
        'bandpass_decision', bandpass_decision);
end


function [filter_range, bandpass_decision] = localInteractiveBandpassDecision( ...
        X, fs, dataset_name, bandpass_sidecar_path, replay_saved_bandpass_decision)
    % The interactive band choice: the range saved in bandpass_decision.json when a replay is requested and the file holds one, otherwise the picker's answer.
    preset_replay = logical(replay_saved_bandpass_decision);
    replayed_from_sidecar = false;
    filter_range = [];

    if preset_replay && isfile(bandpass_sidecar_path)
        try
            saved = jsondecode(fileread(bandpass_sidecar_path));
            if isfield(saved, 'bandpass_range') && numel(saved.bandpass_range) == 2
                filter_range = double(saved.bandpass_range(:)).';
                replayed_from_sidecar = true;
                fprintf('Replaying saved bandpass = [%.3f %.3f] Hz from %s.\n', ...
                    filter_range(1), filter_range(2), bandpass_sidecar_path);
            end
        catch replay_err
            fprintf(2, ['Failed to replay bandpass_decision.json (%s); falling ' ...
                'back to the interactive prompt.\n'], replay_err.message);
        end
    elseif preset_replay
        fprintf(2, ['Saved bandpass replay requested but %s was not found; ' ...
            'falling back to the interactive prompt.\n'], bandpass_sidecar_path);
    end

    if ~replayed_from_sidecar
        [filter_range, bandpass_decision] = ...
            runInteractiveBandpassSelection(X, fs, dataset_name, bandpass_sidecar_path);
    else
        bandpass_decision = struct( ...
            'bandpass_mode',  'preset_replay', ...
            'bandpass_range', filter_range, ...
            'fs',             fs, ...
            'channel_count',  size(X, 2), ...
            'dataset_name',   dataset_name, ...
            'timestamp',      localUtcTimestamp());
    end
end


function T = localAutocorrelationTable(X, channel_ids, boundaries, max_lag)
    % Integrated autocorrelation time of the rank-normal scores, per retained channel.
    %
    % rankIntegratedAutocorrTime assumes one continuous segment (it ranks and autocorrelates the rows it is given), so a trial recording is summarized one segment at a time and the per-channel times are combined as a length-weighted mean.
    % Segments shorter than four samples are skipped.
    % n_eff = n_samples / tau_int_samples is a descriptive effective sample count over the whole recording; nothing in the pipeline decides on it.
    ends = boundaries(:);
    starts = [1; ends(1:end - 1) + 1];
    tau_weighted = zeros(1, size(X, 2));
    weight = 0;
    for s = 1:numel(ends)
        rows = starts(s):ends(s);
        if numel(rows) < 4
            continue;   % too short for a lag-1 autocorrelation to mean anything
        end
        tau_weighted = tau_weighted + numel(rows) * ...
            rankIntegratedAutocorrTime(X(rows, :), min(max_lag, numel(rows) - 2));
        weight = weight + numel(rows);
    end
    assert(weight > 0, 'Phase1:NoUsableSegment', ...
        ['No segment of the %d-sample recording holds the 4 samples the ' ...
        'autocorrelation diagnostic needs.'], size(X, 1));
    tau_int_samples = (tau_weighted / weight).';

    n_samples = repmat(size(X, 1), numel(channel_ids), 1);
    T = table(channel_ids(:), tau_int_samples, n_samples ./ tau_int_samples, ...
        n_samples, repmat(numel(ends), numel(channel_ids), 1), ...
        'VariableNames', {'channel_id', 'tau_int_samples', 'n_eff', ...
        'n_samples', 'n_segments'});
end


function T = localChannelTable(channel_ids_raw, channel_labels_raw, clean_info, ...
        artifact_info, decision_sidecar_path)
    % One row per input channel: its identity, whether it is retained, and the reason when it is not.
    %
    % The five reason columns are the detector's per-channel flags.
    % There is no per-channel flatline flag: the flatline detector marks samples, and its evidence reaches this table only through the amplitude statistics and the exclusion reason.
    n_raw = numel(channel_ids_raw);
    retained = false(n_raw, 1);
    retained(clean_info.retained_channels) = true;
    labels = repmat({''}, n_raw, 1);
    if ~isempty(channel_labels_raw)
        labels = cellstr(reshape(string(channel_labels_raw), [], 1));
    end
    kurtosis_values = localMetric(artifact_info, {'quality_metrics', 'kurtosis_values'}, n_raw);
    if isstruct(artifact_info) && isfield(artifact_info, 'channel_stats')
        robust_std = reshape([artifact_info.channel_stats.robust_std], [], 1);
    else
        robust_std = nan(n_raw, 1);
    end

    T = table(channel_ids_raw(:), labels, retained, ...
        localExclusionReasons(decision_sidecar_path, n_raw), ...
        kurtosis_values, robust_std, ...
        localFlag(artifact_info, 'flag_low_power', n_raw), ...
        localFlag(artifact_info, 'flag_abnormal_kurt', n_raw), ...
        localFlag(artifact_info, 'flag_isolated', n_raw), ...
        localFlag(artifact_info, 'flag_redundant', n_raw), ...
        localFlag(artifact_info, 'flag_temporal_rate', n_raw), ...
        'VariableNames', {'channel_id', 'label', 'retained', 'exclusion_reason', ...
        'kurtosis', 'robust_std', 'low_power', 'abnormal_kurtosis', 'isolation', ...
        'redundancy', 'temporal_rate'});
end


function values = localMetric(artifact_info, path, n)
    % One nested numeric diagnostic as an n x 1 column, NaN when it is absent.
    values = nan(n, 1);
    node = artifact_info;
    for k = 1:numel(path)
        if ~(isstruct(node) && isfield(node, path{k}))
            return;
        end
        node = node.(path{k});
    end
    if numel(node) == n
        values = reshape(double(node), [], 1);
    end
end


function flags = localFlag(artifact_info, name, n)
    % One per-channel detector flag as an n x 1 logical, false when it is absent.
    flags = false(n, 1);
    if isstruct(artifact_info) && isfield(artifact_info, name) && ...
            numel(artifact_info.(name)) == n
        flags = reshape(logical(artifact_info.(name)), [], 1);
    end
end


function reasons = localExclusionReasons(decision_sidecar_path, n)
    % The cleaner's reason string per removed channel, '' for a retained channel.
    %
    % The table quotes the cleaner's sidecar rather than re-deriving reasons from the flags, so it states exactly what the cleaner decided.
    % The sidecar pairs reasons with flagged_channels, the removed set, and auto_detected_reasons with auto_detected_channels, the wider candidate list.
    % A retained candidate is not an exclusion, so only the removed pairing is read; detected-but-kept conditions stay in the five flag columns.
    reasons = repmat({''}, n, 1);
    if ~isfile(decision_sidecar_path)
        return;
    end
    decision = jsondecode(fileread(decision_sidecar_path));
    if ~(isfield(decision, 'flagged_channels') && isfield(decision, 'reasons'))
        return;
    end
    removed = double(decision.flagged_channels(:)).';
    text = localReasonStrings(decision.reasons);
    if numel(removed) ~= numel(text)
        return;   % a sidecar whose two arrays disagree describes nothing reliably
    end
    for k = 1:numel(removed)
        if removed(k) >= 1 && removed(k) <= n
            reasons{removed(k)} = text{k};
        end
    end
end


function text = localReasonStrings(raw)
    % One reason list as a 1 x k cell of char, whatever shape jsondecode returns.
    % A JSON array of strings decodes to a cell, a single string to a char row, and an empty array to [].
    if ischar(raw)
        text = {raw};
    elseif iscell(raw)
        text = cellfun(@char, reshape(raw, 1, []), 'UniformOutput', false);
    elseif isempty(raw)
        text = cell(1, 0);
    else
        text = cellstr(reshape(string(raw), 1, []));
    end
end


function T = localBandProvenanceTable(rec, settings, fs, filter_params)
    % The one row that says which band this recording carries and whether Phase 1 or the upstream preparation applied it.
    % The three bands are written as space-separated text ("" when there is none), because a CSV cell holds one value and an empty band must stay distinct from a zero one.
    T = table(string(rec.dataset), string(rec.modality), double(fs), ...
        rec.filter_signal, localBandText(rec.bandpass), ...
        localBandText(rec.native_band), localBandText(filter_params.filter_range), ...
        settings.filter_order, string(filter_params.method), ...
        string(filter_params.bandpass_decision.bandpass_mode), ...
        'VariableNames', {'dataset_key', 'modality', 'fs', 'filter_signal', ...
        'bandpass', 'native_band', 'filter_range', 'filter_order', 'method', ...
        'band_selection_mode'});
end


function text = localBandText(band)
    text = string(strtrim(sprintf('%g ', double(band))));
end


function T = localStationarityTable(stationarity_info)
    % One row per sliding window, with the window length and step repeated on each row.
    % Descriptive amplitude QC: no verdict, no threshold, no exclusion.
    if ~isfield(stationarity_info, 'window_times')
        T = table(zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
            zeros(0, 1), zeros(0, 1), 'VariableNames', {'window_index', ...
            'window_center_sec', 'window_var', 'window_mean', 'window_sec', 'step_sec'});
        return;
    end
    n = numel(stationarity_info.window_times);
    T = table((1:n).', stationarity_info.window_times(:), ...
        stationarity_info.window_var(:), stationarity_info.window_mean(:), ...
        repmat(stationarity_info.windowSec, n, 1), ...
        repmat(stationarity_info.stepSec, n, 1), ...
        'VariableNames', {'window_index', 'window_center_sec', 'window_var', ...
        'window_mean', 'window_sec', 'step_sec'});
end


function T = localMarginalTable(marginal_stats, channel_ids, channel_labels)
    % One row per retained channel of the artifact-cleaned, pre-band-pass signal.
    % Ranks remove these marginals before any copula is fitted; the table is QC.
    n = numel(channel_ids);
    if ~isfield(marginal_stats, 'kurtosis')
        T = table(channel_ids(:), repmat({''}, n, 1), nan(n, 1), nan(n, 1), ...
            'VariableNames', {'channel_id', 'label', 'kurtosis', 'skewness'});
        return;
    end
    labels = repmat({''}, n, 1);
    if ~isempty(channel_labels)
        labels = cellstr(reshape(string(channel_labels), [], 1));
    end
    T = table(channel_ids(:), labels, ...
        reshape(marginal_stats(1).kurtosis, [], 1), ...
        reshape(marginal_stats(1).skewness, [], 1), ...
        'VariableNames', {'channel_id', 'label', 'kurtosis', 'skewness'});
end


function localWriteRecordingJson(path, payload)
    % Write the sidecar that discoverPhase1Recordings lists, through a rename so a half-written file is never discovered as a complete recording.
    text = jsonencode(payload, 'PrettyPrint', true);
    tmp = localStagingName(path, '.tmp');
    fid = fopen(tmp, 'w');
    assert(fid ~= -1, 'Phase1:RecordingJsonNotWritable', ...
        'Could not open %s for writing.', tmp);
    fprintf(fid, '%s\n', text);
    fclose(fid);
    [ok, msg] = movefile(tmp, path, 'f');
    assert(ok, 'Phase1:RecordingJsonNotWritable', ...
        'Could not move %s into place: %s', tmp, msg);
    fprintf('Wrote the recording sidecar: %s\n', path);
end


function localReportFigureStatus(status, dataset_key)
    % Report a figure-render failure without failing the phase.
    % renderPhaseFigures records ok, exit_code and n_figures in figures_manifest.json when the driver ran, and writes images/render_figures.sh when MATLAB cannot start a shell.
    if ~(isstruct(status) && isfield(status, 'ok') && status.ok)
        fprintf(2, ['[Phase1] %s figure renderer did not complete; the mat and ' ...
            'the CSVs are unaffected.\n'], dataset_key);
        return;
    end
    if ~(isfield(status, 'n_figures') && status.n_figures > 0)
        fprintf(2, ['[Phase1] %s renderer completed without producing a figure; ' ...
            'the data are unaffected.\n'], dataset_key);
    end
end


function ts = localUtcTimestamp()
    ts = char(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end


function localWriteJsonSidecar(path, payload)
    % Write payload as pretty-printed JSON; a file that cannot be opened is skipped silently, and any other I/O error is reported on stderr, never raised.
    try
        fid = fopen(path, 'w');
        if fid ~= -1
            fwrite(fid, jsonencode(payload, 'PrettyPrint', true));
            fclose(fid);
        end
    catch sidecar_err
        fprintf(2, 'Failed to write %s: %s\n', path, sidecar_err.message);
    end
end
