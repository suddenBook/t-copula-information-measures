# Running the t-copula pipeline on a Slurm cluster

The launch scripts are plain rsync plus sbatch.
Nothing on the cluster checks or gates a run: the code is validated before it is deployed, and the cluster only computes.
The only thing that can refuse a submission is `sbatch` itself, and the only thing that sets a job's exit status is MATLAB.
Every phase resumes from its checkpoints, so a job the wall clock cuts is simply followed by another one, and a phase with nothing left to do exits in seconds.

## Site settings

The scripts name no cluster.
Copy `site.env.example` to `site.env` next to it and fill in the values of your own cluster; `site.env` is untracked, and `deploy.sh` carries it to the cluster with the rest of `scripts/`, which is how `submit.sh` and `run_phase.sh` see the values there.

| key                                                          | what it sets                                                                 |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `TC_HPC_HOST`                                                | the ssh alias of the login node, used by `deploy.sh` and `pull.sh`           |
| `TC_HPC_ACCOUNT`, `TC_HPC_PARTITION`, `TC_HPC_QOS`           | the `sbatch` account, partition and QOS                                      |
| `TC_HPC_MATLAB_MODULE`                                       | what `run_phase.sh` gives to `module load`                                   |
| `TC_HPC_BASE`                                                | the remote base directory, `~/tcopula` by default                            |
| `TC_HPC_CPUS`, `TC_HPC_WORKERS`, `TC_HPC_MEM`, `TC_HPC_TIME` | cores and memory per task, MATLAB workers inside the job, and the wall limit |
| `TC_HPC_MAX_JOBS`                                            | how many jobs the QOS runs at once; informational, nothing enforces it       |
| `TC_HPC_UV_PYTHON`                                           | the uv-managed Python of the figure renderers, 3.12 by default               |
| `TC_HPC_MODULE_PREREQS`                                      | optional modules that `run_phase.sh` loads before the MATLAB module          |
| `TC_HPC_SBATCH_EXTRA`                                        | optional extra `sbatch` options, such as `--hint=nomultithread`              |

A `TC_HPC_*` already exported wins over `site.env`, and a command-line option (`--workers`, `--mem`, `--time`, the host argument) wins over both.
`submit.sh` stops with a message naming `site.env.example` when the account, the partition, the QOS or the module is missing; `deploy.sh` and `pull.sh` do the same for the host.

## Connecting

`TC_HPC_HOST` is an alias in the operator's own `~/.ssh/config`: the host name, the user name and any VPN or jump host live there, not in this repository.
MATLAB runs only inside a job, after `module load "$TC_HPC_MATLAB_MODULE"`, never on the login node.
The login node is for `rsync`, `sbatch`, `squeue`, `tail` and the environment build that `deploy.sh` starts there, nothing else.

## Layout on the cluster

Below `TC_HPC_BASE`, `~/tcopula` by default:

| path                | holds                                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `~/tcopula/code`    | the rsynced source tree (`src/`, `scripts/`, `tests/`, `README.md`, `LICENSE`, `.gitignore`) and a `GIT_SHA` file; no `.git`, no `results/`, no `data/` |
| `~/tcopula/data`    | the raw inputs, `TC_DATA_ROOT`                                                                                                                          |
| `~/tcopula/results` | the output tree, `TC_RESULTS_ROOT`, on shared storage because the later jobs of a chain and the finalizers must see the same checkpoints                |
| `~/tcopula/logs`    | Slurm stdout and stderr, plus one `<phase>-<timestamp>.jobs` file per submission                                                                        |

Check your home quota before a production run: the output tree of a full run needs about 110 GB, and the Phase 1 inputs another 2.35 GB.
Node-local `/scratch` holds only what is per job and transient: MATLAB's preferences, the MCR cache, matplotlib's cache and the parallel pool's `JobStorageLocation`.
`run_phase.sh` creates that directory as `/scratch/$USER/<job>.<task>` (under `TMPDIR` when `/scratch` cannot be created) and removes it when the job ends.
`uv` is expected at `~/.local/bin/uv`, which `run_phase.sh` puts on `PATH` for the Python figure renderers.

Export `TC_HPC_BASE` to move the whole layout for one call of any of the scripts; `submit.sh` and `run_phase.sh` also accept `TC_BASE`, which takes precedence there.

## The files here

- `site.env.example` — the site values, as a template; copy to `site.env` (untracked) and edit.
- `prewarm_figures.sh` — on the login node, run by `deploy.sh` after each code transfer to a remote host (not for a local rehearsal path or a dry run); installs the uv-managed interpreter of the renderers and builds every renderer's environment once, so jobs (which run uv with `UV_NO_SYNC=1 UV_OFFLINE=1`) only read them and never race on the shared cache.
- `deploy.sh [--data|--data-all] [--dry-run] [host]` — local; sends the source tree, and with `--data` the Phase 1 inputs of the ten recordings as well.
- `submit.sh <phase> [options]` — on the login node; queues one phase and prints the job ids.
- `pull.sh [host] [all|phaseN]` — local; copies the results and the logs back, and never deletes a local file.
- `phase.sbatch` — the job file; turns the Slurm array index into `TC_RECORDING_INDEX` and `TC_SHARD`, and carries no site value.
- `run_phase.sh <phase>` — what the job runs; loads the modules, sets the roots, prepares the node-local scratch and launches MATLAB.

## Deploying

```bash
scripts/hpc/deploy.sh                     # source only, every time you change the code
scripts/hpc/deploy.sh --data              # source plus the ten recordings' inputs, the first time
scripts/hpc/deploy.sh --dry-run           # list what would be sent and change nothing
```

`--delete` applies to `~/tcopula/code` only, so a file deleted from the local tree disappears there too; no other transfer of these scripts deletes anything.
The remote tree carries no `.git`, so `deploy.sh` writes the local HEAD to `code/GIT_SHA`, which every `run_config.json` records, and prints it.

`deploy.sh` creates the four directories of the layout (`code`, `data`, `results`, `logs`) under `TC_HPC_BASE` before it transfers anything, with one `ssh <host> "mkdir -p ..."` for a remote host and a plain `mkdir -p` for a local rehearsal path, so the first deployment into an empty home works with no preparation.
`--dry-run` prints the `mkdir` it would run and creates nothing.

**What `--data` sends.**
`--data` sends `data/<family>/preprocessed/` for the three families the ten recordings come from, with those paths kept under `~/tcopula/data/`: about 2.35 GB.
That is exactly what Phase 1 reads; the families' other folders, such as `raw/`, belong to data preparation and stay local.
The family list is at the top of `deploy.sh`; it follows the recording table at the top of `Phase1_preprocessing.m`.

`--data-all` sends the whole of `data/` instead, raw archives included.
That is about 102 GB, which on a home quota of a few hundred GB leaves little room for the results, so it is for archive work only.

## A full production run

```bash
scripts/hpc/deploy.sh --data                      # local, once
ssh <your login node>                             # the alias in your ~/.ssh/config
cd ~/tcopula/code/scripts/hpc

./submit.sh phase1 --chain 1                      # array 1-10, one recording per task
./submit.sh phase2 --chain 2 --after <phase1 array id>
./submit.sh phase3 --chain 1 --after <phase2 last array id>
./submit.sh phase4 --after <phase3 finalize id>
./submit.sh phase5 --after <phase2 last array id>
./submit.sh phase6 --after <phase1 array id>      # needs only Phase 1; minutes per recording
./submit.sh phase7 --class heavy --shards 3 ...   # data-free: the heavy memory class, see below
./submit.sh phase7 --class light --shards 4 ...   # the rest, on nodes of their own
./submit.sh phase7 --finalize only --after <heavy array id>:<light array id>
```

Each call prints the job ids it submitted, the last array id and, for phases 3 and 7, the finalize id; the same ids land in `~/tcopula/logs/<phase>-<timestamp>.jobs`.
`--after` takes one id or several separated by colons, and is always an `afterany` dependency: the next phase starts whatever the exit status of the previous one, because a cut job is a normal event and the resuming job sorts it out.
A QOS usually caps how many jobs run at once (`TC_HPC_MAX_JOBS` records the cap); everything else waits in the queue, oldest first.

Shapes and defaults, which `--recordings`, `--shards`, `--workers`, `--mem` and `--time` override:

| phase                                  | array                                         | per task                                                                                                     |
| -------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| phase1, phase2, phase4, phase5, phase6 | `1-10`, one recording per task                | `TC_RECORDING_INDEX=<task>`                                                                                  |
| phase3                                 | `1-20`, recording x surrogate realization     | `TC_RECORDING_INDEX=(t-1)/2+1`, `TC_SHARD=(t-1)%2+1 of 2`, then a finalize array `1-10` with `TC_FINALIZE=1` |
| phase7                                 | `1-S`, one shard of its memory class per task | `TC_SHARD=<task>/S` within `TC_SIM_CLASS`, then one finalize job with `TC_FINALIZE=1`                        |

Phase 6 is serial and light: `TC_HPC_CPUS=2 ./submit.sh phase6 --mem 16G --time 0-01:00:00 --after <phase1 array id>` is ample.
Phase 7 splits its jobs into two memory classes, because a recovery path of the longest window at the densest rate holds several times the memory of the rest: `--class heavy` computes the jobs with more than `heavy_n_obs` observations per path (the Phase 7 settings), `--class light` everything else, including the simulator check and the factorial, and each class is sharded on its own, `TC_SHARD=k/S` within the class.
A class run never publishes, so `--class` implies `--finalize no`, and the one finalize job is queued after both classes with `--finalize only --after <heavy>:<light>`; it merges the whole study, millions of replicate rows, so give it a whole node's memory.
The phase gives a shard its recovery jobs heaviest first, by observations per path, so any shard count balances the observations a shard fits.
Inside a job the replicates run in `parfor` with a per-job worker ceiling: every worker of the pool is charged its baseline, `worker_gb(1)`, and each active one its path, `worker_gb(2)` per million observations, beside the client's `client_gb`, against the job's own allocation (`SLURM_MEM_PER_NODE`); a pool whose baseline alone does not fit stops before it opens (`Phase7:PoolExceedsMemory`), and a job of which not even one path fits stops before it starts (`Phase7:JobExceedsMemory`), since either would only be killed, and killed again on every resume.
So size a class's pool against the node it goes to: a large-memory node for the heavy class, and for the light class a pool whose baseline leaves room for the light class's longest paths.

Every task asks for one node, one task, and the CPUs, memory and wall from `site.env` (64 logical CPUs, 240 GB and a 7-day wall as shipped), and runs MATLAB with `TC_WORKERS` (63) single-threaded workers.
The CPU count is logical, so hyperthreads count: a node with 32 physical cores and two threads per core takes all 64 and 63 workers, and the job prelude raises the pool profile's worker limit to `TC_WORKERS` (MATLAB's own default stops at the physical cores).
Workers are the lever for wall time; memory is the limit: each worker holds its own copy of the recording's rank matrix and of the quantile cache beside its own MATLAB.
Phase 2 at 63 workers needs about 2.2 GB per worker on the SEEG recordings and 1.3 GB on the ECoG and LFP ones, and Phase 3 about 1.6 GB on the ECoG and LFP ones, so size `TC_HPC_WORKERS` against the node's memory (about 64 workers on a 124 GB node for the ECoG and LFP recordings, a node of at least 240 GB for the SEEG ones) and set `TC_HPC_SBATCH_EXTRA=--hint=nomultithread` with `TC_HPC_CPUS` equal to the physical cores if you prefer one worker per physical core.
`--tasks` submits only the listed tasks of an array, so with `TC_HPC_PARTITION` set per call the parts of one array can go to different partitions (one call per partition, the same shard count); it also takes sbatch's throttle (`1-8%4` runs at most four at once).
A whole-node request also means the QOS job cap is the only other limit on throughput.

## Watching a run

```bash
squeue -u $USER -o "%.10i %.28j %.2t %.11M %.11l %R"
tail -n 40 ~/tcopula/logs/tc-phase3-<jobid>_<task>.out
ls ~/tcopula/results/*/*/*/Phase2/mat/checkpoints | tail
```

Array logs are `~/tcopula/logs/<name>-<arrayid>_<task>.out` and the single phase-7 finalize log is `<name>-<jobid>.out`.
A Phase 3 task works through one surrogate realization of one recording: every Phase 2 channel set of every dimension, checkpointed every 1000 sets.
Each job's first line names the phase, the host, the recording, the shard and the worker count; the phase's own diary is in `text/log.txt` under its output directory.

## Resuming after a wall cut

Submit the phase again.
Every phase skips what is already on disk, so the resubmitted array picks up exactly the missing checkpoints and finishes the rest.
`--chain N` does this in advance: it queues the same array N times with `afterany` dependencies, so a cut task is followed by one that resumes it without anybody watching.
The finalize job of phases 3 and 7 depends on the last array of the chain, and it only merges checkpoints and writes the summaries.
To recompute rather than resume, export `TC_OVERWRITE=1` before calling `submit.sh`.

`TC_OVERWRITE` reaches the **first array of the chain only**.
Every successor array and the finalize job are exported `TC_OVERWRITE=0`, so the chain resumes past a wall cut.
That is the point of a chain: a successor that recomputed from the beginning would repeat the first allocation of work for ever and never reach the rest, and a shard needing more than one wall allocation would never finish.
The finalize job only merges checkpoints, so it has nothing to recompute either.
Check it before a production submit with `TC_OVERWRITE=1 ./submit.sh phase3 --chain 2 --dry-run`: `TC_OVERWRITE=1` appears in the first array's `--export` and `TC_OVERWRITE=0` in every later one.

## Pulling the results

```bash
scripts/hpc/pull.sh                     # everything, into results/
scripts/hpc/pull.sh <host> phase3       # only the Phase3 subtrees, plus recording.json, and the Phase 3 logs
```

`pull.sh` uses `--checksum --partial --info=progress2`, so every transferred byte is verified and an interrupted transfer resumes.
It never deletes a local file, and it is safe to run while the jobs are still computing.
Each Slurm log is filed beside the outputs it records, in `results/<family>/<subject>/<condition>/PhaseN/text/cluster_logs/` next to the phase's diary, and Phase 7's in `results/simulation/t_copula/all/Phase7/text/cluster_logs/`.
The folder is the one the log names on its `[PhaseN] <folder>` line, and only a folder whose `PhaseN` output exists locally counts.
A log that names none, such as a resume job that found its work already done, stays on the cluster and is counted in the closing message; the submission stamps (`*.jobs`) stay there too.

## Cost

The figures are approximate guidance.
They are measured where the table says so and otherwise projected from the single-threaded speed of a desktop workstation; a cluster worker (one of 63 on 32 hyperthreaded cores) runs about 2.5 times slower than that.
Such projections can fall well short: Phase 2 projects to 20 to 100 node-hours and measures about 262.

| phase   | cost                                                                                     | note                                                                                                                                                                                                                                                                                                                              |
| ------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 2 | about 262 node-hours (measured)                                                          | 10 array tasks on 64-CPU nodes with 63 workers; the SEEG recordings 48-58 h each, the ECoG and LFP ones 2-14 h                                                                                                                                                                                                                    |
| Phase 3 | about 46 node-hours per realization at workstation speed, about 93 in all                | 20 tasks; the longest, a SEEG realization, about 10.6 h at workstation speed                                                                                                                                                                                                                                                      |
| Phase 4 | 6 to 33 min per recording on a 64-CPU node (measured)                                    |                                                                                                                                                                                                                                                                                                                                   |
| Phase 6 | minutes per recording                                                                    | 10 serial tasks                                                                                                                                                                                                                                                                                                                   |
| Phase 7 | recovery about 26,000 core-hours at workstation speed, about 66,000 cluster worker-hours | measured on a workstation pool: a replicate of 1.2e6 observations takes about 820 s and peaks at 2.85 GiB per worker, one of 4.8e6 about 3,800 s (4,900 at the Gaussian truth) and 5.94 GiB; the heavy class (paths above 1.2e6 observations) is about 17,000 of the core-hours and fits at full width only on large-memory nodes |

Phases 1 and 5 are short and have no separate estimate.
