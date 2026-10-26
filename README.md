# t-copula-information-measures

Code for **[Title of the manuscript]**, [Authors], [Journal], [Year].

- Paper: [link to the published article]
- Preprint: [link to the preprint]
- DOI: [DOI]

The code asks whether the Gaussian copula is an adequate model of the dependence in intracranial recordings when computing information-theoretic measures.
It fits a Student-t copula on ranks as a working model: the correlation matrix comes from Kendall's tau, and the shape p = 1/ν on [0, 3] from a profile likelihood, where p = 0 is the Gaussian copula.
The fitted shapes of each recording are compared with those of Gaussian phase-randomized surrogates of the same recording.
The t-minus-Gaussian corrections to mutual information, total correlation, dual total correlation and O-information then follow in closed form.
A data-free simulation study on a Student-t diffusion measures how well the shape is recovered.

## Repository layout

| Folder     | Contents                                                                                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/` | the seven phase scripts, `Phase1_preprocessing.m` to `Phase7_simulation_study.m`; data preparation for the macaque recordings; Slurm launch scripts in `scripts/hpc/` |
| `src/`     | MATLAB functions, one per file, and the Python figure renderers in `src/visualization/python_figures/`                                                                |
| `tests/`   | `matlab.unittest` suites and the Python reference oracles                                                                                                             |

## Requirements

- MATLAB R2025b with the Statistics and Machine Learning Toolbox, the Signal Processing Toolbox and the Image Processing Toolbox (used by the artifact detection of Phase 1); with the Parallel Computing Toolbox the heavy loops run in parallel.
- Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) for the figures; each figure driver declares its dependencies inline (PEP 723).

## Data

The recordings are not included in this repository.

- Macaque ECoG: eyes-closed and propofol recordings of monkey George from the Neurotycho project (Nagasaka, Shimoda & Fujii (2011), "Multidimensional recording (MDR) and data sharing: an ecological open research and educational platform for neuroscience", _PLoS ONE_ 6, e22561, [doi:10.1371/journal.pone.0022561](https://doi.org/10.1371/journal.pone.0022561)), described with the dataset in Oizumi et al. (2012), "Measuring the level of consciousness based on the amount of integrated information", conference abstract, [doi:10.3389/conf.fnhum.2012.208.00075](https://doi.org/10.3389/conf.fnhum.2012.208.00075).
- Macaque LFP: Brochier et al. (2018), "Massively parallel recordings in macaque motor cortex during an instructed delayed reach-to-grasp task", _Scientific Data_ 5, 180055, [doi:10.1038/sdata.2018.55](https://doi.org/10.1038/sdata.2018.55).
- Human SEEG: the SEEG data were provided by the Academic Center for Epileptology Kempenhaeghe in the Netherlands.

The recordings are declared in the table at the top of `scripts/Phase1_preprocessing.m` and read from `data/`.
`scripts/data_preparation/` holds the scripts that convert the raw macaque recordings into that layout; the header of each script says what it reads and what it writes.

## Running

Run the phases in order from the repository root:

```
matlab -batch "addpath('scripts'); Phase1_preprocessing"
```

Each phase writes to `results/<family>/<subject>/<condition>/PhaseN/` and renders its figures with Python at the end.
Phase 7 needs no data.
Every phase resumes from its checkpoints, so a finished phase exits within seconds when run again.
The settings of each phase sit at the top of its script, together with the environment variables that override them:

| Variable                          | Meaning                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| `TC_DATA_ROOT`, `TC_RESULTS_ROOT` | input and output roots (default `data/` and `results/`)                 |
| `TC_RECORDINGS`                   | comma-separated dataset keys such as `monkey_ECoG/EC1`; empty means all |
| `TC_WORKERS`                      | parallel workers; `0` runs serially                                     |
| `TC_OVERWRITE`                    | `1` recomputes finished work                                            |
| `TC_RENDER_FIGURES`               | `0` skips the figures                                                   |

The cross-recording figures and tables are drawn from the results tree:

```
uv run src/visualization/python_figures/render_cross_recording.py results
```

`scripts/hpc/README.md` describes how to run the phases on a Slurm cluster.

## Tests

```
matlab -batch "addpath(genpath('src')); r = runtests('tests'); disp(table(r))"
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{[key],
  title   = {[Title of the manuscript]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

## License

MIT; see `LICENSE`.
