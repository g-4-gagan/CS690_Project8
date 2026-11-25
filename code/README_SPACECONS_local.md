# SPACECONS: Spatial Trajectory Inference and Consensus

A local-server Jupyter Notebook workflow for running five spatial trajectory inference methods and combining their outputs into consensus pseudotime, trajectory graphs, and milestone results.

**Current configuration:** OSCC2 dataset  
**Notebook:** `SPACECONS_local_updated.ipynb`  
**Execution:** Run locally on a Linux/macOS machine through Jupyter. A CUDA-capable GPU can accelerate parts of the workflow, but the notebook is configured to use CPU for CASCAT by default.

---

## 1. Overview

The notebook runs and compares the following spatial trajectory inference methods:

1. **SpaceFlow** — graph neural network embedding followed by diffusion pseudotime.
2. **SpaTrack** — optimal-transport-based trajectory inference using expression and spatial distance.
3. **StaVia** — spatial smoothing and spatial k-nearest-neighbour graph with random walks.
4. **stLearn PSTS** — spatial trajectory inference using expression and, when available, H&E image morphology.
5. **CASCAT** — causal trajectory inference and cluster-tree construction.

It then combines the method outputs through the SPACECONS workflow:

- **SPACECONS-R:** consensus spot-level pseudotime using weighted rank aggregation.
- **SPACECONS-G:** consensus trajectory graph over shared expression regions.
- **SPACECONS-M:** consensus milestones and milestone graph, while retaining each method's own milestone assignments and graph for comparison.
- **Evaluation:** agreement between methods and, when configured, validation against a reference trajectory.

The notebook also produces plots, tables, intermediate method outputs, and a ZIP archive of the complete results directory.

## 2. Repository / directory layout

Keep the notebook in the project directory and launch Jupyter from that directory (or update `BASE_REL_DIR` in the settings cell).

```text
project/
├── SPACECONS_local_updated.ipynb
├── OSCC2.zip                         # dataset archive (current default)
└── spacecons_work/                   # created automatically by the notebook
    ├── data/                         # downloaded/copied and extracted dataset files
    ├── results_OSCC2/                # final results for the default OSCC2 run
    │   ├── figures/                  # figures captured from notebook displays
    │   ├── fig_*.png                 # method and comparison figures
    │   ├── fig_SPACECONS.png
    │   ├── fig_SPACECONS_milestones.png
    │   ├── spacecons_pseudotime.csv
    │   ├── spacecons_weights.csv
    │   ├── spacecons_graph_edges.csv
    │   ├── spacecons_evaluation.csv
    │   ├── spacecons_summary.json
    │   ├── spacecons_milestones.csv
    │   ├── spacecons_milestone_edges.csv
    │   ├── spacecons_milestone_per_spot.csv
    │   ├── method_milestone_orderings.csv
    │   ├── method_graphs.json
    │   ├── validation_vs_stlearn.csv   # produced when reference validation is enabled
    │   └── ...
    ├── results_OSCC2.zip              # packaged results archive, created at the end
    ├── cascat_run/                    # CASCAT working files and intermediate output
    └── stl_tiles/                     # image tiles used by stLearn when H&E is available
```

The actual output directory is controlled by `WORK_DIR` and `RUN_NAME` in the first settings cell. With the notebook's default values, the final results folder is:

```text
spacecons_work/results_OSCC2/
```

The final compressed archive is:

```text
spacecons_work/results_OSCC2.zip
```

`spacecons_work/` is created relative to the notebook's launch directory by default. The notebook changes its working directory to this folder during setup.

## 3. Dataset download

The dataset archive can be downloaded from Figshare:

**[Download the dataset from Figshare](https://figshare.com/articles/dataset/Data/28342754/3)**

Download the archive and place it in the project directory as:

```text
OSCC2.zip
```

This matches the notebook's current `LOCAL_DATA` setting. The notebook extracts supported ZIP/TAR archives and searches the extracted files for a supported input, such as a 10x Visium directory or an `.h5ad` file.

### Automatic Figshare download

The notebook also contains code to download the files through the Figshare API. To use this instead of a local archive, set `LOCAL_DATA = None` in the settings cell. The notebook uses:

```python
FIGSHARE_ID = 28342754
FIGSHARE_VERSION = 3
```

If the API download is unavailable, download the archive manually from the link above and configure `LOCAL_DATA`.

## 4. Requirements and dependencies

**Dependency installation is incorporated into the notebook.** You do not need to prepare a separate `requirements.txt` for the standard workflow. The setup cells install the main-environment packages and create separate environments for the methods that require incompatible Python/package versions.

### System-level requirements

- Linux or macOS local machine/server.
- Python **3.10–3.12** for the main notebook environment.
- Jupyter Notebook or JupyterLab.
- `pip` and internet access for the initial dependency and model/repository downloads.
- `git` (used to clone the CASCAT repository if it is not already present).
- `uv` is installed by the notebook and is used to create the method-specific environments.
- Sufficient RAM and disk space for spatial transcriptomics data, intermediate matrices, and generated results.
- NVIDIA GPU/CUDA is optional. The notebook's current settings force CASCAT to CPU; other methods may use GPU if configured/supported by their respective packages.

### Main notebook environment

The notebook installs the following package groups when `INSTALL_MAIN = True`:

- Spatial analysis and trajectory packages: `SpaceFlow`, `spaTrack`, `pyVIA`, `pecanpy`, `graphtools`, `phate`, `scprep`, `pygsp`
- Single-cell/spatial analysis: `scanpy`, `anndata`, `igraph`, `leidenalg`
- Scientific computing: `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py`
- Graph and geometry tools: `networkx`, `gudhi`, `pot`, `pygam`
- Visualisation and image processing: `matplotlib`, `cmcrameri`, `plotly`, `datashader`, `scikit-image`, `seaborn`
- Utilities and supporting packages: `requests`, `pyyaml`, `tqdm`, `hnswlib`, `dask`, `gensim`, `umap-learn`, `uv`, and other compatibility/support dependencies.

The notebook also installs PyTorch and PyTorch Geometric-related packages in the main environment.

### Separate method-specific environments

Some methods require versions that conflict with the main environment. The notebook therefore creates these environments automatically under the configured `ENV_DIR`:

| Environment | Python | Main packages / purpose |
|---|---:|---|
| `stl` | 3.11 | `stlearn==1.2.2`, with `setuptools<81` and NumPy below 2 as required by stLearn |
| `cascat` | 3.10 | CASCAT environment with `torch==2.1.1`, `torch_geometric==2.6.1`, compatible PyG extensions, Scanpy and pinned scientific dependencies |

The CASCAT source repository is cloned automatically when it is not already present.

> **Important path note:** In the current notebook, `ENV_DIR` is set to `/home/user/Gagan_env/genomicsProject/envs`. If you are running this notebook on a different server or under a different account, update this path in the settings cell to a directory where you have write permission. The notebook uses this directory for the stLearn and CASCAT environments and the CASCAT checkout.

### Installation behaviour

- Keep `INSTALL_MAIN = True` for the first run so the notebook installs the main dependencies.
- The notebook checks for the separate stLearn and CASCAT Python executables and creates those environments if they do not already exist.
- Once the main environment is prepared, `INSTALL_MAIN` can be set to `False` for subsequent runs to skip the main package installation step.
- Initial setup requires internet access. Subsequent runs can reuse the environments and downloaded data already present on the server.

## 5. Running the notebook on the local server

1. Place `SPACECONS_local_updated.ipynb` and `OSCC2.zip` in your project directory.
2. Open a terminal and activate the Python 3.10–3.12 environment that will run Jupyter.
3. Install/launch Jupyter if it is not already available in that environment.
4. Change directory to the project directory.
5. Start Jupyter Notebook or JupyterLab.
6. Open `SPACECONS_local_updated.ipynb`.
7. Review the settings cell (Cell 1) and update paths or dataset selection if required.
8. Run the notebook **from the first cell to the last cell in order**. Setup, data discovery, preprocessing, and method-specific steps depend on variables and files created by earlier cells.
9. At the end, check the printed results directory and the generated ZIP archive.

Example launch commands:

```bash
cd /path/to/project
jupyter lab
```

or:

```bash
cd /path/to/project
jupyter notebook
```

Do not run the notebook cells out of order on a fresh run. Some method steps may be skipped if their corresponding output files already exist in the results folder; see the rerun note below.

## 6. Changing the dataset or OSCC variant

The current notebook is configured to run on **OSCC2**. To run another OSCC variant or another compatible spatial-transcriptomics sample, update the relevant variables in the first settings cell. You should not need to rewrite the downstream analysis cells when the input follows the supported format.

### Variables to review

| Variable | Current/default value | When to change it |
|---|---|---|
| `RUN_NAME` | `'OSCC2'` | Set a distinct name for the variant/sample. This determines the output folder name, `results_<RUN_NAME>`. |
| `LOCAL_DATA` | `from_base('OSCC2.zip')` | Set this to the path of the other dataset ZIP/TAR archive, extracted folder, or `.h5ad` file. Set to `None` to use the notebook's Figshare download path. |
| `INPUT_PATH` | `None` | Optionally point directly to one Visium folder or one `.h5ad` file when you want to force the input selection. |
| `SAMPLE_HINT` | `None` | If the downloaded/extracted dataset contains multiple samples, set a distinctive part of the desired sample's path, e.g. `'OSCC1'` or another variant identifier. |
| `SAMPLE_COLUMN` | `None` | For a single `.h5ad` containing multiple samples, specify the `.obs` column used to identify samples. |
| `SAMPLE_VALUE` | `None` | Specify the value in `SAMPLE_COLUMN` corresponding to the sample to retain. |
| `COORD_COLS` | `None` | For an `.h5ad` without `obsm['spatial']` and without a sibling Visium `spatial/` folder, provide the coordinate column names, e.g. `('x', 'y')`. |
| `COUNTS_LAYER` | `None` | If `.X` does not contain raw counts, set this to the layer containing raw counts. |
| `VISIUM_COUNT_FILE` | `'filtered_feature_bc_matrix.h5'` | Change only if the Visium count matrix file has a different name. |
| `SPECIES` | `'Human'` | Set to `'Mouse'` for mouse data; this value is passed to CASCAT. |

For a different OSCC variant, a typical settings update would look like:

```python
RUN_NAME = 'OSCC1'
LOCAL_DATA = from_base('OSCC1.zip')
SAMPLE_HINT = 'OSCC1'
```

Use the actual archive name and sample identifier present in your downloaded dataset. If the archive contains only one sample and the notebook detects it unambiguously, `SAMPLE_HINT` may remain `None`.

### Input data formats

The notebook supports:

- **10x Visium directory:** a directory containing `filtered_feature_bc_matrix.h5` (or the configured `VISIUM_COUNT_FILE`) and a sibling `spatial/` directory.
- **AnnData `.h5ad`:** should contain expression data and spatial coordinates in `obsm['spatial']`, or have a sibling `spatial/` directory from which coordinates/image information can be loaded. If coordinates are stored in ordinary `.obs` columns, configure `COORD_COLS`.

When available, the Visium `spatial/` directory can provide spot positions, scalefactors, and H&E tissue images. If no H&E image is available, the notebook can continue with spatial coordinates; the stLearn workflow then uses physical-distance information rather than image morphology.

## 7. Important analysis settings

These settings also appear in the first settings cell. Change them only when appropriate for the dataset or analysis design.

| Variable | Default | Description |
|---|---:|---|
| `MIN_GENES` | `200` | QC threshold: remove spots with fewer detected genes. |
| `MIN_CELLS` | `3` | QC threshold: remove genes detected in fewer spots. |
| `LEIDEN_RES` | `0.8` | Resolution for shared expression-region clustering. |
| `REF_MODE` | `'none'` | Reference trajectory mode: `'none'`, `'annotation'`, or `'markers'`. |
| `ROOT_SPOT` | `None` | Optional exact barcode of the root spot. |
| `ROOT_REGION` | `None` | Optional expression-region label used to select the root when `REF_MODE='none'`. |
| `REF_STATES` | `['early_state', 'late_state']` | Ordered reference states, earliest to latest, for validation. |
| `STATE_MAP` | example labels | Maps annotation labels in the dataset to the states in `REF_STATES`. |
| `MARKERS` | `{}` | Marker genes for each ordered state when `REF_MODE='markers'`. |
| `N_THREADS` | up to `8` | Number of CPU threads used by applicable methods. |
| `CASCAT_TRIALS`, `CASCAT_EPOCHS` | `2`, `50` | CASCAT training settings. The authors' settings are noted in the notebook as `5`, `100`. |
| `CASCAT_KDE` | `0` | `0` uses the faster binned CMI estimator; `1` selects the slower kernel-density estimator. |
| `CASCAT_MAX_MIN` | `60` | Time limit in minutes before the CASCAT run is stopped/skipped. |
| `STL_MAX_MIN` | `120` | Time limit in minutes for the stLearn run. |

### Reference trajectory and validation

`REF_MODE='none'` means no reference trajectory is supplied, so the reference-based validation section is skipped. The root is selected from `ROOT_SPOT`, `ROOT_REGION`, or the notebook's fallback root-selection procedure.

For annotation-based validation:

- Set `REF_MODE = 'annotation'`.
- Set `ANNOT_COLUMN` to the annotation column in `.obs`, or provide `ANNOT_FILE`.
- Define `REF_STATES` in the intended earliest-to-latest order.
- Configure `STATE_MAP` to map the labels found in the data to those ordered states.

For marker-based validation:

- Set `REF_MODE = 'markers'`.
- Define `MARKERS` as an ordered mapping from state names to marker-gene lists.
- Ensure that at least two states are defined and that the marker genes are present in the dataset.

The notebook describes the reference trajectory as being used for root selection and/or validation, not as an input for constructing the method pseudotime outputs.

## 8. Results and output files

All outputs for a run are saved under:

```text
spacecons_work/results_<RUN_NAME>/
```

For the current OSCC2 configuration:

```text
spacecons_work/results_OSCC2/
```

Important outputs include:

| Output | Description |
|---|---|
| `spacecons_pseudotime.csv` | SPACECONS-R consensus pseudotime for individual spots. |
| `spacecons_weights.csv` | Method agreement statistics and consensus weights. |
| `spacecons_graph_edges.csv` | Edges of the SPACECONS-G consensus region graph. |
| `spacecons_evaluation.csv` | Evaluation/comparison of consensus and method orderings. |
| `spacecons_summary.json` | Consensus summary, including milestone ordering and topology-disagreement information. |
| `fig_SPACECONS.png` | SPACECONS consensus pseudotime/trajectory visualisation. |
| `spacecons_milestones.csv` | Consensus milestone assignment information. |
| `spacecons_milestone_edges.csv` | Edges in the consensus milestone graph. |
| `spacecons_milestone_per_spot.csv` | Per-spot consensus milestone and pseudotime. |
| `method_milestone_orderings.csv` | Each method's own milestone ordering. |
| `method_graphs.json` | Saved method-specific graph edges. |
| `fig_SPACECONS_milestones.png` | Consensus milestone graph visualisation. |
| `fig_<method>.png` | Per-method visualisation, where generated. |
| `figures/` | PNG copies of figures displayed by the notebook. |
| `validation_vs_stlearn.csv` | Reference-validation metrics, when reference validation is enabled. |
| `validation_state_means.csv` | Mean pseudotime by reference state, when validation is enabled. |

The notebook also saves method-level intermediate files (for example, raw pseudotime CSVs, embeddings, and CASCAT/stLearn intermediate data) in the same results folder or in the relevant work directory.

At the end of the notebook, the results folder is compressed into:

```text
spacecons_work/results_<RUN_NAME>.zip
```

For the default run, this is `spacecons_work/results_OSCC2.zip`. The final notebook cell displays a link to this ZIP inside Jupyter and prints the results folder and available CSV files.

## 9. Re-running with a different sample

To avoid mixing outputs from different variants:

1. Change `RUN_NAME` to a new, descriptive value (for example, `OSCC1`).
2. Update `LOCAL_DATA` and, if needed, `INPUT_PATH` / `SAMPLE_HINT` / sample-selection variables.
3. Confirm the data format and coordinate availability.
4. Run the notebook from the beginning.
5. Verify that the printed `results` path points to `results_<new RUN_NAME>`.

Use a new `RUN_NAME` for each sample/variant so the output directories remain separate. If you intentionally reuse an existing run name, be aware that some method steps check whether intermediate result files already exist and may reuse them rather than recomputing them. For a clean rerun, back up or remove that run's old results directory and relevant method working outputs before starting.

## 10. Notes and troubleshooting

- **Wrong dataset selected:** set `INPUT_PATH` to the exact Visium directory or `.h5ad` file, or set `SAMPLE_HINT` to a unique part of the desired sample path.
- **Multiple samples in one `.h5ad`:** use `SAMPLE_COLUMN` and `SAMPLE_VALUE`.
- **Missing spatial coordinates:** ensure `obsm['spatial']` is available, provide `COORD_COLS`, or place the corresponding Visium `spatial/` folder alongside the input.
- **No H&E image:** the workflow can still run using spatial coordinates. stLearn will report that morphology is unavailable and use physical-distance information.
- **Environment creation fails:** check internet access, `git`, Python availability, write permissions for `ENV_DIR`, and the selected PyTorch/CUDA wheel configuration.
- **Different server/account:** update the hard-coded `ENV_DIR` path in the settings cell before running.
- **Long-running method:** stLearn and CASCAT have configurable time limits (`STL_MAX_MIN` and `CASCAT_MAX_MIN`). The notebook is designed to continue the comparison if one of these methods does not finish; the consensus uses the methods that produced usable outputs.
- **GPU/CUDA:** CASCAT is explicitly set to CPU in the current settings cell (`CASCAT_DEVICE = 'cpu'`). If you intend to use a GPU, review the CUDA configuration and the later device assignment in that cell before running.
- **Partial method results:** inspect the notebook output for messages indicating that a method was skipped. Consensus results are based on the methods that successfully generated pseudotime outputs.

---

## 11. Reproducibility checklist

Before starting a run, verify:

- [ ] The notebook is being run in the intended local Python/Jupyter environment.
- [ ] The correct dataset archive or input path is configured.
- [ ] `RUN_NAME` uniquely identifies the sample/variant.
- [ ] `ENV_DIR` points to a writable location on this server.
- [ ] The input contains expression data and valid spatial coordinates.
- [ ] The root spot/region is appropriate, or the automatic root selection is acceptable.
- [ ] Reference settings are configured if reference validation is required.
- [ ] The notebook is executed from top to bottom.
- [ ] The final `results_<RUN_NAME>/` directory and ZIP archive are checked after execution.

## 12. Dataset citation / source

The dataset used by the current notebook configuration is available from Figshare:

https://figshare.com/articles/dataset/Data/28342754/3

Please refer to the Figshare record for the dataset's associated metadata and citation details.
