# README — `SPACECONS_v3_1.ipynb` (CS690A Project 8)

**A consensus method for spatial trajectory inference from five methods, tested on annotated benchmark tissues**

Team: Madhavi Singh (261110035), Gagan Kumar Soni (252040606), Nikita Pandit (261110078), Viraat Goyal (251187)
Instructor: Dr. Hamim Zafar, Department of Computer Science and Engineering, IIT Kanpur

This notebook runs the whole pipeline on Kaggle. It does five things:

1. Downloads the four datasets.
2. Runs five spatial trajectory methods from one shared root.
3. Combines them with our consensus method.
4. Validates every result against a reference trajectory.
5. Saves all tables and figures.

It is the version (v3.1) whose run produced every number in the report and the slides.

---

## 1. Names used in the code vs. the report

The notebook keeps the project's working name **SPACECONS**; the report and slides call it **the consensus method**.

| In the notebook / output files | In the report and slides | What it is |
|---|---|---|
| `SPACECONS` (also `SPACECONS (weighted)`) | **Consensus-R** | Weighted rank consensus (pseudotime + 95 % interval) |
| `SPACECONS-M` | **Consensus-M** | Consensus milestones + trajectory graph |
| `SPACECONS-3` (also `SPACECONS-3 (final)`) | **Consensus-3** | Role-based consensus with middle fusion (final ordering) |
| `OT` (in `validation_vs_stlearn.csv`) | **GW** | Gromov–Wasserstein discrepancy between graphs (CASCAT's metric) |
| `validation_vs_stlearn.csv` | Validation tables | Validation against the reference trajectory (for every dataset, not only stLearn's) |

---

## 2. Quick start (about 2–3 hours on Kaggle)

1. Go to kaggle.com → **Create → New Notebook → File → Import Notebook** → upload `SPACECONS_v3_1.ipynb`.
2. In the right panel, **Settings**:
   - **Accelerator:** GPU T4 x1
   - **Internet:** On (needed to install packages and download data)
3. **Optional, but recommended:** add the CASCAT benchmark files as input (see §6, figshare 403).
4. Click **Run All**. Do not run cells one by one the first time: the dataset loop (cell 36) needs every tagged cell to have run once in the session.
5. When it finishes, download the results:
   - Run the last code cell, which zips each `v3_<DATASET>` folder, or
   - Use the **Output** panel (`/kaggle/working/`), or
   - **Save Version → Save & Run All (Commit)** to keep a permanent copy.

> **Tip:** Kaggle's screen sometimes stops refreshing during long cells (especially CASCAT). The run is usually still going: check the cell's elapsed time or the Output panel before stopping anything.

---

## 3. What you need

| Item | Detail |
|---|---|
| Platform | Kaggle notebook, **GPU T4 x1**, **Internet On** |
| Main Python | Kaggle's own Python: SpaceFlow, SpaTrack, StaVia, scanpy, scikit-learn, SciPy, NetworkX, POT |
| stLearn environment | Separate Python 3.11 + NumPy < 2, made with `uv` (stLearn does not install on newer Python) |
| CASCAT environment | Separate Python 3.10 + PyTorch 2.1.1 + PyG 2.6.1, the authors' pinned versions, made with `uv` |
| First-run setup | About 10 minutes for the two environments; later runs in the same session reuse them |
| Disk | About 3–4 GB for data, environments and outputs |

"Dependency conflict" messages during installation are expected and harmless.

---

## 4. Datasets

The datasets to run are chosen in the download cell (cell 5):

```python
DATASETS = ['TBI', 'BC', 'ZHUANG', 'HER2ST']   # development first, then the held-out test
```

| Dataset | Tissue / technology | Spots or cells | Source (downloaded automatically) | Root state | Reference trajectory | Role |
|---|---|---|---|---|---|---|
| **TBI** | Mouse brain 3 days after injury, 10x Visium | 2,440 | NCBI GEO GSE236171, sample GSM7519182 | Hypothalamus | hypothalamus → thalamus → hippocampus → injury (marker-gene states) | Development |
| **BC** | Human breast cancer, Visium Block A Section 1 | 3,798 | 10x Genomics `V1_Breast_Cancer_Block_A_Section_1` + SEDR pathology annotation | DCIS | DCIS → IDC | Development |
| **ZHUANG** | Mouse brain, MERFISH (Zhuang-ABCA-2.005) | 3,161 cells | CASCAT figshare article 28342754 (files 52119332, 52119302) | Supertype 0169 | 0169 → 0167 → 0153 → 0151 → {0150, 0152} | Held-out test |
| **HER2ST** | HER2+ breast cancer, legacy ST | 607 | CASCAT figshare article 28342754 (file 52119398) | Normal breast glands | glands → {immune → connective; in situ → invasive} | Held-out test |

All settings were fixed on TBI and BC **before** ZHUANG and HER2ST were run (held-out test). Reference labels are used only to choose the root state and for validation; **never** to build an ordering, milestones or a graph.

---

## 5. Cell-by-cell guide

Cells marked with a tag (`# [CELL:<tag>]` on the first line) are re-run automatically for every dataset by the dataset loop (cell 36).

| Cell | Tag | What it does | Main outputs |
|---|---|---|---|
| 0 | – | Title and overview | – |
| 1 | – | Installs SpaceFlow, SpaTrack, StaVia and helpers into Kaggle's Python | – |
| 2–3 | – | Builds the separate **stLearn** (Py 3.11) and **CASCAT** (Py 3.10) environments with `uv` | Environments under `/kaggle/working` |
| 4–5 | `download` | Chooses `DATASETS`; downloads TBI (GEO tar), BC (10x + SEDR labels), ZHUANG/HER2ST (figshare API, files checked by size, cell count and labels) | `data/` |
| 6–7 | `load` | Loads the data. QC (Visium: drop spots with < 200 genes); genes in ≥ 3 spots; normalise to 10⁴ and log; 2,000 variable genes; 30 PCs; Leiden regions (resolution 0.8). Picks the label column automatically | `ad` (AnnData), `OUT = v3_<DATASET>/` |
| 8 | – | Saves every figure the notebook shows as a PNG | `v3_<DATASET>/figures/` |
| 9–10 | `root` | **Shared root:** root state from the stated biology → root region = Leiden region with the largest share of root-state spots → root spot = most immature-looking spot (CytoTRACE-like score, stLearn's rule). Also saves 8 spread-out candidate roots and the reference time per spot | `ROOT_BARCODE`, `multi_root_candidates.csv` |
| 11–12 | `spaceflow` | **SpaceFlow:** graph neural network with spatial regularisation 0.1 → 50-D embedding → diffusion pseudotime from the root | `t_spaceflow` |
| 13–14 | `spatrack` | **SpaTrack:** optimal transport, cost = expression + spatial distance (equal weights); 20 start spots nearest the root | `t_spatrack` |
| 15–16 | `stavia` | **StaVia:** spatial smoothing 0.3, spatial kNN 15, memory 5; root = shared root spot | `t_stavia` |
| 17–19 | `stlearn-run` | **stLearn PSTS** in its own environment: stSME with H&E tiles (when an image exists) → Louvain → PSTS from the root spot. Writes and runs `run_stlearn_psts.py` | `raw_stlearn.csv`, `t_stlearn` |
| 20–22 | `cascat-run` | **CASCAT** in its own environment with the authors' YAML settings for the data type; 2 trials × 50 epochs; binned CMI estimator; 60-minute limit; one work folder per dataset. Writes and runs `run_cascat.py` | `raw_cascat.csv`, `cascat_result.h5ad`, `t_cascat` |
| 23–24 | `comparison` | Checks that each method starts in the root region; Spearman agreement between the methods | `comparison_check.csv`, `comparison_agreement.csv`, `all_methods_pseudotime.csv` |
| 25–26 | `plots` | Per-method figure: (a) pseudotime map with the root ★, (b) region graph (PAGA) directed by the method's pseudotime, (c) pseudotime per region | `fig_<method>.png` |
| 27–28 | `consensus` | **Consensus-R:** ranks 0–1, orientation check (flip if the root region is late), LOO² weights, weighted Borda; Kemeny/MC4 and other weightings as checks; RRA p-values per spot (extra); 500 method bootstraps → 95 % interval | `spacecons_pseudotime.csv`, `spacecons_graph_edges.csv`, `fig_SPACECONS.png` |
| 29–30 | `milestones` | **Consensus-M:** each method's own milestones and graph → spot co-membership matrix → average linkage at K = weighted median of method milestone counts → edge transfer through shared spots → maximum spanning tree, directed by pseudotime | `spacecons_milestones.csv`, `spacecons_milestone_per_spot.csv`, `spacecons_milestone_edges.csv`, `spacecons_milestone_network_v2.csv`, `method_milestone_orderings.csv`, `method_graphs.json`, `fig_SPACECONS_milestones.png` |
| 31–32 | `validation` | Validation with CASCAT's metrics (IM, GW labelled "OT", KT, SR, ARI, AMI) + our additions (state-edge F1, stated order recovered, distance-from-root baseline, SR beyond distance) | `validation_vs_stlearn.csv`, `validation_state_means.csv` |
| 33–34 | `spacecons3` | **Consensus-3:** CASCAT cut of the weakest 7 % of neighbour pairs → SNF fused network of 5 views → weighted votes → estimate A (vote solve, root = 0, λ = 0.01) + estimate B (fused walker) → final = rank((A + B)/2); 100 bootstraps; ablations | `spacecons3_per_spot.csv`, `spacecons3_method_contribution.csv`, `spacecons3_summary.json`, `fig_SPACECONS3.png` |
| 35–36 | `second-dataset` | **Dataset loop:** re-runs all tagged cells for each remaining dataset in `DATASETS`; skips finished datasets; stops a dataset cleanly if download, load or root fails | `v3_<DATASET>/` for each dataset |
| 37 | `summary` | All datasets side by side: validation table + milestone counts | `validation_all_datasets.csv` |
| 38–39 | `spacecons20_multiroot` | **Root sensitivity:** SpaceFlow re-run from the 8 candidate roots; agreement between roots | `root_sensitivity.csv`, `multiroot_consensus_per_spot.csv`, `multiroot_consensus_by_root.csv` |
| 40–42 | – | Download links; zips every `v3_<DATASET>` folder | `v3_<DATASET>.zip` |
| 43 | – | Reproducibility notes | – |

---

## 6. Key settings (all fixed before the held-out test)

| Where | Setting |
|---|---|
| Load cell | QC ≥ 200 genes (Visium), genes in ≥ 3 spots, 2,000 HVGs, 30 PCs, Leiden resolution 0.8 |
| SpaceFlow | Spatial regularisation 0.1, 50-D embedding, diffusion pseudotime |
| SpaTrack | Equal expression and spatial cost, 20 start spots |
| StaVia | Spatial smoothing 0.3, spatial kNN 15, memory 5 |
| stLearn | stSME with H&E tiles where an image exists (TBI, BC); PCA of normalised expression otherwise |
| CASCAT | Authors' YAML per data type, 2 trials × 50 epochs, binned CMI, 60-min limit, `percent` 0.07 if missing |
| Consensus-R | `WEIGHTING = 'loo_squared'`, 500 method bootstraps |
| Consensus-M | K = weighted median of the methods' milestone counts; edges kept if support ≥ 0.25 |
| Consensus-3 | `S3 = dict(K_VOTE=6, K_CAND=15, K_SNF=6, SNF_ITERS=20, SNF_MU=0.5, HOOD=18, CASCAT_PERCENT=0.07, B_BOOT=100, PRIOR=0.01)` |

What the Consensus-3 settings mean:
- **K_VOTE:** voting neighbours per spot.
- **K_CAND:** candidate neighbours for the similarity kernels.
- **K_SNF, SNF_ITERS, SNF_MU:** SNF neighbours, rounds and kernel width.
- **HOOD:** neighbourhood size for local gene-trend trust.
- **CASCAT_PERCENT:** share of neighbour pairs cut by CASCAT.
- **B_BOOT:** bootstrap draws.
- **PRIOR:** weak pull towards the methods' mean (λ).

---

## 7. Output files (one folder per dataset: `/kaggle/working/v3_<DATASET>/`)

| File | Contents |
|---|---|
| `all_methods_pseudotime.csv` | Per spot: region, target score (`score_injury` = reference time / injury score), pseudotime of each of the five methods, x, y |
| `raw_stlearn.csv`, `raw_cascat.csv` | Raw outputs of the two methods run in their own environments |
| `cascat_result.h5ad` | CASCAT's full result (embedding, clusters, CMI, tree) |
| `comparison_check.csv` | Each method: mean pseudotime of the root region and of the end region, Spearman with the target score, method family |
| `comparison_agreement.csv` | 5 × 5 Spearman agreement between the methods |
| `fig_<method>.png` | Per-method figure: pseudotime map, region graph, pseudotime per region |
| `spacecons_pseudotime.csv` | **Consensus-R** per spot: pseudotime, 95 % interval (`ci_low`, `ci_high`), local confidence, and RRA p-values for being consistently early or late (`rra_p_early`, `rra_p_late`; extra information, not used for the ordering) |
| `spacecons_graph_edges.csv` | Consensus-R graph on the shared regions (kept for comparison only) |
| `spacecons_milestones.csv` | **Consensus-M** milestones: size, mean pseudotime, majority state |
| `spacecons_milestone_per_spot.csv` | Consensus milestone of every spot |
| `spacecons_milestone_edges.csv`, `spacecons_milestone_network_v2.csv` | Consensus-M trajectory graph: edges, support, direction |
| `method_milestone_orderings.csv` | Each method's own milestones in pseudotime order with majority states |
| `method_graphs.json` | Each method's own milestone graph |
| `validation_vs_stlearn.csv` | **Main validation table:** IM, GW ("OT"), KT, SR, ARI, AMI, state-edge F1, stated order recovered, SR beyond distance, for every method, the baseline and the consensus versions |
| `validation_state_means.csv` | Mean pseudotime of each reference state for each ordering |
| `spacecons3_per_spot.csv` | **Consensus-3** per spot: final pseudotime, 95 % interval, estimate A (`vote_solve`), estimate B (`fused_walker`), A–B `agreement`, `conflict`, `most_trusted` method, each method's share of the vote, fused group, x, y |
| `spacecons3_method_contribution.csv` | Each method: global trust (LOO² weight), mean share of the vote, % of spots where it is most trusted |
| `spacecons3_summary.json` | Run facts: settings, fused views, voting pairs, CASCAT cut pairs, StaVia terminal milestones, A–B Spearman |
| `fig_SPACECONS.png`, `fig_SPACECONS_milestones.png`, `fig_SPACECONS3.png` | Consensus figures (as on slides 17–19) |
| `multi_root_candidates.csv`, `root_sensitivity.csv`, `multiroot_consensus_*.csv` | Candidate roots and root-sensitivity results |
| `figures/` | Every figure shown in the notebook, as PNG |
| `spacecons_version.txt` | `3.1` when the dataset finished; the loop skips datasets that have it |

The working folder also gets:
- `/kaggle/working/validation_all_datasets.csv`: all datasets together.
- `/kaggle/working/v3_<DATASET>.zip`: one ZIP per dataset.

---

## 8. Results produced by this notebook (final run)

SR = Spearman correlation with the reference time (1 = perfect order).

| Ordering | TBI | BC | ZHUANG | HER2ST | Mean |
|---|---|---|---|---|---|
| Best single method | 0.74 (SpaTrack) | 0.48 (stLearn) | 0.73 (stLearn) | 0.32 (CASCAT) | – |
| Average of the 5 methods | 0.52 | 0.37 | 0.64 | 0.02 | 0.39 |
| Distance from root (baseline) | 0.68 | 0.33 | 0.21 | 0.14 | 0.34 |
| **Consensus-R** | 0.57 | 0.40 | **0.74** | −0.01 | 0.42 |
| **Consensus-3** | 0.65 | **0.48** | 0.54 | 0.05 | 0.43 |

Consensus milestones: 15 (TBI), 13 (BC), 10 (ZHUANG), 7 (HER2ST). CASCAT's Combined Score, (KT + SR)/2 − (IM + GW)/2, is computed from `validation_vs_stlearn.csv` by the analysis scripts (report Table A1).

---

## 9. Report figures from the outputs (no Kaggle re-run needed)

After downloading the four `v3_<DATASET>.zip` files:

```bash
pip install -r requirements.txt
mkdir -p results
for d in TBI BC ZHUANG HER2ST; do unzip -q v3_$d.zip -d results/v3_$d; done
python analysis/make_figures.py        --results results --out figs   # Figs. 2-4, summary_SR.csv
python analysis/make_graph_compare.py  --results results --out figs   # Fig. 5 (reference vs graphs)
python analysis/make_appendix_figs.py  --results results --out figs   # Figs. A1-A3
python analysis/make_consensus_flow.py --out figs                     # Fig. 1 (flowchart)
python analysis/make_schematic.py      --out figs                     # pipeline block diagram (slides)
```

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| **HTTP 403** when downloading ZHUANG/HER2ST | figshare sometimes blocks Kaggle | Open https://ndownloader.figshare.com/files/52119332, …/52119302 and …/52119398 in a browser, upload the three files as one Kaggle dataset (**+ Add Input → Upload**), run again. The notebook finds them by file size in `/kaggle/input` |
| CASCAT cell looks stuck | Kaggle's display stopped refreshing; CASCAT takes about 4–10 min per dataset | Wait; check elapsed time or the Output panel. The 60-minute limit stops it if it really hangs |
| `RuntimeError: The "<tag>" cell has not run yet` | The dataset loop was started without Run All | Use **Run All** from the top |
| A dataset ran on the wrong data | An earlier download failed | Fixed in v3.1: the loop stops a dataset if download, load or root fails. Delete the bad `v3_<DATASET>` folder and run again |
| A finished dataset is skipped | `spacecons_version.txt` already says 3.1 | Delete `/kaggle/working/v3_<DATASET>/spacecons_version.txt` (or the folder) to redo it |
| SyntaxError after editing | A cell was pasted twice or cut | Re-import the original notebook |
| `KeyboardInterrupt` stops everything | Stop button pressed during the loop | Intended: the whole loop stops; run the loop cell again (finished datasets are skipped) |

---

## 11. Reproducibility notes

- Reference labels are used only to choose the root state and in the validation cells.
- All Consensus-3 settings were fixed on TBI and BC before ZHUANG and HER2ST were run.
- SpaceFlow, StaVia and CASCAT use the GPU and random initialisation, so a re-run can differ in the **second decimal**. The analysis scripts are deterministic, given the notebook outputs.
- **Bug fixes in v3.1:**
  - One CASCAT work folder per dataset; earlier, a cached CMI file was reused across datasets.
  - The λ = 0.01 prior for patches cut off by CASCAT; earlier they were set to 0. On BC this changed SR from 0.53 to an honest 0.48.
  - Barcodes are read as text.
  - The label column is chosen by its number of unique values.
  - Global variables are reset between datasets.

---

## 12. Methods and references

- **SpaceFlow:** Ren et al., *Nat. Commun.* 2022
- **SpaTrack:** Shen et al., 2025
- **StaVia:** Stassen et al., 2024
- **stLearn PSTS:** Pham et al., *Nat. Commun.* 2023
- **CASCAT:** Yu et al., *Nucleic Acids Research* 2025 (gkaf791), with its benchmark data on figshare 28342754
- **Data:**
  - TBI: GEO GSE236171
  - BC: 10x Genomics, SEDR annotation
  - ZHUANG: Zhang et al., *Nature* 2023
  - HER2ST: Andersson et al., *Nat. Commun.* 2021
