# Cognitive Maps, Planning, and Abstraction

Eight self-contained notebooks that take recent papers in computational neuroscience and cognitive
science, re-implement their models from the equations and the released source, and put them on
**shared tasks** so they can be compared, combined, and falsified rather than admired one at a time.

Each notebook does one of three things:

- **Benchmarks** models from different papers against each other on a single common task (and, where
  the task has them, against human data);
- **Combines** two papers that have never been combined, at a specific technical joint, and tests
  whether the combination does what the theory predicts;
- **Transplants** a framework into a domain its authors never targeted — usually
  [ARC-AGI](https://arcprize.org/arc-agi) — to find out which of its claims are about the framework
  and which were about the original benchmark.

Everything runs end to end on a free Colab CPU instance. Most notebooks are pure NumPy with no
backpropagation anywhere; three use PyTorch.

---

## The notebooks

### Planning algorithms, head to head

| Notebook | What it does | Runtime |
|---|---|---|
| [`MST_Planning_Benchmark.ipynb`](MST_Planning_Benchmark.ipynb) | GCML, Active Predictive Coding, and Jensen et al.'s rollout RNN ported onto the **Maze Search Task** of Kryven et al. (2024), scored on four separate axes: search efficiency, per-participant cross-validated human likelihood, correlation with aggregate human choices, and compute cost. Includes all twelve reference models from the MST paper, a planning-depth sweep, a decision-time analysis, and held-out-maze generalisation. | ~15–25 min CPU |
| [`Benchmarking_Three_Planning_Models_on_a_Shared_Task.ipynb`](Benchmarking_Three_Planning_Models_on_a_Shared_Task.ipynb) | The same three algorithms on a **compositional multi-room building** with a relocating goal — the task family two of the three papers use natively — scored as steps-to-goal over the BFS optimum. | ~5–10 min CPU |

The two are complements, not duplicates: the first has human data and a published model set to
calibrate against; the second uses the navigation task the papers themselves introduce, and is where
the porting decisions (per-edge vs. compass action codes, fully-observed walls, nested rollouts) are
documented in detail.

### Theories of the cognitive map and of sequence memory

| Notebook | What it does | Runtime |
|---|---|---|
| [`Cognitive_Map_Theories_Benchmark.ipynb`](Cognitive_Map_Theories_Benchmark.ipynb) | **TEM-lite** (the factorised structure/sensation branch argued for by Whittington et al. 2022) vs. the **CSCG** (Raju et al. 2024), on one shared formalism — a deterministic labelled multigraph with aliased observations. Six experiments: aliased learning, transitive stitching and shortcut planning, structural transfer to a redrawn sensory world, non-spatial relational graphs, egocentric actions without a compass, and place fields as an artefact of Euclidean plotting. Ends with an ablation of how much of the structural code's success is built-in action algebra. | ~6 min FAST / ~20 min FULL |
| [`CSCG_vs_Slots.ipynb`](CSCG_vs_Slots.ipynb) | **CSCG** vs. Whittington et al. (2025) **activity slots and episodic weights**, on a shared next-observation-prediction suite. Contains the repo's sharpest theoretical result: a CSCG's forward pass *is* a gated linear RNN over a belief state, and the activity-slot architecture is its **max-plus (tropical) limit** — verified numerically. Also an exact capacity scaling law computed without training anything, a compositional-generalisation dissociation, reciprocal methodological critiques applied across, and an EM→gradient-descent hybrid. | ~10–15 min QUICK / ~45–70 min full |
| [`CSCG_Exploring_Replay.ipynb`](CSCG_Exploring_Replay.ipynb) | Antonov & Dayan's (2025) **exploratory replay** machinery — belief-state planning trees, exploratory Gain and Need, EVB prioritisation, non-myopic sequence replay — run on a state space that is *learned by a CSCG from aliased sensations* rather than given as coordinates, with every experiment run side by side against the ground-truth version. Then three predictions unique to the combination, including belief splitter cells emerging with no added machinery. | ~15–20 min (~5 min with `FAST=True`) |

### Concept induction and planning on ARC-AGI

| Notebook | What it does | Runtime |
|---|---|---|
| [`Minimax_Entropy_ARC.ipynb`](Minimax_Entropy_ARC.ipynb) | Zhang et al.'s (2024) **minimax entropy learning** applied to ARC-AGI-1. A descriptive Gibbs model over filter responses is fitted *per task*, from that task's own 2–5 demonstration pairs — no training set, no pretraining, no backprop. Reproduces the paper's three-stage ablation and its few-shot claim, with a genuine MCMC sampler that reaches exact answers from a cold start. | ~0.9 s/task; ~12 min for all 800 |
| [`EC_GCML_ARC.ipynb`](EC_GCML_ARC.ipynb) | **GCML + efficient coding.** GCML needs an observation embedding `Q` and does not learn one in its hardest task; Fang & Sims' (2025) ECPG is precisely a principled way to learn one, and its authors name multi-step planning as their untested extension. Each paper's open problem is the other's contribution. Includes a full validation of the ECPG transcription against the original paper's result before anything is applied to ARC. | ~6–9 min CPU |
| [`R_GCML_ARC.ipynb`](R_GCML_ARC.ipynb) | **GCML + the relational bottleneck** (Webb et al. 2024). The bottleneck supplies the fixed compositional `Q` that GCML hand-designs and cannot learn; GCML supplies the sense of direction and multi-step generative sampler that discriminative relational models lack. Three falsifiable predictions, including exact palette invariance and a held-out-palette transfer test. | ~3–6 min CPU |

---

## Running them

Every notebook is designed to open in Colab and run top to bottom with no setup:

```
https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/<notebook>.ipynb
```

Locally:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
jupyter lab
```

**Dependencies.** `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `pandas` throughout;
`torch` for `MST_Planning_Benchmark`, `Cognitive_Map_Theories_Benchmark`, `CSCG_vs_Slots`, and
`Benchmarking_Three_Planning_Models_on_a_Shared_Task`; `networkx` for the multi-room benchmark.
No GPU is required anywhere — `MST_Planning_Benchmark` will use one if present and is roughly
1.5–2× faster with it.

**Data.** Fetched at runtime, nothing is vendored:

- ARC-AGI-1 and ARC-AGI-2 clone from [`fchollet/ARC-AGI`](https://github.com/fchollet/ARC-AGI) and
  [`arcprize/ARC-AGI-2`](https://github.com/arcprize/ARC-AGI-2). If the runtime has no network, the
  ARC notebooks fall back to a built-in synthetic task generator and **every cell still runs** — only
  the benchmark numbers depend on the real corpus.
- The Maze Search Task mazes, human traces and reference pickles clone from
  [`marta-kryven/MST`](https://github.com/marta-kryven/MST).

**Speed switches.** Notebooks with expensive sweeps expose a flag near the top — `FAST`, `QUICK`, or
`FULL_RUN`. The defaults are set for a first read-through; flip them for the numbers quoted in the
summaries.

---

## How these notebooks are written

There is a consistent house style, and it is the point of the repo as much as any individual result:

**The source code is read before the model is written.** Repeatedly, the released code says something
the paper leaves implicit or states differently — GCML's compositional task uses a fixed hand-designed
encoder and never applies its own eq. (13); its tiling notebook *assigns* the inverse model where the
graph notebook applies a delta rule; Jensen et al.'s maze walls are fully observed and only the goal
is hidden; the `apc-vision` repository implements the perception half of the APC paper, not the
planning half. Each of these changed a design decision, and each is documented inline where it bites.

**Reproduce before extending.** Where a paper has a headline result, the transcription is checked
against it first — `EC_GCML_ARC` reproduces ECPG's inverted-U before touching ARC;
`MST_Planning_Benchmark` validates its decision trees and value function against the pickles shipped
with the original repository; `CSCG_Exploring_Replay` runs every experiment twice, once on ground-truth
coordinates and once on the learned latent graph.

**Separate the ceiling from the mechanism.** On ARC, *coverage* (can this DSL or filter library express
the answer at all?) and *search or ranking efficiency within coverage* are reported separately, always.
Conflating them is how ARC results get oversold, and both ARC-planning notebooks say so explicitly.

**Negative results are kept.** The efficient-coding notebook's central hypothesis — that ECPG would
rescue GCML's `Q`-learning rule — is wrong, and the measurement explaining *why* it is wrong (compression
makes the latent dynamics stochastic, so the prediction error becomes irreducible) is more useful than
the confirmation would have been. GCML's small-sample advantage does not reproduce on ARC, and the
notebook says so in those words. A delay-period decoding analysis that fails to separate two theories
is reported because it fails.

**Comparisons state their own asymmetries.** Where two models are given different training regimes
because their own papers demand it, that is said in the text next to the table rather than smoothed away.

---

## A note on the numbers

Figures quoted in the notebook summaries come from the authors' own runs at the stated settings.
Several results — especially the palette-transfer test in `R_GCML_ARC` (§13) and the head-to-head
tables in `Cognitive_Map_Theories_Benchmark` — are single-seed and the notebooks flag them as such;
wrap the relevant sections in a seed loop and report mean ± sd before quoting them. In
`Benchmarking_Three_Planning_Models_on_a_Shared_Task`, the Jensen-RNN port was reviewed line by line
rather than executed in the environment where the notebook was written, and its numbers deserve more
caution than the other two models' until re-run.

None of the ARC notebooks is a competitive ARC-AGI submission, and each says why: a fixed-primitive
DSL or a small filter library is exactly what the benchmark is built to defeat. The scientific content
is in the controlled comparisons.

---

## Papers implemented

**Planning**
1. Lin, H., Yang, Y., Zhao, R., Pezzulo, G. & Maass, W. (2026). Neural sampling from cognitive maps enables goal-directed imagination and planning. *Nature Machine Intelligence* **8**, 1045–1065. — [`LH-cbicr/GCML`](https://github.com/LH-cbicr/GCML)
2. Stöckl, C., Yang, Y. & Maass, W. (2024). Local prediction-learning in high-dimensional spaces enables neural networks to plan. *Nature Communications* **15**, 2344.
3. Rao, R. P. N., Gklezakos, D. C. & Sathish, V. (2024). Active predictive coding: a unifying neural model for active perception, compositional learning, and hierarchical planning. *Neural Computation* **36**(1), 1–32.
4. Jensen, K. T., Hennequin, G. & Mattar, M. G. (2024). A recurrent network model of planning explains hippocampal replay and human behavior. *Nature Neuroscience* **27**, 1340–1348.
5. Kryven, M. et al. (2024). The Maze Search Task. — [`marta-kryven/MST`](https://github.com/marta-kryven/MST)

**Cognitive maps, sequence memory, replay**

6. Raju, R. V., Guntupalli, J. S., Zhou, G., Wendelken, C., Lázaro-Gredilla, M. & George, D. (2024). Space is a latent sequence: a theory of the hippocampus. *Science Advances* **10**, eadm8470.
7. Whittington, J. C. R., McCaffary, D., Bakermans, J. J. W. & Behrens, T. E. J. (2022). How to build a cognitive map. *Nature Neuroscience* **25**, 1257–1272.
8. Whittington, J. C. R. et al. (2025). A tale of two algorithms: activity slots vs. episodic weights.
9. Antonov, G. & Dayan, P. (2025). Exploring replay. *Nature Communications* **16**, 1657.
10. Mattar, M. G. & Daw, N. D. (2018). Prioritized memory access explains planning and hippocampal replay. *Nature Neuroscience* **21**, 1609–1617.

**Abstraction and concept induction**

11. Zhang, C., Jia, B., Zhu, Y. & Zhu, S.-C. (2024). Human-level few-shot concept induction through minimax entropy learning. *Science Advances* **10**, eadg2488.
12. Fang, Z. & Sims, C. R. (2025). Humans learn generalizable representations through efficient coding. *Nature Communications* **16**, 3989. — [`fangzefunny/ecpg`](https://github.com/fangzefunny/ecpg)
13. Webb, T. W. et al. (2024). The relational bottleneck as an inductive bias for efficient abstraction. *Trends in Cognitive Sciences* **28**(9), 829–843.
14. Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547. — [ARC-AGI](https://arcprize.org/arc-agi)

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── Benchmarking_Three_Planning_Models_on_a_Shared_Task.ipynb
├── MST_Planning_Benchmark.ipynb
├── Cognitive_Map_Theories_Benchmark.ipynb
├── CSCG_vs_Slots.ipynb
├── CSCG_Exploring_Replay.ipynb
├── Minimax_Entropy_ARC.ipynb
├── EC_GCML_ARC.ipynb
└── R_GCML_ARC.ipynb
```

Notebooks are committed **with outputs** so the figures and tables are readable on GitHub without
running anything. If you would rather keep diffs clean, `pip install nbstripout && nbstripout --install`
strips them on commit.

## Contributing

Issues and PRs are welcome, particularly: multi-seed reruns of the single-seed results flagged above,
fidelity corrections where a port diverges from the original source, and extensions of any DSL or
filter library (the binding constraint on both ARC-planning notebooks is coverage, not the learning
algorithm).

## License

MIT for the code in this repository. The papers, their released code, and the ARC and MST datasets
remain under their own licenses — check each upstream repository before redistributing.
