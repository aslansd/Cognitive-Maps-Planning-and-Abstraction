# Cognitive Maps, Planning, and Abstraction

Nine self-contained notebooks that take recent papers in computational neuroscience and cognitive
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
backpropagation anywhere; four use PyTorch.

---

## The notebooks

### Planning algorithms, head to head

| Notebook | What it does | Runtime |
|---|---|---|
| [`MST_Planning_Benchmark.ipynb`](MST_Planning_Benchmark.ipynb) | GCML, Active Predictive Coding, and Jensen et al.'s rollout RNN ported onto the **Maze Search Task** of Kryven et al. (2024), scored on four separate axes: search efficiency, per-participant cross-validated human likelihood, correlation with aggregate human choices, and compute cost. Includes all twelve reference models from the MST paper, a planning-depth sweep, a decision-time analysis, and held-out-maze generalisation. | ~15–25 min CPU |
| [`Benchmarking_Four_Planning_Models_on_a_Shared_Task.ipynb`](Benchmarking_Four_Planning_Models_on_a_Shared_Task.ipynb) | Those three algorithms **plus SSP-BO** (Furlong et al. 2026) on a **compositional multi-room building** with a relocating goal — the task family two of the papers use natively — scored as steps-to-goal over the BFS optimum, pooled over 3 repeats × 15 goal relocations, with planning wall-clock reported next to path quality rather than folded into it. Ends with a section putting SSP-BO on the axis its own paper argues about: sample-selection cost as *n* grows, head to head against the GP baseline shipped in its own repository. | ~15–25 min CPU |

The two are complements, not duplicates: the first has human data and a published model set to
calibrate against; the second uses the navigation task the papers themselves introduce, and is where
the porting decisions (per-edge vs. compass action codes, fully-observed walls, nested rollouts) are
documented in detail.

The fourth model is the odd one out on purpose. **SSP-BO is not a navigation-policy learner** — it is a
sample-efficient black-box optimizer whose claim is that sample selection costs *O(d²)* in a fixed
embedding dimension rather than *O(n³)* in the number of observations. It enters the benchmark through
its own repository's `SSPTrajectoryAgent`, which encodes a whole waypoint sequence as a single
time-bound SSP sum `φ(traj) = Σₜ Tᵗ ⊛ φ(xₜ)`, so "planning" becomes Bayesian optimization over
trajectory space. The waypoints are executed by a deliberately myopic follower that hill-climbs
straight-line distance and therefore gets trapped inside rooms on its own, which is what forces the
optimizer to discover the doorways rather than the controller. Three of the four models amortize —
they learn a map, a model, or a policy once and emit a path per goal in milliseconds; SSP-BO carries
nothing between goals and re-solves from scratch every time. The notebook reports that asymmetry as a
cost column instead of hiding it, and the library is vendored near-verbatim from the authors' source
rather than reimplemented.

### Theories of the cognitive map and of sequence memory

| Notebook | What it does | Runtime |
|---|---|---|
| [`Cognitive_Map_Theories_Benchmark.ipynb`](Cognitive_Map_Theories_Benchmark.ipynb) | **TEM-lite** (the factorised structure/sensation branch argued for by Whittington et al. 2022) vs. the **CSCG** (Raju et al. 2024), on one shared formalism — a deterministic labelled multigraph with aliased observations. Six experiments: aliased learning, transitive stitching and shortcut planning, structural transfer to a redrawn sensory world, non-spatial relational graphs, egocentric actions without a compass, and place fields as an artefact of Euclidean plotting. Ends with an ablation of how much of the structural code's success is built-in action algebra. | ~6 min FAST / ~20 min FULL |
| [`CSCG_vs_Slots.ipynb`](CSCG_vs_Slots.ipynb) | **CSCG** vs. Whittington et al. (2025) **activity slots and episodic weights**, on a shared next-observation-prediction suite. Contains the repo's sharpest theoretical result: a CSCG's forward pass *is* a gated linear RNN over a belief state, and the activity-slot architecture is its **max-plus (tropical) limit** — verified numerically. Also an exact capacity scaling law computed without training anything, a compositional-generalisation dissociation, reciprocal methodological critiques applied across, and an EM→gradient-descent hybrid. | ~10–15 min QUICK / ~45–70 min full |
| [`CSCG_Exploring_Replay.ipynb`](CSCG_Exploring_Replay.ipynb) | Antonov & Dayan's (2025) **exploratory replay** machinery — belief-state planning trees, exploratory Gain and Need, EVB prioritisation, non-myopic sequence replay — run on a state space that is *learned by a CSCG from aliased sensations* rather than given as coordinates, with every experiment run side by side against the ground-truth version. Then three predictions unique to the combination, including belief splitter cells emerging with no added machinery. | ~15–20 min (~5 min with `FAST=True`) |

### Relational abstraction and order learning

| Notebook | What it does | Runtime |
|---|---|---|
| [`Relational_Abstraction_Benchmark.ipynb`](Relational_Abstraction_Benchmark.ipynb) | Four papers on how relational abstractions are acquired, compared in prose and then reduced to a benchmark. **Lippl et al. (2024)** — where a rank *emerges* from norm minimisation on a partly conjunctive representation — is run head to head against **Yang & Maass (2026)** — where the rank is *built in* as a single linear neuron with a local plasticity rule and a convergence proof. Eight shared tasks span abstraction (transitive inference, curriculum robustness, sample efficiency, transverse patterning), planning (list linking, route planning on a learned two-attribute map), and cognitive maps (multi-attribute queries, Pareto fronts, grid-cell embedding, the parietal horseshoe). Webb et al. (2023) and Krause & Reimann (2024) supply the diagnostics rather than models: a behavioural relational-bottleneck score and a symmetry-equivariance score, both computable from behaviour alone. | ~3–6 min CPU (`QUICK=True`) |

The comparison is structured so that the two review-and-analysis papers become **instruments** applied
to the two mechanistic ones. The design pays off in §B4: on standard transitive inference the two
theories are near-indistinguishable, and it takes a non-transitive relation to separate them.

### Concept induction and planning on ARC-AGI

| Notebook | What it does | Runtime |
|---|---|---|
| [`Minimax_Entropy_ARC.ipynb`](Minimax_Entropy_ARC.ipynb) | Zhang et al.'s (2024) **minimax entropy learning** applied to ARC-AGI-1. A descriptive Gibbs model over filter responses is fitted *per task*, from that task's own 2–5 demonstration pairs — no training set, no pretraining, no backprop. Reproduces the paper's three-stage ablation and its few-shot claim, with a genuine MCMC sampler that reaches exact answers from a cold start. | ~0.9 s/task; ~12 min for all 800 |
| [`EC_GCML_ARC.ipynb`](EC_GCML_ARC.ipynb) | **GCML + efficient coding.** GCML needs an observation embedding `Q` and does not learn one in its hardest task; Fang & Sims' (2025) ECPG is precisely a principled way to learn one, and its authors name multi-step planning as their untested extension. Each paper's open problem is the other's contribution. Includes a full validation of the ECPG transcription against the original paper's result before anything is applied to ARC. | ~6–9 min CPU |
| [`R_GCML_ARC.ipynb`](R_GCML_ARC.ipynb) | **GCML + the relational bottleneck** (Webb et al. 2024). The bottleneck supplies the fixed compositional `Q` that GCML hand-designs and cannot learn; GCML supplies the sense of direction and multi-step generative sampler that discriminative relational models lack. Three falsifiable predictions, including exact palette invariance and a held-out-palette transfer test. | ~3–6 min CPU |

`R_GCML_ARC` and `Relational_Abstraction_Benchmark` are the two entry points into Webb et al.'s
relational bottleneck, from opposite directions: the first *uses* it as an architecture, the second
*measures* it as a property of behaviour.

---

## Running them

Every notebook is designed to open in Colab and run top to bottom with no setup:

```
https://colab.research.google.com/github/aslansd/Cognitive-Maps-Planning-and-Abstraction/blob/main/<notebook>.ipynb
```

**Dependencies.** `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `pandas` throughout;
`torch` for `MST_Planning_Benchmark`, `Cognitive_Map_Theories_Benchmark`, `CSCG_vs_Slots`, and
`Benchmarking_Four_Planning_Models_on_a_Shared_Task`; `networkx` for the multi-room benchmark and
for the partial-order transitive closure in `Relational_Abstraction_Benchmark`.
`Benchmarking_Four_Planning_Models_on_a_Shared_Task` vendors the SSP-BO library inline — it needs no
`pip install` of `ssp-bayes-opt`, and in particular does **not** require Nengo, which the upstream
package pulls in at import time.
No GPU is required anywhere — `MST_Planning_Benchmark` will use one if present and is roughly
1.5–2× faster with it.

**Data.** Fetched at runtime, nothing is vendored:

- ARC-AGI-1 and ARC-AGI-2 clone from [`fchollet/ARC-AGI`](https://github.com/fchollet/ARC-AGI) and
  [`arcprize/ARC-AGI-2`](https://github.com/arcprize/ARC-AGI-2). If the runtime has no network, the
  ARC notebooks fall back to a built-in synthetic task generator and **every cell still runs** — only
  the benchmark numbers depend on the real corpus.
- The Maze Search Task mazes, human traces and reference pickles clone from
  [`marta-kryven/MST`](https://github.com/marta-kryven/MST).
- `Relational_Abstraction_Benchmark` needs no data and no network at all: every task is generated in
  the notebook, and the model library is written to disk by a `%%writefile` cell.

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
planning half; the `ssp-bayes-opt` package accumulates a `constraint_ssp` term that its acquisition
function never reads, so a start-position constraint has to be enforced by construction instead; Yang &
Maass evaluate their rank profile at the moment the ranking *first* becomes correct rather than at
convergence, which turns out to change the size of the terminal item effect by nearly a factor of two. Each of these changed a design decision, and each is documented inline where
it bites.

**Reproduce before extending.** Where a paper has a headline result, the transcription is checked
against it first — `EC_GCML_ARC` reproduces ECPG's inverted-U before touching ARC;
`MST_Planning_Benchmark` validates its decision trees and value function against the pickles shipped
with the original repository; `CSCG_Exploring_Replay` runs every experiment twice, once on ground-truth
coordinates and once on the learned latent graph; `Relational_Abstraction_Benchmark` opens with a
validation section that matches its ridge and gradient-flow simulations against the closed forms
ported from Lippl et al.'s R package to ~1e-15, and recovers the analytic conjunctivity factor
α = 0.14716 for a one-hidden-layer ReLU network from random features.

**Separate the ceiling from the mechanism.** On ARC, *coverage* (can this DSL or filter library express
the answer at all?) and *search or ranking efficiency within coverage* are reported separately, always.
Conflating them is how ARC results get oversold, and both ARC-planning notebooks say so explicitly.
The same discipline applies elsewhere: in `Relational_Abstraction_Benchmark`, ordinal accuracy on a
learned cognitive map is reported separately from the metric-dependent quantities built on it, because
every model scores 1.000 on the former while differing substantially on the latter.

**Negative results are kept.** The efficient-coding notebook's central hypothesis — that ECPG would
rescue GCML's `Q`-learning rule — is wrong, and the measurement explaining *why* it is wrong (compression
makes the latent dynamics stochastic, so the prediction error becomes irreducible) is more useful than
the confirmation would have been. GCML's small-sample advantage does not reproduce on ARC, and the
notebook says so in those words. Jensen-RNN finishes the four-way planning benchmark at 24× the optimum,
close to the step budget it is allowed, and the row stays in the summary table rather than being quietly
dropped or rescued by a training-budget bump chosen after seeing the number. In
the relational benchmark, all four models sit at chance on the
pairs a partial order does not entail — they interpolate an order without representing their own
uncertainty about it — and that row is left in the scorecard rather than dropped for being flat. A
delay-period decoding analysis that fails to separate two theories is reported because it fails.

**Shared randomness is treated as a confound.** Adding SSP-BO to the four-way planning benchmark moved
APC's mean from 1.26 to 2.09 without a line of APC's code changing: GCML's theta-cycle rollouts and APC's
random-shooting MPC both draw from the *global* `np.random` stream, so evaluating models interleaved
inside one loop makes each model's score depend on how many draws the others consumed first. Every model
now gets its own pass with the RNG reset beforehand. Anyone adding a fifth model should keep that
structure, or they will silently perturb the other four.

**Comparisons state their own asymmetries.** Where two models are given different training regimes
because their own papers demand it, that is said in the text next to the table rather than smoothed
away. The list-linking benchmark pits an incremental learner against a batch estimator, which is an
architectural mismatch rather than a defeat, and the notebook says so and names the fair version.

---

## A note on the numbers

Figures quoted in the notebook summaries come from the authors' own runs at the stated settings.
Several results — especially the palette-transfer test in `R_GCML_ARC` (§13) and the head-to-head
tables in `Cognitive_Map_Theories_Benchmark` — are single-seed and the notebooks flag them as such;
wrap the relevant sections in a seed loop and report mean ± sd before quoting them.

`Benchmarking_Four_Planning_Models_on_a_Shared_Task` pools every model over 3 repeats × 15 goal
relocations, because single-pass numbers on this task turned out to be seed artifacts: GCML scored 1.42
under one global seed and 2.08 under another. Its committed results are

| Model | steps / optimal | planning s / goal |
|---|---|---|
| SSP-BO | **1.180 ± 0.449** | 8.12 |
| APC | 1.361 ± 0.966 | 0.0012 |
| GCML | 1.649 ± 1.315 | 0.0259 |
| Jensen-RNN | 23.992 ± 8.629 | 0.0333 |

Two things in that table need saying out loud rather than being left for the reader to notice.

**SSP-BO wins on path quality and loses on cost by about four orders of magnitude.** It reached the goal
on 45/45 runs at 1.18× the BFS optimum, but spends ~8 s and 92 imagined rollouts per goal doing it, against
APC's ~1 ms. Reading the first column without the second is reading half the result. Its §9 scaling
section is where the trade actually pays: sample-selection cost stayed flat from *n* = 12 to *n* = 262
while the GP baseline climbed from ~72 ms to ~220 ms per step — ~8× and still widening, well short of the
paper's 60–200 ×, which lives at much larger *n*. In that run the GP reached a marginally *better* final
objective score (−1.04 vs −1.20), so the honest reading of §9 is "same accuracy, flat cost curve", not
"better on both".

**Jensen-RNN does not solve this task, and that row is left in.** At 24× the optimum it is effectively
failing, and its per-repeat means are identical because evaluation is greedy `argmax` over fixed weights,
so the three repeats vary nothing. That number is a statement about *this port at 400 meta-training
episodes on this building*, not a verdict on Jensen et al.'s model — the notebook meta-trains across
freshly sampled mazes while GCML and APC pretrain on the very building they are tested in, which is the
asymmetry their own papers dictate and the one most likely to explain the gap. Raise `n_episodes` before
quoting it anywhere.

`Relational_Abstraction_Benchmark` averages every stochastic result over `N_SEEDS` (8 with
`QUICK=True`, 25 without), and its two known fidelity gaps are stated inline: the replicated update
count for the rank rule is 61–84 against the 93.6 the paper reports, with a sparsity sweep printed so
the reader can see that the discrepancy is a code-sparsity effect rather than a mechanism error; and
the rich-regime network is trained with full-batch gradient descent under an automatic step-size
search rather than the original's schedule, so the *pattern* across initialisation scales should be
trusted ahead of the decimals.

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
5. Furlong, P. M., Dumont, N. S.-Y., Antonova, R., Orchard, J. & Eliasmith, C. (2026). Compositional neurosymbolic representations enable efficient active exploration. *Nature Communications*, [s41467-026-75703-4](https://www.nature.com/articles/s41467-026-75703-4). — `ssp-bayesopt`
6. Kryven, M. et al. (2024). The Maze Search Task. — [`marta-kryven/MST`](https://github.com/marta-kryven/MST)

**Cognitive maps, sequence memory, replay**

7. Raju, R. V., Guntupalli, J. S., Zhou, G., Wendelken, C., Lázaro-Gredilla, M. & George, D. (2024). Space is a latent sequence: a theory of the hippocampus. *Science Advances* **10**, eadm8470.
8. Whittington, J. C. R., McCaffary, D., Bakermans, J. J. W. & Behrens, T. E. J. (2022). How to build a cognitive map. *Nature Neuroscience* **25**, 1257–1272.
9. Whittington, J. C. R. et al. (2025). A tale of two algorithms: activity slots vs. episodic weights.
10. Antonov, G. & Dayan, P. (2025). Exploring replay. *Nature Communications* **16**, 1657.
11. Mattar, M. G. & Daw, N. D. (2018). Prioritized memory access explains planning and hippocampal replay. *Nature Neuroscience* **21**, 1609–1617.

**Abstraction and concept induction**

12. Zhang, C., Jia, B., Zhu, Y. & Zhu, S.-C. (2024). Human-level few-shot concept induction through minimax entropy learning. *Science Advances* **10**, eadg2488.
13. Fang, Z. & Sims, C. R. (2025). Humans learn generalizable representations through efficient coding. *Nature Communications* **16**, 3989. — [`fangzefunny/ecpg`](https://github.com/fangzefunny/ecpg)
14. Webb, T. W. et al. (2024). The relational bottleneck as an inductive bias for efficient abstraction. *Trends in Cognitive Sciences* **28**(9), 829–843. — arXiv:2309.06629
15. Lippl, S., Kay, K., Jensen, G., Ferrera, V. P. & Abbott, L. F. (2024). A mathematical theory of relational generalization in transitive inference. *PNAS* **121**(28), e2314511121. — [`sflippl/relational-generalization-in-ti`](https://github.com/sflippl/relational-generalization-in-ti)
16. Yang, Y. & Maass, W. (2026). The inherent capacity of neurons to learn order relations and support abstract reasoning. *Nature Communications*, s41467-026-76102-5. Preprint: bioRxiv 2025.03.17.642834.
17. Krause, R. & Reimann, M. W. (2024). Items or Relations: what do Artificial Neural Networks learn? arXiv:2404.12401
18. Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547. — [ARC-AGI](https://arcprize.org/arc-agi)

---

## Repository layout

```
.
├── README.md
├── Benchmarking_Four_Planning_Models_on_a_Shared_Task.ipynb
├── MST_Planning_Benchmark.ipynb
├── Cognitive_Map_Theories_Benchmark.ipynb
├── CSCG_vs_Slots.ipynb
├── CSCG_Exploring_Replay.ipynb
├── Relational_Abstraction_Benchmark.ipynb
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
algorithm). `Relational_Abstraction_Benchmark` ends with a list of extensions its library is built to
accept — the most interesting being to give the rank rule a small conjunctive population and re-run
the transverse-patterning benchmark, which is the synthesis the two papers point at and neither
performs.

## License

MIT for the code in this repository. The papers, their released code, and the ARC and MST datasets
remain under their own licenses — check each upstream repository before redistributing.
