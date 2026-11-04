# Current user decisions

Active decisions, in order of precedence. A later item governs only what the
earlier ones leave open.

Paths of the form `<paper>/…` refer to the sibling checkout of the paper
repository `ResearchPaper-sensitivities`. On the machine used on 2026-11-04 it
is checked out as `../RSS27/`.

1. **Research framing v9**: `<paper>/docs/interface_framing_v9.md`
   (2026-11-04; decisions Q1–Q40, recorded in
   `<paper>/docs/framing_grill_1104.md`). It owns:
   - the research question, viewpoint, claims and scope (§2);
   - the definitions and vocabulary (§3) and the analysis (§4);
   - the central hypothesis, the Prediction, the learning test and their
     decision rules (§5);
   - the inner/outer division and the placement rule (§6);
   - the scenes, tasks, outputs, policy inputs and encoder (§7);
   - the stages, comparisons and baselines (§8);
   - the protocol and the dev/held-out split (§9).

   Q18, the Stage 1 completion scope, is still open. The derivations and toy
   checks are in `<paper>/docs/sensitivity_derivation_1104.md`; their numbers
   come from Python toys, not from pivopt. Read the relevant section before
   implementing or running an experiment.
2. **Historical framings**: v8 (`<paper>/docs/interface_framing_v8.md`) and its
   predecessors are historical only; v9 §13 maps v8 to v9. The v8 dev checks
   (`artifacts/dev_conflict_1028/`) and the 10/31 checks
   (`artifacts/dev_framework_1031/`, cited in
   `<paper>/docs/sensitivity_framework_1031.md`) were absent from the
   2026-11-04 checkout. None of their numbers is used until it has been
   regenerated at the new pin (v9 §8, stage S0).
3. **Re-pin**: `docs/repin_2026-11-04.md` records the user's decisions Q41–Q44
   and the re-pin to pivopt `01654104`: the one-time migration of `examples/`,
   the reference outputs generated at `94f7c1cd`, the equivalence criterion
   and its results (52 passing tests).
4. **Scaffold**: `docs/scaffold_2026-10-26.md` records how `examples/` was
   imported from pivopt `94f7c1cd`, the environment, and the verification
   baseline (42 passing tests).
5. **Exploratory pilots**: `artifacts/pilot_1026/` (git-ignored and
   machine-local). They used a single seed, constants tuned on the test batch,
   untuned ES/FR and a saturating sigmoid decoder. They motivate the design and
   are never reported as results.

If a request conflicts with these decisions, say so and ask before acting.
Changing the research design (hypothesis, prediction, decision rules,
thresholds, seeds, outer losses, tasks, the experiment set) or the pivopt pin
requires the user's explicit decision. Write confirmed changes back into the
framing document so that code and framing never silently diverge.

# Working in RSS27_exp

RSS27_exp holds the experiments for the RSS 2027 two-stage paper *What Should
a Policy Predict for a Differentiable Trajectory Planner? Task Demand,
Sensitivity and Alignment*. A policy reads a scene grid with task channels and
predicts the parameters of one global plan: corridor boxes, per-segment motion
bounds and segment timing. It is trained through the planner only.

The gradient that the policy receives for a planner parameter is a product of
three factors computed at the nominal plan: the task demand, the parameter's
sensitivity in its current regime, and their alignment (v9 §3–§4). The central
hypothesis is that these factors predict which parameters a policy should
predict, and which change restores learning when the gradient through a
parameter is too weak (v9 §5).

There are two scenes:
- **2D:** a differential-drive robot crosses three doorways. Tasks: transit,
  slow zone, door time window and keep-out zone. K ∈ {8, 16, 32}.
- **Arm:** a UR5e moves objects between two cabinets around a pillar. Tasks:
  transfer, open container, inspection and shared-zone timing.
  K ∈ {6, 12}.

The planner receives the boundary conditions and the robot's fixed
constraints. Task requirements reach it only through the predicted parameters
(the placement rule, v9 §6). Plans are executed in MuJoCo for evaluation.

Optimize for evidence that a skeptical reviewer can reproduce from a frozen
configuration and a commit. Readable code, honest reporting and an auditable
trail matter more than speed or generality. Documents and pilot notes are
leads, not proof: check the code and the raw outputs before relying on a claim.

## Repository layout and ownership

| Path | Role | Rule |
|---|---|---|
| `examples/` | pivopt examples imported at `94f7c1cd` and migrated once to `01654104` at the re-pin; `docs/upstream_examples.json` holds their source hashes and every local change | **Frozen at the current pin.** Never edit them or run fixers on them. They change only in a re-pin milestone that the user approves, which migrates them to the new API and proves the legacy behavior with the equivalence tests. Only the example tests and the equivalence tests may import them, read-only |
| `experiments/` | Reusable package for the paper's experiments (framing v9 §5 and §7–§9). The re-pin milestone is complete, so it may now be created | Run modules as `python -m experiments.<module>` from the repository root and import them as `experiments.<module>`. Never put `experiments/` itself on `sys.path`. When creating the package, add `"."` to pytest's `pythonpath`. Give each module one responsibility |
| `tests/` | pytest suite | Tests use the same public interfaces as the code they check |
| `artifacts/` | All generated outputs (git-ignored) | One directory per experiment, each with a manifest. Never commit outputs |
| `docs/` | Repository notes and provenance | Add one short note per completed milestone |
| `<paper>/` | The paper and the framing (a separate repository) | Do not edit it from here unless the user asks. Never edit `<paper>/main.tex` without an explicit request |

pivopt is a dependency pinned in `pyproject.toml` and `uv.lock` at
`01654104` (master on 2026-11-04; v9 §1.3). The re-pin from `94f7c1cd` is
recorded in `docs/repin_2026-11-04.md`.

- `01654104` removed `FixedTotalTime` without an alias. The examples now use
  `AllocatedTime` with `FreeFractions(logit_penalty=0)` and, as upstream did,
  bind a constant total as an input.
- `01654104` changed the defaults of `OptimizerSettings.past`,
  `OptimizerSettings.maximum_iterations`, `LBFGS.memory_size` and
  `BackwardChecks` (every gate is now disabled). The migrated examples state
  the old values explicitly. Every new experiment states all optimizer
  settings and backward checks explicitly.
- `tests/fixtures/repin_94f7c1cd/` holds reference outputs generated at
  `94f7c1cd` with the unmigrated examples, and `tests/equivalence/` compares
  the migrated configurations against them. Regenerating the references
  requires an old-pin tree; `tests/equivalence/generate_reference.py` explains
  how.
- softplus, which v9 §5 and §8 use, exists only from this pin on.

Never edit pivopt from this repository. pivopt may change for the research,
for example to add the direct-TO cost terms of v9 §1.3 and §6. Such changes go
only through pivopt's own repository: a plan in `pivopt/agent_docs/` that the
user reviews, pivopt's own acceptance checks, and then a re-pin here that the
user approves, with the equivalence tests above. Use only its public API: no
`pivopt._*` imports and no repository-internal tools.

## Environment and commands

Run everything from the repository root. The project uses Python 3.12 and uv
and is CPU-only.

```bash
PIVOPT_BUILD_MODE=full CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --locked
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
uv run --no-sync pytest -q
uv run --no-sync ruff check --fix experiments tests
uv run --no-sync ruff format experiments tests
```

- Use `uv run --no-sync` for every Python command. Do not `pip install` or
  change dependencies or the lockfile without the user's decision.
- Ruff 0.17.2 with `ruff.toml` defines the style, and the pre-commit hook runs
  the same checks. If a hook would rewrite a file under `examples/`, stop and
  ask. Hooks do not run tests, so run pytest yourself whenever experiment
  behavior changes.
- In a sandbox, keep caches inside the checkout: `UV_CACHE_DIR=$PWD/.cache/uv`,
  `XDG_CACHE_HOME=$PWD/.cache` and
  `TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/torchinductor`.
- Run long jobs in the background and record their command lines in the
  manifest. Stop only processes you started.

## Research protocol (hard rules)

These rules protect the pre-registration. Breaking one can invalidate the
paper's central claim, so ask whenever you are unsure.

- **Use development data for every decision.** Tune, debug, calibrate and fit
  thresholds or predictions only on development data (framing v9 §9):
  - development seeds 1000–1002;
  - the 2D development configurations: K = 8, the straight and S layouts,
    the lowest and highest of the three pressure margins, and all four tasks;
  - the mechanism worlds and toys of stage S0.

  Never base a decision on test seeds or on a held-out configuration. The
  held-out configurations are:
  - interpolation: the middle pressure margin and the intermediate layout;
  - extrapolation: K = 16 and K = 32 in 2D, and every arm configuration.

  Task-level calibration on development seeds is allowed for the arm (scene
  ranges, loss weights, convergence settings). Test results come from seeds
  30001–30005.
- **Fix on development data, before any held-out reading:**
  - the flatness threshold ε and the normalization of the nudge
    size β;
  - the two update budgets c;
  - the stall rule (M checkpoints, x%), the continuation length and the
    learning rate;
  - the number of instances per configuration, with at least 20 per 2D
    configuration and 10 per arm configuration;
  - the strongest baseline of the Prediction, which is the one with the lowest
    regret on development data.

  The scene encoder is pretrained on development scenes only and frozen before
  any held-out reading. Its R² on held-out scenes is reported (v9 §5.5 and
  §7.3).
- **Freeze before reading.** Compute or read no result from test seeds or a
  held-out configuration before the freeze. The freeze commits, in both
  repositories:
  - the framing's §5–§9 and every configuration;
  - the held-out configurations and their instance generators;
  - the measurement definitions, decision rules and thresholds;
  - the frozen encoder and the analysis code.

  The manifest records both commit hashes. The user sets the freeze date.
  After the freeze, label every change "post-freeze" and state why it was made.
- **Count everything.** The seed is the statistical unit for training runs.
  For the Prediction, the unit is the nominal plan instance, clustered by
  configuration (v9 §5.5). Report every per-seed and per-instance value, every
  run and every failure. Never drop or rerun a run because its result is
  inconvenient. A run with more than 10% skipped batches, or more than 10%
  clipped gradient steps, is an "estimator failure" and stays in the results.
- **Keep every number traceable.** Each reported number must trace to a
  configuration, a commit and raw outputs under `artifacts/`. Never write a
  number from memory, regenerate reference values from a new implementation,
  or widen a tolerance to make a check pass.
- **Label status.** Mark each result as complete, preliminary or planned.

## Experiment code

- Reimplement inside `experiments/` whatever the experiments need from the
  frozen examples; do not copy blocks out of `examples/`.
- No development run starts until the equivalence tests in
  `tests/equivalence/` pass. They run the legacy configurations that v9 uses
  (the gate planning problems of `cpu_gate_certified_training`,
  `cpu_gate_training` and `cpu_gate_probe` with the sigmoid decoder and the
  baseline calibration on fixed batches, `examples/arm.py`,
  `examples/mujoco/arm.py` and `examples/mujoco/unicycle.py`) against the
  reference outputs generated at `94f7c1cd`, and require:
  - planner inputs bitwise identical, and declarations, including every
    optimizer setting and backward check, identical apart from the documented
    time migration;
  - identical lane status and cause codes;
  - outputs bitwise where pivopt records a path as bitwise unchanged;
  - elsewhere, pivopt's approved endpoint envelope (f64: rtol 2e-6, atol 2e-7,
    objective 1e-8 relative to max(1, |value|)) on lanes that are stationary
    to 1e-7 on both pins, and for every lane, zero-iteration solves and
    pullbacks at the same trajectory point inside the same envelope;
  - planner-derived example values, such as outer losses, inside the envelope
    when every lane qualified, and otherwise reported;
  - optimizer work counts reported, not compared.
- New experiments use the centered log decoder, which gives every
  initialization level the same initial relative sensitivity. The last layer of
  the policy head starts at zero, so training starts from the decoder's nominal
  values. The sigmoid decoder exists only for the equivalence tests and for
  reproducing pilots.
- Train the policy head with SGD with momentum 0.9 and one global learning
  rate. Use no per-parameter adaptive optimizer and no gradient normalization.
  Gradient clipping is only a logged safety measure (v9 §5.5).
- Use float64, explicit seeds and no hidden global state. A run must be a pure
  function of its configuration and seed.
- Check every derivative a result relies on against a re-solve central
  difference. This covers:
  - implicit gradients and the three-factor decomposition;
  - finite-nudge signals on each side;
  - planner-path responses and re-timing (direct) terms;
  - the residual of the scaling identity (v9 §4).
- Keep direct terms (outer-loss terms that depend on θ without the planner)
  separate from the planner-path response, and record them on their own.
- Every manifest records:
  - the command line, configuration and seeds;
  - the commits of both repositories and the pivopt pin;
  - environment versions and output hashes.
- Write the minimum code the planned experiments need. Give each module one
  responsibility. Add no speculative options and no abstraction with a single
  user. Match the surrounding style and keep Ruff clean.
- Use the framing's vocabulary (v9 §3) in names and logs:
  - **gradient factors:** nudge, nudge response, parameter sensitivity, task
    demand, sensitivity, alignment;
  - **penalty states:** flat region (before the threshold, linear part),
    curved part, capability;
  - **finite nudges:** finite nudge (PLUS, MINUS, SYMMETRIC), side rule,
    exposure threshold;
  - **gradient terms:** planner-path response versus direct (re-timing) term,
    scaling identity;
  - **problem structure:** inner planner versus outer task loss, placement
    rule (a)/(b)/(c), direct TO, capability reference;
  - **study design:** Prediction, learning test, stall rule, predicted versus
    mismatched intervention, configuration versus instance, held-out
    configuration, regret, nominal versus realized.

## Paper-facing material

The paper is double-blind. In anything that may reach the paper (figures,
tables, captions, text):
- Do not name pivopt or link a repository; describe the planner generically.
- Do not cite the unpublished Diff² work and do not overlap with it (framing
  v9 §1.2).
  - **What Diff² uses:** depth-image inputs with a ResNet18 encoder,
    frame-local occupancy and ESDF, polytopes built from occupancy, and
    receding-horizon flight.
  - **Never use:** image or depth inputs, ESDF supervision or ESDF losses,
    receding-horizon planning, training through tracking or dynamics, or
    Diff²'s scenes, figures and results.
  - **Allowed:**
    - a known global occupancy grid or voxel map with task channels, encoded
      by the small coordinate-channel CNN of v9 §7.3; the outer clearance loss
      uses analytic distances to obstacle primitives;
    - policy-predicted segment durations and allocations, as interfaces under
      study, not as contributions;
    - reimplementations in `experiments/` of ideas such as the bounded
      axis-angle box rotation, OBB geometry and the MuJoCo evaluation models.
- Cite only references already verified against their source (framing v9
  §14).

## Execution and closeout

1. Before changing anything, capture the baseline: `git status`, the pivopt
   pin and the current test result. Preserve unrelated user changes; never
   commit, revert or take credit for them.
2. For multi-step work, state a short plan with a check for each step, then
   keep working until the checks pass.
3. At each milestone, run the affected tests and Ruff. Report the commands, the
   results and anything that was not run.
4. Commit only when the user asks, one milestone per commit. Use a
   conventional prefix (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), and state
   the motivation and the verification in the message.
5. Leave a short milestone note in `docs/` with commands, results, deviations
   and remaining work, so the work can resume without the chat history.

Make routine implementation choices yourself. Propose concretely, then wait for
the user, before anything that changes the research design, the protocol,
frozen files or the pivopt pin.
