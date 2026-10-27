# Current user decisions

Active decisions, in order of precedence. A later item governs only what the
earlier ones leave open.

1. **Research framing v6**:
   `../ResearchPaper-sensitivities/docs/interface_framing_v6.md` (2026-10-27).
   It owns the research question and thesis (§2), the theory (§4), the
   hypotheses H1–H4, their thresholds and the H4 prediction table (§5), the
   tasks, policy and two-phase training (§6), the experiment matrix and the
   dev/held-out/test split (§7), the protocol and freeze contents (§8), the
   pivopt changes (§9) and the implementation modules (§14). Read the relevant
   section before implementing or running an experiment. The theory is
   explained in `theory_notes_1027.md` and `theory_tutorial_1027.md` in the
   same directory. The predecessors `interface_framing_v5.md`,
   `interface_framing_v4.md`, `interface_framing_1026.md` (v3) and
   `learnability_framing_1026.md` (v2.1) are historical only; v4 contains
   errors that v6 §16 corrects.
2. **Scaffold**: `docs/scaffold_2026-10-26.md` records how `examples/` was
   imported from pivopt `94f7c1cd`, the environment, and the verification
   baseline (42 passing tests).
3. **Exploratory pilots**: `artifacts/pilot_1026/` (git-ignored, this machine
   only). They used a single seed, constants tuned on the test batch, untuned
   ES/FR and a saturating sigmoid decoder. They motivate the design and are
   never reported as results.

If a request conflicts with these decisions, say so and ask before acting.
Changing the research design (hypotheses, thresholds, seeds, outer losses,
tasks, the experiment set) or the pivopt pin requires the user's explicit
decision. Write confirmed changes back into the framing document so that code
and framing never silently diverge.

# Working in RSS27_exp

RSS27_exp holds the experiments for the RSS 2027 two-stage paper *Cost or
Constraint? What Learned Limits Become in Soft-Constrained Trajectory
Planners*. A policy reads a known map and predicts a corridor of boxes and the
motion limits for one global plan, and it is trained through the planner only.
The thesis is that a learned limit acts as a constraint only while it binds.
Once its relaxation saturates, an offset-only limit under fixed timing stops
affecting the plan. A limit that also sets the residual scale, or that runs
under optimized timing, keeps acting as a cost weight whose nominal value no
longer bounds the motion. The tasks are a 2D unicycle and a 3D quadrotor that
pass walls rotated about z, with the segment count swept over K ∈ {8, 16, 24}.
Plans are also executed in MuJoCo for evaluation.

Optimize for evidence that a skeptical reviewer can reproduce from a frozen
configuration and a commit. Readable code, honest reporting and an auditable
trail matter more than speed or generality. Documents and pilot notes are
leads, not proof: check the code and the raw outputs before relying on a claim.

## Repository layout and ownership

| Path | Role | Rule |
|---|---|---|
| `examples/` | pivopt examples imported at `94f7c1cd`; `docs/upstream_examples.json` holds their source hashes | **Frozen.** Never edit them or run fixers on them. Only the existing example tests and the new equivalence tests may import them, read-only |
| `experiments/` | Reusable package for the paper's experiments (framing v6 §14): `scenes`, `geometry`, `corridor`, `planners`, `policy`, `training`, `diagnostics`, `mechanism_worlds`, `mujoco_eval`, `study` | Run modules as `python -m experiments.<module>` from the repository root and import them as `experiments.<module>`. Never put `experiments/` itself on `sys.path`. When creating the package, add `"."` to pytest's `pythonpath` |
| `tests/` | pytest suite | Tests use the same public interfaces as the code they check |
| `artifacts/` | All generated outputs (git-ignored) | One directory per experiment, each with a manifest. Never commit outputs |
| `docs/` | Repository notes and provenance | Add one short note per completed milestone |
| `../ResearchPaper-sensitivities/` | The paper and the framing (a separate repository) | Do not edit it from here unless the user asks |

pivopt is a dependency pinned in `pyproject.toml` and `uv.lock`, currently at
`94f7c1cd`. Never edit pivopt from this repository. pivopt may change for the
research (framing v6 §1.4, §9), but only through its own repository: a plan in
`pivopt/agent_docs/` that the user reviews, pivopt's own acceptance checks, and
then a re-pin here that the user approves, with the equivalence tests showing
that the legacy configuration is bitwise unchanged. Use only its public API:
no `pivopt._*` imports and no repository-internal tools.

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
  thresholds or predictions only on development seeds (1000–1002), the 2D
  development condition (K = 8, speed limit, both encodings and both time
  modes) and the mechanism worlds. Never base a decision on a test pool or on
  a held-out cell (framing v6 §7.1). The held-out cells are:
  - the 2D centripetal and forward acceleration limits;
  - the 2D speed limit at K = 16 and K = 24;
  - the 3D speed, acceleration and tilt limits;
  - the corridor parameters (centers, half-extents, rotations) of both tasks.

  Task-level calibration on development seeds is allowed for the 3D task
  (scene ranges, Phase A loss weights, convergence) and for the model-mismatch
  acceptance checks, which use constant limits only. Test results come from
  seeds 30001–30005 on the frozen 2D and 3D test pools.
- **Freeze before reading.** Compute or read no result from a test pool or a
  held-out cell before the freeze. The freeze commits, in both repositories,
  the framing's §5–§7, every configuration, the H4 prediction table, the
  measurement definitions, the thresholds and the analysis code (framing v6
  §8.4). The manifest records both commit hashes. The user sets the freeze
  date. After the freeze, label every change "post-freeze" and state why it was
  made.
- **Count everything.** The seed is the statistical unit. Report every
  per-seed value, every run and every failure. Never drop or rerun a run
  because its result is inconvenient. A run with more than 10% skipped batches
  is an "estimator failure" and stays in the results.
- **Keep every number traceable.** Each reported number must trace to a
  configuration, a commit and raw outputs under `artifacts/`. Never write a
  number from memory, regenerate reference values from a new implementation,
  or widen a tolerance to make a check pass.
- **Label status.** Mark each result as complete, preliminary or planned.

## Experiment code

- Reimplement inside `experiments/` whatever the experiments need from the
  frozen examples; do not copy blocks out of `examples/`.
- No development run starts until the equivalence tests pass. They run the
  legacy configuration (sigmoid decoder, baseline calibration) on fixed batches
  and require:
  - planner inputs and outputs bitwise identical to the frozen examples;
  - outer losses and utility within 1e-12 relative;
  - frozen dataset hashes that match.
- New experiments use the centered log decoder, which gives every
  initialization level the same initial relative sensitivity. The sigmoid
  decoder exists only for the equivalence tests and for reproducing pilots.
- Use float64, explicit seeds and no hidden global state. A run must be a pure
  function of its configuration and seed.
- Check every derivative a result relies on against a re-solve central
  difference. This covers implicit gradients, following slopes, planner-path
  responses and the boundary/force/measure decomposition.
- Keep direct terms (outer-loss terms that depend on θ without the planner)
  separate from the planner-path response, and record them on their own.
- Every manifest records:
  - the command line, configuration and seeds;
  - the commits of both repositories and the pivopt pin;
  - environment versions and output hashes.
- Write the minimum code the planned experiments need. Give each module one
  responsibility. Add no speculative options and no abstraction with a single
  user. Match the surrounding style and keep Ruff clean.
- Use the framing's vocabulary (v6 §3.3) in names and logs: interface, limit,
  inactive/binding/saturated regime, boundary/force/measure term,
  constraint-like/cost-like/inert, offset-only/scale-coupled/timing-coupled/
  shape-coupled, fixed/optimized timing, myopic/truthful zero, following
  slope, planner-path response, matched intervention, hardening, Phase A/B,
  held-out cell, nominal versus realized.

## Paper-facing material

The paper is double-blind. In anything that may reach the paper (figures,
tables, captions, text):
- Do not name pivopt or link a repository; describe the planner generically.
- Do not cite the unpublished Diff² work and do not overlap with it (framing
  v6 §1.3). Ideas such as the bounded axis-angle box rotation, OBB geometry and
  MuJoCo evaluation models may be reimplemented in `experiments/`. Never use
  image or depth inputs, ESDF supervision, policy-predicted time allocation,
  receding-horizon planning, training through tracking or dynamics, or Diff²'s
  scenes, figures and results.
- Cite only references already verified against their source (framing §15).

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
