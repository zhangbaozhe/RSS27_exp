# Current user decisions

Active decisions, in order of precedence. A later item governs only what the
earlier ones leave open.

1. **Research framing v3**:
   `../ResearchPaper-sensitivities/docs/interface_framing_1026.md` (decisions
   Q1–Q43, 2026-10-26). It owns the research question, the hypotheses and
   their thresholds (§5), the experiment designs and protocol (§6), the Stage 1
   deliverables, schedule and contingencies (§8), and the implementation list
   (§12). Read the relevant section before implementing or running an
   experiment. Its predecessor, `learnability_framing_1026.md` (v2.1), is
   historical only.
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
Constraint? Predicting What a Policy Can Learn Through a Trajectory Planner*.
The paper asks which parameter interfaces of a differentiable trajectory
planner receive a task's learning signal when a policy is trained end to end
through it. The interfaces are constraint offsets, constraint softness, cost
weights and timing. It also asks whether diagnostics measured early in training
can predict this in held-out settings.

Optimize for evidence that a skeptical reviewer can reproduce from a frozen
configuration and a commit. Readable code, honest reporting and an auditable
trail matter more than speed or generality. Documents and pilot notes are
leads, not proof: check the code and the raw outputs before relying on a claim.

## Repository layout and ownership

| Path | Role | Rule |
|---|---|---|
| `examples/` | pivopt examples imported at `94f7c1cd`; `docs/upstream_examples.json` holds their source hashes | **Frozen.** Never edit them or run fixers on them. Only the existing example tests and the new equivalence tests may import them, read-only |
| `experiments/` | Reusable package for the paper's experiments (framing §12): `sgate`, `policy`, `diagnostics`, `estimators`, `qp_planner`, `mechanism_worlds`, `study` | Run modules as `python -m experiments.<module>` from the repository root and import them as `experiments.<module>`. Never put `experiments/` itself on `sys.path`. When creating the package, add `"."` to pytest's `pythonpath` |
| `tests/` | pytest suite | Tests use the same public interfaces as the code they check |
| `artifacts/` | All generated outputs (git-ignored) | One directory per experiment, each with a manifest. Never commit outputs |
| `docs/` | Repository notes and provenance | Add one short note per completed milestone |
| `../ResearchPaper-sensitivities/` | The paper and the framing (a separate repository) | Do not edit it from here unless the user asks |

pivopt is a dependency pinned in `pyproject.toml` and `uv.lock` at `94f7c1cd`.
Do not change the pin and do not edit pivopt from this repository. Use only its
public API: no `pivopt._*` imports and no repository-internal tools.

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
  thresholds or prediction rules only on development seeds (1000–1002),
  development initialization levels and the mechanism worlds. Never base a
  decision on the frozen test pool or on a held-out setting. The held-out
  settings are:
  - held-out seeds;
  - initialization levels 0.35×, 0.55× and 0.80×;
  - distribution shifts;
  - the QP policy task;
  - the quadrotor.
- **Freeze before reading.** Compute or read no result from the test pool or a
  held-out setting before the freeze. The freeze commits, in both
  repositories, the framing's §5–§6, every configuration, the diagnostic
  definitions, the thresholds, the prediction rules and the analysis code. The
  manifest records both commit hashes. The freeze is planned for 2026-11-10,
  latest 2026-11-12. After the freeze, label every change "post-freeze" and
  state why it was made.
- **Count everything.** The seed is the statistical unit. Report every
  per-seed value, every run and every failure. Never drop or rerun a run
  because its result is inconvenient. A run with more than 10% skipped batches
  is an "estimator failure" and stays in the results.
- **Keep every number traceable.** Each reported number must trace to a
  configuration, a commit and raw outputs under `artifacts/`. Never write a
  number from memory, regenerate reference values from a new implementation,
  or widen a tolerance to make a check pass.
- **Label status.** Mark each result as complete, preliminary or planned,
  exactly as framing §8.1 defines.

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
  difference. This covers implicit gradients, κ measured by pullback, and the
  QP's KKT gradients for the hard, L1 and L2 variants.
- Every manifest records:
  - the command line, configuration and seeds;
  - the commits of both repositories and the pivopt pin;
  - environment versions and output hashes.
- Write the minimum code the planned experiments need. Give each module one
  responsibility. Add no speculative options and no abstraction with a single
  user. Match the surrounding style and keep Ruff clean.
- Use the framing's vocabulary in names and logs: interface, offset, softness,
  cost weight, κ, occupancy, dead/binding/shaping regime, bypass, hardening,
  nominal versus realized.

## Paper-facing material

The paper is double-blind. In anything that may reach the paper (figures,
tables, captions, text):
- Do not name pivopt or link a repository; describe the planner generically.
- Do not cite or reuse the unpublished Diff² work, and avoid claims or figures
  that overlap with it.
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
