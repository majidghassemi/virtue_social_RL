# Pilot of the teacher training (fixed before the pilot runs)

## What runs

The first 70 of 347 updates (about 20,000 episodes, 2 x 10^7 environment steps) of two real runs from `configs/train_prosocial_v1.yaml`: lam 0 seed 0 and lam 1 seed 0. Job: `scripts/slurm/pilot_prosocial.sh`. The runs keep their checkpoints and are continued later by the full job, so the pilot is not extra compute.

## What it is for

Only to check that training works before spending the allocation on all 9 runs. It is not a test of the research question, and its results must not be used to change the disposition reward, the lam values, the maps or the evaluation.

## Checks (printed by `scripts/plot_training.py runs/train/prosocial_v1`)

1. Numerics: every logged value is finite.
2. Learning: apples per episode (all agents) in the last 10% of pilot updates is higher than in the first 10%, for both runs.
3. Policy: entropy falls by more than 0.05 from its start (about 2.08, the uniform policy over 8 actions).
4. Cost: time per update matches the benchmark of the GPU check job, and the job is not killed for memory.

## What may be changed if a check fails

Engineering only: a bug fix, or a PPO setting that stops training from working at all (for example the learning rate if values diverge). Every change is written in NOTES.md with the pilot curves that caused it, and the pilot is rerun. If all checks pass, nothing changes and the two runs continue.

## What is looked at but not judged

Differences between lam 0 and lam 1 (apples, last apples taken, zaps, regrowth) are shown in the plots. They are early, one seed each, and are not a result. No decision depends on them.
