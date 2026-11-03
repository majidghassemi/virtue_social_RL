# Running Phase 4 on Nibi (Alliance, Waterloo)

Nibi's GPU nodes have 8 NVIDIA H100 SXM (80 GB) each. The job scripts in `scripts/slurm/` ask for one H100 per job (`--gpus-per-node=h100:1`). Run every command from the repository root.

## 1. Log in

```bash
ssh YOURUSER@nibi.alliancecan.ca      # nibi.sharcnet.ca also appears in SHARCNET's guide
```

Approve the multifactor prompt (Duo).

## 2. Find your account and project space

```bash
ls ~/projects        # e.g. def-PINAME, maybe also rrg-... if the group has a RAC allocation
```

Use the `def-...` name unless your supervisor tells you to use an `rrg-...` one. Then make it the default for every job, so the scripts need no editing:

```bash
echo 'export SBATCH_ACCOUNT=def-PINAME'  >> ~/.bashrc
echo 'export SALLOC_ACCOUNT=def-PINAME'  >> ~/.bashrc
source ~/.bashrc
```

## 3. Get the code (login node, which has internet)

```bash
cd ~/projects/def-PINAME/$USER        # create it with mkdir -p if it does not exist
git clone --recurse-submodules https://github.com/majidghassemi/virtue_social_RL.git
cd virtue_social_RL
mkdir -p logs/slurm                   # Slurm does not create the log folder itself
```

If the repository is private, git asks for your GitHub username and a personal access token (not your password).

## 4. Python environment (login node, about 5 to 10 minutes)

Alliance's pip ignores the usual binary wheels from PyPI, so compiled packages come from the Alliance wheelhouse. `requirements-nibi.txt` lists versions that exist there (JAX 0.6.0 instead of 0.6.2, see the file header). The test suite passes with them.

```bash
module load python/3.11
virtualenv --no-download .venv               # skip if .venv already exists
.venv/bin/pip install --no-index --upgrade pip
.venv/bin/pip install -r requirements-nibi.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -c "import virtue_rl, socialjax, jax; print('ok', jax.__version__)"
```

The last line should print `ok 0.6.0`. The login node has no GPU, so JAX may print a warning that it falls back to the CPU.

## 5. Check the GPU and measure speed (one job, up to 1 hour)

```bash
sbatch scripts/slurm/check_gpu.sh
squeue -u $USER                       # PD = waiting, R = running
```

When it is gone from `squeue`, look at `logs/slurm/check_gpu_<jobid>.out`. It should show the H100, `backend gpu` (and whether that needed `module load cuda`), the trainer tests passing, and a line like:

```
first call (compile + 1 update) ...s, ...s per update, ... env steps/s, full run about X h
```

Send that file back before going on. The benchmark sets the time limits for steps 6 and 7.

## 6. Pilot: the first 20% of two teacher runs

Before spending the allocation on all 9 runs, run the first 70 of 347 updates of two of them (lam 0 seed 0, lam 1 seed 0). What the pilot checks, and what may change if it fails, is fixed in `docs/pilot.md`.

```bash
sbatch scripts/slurm/pilot_prosocial.sh
```

When both have finished (`stopped at update 70` at the end of their logs):

```bash
.venv/bin/python scripts/plot_training.py runs/train/prosocial_v1
```

Send the printed checks and `runs/train/prosocial_v1/curves.png`. The two runs keep their checkpoints, so step 7 continues them instead of starting again.

## 7. Train the 9 teachers

After the time limit in `scripts/slurm/train_prosocial.sh` is set:

```bash
sbatch scripts/slurm/train_prosocial.sh       # array 0-8: lam0 seeds 0-2, lam0.5 seeds 0-2, lam1 seeds 0-2
squeue -u $USER
tail -f logs/slurm/prosocial_v1_*_0.out       # progress of the first run (Ctrl-C to stop watching)
```

Each run writes to `runs/train/prosocial_v1/<condition>_seed<seed>/` and is finished when `params_final.msgpack` exists. A run that hits the time limit resumes from its checkpoint when you resubmit its index, for example `sbatch --array=4 scripts/slurm/train_prosocial.sh`.

## 8. Train the 12 learners (after all 9 teachers have finished)

```bash
salloc --gpus-per-node=h100:1 --cpus-per-task=4 --mem=32G --time=0:30:00
.venv/bin/python scripts/train.py --config configs/train_social_v1.yaml --bench 3
exit
```

Send the benchmark line, set the time in `scripts/slurm/train_social.sh`, then:

```bash
sbatch scripts/slurm/train_social.sh          # array 0-11, order in the script header
```

## 9. Bring results back

```bash
tar czf prosocial_v1.tgz runs/train/prosocial_v1/*/{config.json,metrics.jsonl,params_final.msgpack,params_final.json}
```

Download it from your laptop with `scp YOURUSER@nibi.alliancecan.ca:~/projects/def-PINAME/$USER/virtue_social_RL/prosocial_v1.tgz .` and attach it in the chat.

## If something fails

* `module load python/3.11` not found: `module avail python` and use a 3.11 or 3.12 version (the project needs Python 3.10 to 3.12).
* `pip install` fails on one package: send the last 30 lines.
* `JAX does not see a GPU` in the check job: send the first 10 lines of the output (GPU name, driver version, JAX line).
* Out of memory: send the error. Do not change `envs_per_map` without telling me, since it changes the batch size.
* `sbatch: error: ... account`: `SBATCH_ACCOUNT` is not set in that shell (`echo $SBATCH_ACCOUNT`).
