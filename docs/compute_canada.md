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

```bash
module load python/3.11
virtualenv --no-download .venv
.venv/bin/pip install --no-index --upgrade pip
.venv/bin/pip install -e ".[gpu,dev]"
.venv/bin/python -c "import virtue_rl, socialjax, jax; print('ok', jax.__version__)"
```

The last line should print `ok 0.6.2`. The login node has no GPU, so JAX falls back to the CPU there and may print a warning about it.

## 5. Check the GPU and measure speed (one job, up to 1 hour)

```bash
sbatch scripts/slurm/check_gpu.sh
squeue -u $USER                       # PD = waiting, R = running
```

When it is gone from `squeue`, look at `logs/slurm/check_gpu_<jobid>.out`. It should show the H100, `backend gpu`, the trainer tests passing, and a line like:

```
first call (compile + 1 update) ...s, ...s per update, ... env steps/s, full run about X h
```

Send that file back before going on. The benchmark sets the time limits for steps 6 and 7.

## 6. Train the 9 teachers

After the time limit in `scripts/slurm/train_prosocial.sh` is set:

```bash
sbatch scripts/slurm/train_prosocial.sh       # array 0-8: lam0 seeds 0-2, lam0.5 seeds 0-2, lam1 seeds 0-2
squeue -u $USER
tail -f logs/slurm/prosocial_v1_*_0.out       # progress of the first run (Ctrl-C to stop watching)
```

Each run writes to `runs/train/prosocial_v1/<condition>_seed<seed>/` and is finished when `params_final.msgpack` exists. A run that hits the time limit resumes from its checkpoint when you resubmit its index, for example `sbatch --array=4 scripts/slurm/train_prosocial.sh`.

## 7. Train the 12 learners (after all 9 teachers have finished)

```bash
salloc --gpus-per-node=h100:1 --cpus-per-task=4 --mem=32G --time=0:30:00
.venv/bin/python scripts/train.py --config configs/train_social_v1.yaml --bench 3
exit
```

Send the benchmark line, set the time in `scripts/slurm/train_social.sh`, then:

```bash
sbatch scripts/slurm/train_social.sh          # array 0-11, order in the script header
```

## 8. Bring results back

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
