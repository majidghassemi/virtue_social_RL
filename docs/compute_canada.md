# Running Phase 4 on Alliance (Compute Canada) clusters

Everything below runs from the repository root. Job scripts are in `scripts/slurm/`. Each one has a commented line `##SBATCH --account=def-YOURPI`. Change it to your group's account (shown by `sshare -U` or in CCDB) and remove one `#`.

## 1. Set up once, on a login node

Login nodes have internet access, compute nodes usually do not, so install everything here.

```bash
cd ~/projects/def-YOURPI/$USER          # or wherever you keep code
git clone --recurse-submodules https://github.com/majidghassemi/virtue_social_RL.git
cd virtue_social_RL
module load python/3.11                 # if this name is not found: module avail python
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[gpu,dev]"   # JAX 0.6.2 with CUDA 12, plus pytest
mkdir -p logs/slurm
```

`--recurse-submodules` matters: SocialJax is a submodule. If you cloned without it, run `git submodule update --init`.

## 2. Check the GPU and measure speed (about 15 to 30 minutes)

```bash
sbatch scripts/slurm/check_gpu.sh
```

When it finishes, the output is in `logs/slurm/check_gpu_<jobid>.out`. It should show:

1. the GPU name and `backend gpu`
2. the trainer tests passing (this runs the GPU code path, which was checked against the CPU path)
3. a benchmark line like `first call ... s, ... s per update, ... env steps/s, full run about X h`

Send me that file before step 3. The benchmark sets the job time.

## 3. Train the 9 teachers

Set `#SBATCH --time` in `scripts/slurm/train_prosocial.sh` from the benchmark (the estimate plus about 30%). Jobs of 24 hours or less usually start sooner. A run that hits the limit resumes from its last checkpoint when you resubmit the same index, for example `sbatch --array=4 scripts/slurm/train_prosocial.sh`.

```bash
sbatch scripts/slurm/train_prosocial.sh     # array 0-8: lam0 seeds 0-2, lam0.5 seeds 0-2, lam1 seeds 0-2
squeue -u $USER                             # job status
tail -f logs/slurm/prosocial_v1_*_0.out     # progress of one run
```

Each run writes to `runs/train/prosocial_v1/<condition>_seed<seed>/`. A run is finished when `params_final.msgpack` exists.

## 4. Train the 12 learners (after all teachers have finished)

The learners load `runs/train/prosocial_v1/lam0_seed*/` and `lam1_seed*/params_final.msgpack`. Benchmark them once in a GPU job:

```bash
.venv/bin/python scripts/train.py --config configs/train_social_v1.yaml --bench 3
```

Then set `--time` in `scripts/slurm/train_social.sh` and submit:

```bash
sbatch scripts/slurm/train_social.sh        # array 0-11, see the header for the order
```

## 5. Bring results back

The files needed for analysis are small:

```bash
tar czf prosocial_v1.tgz runs/train/prosocial_v1/*/{config.json,metrics.jsonl,params_final.msgpack,params_final.json}
```

Attach the archive in the chat, or copy it somewhere I can read.

## If something fails

* `JAX does not see a GPU`: send me the first lines of the check output (GPU name, driver version, the JAX line).
* Out of memory: lower `envs_per_map` in the config (24 is 288 parallel environments) and tell me, since it changes the batch size.
* Any other error: send the end of the `.out` file.
