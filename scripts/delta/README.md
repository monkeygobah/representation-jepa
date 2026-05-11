# DeltaAI Setup

These scripts keep the DeltaAI setup explicit and project-aware. The default
project is `bhgj`, which maps to:

```bash
DELTA_ACCOUNT=bhgj-dtai-gh
DELTA_WORK=/work/nvme/bhgj/$USER/representation-jepa
```

To switch projects later:

```bash
export DELTA_PROJECT=bhgk
```

## Interactive Setup

Start a GPU shell from the login node:

```bash
scripts/delta/srun_interactive.sh
```

Inside the compute-node shell, load the repo environment:

```bash
cd /u/$USER/representation-jepa
source scripts/delta/env.sh
```

Install the missing Python packages once per project:

```bash
scripts/delta/install_python_deps.sh
```

The extra packages are installed under:

```bash
/work/nvme/$DELTA_PROJECT/$USER/python-userbase
```

## Storage Layout

Use home for code:

```bash
/u/$USER/representation-jepa
```

Use NVMe work storage for heavy training files:

```bash
/work/nvme/$DELTA_PROJECT/$USER/representation-jepa/data
/work/nvme/$DELTA_PROJECT/$USER/representation-jepa/runs
/work/nvme/$DELTA_PROJECT/$USER/representation-jepa/models
```

Delta-specific training configs should use those absolute paths directly.

## Batch Training

Submit a config to the normal training partition:

```bash
sbatch -A bhgj-dtai-gh scripts/delta/train_ssl.sbatch configs/delta/my_config.yaml
```

The batch script intentionally does not hard-code an account so it can be
reused across ACCESS projects.
