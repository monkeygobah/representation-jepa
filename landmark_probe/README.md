# Landmark Probe

Config-driven downstream landmark probe pipeline for frozen SSL backbones.

Stages:

- `prepare`: rebuild canonical `224x224` periorbital dataset with fixed splits
- `extract`: reopen training runs/checkpoints and write pooled backbone embeddings
- `probe`: train an MLP landmark regressor on frozen embeddings
- `aggregate`: summarize completed probe runs into flat CSV tables

Entry points live in `scripts/`:

- `run_landmark_prepare.py`
- `run_landmark_extract.py`
- `run_landmark_probe.py`
- `run_landmark_aggregate.py`
