Round 1 baseline pretraining configs for the embedding-geometry table.

These cover the 9 training runs:
- `lejepa` with `V=2`
- `infonce` with `V=2`
- `vicreg` with `V=2`

Each objective is paired with:
- `random` init
- `imagenet` init
- `seg_init` init

The remaining 3 table rows are the no-pretraining backbone baselines and do not
need `train_ssl.py` configs.
