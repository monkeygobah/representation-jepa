Study configs for the embedding extraction and geometry analysis pipeline live
in this directory. One config represents one analysis study.

Use `/workspace/...` paths in study YAMLs for Docker portability. The config
loader maps those paths back to the current repo root when the checkout is not
literally mounted at `/workspace`.
