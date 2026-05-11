# EP/BHEP 1m 50k-step LE-JEPA follow-up

Four matched LE-JEPA runs on the 1m `subset6_minus_7` training set:

- `eppartial` regularizer with random init
- `eppartial` regularizer with ImageNet init
- `bhep` regularizer with random init
- `bhep` regularizer with ImageNet init

Run names keep the existing parser shape:

`geometry-fixedcompute-{scale}-{objective}-{init}-{budget}`

The objective slot is `eppartial` or `bhep`, while the actual training method
remains `ssl.method: "lejepa"`.
