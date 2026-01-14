param(
  [string]$Config = "configs/multitask.yaml"
)
python -m chexpert_mvp.train --config $Config
