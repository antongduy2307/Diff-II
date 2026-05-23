# Diff-II

Code and teammate-ready inference setup for the CVPR 2025 paper **Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification**.

The original paper PDF is kept in the repository root for reference:
`Inversion Circle Interpolation Diffusion-based Image Augmentation for Data-scarce Classification.pdf`

![](teaser.jpg)

Paper link: [arXiv 2408.16266](https://arxiv.org/pdf/2408.16266)

## What this repo now supports

This fork is prepared so a teammate can:

1. clone the repo,
2. create the environment,
3. download the required assets,
4. run the inference pipeline without editing shell scripts.

The original research scripts are still here. The recommended entrypoints for setup and inference are:

- `python scripts/download_assets.py`
- `python scripts/run_inference.py ...`

## Prerequisites

- NVIDIA GPU with CUDA support
- Conda or Mamba
- Python 3.9

The paper authors ran larger experiments on 8 GPUs, but the helper script also supports smaller GPU sets by remapping visible devices.

## Quickstart

### 1. Create the environment

```bash
conda env create -f environment.yaml
conda activate diff-ii
```

If dependency solving is slow:

```bash
conda config --set solver libmamba
```

### 2. Download the required inference assets

Cross-platform helper:

```bash
python scripts/download_assets.py
```

This downloads and extracts:

- `datasets/`
- `ft_ti_db/`
- `suffix/`

### 3. Run inference

Example: run the full pipeline on the `aircraft` `5shot` split with one GPU:

```bash
python scripts/run_inference.py --dataset aircraft --shot 5shot --gpus 0
```

Example: run with four GPUs:

```bash
python scripts/run_inference.py --dataset aircraft --shot 5shot --gpus 0,1,2,3
```

Example: regenerate only the image synthesis stage after inversions already exist:

```bash
python scripts/run_inference.py --dataset aircraft --shot 5shot --gpus 0,1 --stage generation
```

### 4. Expected outputs

After a successful full run, the main generated artifacts are:

- `inversions/<dataset>/<shot>/...`
- `syn/<dataset>/<shot>/ours_<strength>/...`

## Main inference knobs

You can adjust the important settings directly from the wrapper script CLI:

- `--dataset`: `aircraft`, `car`, `cub`, `flower`, `pet`
- `--shot`: usually `5shot` or `10shot`
- `--gpus`: visible GPU ids, for example `0` or `0,1,2,3`
- `--strength`: interpolation strength for generation
- `--expansion-rate`: number of generated samples relative to source images
- `--stage`: `all`, `inversion`, or `generation`

See all options with:

```bash
python scripts/run_inference.py --help
```

## Recommended teammate workflow

When a teammate clones this repo, the default path should be:

```bash
conda env create -f environment.yaml
conda activate diff-ii
python scripts/download_assets.py
python scripts/run_inference.py --dataset aircraft --shot 5shot --gpus 0
```

## Notes

- `scripts/download.sh` and `scripts/run.sh` are the original shell-oriented helpers from the source repo.
- `scripts/run_inference.py` is the preferred wrapper because it avoids manual edits and works without Bash.
- `get_suffix.py` is only needed if you want to regenerate suffix prompts instead of using the downloaded ones.
- To run `get_suffix.py`, set `DIFF_II_OPENAI_API_KEY` in your environment first.

## Citing Diff-II

```bibtex
@article{wang2024improving,
  title={Improving diffusion-based data augmentation with inversion spherical interpolation},
  author={Wang, Yanghao and Chen, Long},
  journal={arXiv preprint arXiv:2408.16266},
  year={2024}
}
```

