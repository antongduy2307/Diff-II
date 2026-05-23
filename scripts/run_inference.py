import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASET_CHOICES = ("aircraft", "car", "cub", "flower", "pet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Diff-II inference pipeline without editing shell scripts."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--shot", default="5shot", help="Few-shot split, for example 5shot or 10shot.")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids to expose via CUDA_VISIBLE_DEVICES, for example 0 or 0,1,2,3.",
    )
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--source-dir", default="datasets/")
    parser.add_argument("--inversion-dir", default="inversions/")
    parser.add_argument("--output-dir", default="syn/")
    parser.add_argument("--inversion-step", type=int, default=200)
    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--generation-step", type=int, default=25)
    parser.add_argument("--generation-guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--expansion-rate", type=float, default=5.0)
    parser.add_argument(
        "--stage",
        choices=("all", "inversion", "generation"),
        default="all",
        help="Run the full pipeline or only one stage.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not run the asset downloader before inference.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify assets, then stop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of this script.",
    )
    return parser.parse_args()


def run_command(command: list[str], repo_root: Path, env: dict[str, str], dry_run: bool) -> None:
    print(f"[cmd] {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    return (repo_root / path_value).resolve()


def verify_required_inputs(repo_root: Path, dataset: str, shot: str, source_dir: str) -> None:
    source_root = resolve_repo_path(repo_root, source_dir)
    train_dir = source_root / dataset / shot / "train"
    lora_file = repo_root / "ft_ti_db" / dataset / shot / "pytorch_lora_weights.safetensors"
    suffix_file = repo_root / "suffix" / dataset / f"{shot}.txt"

    missing = [path for path in (train_dir, lora_file, suffix_file) if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required downloaded assets. Expected the following paths to exist:\n"
            f"{missing_text}"
        )


def verify_inversion_inputs(repo_root: Path, dataset: str, shot: str, inversion_dir: str) -> None:
    inversion_root = resolve_repo_path(repo_root, inversion_dir) / dataset / shot

    if not inversion_root.exists():
        raise FileNotFoundError(f"Inversion directory does not exist: {inversion_root}")


def verify_outputs(
    repo_root: Path, dataset: str, shot: str, strength: float, inversion_dir: str, output_dir: str
) -> None:
    inversion_root = resolve_repo_path(repo_root, inversion_dir) / dataset / shot
    generation_root = resolve_repo_path(repo_root, output_dir) / dataset / shot / f"ours_{strength}"

    if not inversion_root.exists():
        raise FileNotFoundError(f"Inversion output directory was not created: {inversion_root}")
    if not generation_root.exists():
        raise FileNotFoundError(f"Generation output directory was not created: {generation_root}")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    env = os.environ.copy()

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided via --gpus.")

    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    worker_count = len(gpu_ids)
    python_executable = sys.executable

    if not args.skip_download:
        download_command = [python_executable, "scripts/download_assets.py"]
        if args.dry_run:
            print("[info] Asset download is enabled.")
        run_command(download_command, repo_root, env, args.dry_run)

    if args.download_only:
        return 0

    if args.dry_run:
        print("[info] Skipping asset existence checks because --dry-run was requested.")
    else:
        verify_required_inputs(repo_root, args.dataset, args.shot, args.source_dir)

    inversion_command = [
        python_executable,
        "get_inversion.py",
        "--datasets",
        args.dataset,
        "--shot",
        args.shot,
        "--model_id",
        args.model_id,
        "--inversion_step",
        str(args.inversion_step),
        "--condiction_scale",
        str(args.inversion_guidance_scale),
        "--source_dir",
        args.source_dir,
        "--des_dir",
        args.inversion_dir,
        "--n_workers",
        str(worker_count),
    ]
    generation_command = [
        python_executable,
        "interpolation_le.py",
        "--datasets",
        args.dataset,
        "--shot",
        args.shot,
        "--model_id",
        args.model_id,
        "--inversion_step",
        str(args.generation_step),
        "--condiction_scale",
        str(args.generation_guidance_scale),
        "--ddim_inversion_dir",
        args.inversion_dir,
        "--des_dir",
        args.output_dir,
        "--strength",
        str(args.strength),
        "--expansion_rate",
        str(args.expansion_rate),
        "--n_workers",
        str(worker_count),
    ]

    if args.stage in ("all", "inversion"):
        run_command(inversion_command, repo_root, env, args.dry_run)

    if args.stage in ("all", "generation"):
        if not args.dry_run:
            verify_inversion_inputs(repo_root, args.dataset, args.shot, args.inversion_dir)
        run_command(generation_command, repo_root, env, args.dry_run)

    if not args.dry_run and args.stage == "all":
        verify_outputs(
            repo_root,
            args.dataset,
            args.shot,
            args.strength,
            args.inversion_dir,
            args.output_dir,
        )
        print("[ok] Inference outputs were created.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
