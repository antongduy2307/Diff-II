import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ASSETS = (
    {
        "name": "datasets",
        "url": "https://huggingface.co/datasets/Luka-Wang/Diff-II/resolve/main/datasets.zip",
        "archive": "datasets.zip",
        "target": "datasets",
    },
    {
        "name": "ft_ti_db",
        "url": "https://huggingface.co/datasets/Luka-Wang/Diff-II/resolve/main/ft_ti_db.zip",
        "archive": "ft_ti_db.zip",
        "target": "ft_ti_db",
    },
    {
        "name": "suffix",
        "url": "https://huggingface.co/datasets/Luka-Wang/Diff-II/resolve/main/suffix.zip",
        "archive": "suffix.zip",
        "target": "suffix",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the Diff-II assets required for inference."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload archives and re-extract targets even if they already exist.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded zip files after extraction.",
    )
    return parser.parse_args()


def download_with_progress(url: str, destination: Path) -> None:
    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100 // total_size
        message = f"\rDownloading {destination.name}: {percent:3d}% ({downloaded}/{total_size} bytes)"
        sys.stdout.write(message)
        sys.stdout.flush()

    urllib.request.urlretrieve(url, destination, reporthook=report)
    sys.stdout.write("\n")


def ensure_asset(repo_root: Path, asset: dict[str, str], force: bool, keep_archives: bool) -> None:
    archive_path = repo_root / asset["archive"]
    target_path = repo_root / asset["target"]

    if target_path.exists() and not force:
        print(f"[skip] {asset['name']} already exists at {target_path}")
        return

    if archive_path.exists() and force:
        archive_path.unlink()

    if target_path.exists() and force:
        shutil.rmtree(target_path)

    if not archive_path.exists():
        print(f"[download] {asset['url']}")
        download_with_progress(asset["url"], archive_path)
    else:
        print(f"[reuse] archive already exists: {archive_path}")

    print(f"[extract] {archive_path.name} -> {target_path.name}")
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(repo_root)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Expected extracted directory '{target_path}' was not created by {archive_path.name}."
        )

    if not keep_archives:
        archive_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    for asset in ASSETS:
        ensure_asset(repo_root, asset, force=args.force, keep_archives=args.keep_archives)

    print("[ok] All required inference assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
