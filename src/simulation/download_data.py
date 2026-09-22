"""
Step 1 - download the data.

  python src/simulation/download_data.py                 # PaySim + first 6 Alibaba call-graph files (~1 GB)
  python src/simulation/download_data.py --calls 12      # more call-graph files = more time windows (~177 MB each)
  python src/simulation/download_data.py --extra         # also Node_0, MSResource_0, MSRTQps_0 (~3 GB, for CPU/memory features later)
  python src/simulation/download_data.py --skip-paysim   # only Alibaba

Downloads resume automatically if the connection drops: just run the same command again.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
import argparse
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from config import ALI_RAW, PAYSIM_RAW

# Official base URL, taken from the fetchData.sh in the alibaba/clusterdata repo.
BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces"


def get_paysim():
    if list(PAYSIM_RAW.glob("*.csv")):
        print("PaySim: already present, skipping.")
        return
    PAYSIM_RAW.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub  # pip install kagglehub

        src = Path(kagglehub.dataset_download("ealaxi/paysim1"))
        for f in src.glob("*.csv"):
            shutil.copy(f, PAYSIM_RAW / f.name)
        print("PaySim: done ->", PAYSIM_RAW)
    except Exception as e:  # noqa: BLE001 - any failure should fall through to the manual route
        print(f"PaySim auto-download failed: {e}")
        print("Manual fallback:")
        print("  1. Open https://www.kaggle.com/datasets/ealaxi/paysim1 and click Download (free Kaggle login)")
        print(f"  2. Unzip the CSV into: {PAYSIM_RAW}")


def fetch(url: str, dest: Path):
    """Download with resume support (HTTP Range) and a simple progress print."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable -> file is already complete
            return
        raise
    mode = "ab" if have and resp.status == 206 else "wb"
    done = have if mode == "ab" else 0
    next_report = done + 50_000_000
    with open(dest, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"    {dest.name}: {done / 1e6:.0f} MB")
                next_report += 50_000_000


def get_alibaba(files):
    for table, name in files:
        folder = ALI_RAW / table
        archive = folder / f"{name}.tar.gz"
        already = list(folder.rglob(f"{name}*.csv")) if folder.exists() else []
        if already:
            print(f"{name}: already extracted, skipping.")
            continue
        print(f"{name}: downloading ...")
        fetch(f"{BASE}/{table if table != 'Node' else 'node'}/{name}.tar.gz", archive)
        print(f"{name}: extracting ...")
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(folder, filter="data")  # safer extraction on Python 3.12+
            except TypeError:
                tar.extractall(folder)
        archive.unlink()  # free the disk space; the CSVs are what we need
    print("Alibaba: done ->", ALI_RAW)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=6, help="number of MSCallGraph files (each = 5 min of trace)")
    ap.add_argument("--extra", action="store_true", help="also fetch Node_0, MSResource_0, MSRTQps_0")
    ap.add_argument("--skip-paysim", action="store_true")
    a = ap.parse_args()

    if not a.skip_paysim:
        get_paysim()
    wanted = [("MSCallGraph", f"MSCallGraph_{i}") for i in range(a.calls)]
    if a.extra:
        wanted += [("Node", "Node_0"), ("MSResource", "MSResource_0"), ("MSRTQps", "MSRTQps_0")]
    get_alibaba(wanted)
