"""Download the Lee & Moser (2015) channel-flow files needed for Paper 1.

Run this on your own computer (standard-library Python 3, no packages needed):

    python download_lm.py --out lee_moser_2015

It fetches 35 small text files (~3 MB in total) from the Oden Institute server
and writes a SHA-256 checksum list. Then zip the folder and send it back.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://turbulence.oden.utexas.edu/channel2015/data/"
LABELS = ["0180", "0550", "1000", "2000", "5200"]
TAGS = ["mean_prof", "vel_fluc_prof"] + [f"RSTE_{c}_prof" for c in ("uu", "vv", "ww", "uv", "k")]


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for lab in LABELS:
        for tag in TAGS:
            name = f"LM_Channel_{lab}_{tag}.dat"
            dest = out / name
            if not dest.exists():
                print("downloading", name)
                urllib.request.urlretrieve(BASE + name, dest)
            lines.append(f"{hashlib.sha256(dest.read_bytes()).hexdigest()}  {name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"Done: {len(lines)} files in {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lee_moser_2015")
    main(ap.parse_args().out)
