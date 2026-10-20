"""Final audit: reconstruct the three runner versions recorded in the run manifest.

code_version() hashes src/ps2/*.py (sorted) followed by scripts/paper2/run_protocol.py. Removing the
--drain option, and then also next_job()/--next, from the current runner reproduces the two earlier
hashes exactly, which proves (i) the earlier runner files byte for byte and (ii) that src/ps2 was
identical for every recorded version.

    python3 scripts/paper2/audit_code_versions.py
"""
import difflib
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
OUT = R / "results/paper2/audit/code_versions"
OUT.mkdir(parents=True, exist_ok=True)
src = b"".join(p.read_bytes() for p in sorted((R / "src/ps2").glob("*.py")))
cur = (R / "scripts/paper2/run_protocol.py").read_text()
H = lambda t: hashlib.sha256(src + t.encode()).hexdigest()[:16]  # noqa: E731

drain_args = '    ap.add_argument("--drain", choices=["main", "ablations"])\n    ap.add_argument("--max-min", type=float, default=95)\n'
drain_blk = '''    if a.drain:   # claim jobs one after another (locks make parallel drains safe); none started after max-min
        t0 = time.time()
        while (time.time() - t0) / 60 < a.max_min and next_job(a.drain):
            pass
    elif a.next:
'''
d0ee = cur.replace(drain_args, "").replace(drain_blk, "    if a.next:\n")
nj = cur[cur.index("def next_job(kind):"):cur.index("def pilot():")]
e547 = d0ee.replace(nj, "").replace('    ap.add_argument("--next", choices=["main", "ablations"])\n', "") \
           .replace("    if a.next:\n        next_job(a.next)\n    elif a.pilot:\n", "    if a.pilot:\n")
versions = {"e547cdf35c1f9042": e547, "d0ee1435baa53e33": d0ee, "af8d235b9b547dfc": cur}
res = {}
for h, text in versions.items():
    res[h] = {"reconstructed_hash": H(text), "matches": H(text) == h}
    (OUT / f"run_protocol_{h}.py").write_text(text)
for a, b in [("e547cdf35c1f9042", "d0ee1435baa53e33"), ("d0ee1435baa53e33", "af8d235b9b547dfc")]:
    diff = difflib.unified_diff(versions[a].splitlines(True), versions[b].splitlines(True), f"run_protocol_{a}.py", f"run_protocol_{b}.py")
    (OUT / f"diff_{a[:4]}_to_{b[:4]}.patch").write_text("".join(diff))
(OUT / "reconstruction.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
