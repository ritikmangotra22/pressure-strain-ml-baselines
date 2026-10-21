"""Test-set guard: the Moser-Kim-Mansour confirmatory table may be read only by the scripts that
build it and run the single confirmatory evaluation, and its content must match the frozen hash."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {"scripts/paper3/build_mkm_tables.py", "scripts/paper3/confirmatory_test.py", "tests/paper3/test_mkm_guard.py"}
FROZEN = "f13dd1d097bc77c47b68e57d50661fe70f1b8ced24bc284cb500718fbb3353d7"


def test_only_allowed_files_reference_the_confirmatory_table():
    hits = set()
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        if "processed_table_mkm.csv" in p.read_text():
            hits.add(str(p.relative_to(ROOT)))
    assert hits <= ALLOWED, f"confirmatory table referenced outside the allowed scripts: {sorted(hits - ALLOWED)}"


def test_confirmatory_table_hash_frozen():
    p = ROOT / "results/paper3/mkm_data/processed_table_mkm.csv"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == FROZEN
