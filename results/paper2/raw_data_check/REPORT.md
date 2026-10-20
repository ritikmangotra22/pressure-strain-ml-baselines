# Raw-data verification, 20 Oct 2026
Machine: Apple Silicon, macOS 26.2, Python 3.13.16 (uv), Paper 1 commit 5b2752d.
- Lee & Moser: 35/35 files byte-identical to the committed checksum list (order differs only).
- Hoyas & Jimenez: 67/67 checksums OK.
- Paper 1 tests: 20 passed, 0 skipped, 0 failed (pytest_log.txt).
- Rebuilt tables: same rows, columns and order as the frozen tables; only III_b differs
  (max abs 1.1e-16; table_comparison.txt). Byte hashes differ (hashes.txt) for that reason only.
- Paper 2 does not use III_b; all columns it uses are identical.
VERDICT: VERIFIED at numerical precision.
