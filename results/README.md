# Nexus Benchmark Results

This directory contains the replication and canonical evaluation artifacts for the Nexus v2.0 paper. All stale, pre-v2 benchmark logs and legacy intermediate files have been removed or consolidated.

---

## Directory Structure

*   [`v2.0_canonical/`](v2.0_canonical/): The single source of truth for the paper's final measurements. Includes:
    *   `raw/`: Raw JSON logs from each independent evaluation run.
    *   `README.md`: Mapping of raw JSON files to paper tables and figures.
    *   `EVIDENCE_LEDGER.csv`: Master audit registry mapping claims to code and artifacts.
    *   `REFERENCE_AUDIT.csv`: Verification trace database tracking inline values in LaTeX.
*   [`v2_phase3_vtrans/`](v2_phase3_vtrans/): Reports and capstone validation runs evaluating transposed-V splicing on soft-capped architectures (specifically Gemma-2-9B).

---

## Local Replication

To run the evaluation harnesses and reproduce each table and figure, follow the step-by-step instructions in the root [REPRODUCE.md](../REPRODUCE.md).
