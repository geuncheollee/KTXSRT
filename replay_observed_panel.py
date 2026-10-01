"""Rebuild the panel from locally downloaded public source tables."""
from public_inputs import ROOT, build_panel

if __name__ == "__main__":
    panel = build_panel(ROOT / "data/public_sources/kric_carkind_raw.csv", ROOT / "data/public_sources/srt_20251231.csv")
    panel.to_csv(ROOT / "revision/data/monthly_panel_revision_v1.csv", index=False, encoding="utf-8-sig")
    print("Rebuilt 60 months from public-source files.", flush=True)
