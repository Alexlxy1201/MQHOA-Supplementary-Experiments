# Presentation after the completed campaign

After `python -m mqhoa_supp analyze --out <runs>` finishes, run:

```bash
python analysis_tools/finalize.py <runs>
```

This adds a Chinese result summary, seed/instance-level distribution plots, and the verified public code URL to the reviewer-response draft. It reads completed CSV/JSONL only; it does not evaluate schedules or change optimisation parameters. Its own SHA256 and input hashes are saved in `analysis/final_presentation_manifest.json`. The frozen experiment-code hash remains separate.

The extra distributions are descriptive: seed-level within each archived instance, independent-instance means for the synthetic set. They are not extra independent experiments or additional hypothesis tests.

Once the full quality matrix is complete, `python analysis_tools/quality_preview.py <runs>` can compute its predeclared quality statistics while the separate timing phase continues. It writes `quality_preview/` with an explicit provisional label. The final combined report still requires the complete timing matrix. Standalone synthetic configurations must also pass `--config configs/synthetic_only.json` to the preview command.
