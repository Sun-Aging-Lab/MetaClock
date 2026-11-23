"""Reproduce the three Figure 5 screens and the 68 external MetaClock-19 contrasts."""

import argparse
import fcntl
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .manuscript_screen import RESULT_NAME, write_external_results
from .paths import PROJECT_ROOT, plot_input, relative_path, repo_path
from .plot_geo_biometaclock_volcano_fdr import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=repo_path, default=PROJECT_ROOT / "results" / "figure5")
    parser.add_argument("--run-dir", type=plot_input,
                        help="Re-score complete component and sample-design shards instead of using donor summaries.")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summaries = Path(__file__).resolve().parent / "screen_summaries"
    with TemporaryDirectory(dir=output) as folder:
        dest = Path(folder)
        write_external_results(dest, args.run_dir)
        for cohort, model in (("fhs", "metaclock5"), ("whi", "metaclock19")):
            plot_results(summaries / f"{cohort}_{model}.csv", dest / f"{cohort}_volcano", cohort=cohort)
        plot_results(dest / RESULT_NAME, dest / "external_volcano", combined_geo_translage=True)
        # plot_results records paths relative to the repository. Publish stable paths
        # in its audit before atomically replacing any of the prior run's files.
        for audit in dest.glob("*_audit.json"):
            record = json.loads(audit.read_text())
            for key in ("results", "output_pdf", "output_svg", "audit_json"):
                location = PROJECT_ROOT / record[key]
                if location.parent == dest:
                    record[key] = relative_path(output / location.name)
            audit.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        summary = dest / "external_analysis_summary.json"
        with (output / ".figure5-publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for path in sorted(dest.iterdir()):
                if path != summary:
                    path.replace(output / path.name)
            summary.replace(output / summary.name)


if __name__ == "__main__":
    main()
