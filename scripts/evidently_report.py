from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise SystemExit(
            "Install optional dependencies first: pip install evidently==0.4.40 pandas==2.2.3"
        ) from exc

    reference_data = joblib.load(ROOT / "reference_stats.joblib")
    reference = pd.DataFrame(
        reference_data["X"],
        columns=reference_data["feature_names"],
    )

    rng = np.random.default_rng(0)
    current = reference.sample(n=200, random_state=0, replace=True).copy()
    current["petal_length"] = current["petal_length"] + 1.5
    current["petal_width"] = current["petal_width"] + rng.normal(
        loc=0.5,
        scale=0.1,
        size=len(current),
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    output = ROOT / "drift_report.html"
    report.save_html(str(output))
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
