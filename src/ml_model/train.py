"""
Step 2 of the fraud-risk model: train, evaluate, save.

    python src/ml_model/train.py

Reads   dataset/processed/paysim_features.parquet
Writes  src/ml_model/model.pkl                 the trained model (+ the settings needed to reuse it)
        results/accuracy.xlsx                  metrics per split (formulas for accuracy / precision / recall / F1)
        results/graphs/*.png                   precision-recall curve, confusion matrix, feature weights
        dataset/processed/paysim_meta.json     split boundaries and class counts

The model is a class-weighted logistic regression. It gives every transaction a fraud_risk between 0 and 1,
which the RL agent uses as a state feature and as a term in its reward.

Class imbalance (only ~0.13% of transactions are fraud) is handled inside the model with class_weight="balanced".
The data itself is NOT resampled, so validation and test keep the real fraud rate.
The decision threshold is chosen on the VALIDATION split (maximum F1) and then applied unchanged to the test split.
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, ...)
from config import GRAPHS, MODEL_PATH, OUT, RESULTS, TRAIN_MAX_STEP, VAL_MAX_STEP  # noqa: E402
from preprocessing import FEATURES, design_matrix  # noqa: E402


def write_accuracy_xlsx(rows, threshold, path):
    """One row per split. Counts are inputs (blue); accuracy, precision, recall, F1 and totals are FORMULAS."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Fraud model"
    base = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True)
    head_fill = PatternFill("solid", start_color="D9E2F3", end_color="D9E2F3")

    ws["A1"] = "Fraud-risk model: results per split (PaySim)"
    ws["A1"].font = Font(name="Arial", size=13, bold=True)
    ws["A2"] = "Decision threshold (chosen on the validation split to maximise F1)"
    ws["A2"].font = base
    ws["E2"] = threshold
    ws["E2"].font = Font(name="Arial", size=10, color="0000FF")   # blue = an input value
    ws["E2"].number_format = "0.0000"

    headers = ["Split", "Steps (hours)", "Transactions", "Fraud cases", "TP", "FP", "FN", "TN",
               "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=4, column=c, value=h)
        cell.font, cell.fill = bold, head_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for i, r in enumerate(rows):
        n = 5 + i
        ws.cell(n, 1, r["split"]); ws.cell(n, 2, r["steps"])
        ws.cell(n, 3, f"=E{n}+F{n}+G{n}+H{n}")                       # transactions = TP + FP + FN + TN
        ws.cell(n, 4, f"=E{n}+G{n}")                                 # fraud cases   = TP + FN
        for col, key in zip("EFGH", ("tp", "fp", "fn", "tn")):
            ws[f"{col}{n}"] = int(r[key])
            ws[f"{col}{n}"].font = Font(name="Arial", size=10, color="0000FF")
        ws[f"I{n}"] = f"=IFERROR((E{n}+H{n})/C{n},0)"
        ws[f"J{n}"] = f"=IFERROR(E{n}/(E{n}+F{n}),0)"
        ws[f"K{n}"] = f"=IFERROR(E{n}/(E{n}+G{n}),0)"
        ws[f"L{n}"] = f"=IFERROR(2*J{n}*K{n}/(J{n}+K{n}),0)"
        ws[f"M{n}"] = round(float(r["roc_auc"]), 4)
        ws[f"N{n}"] = round(float(r["pr_auc"]), 4)
        for col in "ABCDIJKLMN":
            ws[f"{col}{n}"].font = base
        for col in "CDEFGH":
            ws[f"{col}{n}"].number_format = "#,##0"
        for col in "IJKL":
            ws[f"{col}{n}"].number_format = "0.00%"
        for col in "MN":
            ws[f"{col}{n}"].number_format = "0.000"

    widths = [10, 14, 14, 12, 9, 9, 9, 12, 11, 11, 11, 9, 10, 10]
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + c)].width = w
    ws.row_dimensions[4].height = 30
    ws.freeze_panes = "A5"

    notes = wb.create_sheet("Notes")
    lines = [
        ("Notes and assumptions", True),
        (f"Generated {date.today().isoformat()} by src/ml_model/train.py (scikit-learn {sklearn.__version__}).", False),
        ("Data: PaySim synthetic mobile-money transactions (Kaggle ealaxi/paysim1), split by TIME: "
         f"train = steps 1-{TRAIN_MAX_STEP}, validation = {TRAIN_MAX_STEP + 1}-{VAL_MAX_STEP}, test = {VAL_MAX_STEP + 1}-743.", False),
        ("Model: logistic regression with class_weight='balanced'. Features: transaction type, log(amount), origin-balance "
         "error flag, origin-account-emptied flag, merchant-destination flag.", False),
        ("Blue cells are values written by the training script (confusion-matrix counts, threshold). Black cells are formulas.", False),
        ("Accuracy is shown for completeness only: fraud is ~0.13% of transactions, so a model that never flags anything "
         "already scores ~99.87%. Read precision, recall, F1 and PR-AUC instead.", False),
        ("Caveat: PaySim fraud follows a simple simulated pattern (the origin account is emptied), so scores look near-perfect. "
         "Treat them as a check that the pipeline works, not as a claim about real-world fraud detection.", False),
        ("Purpose in the project: the model output (fraud_risk, 0 to 1) is a state feature and a reward term for the RL agent.", False),
    ]
    for i, (text, is_bold) in enumerate(lines, 1):
        cell = notes.cell(i, 1, text)
        cell.font = Font(name="Arial", size=12 if i == 1 else 10, bold=is_bold)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    notes.column_dimensions["A"].width = 120
    wb.save(path)


def main():
    df = pd.read_parquet(OUT / "paysim_features.parquet")
    y = df["is_fraud"].to_numpy()
    tr, va, te = (df["split"] == s for s in ("train", "val", "test"))

    # ---- train (standardisation statistics come from TRAIN only) ----
    log_mean, log_std = float(df.loc[tr, "log_amount"].mean()), float(df.loc[tr, "log_amount"].std())
    X = design_matrix(df, log_mean, log_std)
    clf = LogisticRegression(class_weight="balanced", max_iter=500).fit(X[tr], y[tr])
    p = clf.predict_proba(X)[:, 1]

    # ---- threshold from the validation split, then frozen ----
    prec, rec, thr = precision_recall_curve(y[va], p[va])
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[int(np.argmax(f1[:-1]))])

    rows = []
    for name, mask, steps in (("train", tr, f"1-{TRAIN_MAX_STEP}"), ("val", va, f"{TRAIN_MAX_STEP + 1}-{VAL_MAX_STEP}"),
                              ("test", te, f"{VAL_MAX_STEP + 1}-{int(df['step'].max())}")):
        tn, fp, fn, tp = confusion_matrix(y[mask], p[mask] >= threshold, labels=[0, 1]).ravel()
        rows.append({"split": name, "steps": steps, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "roc_auc": roc_auc_score(y[mask], p[mask]), "pr_auc": average_precision_score(y[mask], p[mask])})
        print(f"{name:5s} PR-AUC {rows[-1]['pr_auc']:.3f}  ROC-AUC {rows[-1]['roc_auc']:.3f}  TP {tp} FP {fp} FN {fn}")

    # ---- save the model with everything needed to reuse it ----
    joblib.dump({"model": clf, "features": FEATURES, "log_amount_mean": log_mean, "log_amount_std": log_std,
                 "threshold": threshold, "trained_on": f"PaySim train split, steps 1-{TRAIN_MAX_STEP}",
                 "sklearn_version": sklearn.__version__}, MODEL_PATH)

    # ---- results ----
    RESULTS.mkdir(exist_ok=True)
    GRAPHS.mkdir(parents=True, exist_ok=True)
    write_accuracy_xlsx(rows, threshold, RESULTS / "accuracy.xlsx")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, mask in (("validation", va), ("test", te)):
        pr, rc, _ = precision_recall_curve(y[mask], p[mask])
        ax.plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(y[mask], p[mask]):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Fraud-risk model: precision-recall curve", ylim=(0, 1.02))
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(GRAPHS / "pr_curve.png", dpi=150); plt.close(fig)

    tn, fp, fn, tp = (rows[2][k] for k in ("tn", "fp", "fn", "tp"))
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not fraud", "fraud"], yticklabels=["not fraud", "fraud"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix, test split (threshold {threshold:.3f})")
    fig.tight_layout(); fig.savefig(GRAPHS / "confusion_matrix_test.png", dpi=150); plt.close(fig)

    order = np.argsort(clf.coef_[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(FEATURES)[order], clf.coef_[0][order], color="#4472C4")
    ax.set(xlabel="Weight in the model (log-odds)", title="Fraud-risk model: what drives the score")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(GRAPHS / "feature_weights.png", dpi=150); plt.close(fig)

    train = df[tr]
    n_pos, n_neg = int(train["is_fraud"].sum()), int((train["is_fraud"] == 0).sum())
    meta = {"rows_clean": int(len(df)),
            "split_steps": {"train": [1, TRAIN_MAX_STEP], "val": [TRAIN_MAX_STEP + 1, VAL_MAX_STEP], "test": [VAL_MAX_STEP + 1, int(df["step"].max())]},
            "rows_per_split": df["split"].value_counts().to_dict(),
            "fraud_per_split": df.groupby("split")["is_fraud"].sum().astype(int).to_dict(),
            "train_pos": n_pos, "train_neg": n_neg, "scale_pos_weight": round(n_neg / max(n_pos, 1), 2),
            "features_for_fraud_risk": FEATURES, "threshold": round(threshold, 4)}
    (OUT / "paysim_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", MODEL_PATH.name, "| results/accuracy.xlsx | results/graphs/{pr_curve,confusion_matrix_test,feature_weights}.png")


if __name__ == "__main__":
    main()
