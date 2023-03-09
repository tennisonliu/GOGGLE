# 3rd Party
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Synthcity
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

generators = Plugins()


def evaluate_baselines(data, seed):
    quality_evaluator = eval_statistical.AlphaPrecision()
    xgb_evaluator = eval_performance.PerformanceEvaluatorXGB()
    linear_evaluator = eval_performance.PerformanceEvaluatorLinear()
    mlp_evaluator = eval_performance.PerformanceEvaluatorMLP()

    xgb_detector = eval_detection.SyntheticDetectionXGB()
    mlp_detector = eval_detection.SyntheticDetectionMLP()
    gmm_detector = eval_detection.SyntheticDetectionGMM()

    X_train, X_test = train_test_split(
        data, random_state=seed + 42, test_size=0.33, shuffle=False
    )

    results = {}

    for model in ["nflow", "ctgan", "tvae", "bayesian_network", "copulagan"]:
        gen = generators.get(model, device="cpu")
        gen.fit(X_train)
        X_synth = gen.generate(count=X_test.shape[0])

        X_test_loader = GenericDataLoader(
            X_test,
            target_column="target",
        )
        X_synth_loader = GenericDataLoader(
            X_synth,
            target_column="target",
        )

        xgb_score = xgb_evaluator.evaluate(X_test_loader, X_synth_loader)
        linear_score = linear_evaluator.evaluate(X_test_loader, X_synth_loader)
        mlp_score = mlp_evaluator.evaluate(X_test_loader, X_synth_loader)
        xgb_det = xgb_detector.evaluate(X_test_loader, X_synth_loader)
        mlp_det = mlp_detector.evaluate(X_test_loader, X_synth_loader)
        gmm_det = gmm_detector.evaluate(X_test_loader, X_synth_loader)
        data_qual = quality_evaluator.evaluate(X_test_loader, X_synth_loader)
        naive_data_qual = {k: v for (k, v) in data_qual.items() if "naive" in k}

        gt_perf = np.mean([xgb_score["gt"], linear_score["gt"], mlp_score["gt"]])
        synth_perf = np.mean(
            [xgb_score["syn_ood"], linear_score["syn_ood"], mlp_score["syn_ood"]]
        )
        det_score = np.mean([xgb_det["mean"], gmm_det["mean"], mlp_det["mean"]])
        qual_score = np.mean(list(naive_data_qual.values()))

        results[model] = [gt_perf, synth_perf, det_score, qual_score]

    return results


if __name__ == "__main__":

    dataset = "breast"
    data_path = Path("src/goggle/exps/data/breast_cancer_2.csv").resolve()

    X = pd.read_csv(data_path)

    ind = list(range(len(X.columns)))
    ind = [x for x in ind if x != X.columns.get_loc("target")]
    col_list = X.columns[ind]
    ct = ColumnTransformer(
        [("scaler", StandardScaler(), col_list)], remainder="passthrough"
    )

    X_ = ct.fit_transform(X)
    X = pd.DataFrame(X_, index=X.index, columns=X.columns)

    X.head()

    print(evaluate_baselines(X, 0))
