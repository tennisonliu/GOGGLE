import sys
import json
import numpy as np

import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.metrics import eval_statistical
from synthcity.metrics import eval_performance
from synthcity.metrics import eval_detection

from sklearn.model_selection import train_test_split

generators = Plugins()


def evaluate_baselines(data, dataset_name, seed):
    quality_evaluator = eval_statistical.AlphaPrecision()
    xgb_evaluator = eval_performance.PerformanceEvaluatorXGB()
    linear_evaluator = eval_performance.PerformanceEvaluatorLinear()
    mlp_evaluator = eval_performance.PerformanceEvaluatorMLP()

    xgb_detector = eval_detection.SyntheticDetectionXGB()
    mlp_detector = eval_detection.SyntheticDetectionMLP()
    gmm_detector = eval_detection.SyntheticDetectionGMM()

    X_train, X_test = train_test_split(
        data, random_state=seed+42, test_size=0.33, shuffle=False)

    results = {}

    for model in ['nflow', 'ctgan', 'tvae', 'bayesian_network', 'copulagan']:
        gen = generators.get(model)
        gen.fit(X_train)
        X_synth = gen.generate(count=X_test.shape[0])

        xgb_score = xgb_evaluator.evaluate(X_test, X_synth)
        linear_score = linear_evaluator.evaluate(X_test, X_synth)
        mlp_score = mlp_evaluator.evaluate(X_test, X_synth)
        xgb_det = xgb_detector.evaluate(X_test, X_synth)
        mlp_det = mlp_detector.evaluate(X_test, X_synth)
        gmm_det = gmm_detector.evaluate(X_test, X_synth)
        data_qual = quality_evaluator.evaluate(X_test, X_synth)

        gt_perf = np.mean([xgb_score['gt'],
                           linear_score['gt'],
                           mlp_score['gt']])
        synth_perf = np.mean([xgb_score['syn'],
                              linear_score['syn'],
                              mlp_score['syn']])
        det_score = np.mean([xgb_det['mean'],
                             gmm_det['mean'],
                             mlp_det['mean']])
        qual_score = np.mean(list(data_qual.values()))

        results[model] = [gt_perf, synth_perf, det_score, qual_score]

    return results
