import os
import json
import numpy as np
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from tesseract import evaluation, temporal, metrics, mock, spatial


PROJ_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJ_ROOT, "data", "processed")

features_file, labels_file, meta_file = (
    os.path.join(DATA_PATH, "drebin-parrot-v2-down-features-X.json"),
    os.path.join(DATA_PATH, "drebin-parrot-v2-down-features-Y.json"),
    os.path.join(DATA_PATH, "drebin-parrot-v2-down-features-meta.json"),
)

with open(features_file) as json_file:
    D = json.load(json_file)
    for datapoint in D:
        del datapoint["sha256"]
    vec = DictVectorizer()
    X = vec.fit_transform(D)  # transform key-value (JSON) into sparse feature vector

with open(labels_file) as json_file:
    labels = json.load(json_file)
    y = np.array([l[0] for l in labels])

with open(meta_file) as json_file:
    meta = json.load(json_file)
    t = list()
    for m in meta:
        timestamp = datetime.strptime(m["dex_date"], "%Y-%m-%dT%H:%M:%S")
        t.append(timestamp)
    t = np.array(t)

X_train, X_test, y_train, y_test, temp_train, temp_test = (
    temporal.time_aware_train_test_split(
        X, y, t, train_size=12, test_size=1, granularity="month"
    )
)


# Perform a timeline evaluation
clf = LinearSVC()
results = evaluation.fit_predict_update(
    clf, X_train, X_test, y_train, y_test, temp_train, temp_test
)

# View results
metrics.print_metrics(results)

# View AUT(F1, 24 months) as a measure of robustness over time
print(metrics.aut(results, "f1"))


## Try to reproduce Figure 6
spatial_clf = LinearSVC()
spatial_results = spatial.find_optimal_train_ratio(
    spatial_clf,
    X_train=X_train,
    y_train=y_train,
    t_train=temp_train,
    proper_train_size=8,
    validation_size=4,
    granularity="month",
    start_tr_rate=0.1,
)
