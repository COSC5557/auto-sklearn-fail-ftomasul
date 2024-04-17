import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)

# Visualize data distribution (if plot code is run with train code it crashes ¯\_(ツ)_/¯)
# data = X.copy()
# data['quality'] = y
# plt.figure(figsize=(15, 10))
# for i, feature in enumerate(data.columns):
#     plt.subplot(3, 4, i+1)
#     data[feature].hist()
#     plt.title(feature)
# plt.tight_layout()
# plt.savefig("distributions.png")
# plt.close()

# Try normalizing the data
scaler = sklearn.preprocessing.StandardScaler()
X_norm = scaler.fit_transform(X)

# Fix imbalance
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):
   X_train, X_test = X_norm[train_index], X_norm[test_index]  
   y_train, y_test = y.iloc[train_index], y.iloc[test_index]

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))

from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(time_left_for_this_task=300,
                               resampling_strategy="cv",
                               resampling_strategy_arguments={"folds": 10})
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))
