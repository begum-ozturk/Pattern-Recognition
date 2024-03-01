import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report

df = pd.read_csv(r'C:\Users\begum\Desktop\Projects\Pattern Recognition\otu.csv', low_memory=False)

X = df.iloc[2:, :].T
y = df.iloc[1, :]

X = X.apply(pd.to_numeric, errors='coerce')

X = X.fillna(X.mean())

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

n_splits_adjusted = min(5, min(pd.Series(y).value_counts()))
ss = ShuffleSplit(n_splits=n_splits_adjusted, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=ss)
print(f'Cross-validation scores: {scores}')

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

if len(set(y)) == 2:
    auc = roc_auc_score(y_test, y_pred)
    print(f'AUC: {auc}')

report = classification_report(y_test, y_pred, zero_division=1)
print(report)
