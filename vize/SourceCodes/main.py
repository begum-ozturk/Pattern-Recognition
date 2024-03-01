import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv(r'C:\Users\begum\Desktop\Projects\Pattern Recognition\FinalCSV\train.csv')
test_data = pd.read_csv(r'C:\Users\begum\Desktop\Projects\Pattern Recognition\FinalCSV\test.csv')

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

selector = SelectKBest(f_classif, k=100)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_new, y_train)

y_pred = model.predict(X_test_new)
report = classification_report(y_test, y_pred)
print(report)

scores = cross_val_score(model, X_train_new, y_train, cv=5)
print('Cross-validation scores:', scores)
