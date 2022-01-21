import pandas as pd

dataset = pd.read_csv('breast_cancer.csv')

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 0).fit(X_train, Y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true=Y_test, y_pred=y_pred)
print(cm)
# (84 + 47) / (84+47+3+3)
print(accuracy_score(Y_test, y_pred))

from sklearn.model_selection import cross_val_score
accur = cross_val_score(estimator=clf, X = X_train, y = Y_train, cv = 10)

print('Accuracy: {:.2f}%'.format(accur.mean() * 100))
print('Standard Deviation: {:.2f} %'.format(accur.std() * 100))