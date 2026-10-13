import numpy as np
import matplotlib.pyplot as plt
from utilities import visualize_classifier

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score

# Завантаження даних
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
print("Data shape:", X.shape, " classes:", np.unique(y))

# Навчання на всіх даних
classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")
print("Errors:", (y != y_pred).sum(), "of", X.shape[0])

visualize_classifier(classifier, X, y)

# Перший прогін: розбиття з random_state=3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier_new, X_test, y_test)

# Перехресна перевірка
num_folds = 3
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    values = cross_val_score(classifier, X, y, scoring=metric, cv=num_folds)
    print(metric + ": " + str(round(100 * values.mean(), 2)) + "%")

# Другий прогін: розбиття з random_state=10
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=10)
classifier_2 = GaussianNB()
classifier_2.fit(X_train2, y_train2)
y_test_pred2 = classifier_2.predict(X_test2)

accuracy = 100.0 * (y_test2 == y_test_pred2).sum() / X_test2.shape[0]
print("Accuracy of the classifier (second run) =", round(accuracy, 2), "%")
print("Errors:", (y_test2 != y_test_pred2).sum(), "of", X_test2.shape[0])

visualize_classifier(classifier_2, X_test2, y_test2)
