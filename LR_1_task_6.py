import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

from sklearn.svm import SVC
import pandas as pd

# Завантаження та розбиття даних
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

models = {
    'SVM (linear)': SVC(kernel='linear'),
    'SVM (rbf)': SVC(kernel='rbf'),
    'Naive Bayes': GaussianNB(),
}

# Якість на тестовій вибірці
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = 100.0 * (model.predict(X_test) == y_test).mean()
    print(name, "- accuracy on test data =", round(acc, 2), "%")

# Перехресна перевірка (3 частини)
metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
results = {}
for name, model in models.items():
    results[name] = [round(100 * cross_val_score(model, X, y, scoring=m, cv=3).mean(), 2) for m in metrics]
print()
print(pd.DataFrame(results, index=metrics))

# Межі класів SVM з ядром rbf на тестових даних
visualize_classifier(models['SVM (rbf)'], X_test, y_test)
