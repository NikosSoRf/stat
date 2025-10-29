import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
# Сбалансированный набор данных
import comb
# Несбалансированный набор данных
import comb2
import visualization
import vist

# Загрузка данных
data = comb.comb()
data = pd.read_csv('combined_data.csv')

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.iloc[:, 1:]  # Все столбцы, кроме первого (класс)
y = data.iloc[:, 0]   # Первый столбец (класс)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Стандартизация данных (для SVM и KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Определение гиперпараметров для Grid Search
param_grids = {
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
}

# Инициализация моделей
models = {
    "SVM": SVC(probability=True, random_state=42),
    #"Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    #"Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Для ROC-AUC нужно бинаризировать метки классов 
# Получаем уникальные классы
classes = np.unique(y)
n_classes = len(classes)

# Бинаризируем метки тестовой выборки
y_test_binarized = label_binarize(y_test, classes=classes)

# Обучение, предсказание и оценка моделей
results = {}
best_params = {}
timing_results = {}  # Словарь для хранения времени выполнения

print("=== GRID SEARCH RESULTS ===")

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Выполнение Grid Search для моделей
    if name in param_grids:
        if name in ["SVM", "KNN"]:
            X_train_used = X_train_scaled
            X_test_used = X_test_scaled
        else:
            X_train_used = X_train
            X_test_used = X_test
            
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_used, y_train)
        
        # Сохранение лучших параметров
        best_params[name] = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    else:
        # Для моделей без гиперпараметров
        best_params[name] = "No hyperparameters to tune"
        print("No hyperparameter tuning needed")
        
        if name in ["SVM", "KNN"]:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        best_model = model
    
    # ИЗМЕРЕНИЕ ВРЕМЕНИ ТОЛЬКО ДЛЯ ЛУЧШЕЙ МОДЕЛИ
    # Время обучения лучшей модели
    start_train_time = time.time()
    
    if name in ["SVM", "KNN"]:
        best_model.fit(X_train_scaled, y_train)
    else:
        best_model.fit(X_train, y_train)
    
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    
    # Время предсказания лучшей модели
    start_predict_time = time.time()
    
    if name in ["SVM", "KNN"]:
        y_pred = best_model.predict(X_test_scaled)
        y_proba = best_model.predict_proba(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
    
    end_predict_time = time.time()
    predict_time = end_predict_time - start_predict_time
    
    # Общее время
    total_time = train_time + predict_time
    
    # Сохранение времени выполнения
    timing_results[name] = {
        "Training Time": train_time,
        "Prediction Time": predict_time,
        "Total Time": total_time
    }
    
    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC-AUC (для многоклассовой классификации)
    # Проверяем, является ли задача бинарной классификацией
    if n_classes == 2:
        # Для бинарной классификации используем вероятности положительного класса
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        # Для многоклассовой классификации
        roc_auc = roc_auc_score(
            y_test_binarized,
            y_proba,
            multi_class='ovr',
            average='weighted'
        )
    
    # Сохранение результатов
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "ROC-AUC": roc_auc,
        "Probability Predictions": y_proba,
        "Best Model": best_model
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Training Time (best model): {train_time:.4f} seconds")
    print(f"Prediction Time (best model): {predict_time:.4f} seconds")
    print(f"Total Time (best model): {total_time:.4f} seconds")

# Вывод итоговой информации о лучших гиперпараметрах
print("\n" + "="*50)
print("SUMMARY OF BEST HYPERPARAMETERS")
print("="*50)
for name, params in best_params.items():
    print(f"{name}: {params}")

# Передаем дополнительные параметры в функцию визуализации
visualization.visualization(results, y_test, y_test_binarized, n_classes, timing_results)