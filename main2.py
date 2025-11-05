import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import time
import comb
import comb2
import vist
import visualization

from grid_search import grid_search_base, grid_search_meta

# Загрузка данных
data = comb.comb()
#data = comb.augment_data()
#data = pd.read_csv('combined_data.csv')

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.iloc[:, 1:]  # Все столбцы, кроме первого (класс)
y = data.iloc[:, 0]   # Первый столбец (класс)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Стандартизация данных (для SVM и KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Базовые модели с Grid Search
base_models = {
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Метамодель (финальный классификатор)
meta_model = LogisticRegression(random_state=42)

# Для ROC-AUC нужно бинаризировать метки классов (One-vs-Rest)
label_binarizer = LabelBinarizer()
y_test_binarized = label_binarizer.fit_transform(y_test)

# Получаем количество классов
n_classes = len(np.unique(y))

# Словарь для хранения результатов
results = {}
print("=== GRID SEARCH FOR BASE MODELS ===")

# Оптимизация гиперпараметров базовых моделей
optimized_base_models, best_params = grid_search_base(base_models, X_train_scaled, X_train, y_train)

print("\n=== GRID SEARCH FOR META MODEL ===")

# Создаем фичи для метамодели для оптимизации
print("Creating meta-features for meta-model optimization...")
svm_proba = cross_val_predict(optimized_base_models["SVM"], X_train_scaled, y_train, cv=4, method='predict_proba')
knn_proba = cross_val_predict(optimized_base_models["KNN"], X_train_scaled, y_train, cv=4, method='predict_proba')
nb_proba = cross_val_predict(optimized_base_models["Naive Bayes"], X_train_scaled, y_train, cv=4, method='predict_proba')

X_meta_train = np.column_stack([svm_proba, knn_proba, nb_proba])

best_meta_model, best_params["Meta_Model"] = grid_search_meta(meta_model, X_meta_train, y_train, best_params)

# Функция для создания стекинга из двух моделей с оптимизированными гиперпараметрами
def create_stacking_pair(model1_name, model1, model2_name, model2, meta, X_tr, X_te, y_tr, scaler_required=True):
    """
    Создает стекинг из двух моделей с измерением времени
    """
    stack_name = f"Stacking_{model1_name}_{model2_name}"
    
    # Определяем, нужно ли масштабирование для этих моделей
    if scaler_required:
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
    else:
        X_train_used = X_train
        X_test_used = X_test
    
    # Измеряем время обучения
    start_train_time = time.time()
    
    # Создаем прогнозы для метамодели с помощью cross-validation
    model1_proba = cross_val_predict(model1, X_train_used, y_tr, cv=4, method='predict_proba')
    model2_proba = cross_val_predict(model2, X_train_used, y_tr, cv=4, method='predict_proba')
    
    # Объединяем прогнозы в новые признаки
    X_meta_train = np.column_stack([model1_proba, model2_proba])
    
    # Обучаем метамодель на новых признаках
    meta.fit(X_meta_train, y_tr)
    
    # Теперь обучаем базовые модели на всех данных для финального предсказания
    model1.fit(X_train_used, y_tr)
    model2.fit(X_train_used, y_tr)
    
    train_time = time.time() - start_train_time
    
    # Измеряем время предсказания
    start_pred_time = time.time()
    
    # Создаем тестовые данные для метамодели
    model1_test_proba = model1.predict_proba(X_test_used)
    model2_test_proba = model2.predict_proba(X_test_used)
    X_meta_test = np.column_stack([model1_test_proba, model2_test_proba])
    
    # Предсказания метамодели
    y_pred = meta.predict(X_meta_test)
    y_proba = meta.predict_proba(X_meta_test)
    
    pred_time = time.time() - start_pred_time
    
    # Общее время
    total_time = train_time + pred_time
    
    return stack_name, y_pred, y_proba, train_time, pred_time, total_time

# Создаем все попарные комбинации стекинга с оптимизированными моделями
print("\n=== CREATING OPTIMIZED STACKING ENSEMBLES ===")
model_pairs = [
    ("SVM", "KNN", True),
    ("SVM", "Naive Bayes", True),
    ("KNN", "Naive Bayes", True)
]

for model1_name, model2_name, scale_required in model_pairs:
    print(f"Создание оптимизированного стекинга: {model1_name} + {model2_name}")
    
    model1 = optimized_base_models[model1_name]
    model2 = optimized_base_models[model2_name]
    
    # Используем оптимизированную метамодель для каждого стекинга
    stack_name, y_pred, y_proba, train_time, pred_time, total_time = create_stacking_pair(
        model1_name, model1, 
        model2_name, model2, 
        best_meta_model,  # Используем оптимизированную метамодель
        X_train, X_test, y_train,
        scaler_required=scale_required
    )
    
    # Вычисление метрик для стекинга
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Исправленное вычисление ROC-AUC
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
    
    # Сохраняем результаты
    results[stack_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "ROC-AUC": roc_auc,
        "Probability Predictions": y_proba,
        "Training Time (s)": train_time,
        "Prediction Time (s)": pred_time,
        "Total Time (s)": total_time
    }

# Выводим результаты для сравнения
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ОПТИМИЗИРОВАННЫХ МОДЕЛЕЙ (ВКЛЮЧАЯ ВРЕМЯ):")
print("="*80)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy:       {metrics['Accuracy']:.4f}")
    print(f"  Precision:      {metrics['Precision']:.4f}")
    print(f"  Recall:         {metrics['Recall']:.4f}")
    print(f"  F1 Score:       {metrics['F1 Score']:.4f}")
    print(f"  ROC-AUC:        {metrics['ROC-AUC']:.4f}")
    print(f"  Training Time:  {metrics['Training Time (s)']:.4f} сек")
    print(f"  Prediction Time:{metrics['Prediction Time (s)']:.4f} сек")
    print(f"  Total Time:     {metrics['Total Time (s)']:.4f} сек")

# Дополнительная таблица сравнения времени
print("\n" + "="*80)
print("СРАВНЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
print("="*80)
print(f"{'Модель':<30} {'Время обучения (сек)':<20} {'Время предсказания (сек)':<25} {'Общее время (сек)':<15}")
for name, metrics in results.items():
    print(f"{name:<30} {metrics['Training Time (s)']:<20.4f} {metrics['Prediction Time (s)']:<25.4f} {metrics['Total Time (s)']:<15.4f}")

# Вывод лучших гиперпараметров
print("\n" + "="*80)
print("ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ ДЛЯ ВСЕХ МОДЕЛЕЙ:")
print("="*80)
for model_name, params in best_params.items():
    print(f"{model_name}: {params}")

# Визуализация результатов
vist.visualization(results, y_test_binarized, y_test)