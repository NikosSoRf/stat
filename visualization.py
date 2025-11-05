import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def visualization(results, y_test, y_test_binarized, n_classes, timing_results=None):
    # Вывод результатов
    def vis_results():
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1 Score: {metrics['F1 Score']:.4f}")
            print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
            print("-" * 50)
    
    # Визуализация матрицы ошибок
    def vis_CM():
        for model, metrics in results.items():
            cm = metrics["Confusion Matrix"]
            
            plt.figure(figsize=(6, 5))
            plt.rcParams.update({'font.size': 18})
            # Create heatmap using matplotlib
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(f"Confusion Matrix - {model}")
            plt.colorbar()
            
            # Add annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]),
                            horizontalalignment="center",
                            verticalalignment="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            
            plt.xlabel("Predicted")
            plt.ylabel("True")
            
            # Set tick labels if needed (assuming binary classification)
            if cm.shape[0] == 2:
                plt.xticks([0, 1], ["Negative", "Positive"])
                plt.yticks([0, 1], ["Negative", "Positive"])
            
            plt.show()
    
    # Визуализация ROC-кривых (для каждого класса)
    def vis_ROC():
        for model, metrics in results.items():
            y_proba = metrics["Probability Predictions"]
            
            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.size': 18})
            # Получаем уникальные классы
            unique_classes = np.unique(y_test)
            
            # Для каждого класса строим ROC-кривую
            for class_id in range(n_classes):
                # Для бинарной классификации используем специальный подход
                if n_classes == 2:
                    # Определяем текущий класс
                    current_class = unique_classes[class_id]
                    
                    # Создаем бинарные метки для текущего класса
                    y_binary = (y_test == current_class).astype(int)
                    
                    # Используем вероятности для текущего класса
                    RocCurveDisplay.from_predictions(
                        y_binary,
                        y_proba[:, class_id],
                        name=f"Class {current_class}",
                        ax=plt.gca()
                    )
                else:
                    # Для многоклассовой классификации
                    RocCurveDisplay.from_predictions(
                        y_test_binarized[:, class_id],
                        y_proba[:, class_id],
                        name=f"Class {unique_classes[class_id]}",
                        ax=plt.gca()
                    )
            
            plt.title(f"ROC Curves - {model}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.show()
    
    # Сравнение метрик на одном графике + сетка
    def vis_metrics():
        metrics_df = pd.DataFrame(results).T
        metrics_df[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
            kind='bar',
            figsize=(12, 6),
            rot=0,
            grid=True  # Добавляем сетку
        )
        plt.title("Сравнение моделей по метрикам", pad=20)
        plt.ylabel("Значение метрики")
        plt.ylim(0, 1.1)  # Фиксируем диапазон для наглядности
        plt.tight_layout()
        plt.rcParams.update({'font.size': 14})
        plt.show()

        # Создаём DataFrame с метриками
        metrics_df = pd.DataFrame(results).T
        metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]

        # Сохраняем в Excel (без форматирования)
        metrics_df.to_excel("model_comparison.xlsx", index_label="Model")

        print("Данные сохранены в 'model_comparison.xlsx'")
    
    # Визуализация времени выполнения (упрощенная версия)
    def vis_timing():
        if timing_results is None:
            print("No timing data available")
            return
        
        # Создаем DataFrame с временными метриками
        timing_df = pd.DataFrame(timing_results).T
        
        # Переименовываем колонки для consistency
        timing_df = timing_df.rename(columns={
            "Training Time": "Training Time (s)",
            "Prediction Time": "Prediction Time (s)", 
            "Total Time": "Total Time (s)"
        })
        
        # Временные метрики (если есть)
        time_metrics = ["Training Time (s)", "Prediction Time (s)", "Total Time (s)"]
        available_time_metrics = [m for m in time_metrics if m in timing_df.columns]
        
        if available_time_metrics:
            timing_df[available_time_metrics].plot(
                kind='bar',
                figsize=(12, 6),
                rot=45,
                grid=True
            )
            plt.title("Сравнение времени выполнения моделей", pad=20)
            plt.ylabel("Время (секунды)")
            plt.tight_layout()
            plt.rcParams.update({'font.size': 18})
            plt.show()
            
            # Вывод таблицы с временем
            print("\n" + "="*60)
            print("ВРЕМЯ ВЫПОЛНЕНИЯ МОДЕЛЕЙ")
            print("="*60)
            for model, times in timing_results.items():
                print(f"{model}:")
                print(f"  Обучение: {times['Training Time']:.4f} сек.")
                print(f"  Предсказание: {times['Prediction Time']:.4f} сек.")
                print(f"  Общее время: {times['Total Time']:.4f} сек.")
                print("-" * 40)
            
            # Сохранение данных о времени в Excel
            timing_df.to_excel("model_timing.xlsx", index_label="Model")
            print("Данные о времени сохранены в 'model_timing.xlsx'")
    
    # Вызываем все функции визуализации
    vis_results()
    vis_CM()
    vis_ROC()
    vis_metrics()
    vis_timing()