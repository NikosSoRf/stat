import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def visualization(results, y_test_binarized, y_test):
    # Получаем количество классов из тестовых данных
    n_classes = len(np.unique(y_test))
    
    # Вывод результатов
    def vis_results():
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1 Score: {metrics['F1 Score']:.4f}")
            print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
            if "Training Time (s)" in metrics:
                print(f"Training Time: {metrics['Training Time (s)']:.4f} сек")
                print(f"Prediction Time: {metrics['Prediction Time (s)']:.4f} сек")
                print(f"Total Time: {metrics['Total Time (s)']:.4f} сек")
            print("-" * 50)
    
    # Визуализация матрицы ошибок
    def vis_CM():
        for model, metrics in results.items():
            cm = metrics["Confusion Matrix"]
            
            plt.figure(figsize=(6, 5))
            plt.rcParams.update({'font.size': 18})
            
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(f"{model}")
            plt.colorbar()
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]),
                            horizontalalignment="center",
                            verticalalignment="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            
            plt.xlabel("Predicted")
            plt.ylabel("True")
            
            if cm.shape[0] == 2:
                plt.xticks([0, 1], ["Negative", "Positive"])
                plt.yticks([0, 1], ["Negative", "Positive"])
            
            plt.show()
    
    # Визуализация ROC-кривых
    def vis_ROC():
        for model, metrics in results.items():
            y_proba = metrics["Probability Predictions"]
            
            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.size': 18})
            # Для бинарной классификации
            if n_classes == 2:
                # Получаем уникальные классы
                unique_classes = np.unique(y_test)
                
                # Определяем положительный класс 
                pos_label = unique_classes[1] 
                
                # ROC-кривая для бинарной классификации
                RocCurveDisplay.from_predictions(
                    y_test,
                    y_proba[:, 1],  # Вероятности положительного класса
                    pos_label=pos_label, 
                    name="ROC Curve",
                    ax=plt.gca()
                )
            else:
                # Для многоклассовой классификации
                for class_id in range(n_classes):
                    RocCurveDisplay.from_predictions(
                        y_test_binarized[:, class_id],
                        y_proba[:, class_id],
                        name=f"Class {class_id}",
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
        
        # Основные метрики
        main_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        available_metrics = [m for m in main_metrics if m in metrics_df.columns]
        plt.rcParams.update({'font.size': 18})
        metrics_df[available_metrics].plot(
            kind='bar',
            figsize=(12, 6),
            rot=0,  # горизонтальные подписи
            grid=True
        )
        plt.title("Сравнение моделей по метрикам", pad=20)
        plt.ylabel("Значение метрики")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        
        plt.show()

        # Временные метрики
        time_metrics = ["Training Time (s)", "Prediction Time (s)", "Total Time (s)"]
        available_time_metrics = [m for m in time_metrics if m in metrics_df.columns]
        
        if available_time_metrics:
            metrics_df[available_time_metrics].plot(
                kind='bar',
                figsize=(12, 6),
                rot=0,
                grid=True
            )
            plt.title("Сравнение времени выполнения моделей", pad=20)
            plt.ylabel("Время (секунды)")
            plt.tight_layout()
            plt.rcParams.update({'font.size': 18})
            plt.show()

        # схранение в Excel
        metrics_df.to_excel("model_comparison.xlsx", index_label="Model")
        print("Данные сохранены в 'model_comparison.xlsx'")
    
    #vis_results()
    #vis_CM()
    #vis_ROC()
    vis_metrics()