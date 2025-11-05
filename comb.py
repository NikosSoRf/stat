import os
import pandas as pd
import numpy as np

def comb():
    # Пути к папкам с данными
    folders = {
        1: 'data\\1_Dangerous_VFL_VF\\full',
        #2: 'data\\2_Special_Form_VTTdP\\full',
        3: 'data\\3_Threatening_VT\\full',
        #4: 'data\\4_Potential_Dangerous\\full',
        5: 'data\\5_Supraventricular\\full',
        6: 'data\\6_Sinus_rhythm\\full'
    }

    # Создаем пустой DataFrame для объединенных данных
    data = pd.DataFrame()
    
    # Определяем минимальное количество файлов в папке
    min_files = float('inf')
    for folder_path in folders.values():
        files = os.listdir(folder_path)
        if len(files) < min_files:
            min_files = len(files)
    
    print(f"Из каждой папки взято файлов: {min_files}")
    
    # Проходим по каждой папке и берем min_files файлов
    for class_label, folder_path in folders.items():
        # Получаем список файлов в папке и берем первые min_files
        files = os.listdir(folder_path)[:min_files]
        
        # Проходим по каждому файлу в папке
        for file in files:
            # Читаем данные из файла
            file_path = os.path.join(folder_path, file)
            patient_data = pd.read_csv(file_path)
            
            # Преобразуем данные в одну строку
            patient_data_flattened = patient_data.values.flatten()
            
            # Добавляем метку класса в начало строки
            patient_data_flattened = pd.Series([class_label] + list(patient_data_flattened))
            
            # Добавляем данные пациента в общий DataFrame
            data = pd.concat([data, patient_data_flattened.to_frame().T], ignore_index=True)

    # Сохраняем объединенные данные в CSV-файл
    data.to_csv('combined_data.csv', index=False)

    return data

def augment_data(data = comb(), augmentation_factor=2):
    """
    Увеличивает выборку в augmentation_factor раз с помощью аугментации данных
    
    Parameters:
    data - исходный DataFrame с данными
    augmentation_factor - во сколько раз увеличить выборку (по умолчанию 2)
    """
    if augmentation_factor < 2:
        return data
    
    augmented_data = data.copy()
    
    # Определяем количество новых образцов для создания
    original_size = len(data)
    samples_to_create = original_size * (augmentation_factor - 1)
    
    print(f"Исходный размер выборки: {original_size}")
    print(f"Создаем {samples_to_create} новых образцов...")
    
    # Получаем только числовые данные (без метки класса в первом столбце)
    numeric_data = data.iloc[:, 1:].values
    labels = data.iloc[:, 0].values
    
    # Создаем новые образцы
    for i in range(samples_to_create):
        # Случайно выбираем исходный образец
        original_idx = np.random.randint(0, original_size)
        original_sample = numeric_data[original_idx].copy()
        original_label = labels[original_idx]
        
        # Выбираем метод аугментации 'noise', 'scaling', 'smoothing'
        method = 'smoothing'
        if method == 'noise':
            # Добавление гауссовского шума
            noise = np.random.normal(0, 0.01, original_sample.shape)
            augmented_sample = original_sample + noise
            
        elif method == 'scaling':
            # Масштабирование амплитуды
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented_sample = original_sample * scale_factor
                
        elif method == 'smoothing':
            # Легкое сглаживание
            window_size = 3
            if len(original_sample) > window_size:
                augmented_sample = np.convolve(original_sample, np.ones(window_size)/window_size, mode='same')
            else:
                augmented_sample = original_sample
        
        # Добавляем метку класса в начало
        augmented_row = pd.Series([original_label] + list(augmented_sample))
        
        # Добавляем новый образец в DataFrame
        augmented_data = pd.concat([augmented_data, augmented_row.to_frame().T], ignore_index=True)
    
    print(f"Финальный размер выборки: {len(augmented_data)}")
    
    # Сохраняем аугментированные данные
    augmented_data.to_csv('augmented_combined_data.csv', index=False)
    
    return augmented_data
