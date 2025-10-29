import os
import pandas as pd

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

# Вызываем функцию
if __name__ == "__main__":
    data = comb()