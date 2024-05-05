from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd


def data_process(results, data_file):
    # Создаем пустые списки для хранения результатов
    fractal_dimensions = []

    count = []

    for i, value in enumerate(results):
        count.append(i + 1)
        fractal_dimensions.append(value)

    # Создаем DataFrame из списков
    results_df = pd.DataFrame({
        '№ Изображения': count,
        'Значение фрактальной размерности (D)': fractal_dimensions
    })

    # Записываем результаты в CSV-файл с разделителем ';'
    results_df.to_excel(data_file, index=False)