import numpy as np
import pandas as pd
import random
import pycuda.driver as cuda
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from time import time

df = pd.read_csv('Worldbank-data-all.csv')
capitals = pd.read_csv('capitals-location.csv')

def make_indicator_dictionary(first_column, last_column):
    first_elements = df[first_column].to_numpy()  # перетворюємо у масив NumPy
    last_elements = df[last_column].to_numpy()  # перетворюємо у масив NumPy

    # Заміна значень '..' на 0 та перетворення на float
    last_elements = np.where(last_elements == '..', 0, last_elements)  # замінюємо '..' на 0
    last_elements = last_elements.astype(float)  # перетворюємо в float

    # Тепер розділимо цей масив на групи по 13/1492 елементів
    #group_size = 13
    group_size = 1492
    num_groups = len(last_elements) // group_size
    grouped_array = np.array(np.array_split(last_elements[:num_groups * group_size], num_groups))

    # Додаємо назву країни до кожної групи
    grouped_with_country = []

    for idx, group in enumerate(grouped_array):
        country_name = first_elements[idx * 1492]
        # Створюємо словник з назвою країни та підмасивом даних
        country_data = {country_name: group}
        grouped_with_country.append(country_data)

    return grouped_with_country


def make_capital_dictionary(name_column, coordinate_column):
    name_elements = capitals[name_column].to_numpy()  # перетворюємо у масив NumPy
    coordinate_elements = capitals[coordinate_column].to_numpy()  # перетворюємо у масив NumPy
    # print(coordinate_elements)

    grouped_with_country = []

    for idx, group in enumerate(coordinate_elements):
        country_name = name_elements[idx]
        # Створюємо словник з назвою країни та підмасивом даних
        country_data = {country_name: group}
        grouped_with_country.append(country_data)

    return grouped_with_country

def convert_clusters_to_arrays(clusters):
    # Перетворює кластери на 2D масиви даних і 1D масиви міток
    X = []
    labels = []
    for cluster_index, cluster_data in clusters.items():
        for item in cluster_data:
            values = list(item.values())[0]
            X.append(values)
            labels.append(cluster_index)

    return np.array(X), np.array(labels)


# Функція групує дані обох типів кластеризації у єдиний масив
def next_step_clustering_list(clusters_f, clusters_s):
    new_dict = {}
    # Перший цикл: додаємо перший елемент в список для кожного ключа
    for cluster_f_index, cluster_f_data in clusters_f.items():
        for item in cluster_f_data:
            key = list(item.keys())[0]  # передбачається, що keys() повертає щонайменше один ключ
            if key not in new_dict:
                new_dict[key] = np.array([0, 0])  # ініціалізуємо список з двома елементами
            new_dict[key][0] = cluster_f_index  # встановлюємо перший елемент списку
            new_dict[key][1] = cluster_f_index

    # Другий цикл: додаємо другий елемент в список для кожного ключа
    for cluster_s_index, cluster_s_data in clusters_s.items():
        for item in cluster_s_data:
            key = list(item.keys())[0]  # отримуємо перший ключ
            if key not in new_dict:
                new_dict[key] = np.array([0, 0])  # ініціалізуємо список, якщо його ще немає
            new_dict[key][1] = cluster_s_index  # встановлюємо другий елемент списку
            if new_dict[key][0] == 0: new_dict[key][0] = new_dict[key][1]

    # Перетворення словника у список словників
    result = [{key: value} for key, value in new_dict.items()]
    return result