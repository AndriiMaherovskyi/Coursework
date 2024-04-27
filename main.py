import pycuda.driver as cuda
import numpy as np
import pandas as pd
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import csv

import random


def initialize_centroids(data, k):
    # Ініціалізує центроїди випадковими точками з набору даних.
    centroids = random.sample(data, k)
    return centroids


def assign_to_clusters(data, centroids):
    clusters = {}
    k = len(centroids)

    # Перебираємо кожен словник у data
    for idx, item in enumerate(data):
        country_name = list(item.keys())[0]
        values = np.array(list(item.values())[0])
        closest_centroid_index = np.argmin([np.linalg.norm(values - list(c.values())[0]) for c in centroids])
        if closest_centroid_index not in clusters:
            clusters[closest_centroid_index] = []

        clusters[closest_centroid_index].append({country_name: values})
    return clusters


def update_centroids(clusters):
    centroids = []
    for cluster_index in clusters:
        # Отримуємо список масивів даних у кластері
        cluster_data = [list(v.values())[0] for v in clusters[cluster_index]]
        # Розраховуємо середнє значення для кожного виміру
        centroid = np.mean(cluster_data, axis=0)
        # Створюємо новий центроїд у вигляді словника з ідентифікатором кластера
        centroids.append({f"Cluster_{cluster_index}": centroid})

    return centroids


def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        # Перевірка, чи збігаються нові центроїди зі старими
        is_converged = all(
            np.allclose(list(centroid.values())[0], list(new_centroid.values())[0])
            for centroid, new_centroid in zip(centroids, new_centroids)
        )

        if is_converged:
            break

        centroids = new_centroids

    return clusters, centroids


if __name__ == "__main__":
    # Читаємо CSV-файл
    df = pd.read_csv('Worldbank-data-2020.csv')

    # Перша колонка містить назву країни
    first_column = df.columns[0]
    # Остання колонка містить числові дані
    last_column = df.columns[-1]

    first_elements = df[first_column].to_numpy()  # перетворюємо у масив NumPy
    last_elements = df[last_column].to_numpy()  # перетворюємо у масив NumPy

    # Заміна значень '..' на 0 та перетворення на float
    last_elements = np.where(last_elements == '..', 0, last_elements)  # замінюємо '..' на 0
    last_elements = last_elements.astype(float)  # перетворюємо в float

    # Тепер розділимо цей масив на групи по 13 елементів
    group_size = 13
    num_groups = len(last_elements) // group_size
    grouped_array = np.array(np.array_split(last_elements[:num_groups * group_size], num_groups))

    # Додаємо назву країни до кожної групи
    grouped_with_country = []

    for idx, group in enumerate(grouped_array):
        country_name = first_elements[idx*13]
        # Створюємо словник з назвою країни та підмасивом даних
        country_data = {country_name: group}
        grouped_with_country.append(country_data)


    # Виклик K-means алгоритму
    k = 8  # Кількість кластерів
    clusters, centroids = k_means(grouped_with_country, k)
    print("Clusters:", clusters[0])
    print("Centroids:", centroids)


# with open('dataset-ucdp-prio-conflict.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     i = 2
#     for i in reader:
#         print(i[0])
#         print()
#         print(i[1])
# driver.init()
#
# # Create a CUDA context
# device = driver.Device(0) # Визначає кількість підключених відеокарт, рахуємо від 0
# context = device.make_context()
#
# # Define the CUDA kernel
# kernel_code = """
# __global__ void add_arrays(float *a, float *b, float *c) {
#     int i = threadIdx.x;
#     c[i] = a[i] + b[i];
# }
# """
#
# # Compile the CUDA kernel
# module = compiler.SourceModule(kernel_code)
#
# '''
#     Поки залишаю приклад з сайту по масивах, потім будемо щось тут створювати
# '''
# # Allocate memory on the GPU
# a_gpu = gpuarray.to_gpu(np.random.randn(100).astype(np.float32))
# b_gpu = gpuarray.to_gpu(np.random.randn(100).astype(np.float32))
# c_gpu = gpuarray.empty_like(a_gpu)
#
# # Create events for timing
# start = driver.Event()
# end = driver.Event()
#
# # Start timing
# start.record()
#
# # Launch the CUDA kernel
# add_arrays = module.get_function("add_arrays")
# add_arrays(a_gpu, b_gpu, c_gpu, block=(100,1,1))
#
# # Stop timing
# end.record()
# end.synchronize()  # Очікування завершення всіх операцій
# elapsed_time_ms = start.time_till(end)  # Час у мілісекундах
#
# # Copy the result back to the CPU
# c_cpu = c_gpu.get()
# print(c_cpu)
# print("Elapsed time (ms):", elapsed_time_ms)
#
# # Clean up context
# context = pycuda.autoinit.context
#
# # Clean up
# context.pop()