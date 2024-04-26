import pycuda.driver as cuda
import numpy as np
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import csv

import random


def initialize_centroids(data, k):
    # Ініціалізує центроїди випадковими точками з набору даних.
    data_list = data.tolist()
    centroids = random.sample(data_list, k)
    return centroids


def assign_to_clusters(data, centroids):
    # Призначає кожен зразок даних до найближчого центроїда.
    clusters = {}
    for point in data:
        closest_centroid_index = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])
        if closest_centroid_index not in clusters:
            clusters[closest_centroid_index] = []
        clusters[closest_centroid_index].append(point)
    return clusters


def update_centroids(clusters):
    # Перераховує центроїди на основі зразків, що належать до кластерів.
    centroids = []
    for cluster_index in clusters:
        centroids.append(np.mean(clusters[cluster_index], axis=0)) # обчислює середнє значення
    return centroids


def k_means(data, k, max_iterations=100):
    # Реалізація алгоритму k-середніх.
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        # Перевірка на збіг центроїдів
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids


if __name__ == "__main__":
    # Припустимо, що data - це ваш набір даних у форматі numpy array
    data = np.array([[1, 2], [2, 3], [4, 5], [7, 8], [10, 11], [12, 13]])
    k = 2  # Кількість кластерів
    clusters, centroids = k_means(data, k)
    print("Clusters:", clusters)
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