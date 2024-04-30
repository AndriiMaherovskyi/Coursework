import numpy as np
import pandas as pd
import random
import pycuda.driver as cuda
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def plot_clusters(clusters, centroids, dataset_name, iteration=None):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'purple', 'orange']
    markers = ['o', 's', 'D', '^', 'P', 'X', 'p']

    # Plot clusters
    for cluster_index, cluster_data in clusters.items():
        cluster_points = [list(item.values())[0] for item in cluster_data]
        cluster_points = np.array(cluster_points)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[cluster_index % len(colors)],
                    marker=markers[cluster_index % len(markers)],
                    label=f'Cluster {cluster_index}')

    # Plot centroids
    for idx, centroid in enumerate(centroids):
        centroid_values = list(centroid.values())[0]
        plt.scatter(centroid_values[0], centroid_values[1],
                    c='black', marker='x', s=100, linewidths=3,
                    label=f'Centroid {idx}')

    if iteration == 0:
        plt.title(f'{dataset_name} Clusters (First iteration)')
    else:
        plt.title(f'{dataset_name} Clusters')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def assign_to_clusters(data, centroids):
    clusters = {}

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
    centroids = random.sample(data, k)
    clusters = {}
    for i in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        # Check convergence
        is_converged = all(
            np.allclose(list(centroid.values())[0], list(new_centroid.values())[0])
            for centroid, new_centroid in zip(centroids, new_centroids)
        )

        if i == 0:
            plot_clusters(clusters, new_centroids, 'K-means Clusters', 0)

        if is_converged:
            break

        centroids = new_centroids

    return clusters, centroids


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


def show_clusters(clusters):
    for cluster_index, cluster_data in clusters.items():
        country_names = [list(item.keys())[0] for item in cluster_data]
        print(f"Cluster {cluster_index}: {country_names}")


def evaluate_clustering(X, labels):
    # Оцінюємо за допомогою різних метрик:
    silhouette_avg = silhouette_score(X, labels)
    print(f"Силуетний індекс: {silhouette_avg}")

    db_score = davies_bouldin_score(X, labels)
    print(f"Індекс Девіса-Болдіна: {db_score}")

    ch_score = calinski_harabasz_score(X, labels)
    print(f"Індекс Каліна-Харабаша: {ch_score}")


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




if __name__ == "__main__":
    # Читаємо CSV-файл
    df = pd.read_csv('Worldbank-data-all.csv')

    first_column = df.columns[0]  # Перша колонка містить назву країни
    last_column = df.columns[-1]  # Остання колонка містить числові дані
    grouped_with_country = make_indicator_dictionary(first_column, last_column)
    print(f"Countries quantity(indicators): {len(grouped_with_country)}")
    k = 7  # Кількість кластерів
    clusters_f, centroids_f = k_means(grouped_with_country, k)

    print("Indicators clusters:")
    show_clusters(clusters_f)
    plot_clusters(clusters_f, centroids_f, 'Indicators')

    # вивід метрик
    X_f, labels_f = convert_clusters_to_arrays(clusters_f)
    evaluate_clustering(X_f, labels_f)

    capitals = pd.read_csv('capitals-location.csv')
    name_column = capitals.columns[0]  # Перша колонка містить назву країни
    coordinate_column = capitals.columns[2:4]  # Остання колонка містить числові дані

    grouped_with_country = make_capital_dictionary(name_column, coordinate_column)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Countries quantity(capitals): {len(grouped_with_country)}")
    k = 7  # Кількість кластерів
    clusters_s, centroids_s = k_means(grouped_with_country, k)

    print("Capitals clusters:")
    show_clusters(clusters_s)
    plot_clusters(clusters_s, centroids_s, 'Capitals')  # Plot відображення утворених кластерів

    # вивід метрик
    X_s, labels_s = convert_clusters_to_arrays(clusters_s)
    evaluate_clustering(X_s, labels_s)

    # Комбінована кластеризація

    nex_step_dict = next_step_clustering_list(clusters_f, clusters_s)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Indicators and capitals(Combine): {len(grouped_with_country)}")
    k = 7  # Кількість кластерів
    clusters_l, centroids_l = k_means(nex_step_dict, k)

    print("Combine clusters:")
    show_clusters(clusters_l)
    plot_clusters(clusters_l, centroids_l, 'Combine')  # Plot відображення утворених кластерів

    # вивід метрик
    X_l, labels_l = convert_clusters_to_arrays(clusters_l)
    evaluate_clustering(X_l, labels_l)

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
