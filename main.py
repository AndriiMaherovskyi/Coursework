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


region_names = ["Західна Європа", "Східна Європа", "Північна Америка", "Південна Америка", "Азія", "Африка", "Океанія"]


def cuda_kernel(data_o, centroids_o):
    # Функція на CUDA для знаходження найближчого центроїда
    cuda_code = """
    __global__ void assign_to_clusters(float *data, float *centroids, int *result, int num_data, int num_centroids, int data_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_data) return;

        float min_distance = 1e10;
        int closest_centroid = -1;

        for (int i = 0; i < num_centroids; i++) {
            float distance = 0;
            for (int j = 0; j < data_size; j++) {
                float diff = data[idx * data_size + j] - centroids[i * data_size + j];
                distance += diff * diff;
            }
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = i;
            }
        }

        result[idx] = closest_centroid;
    }
    """

    # Скомпілюємо ядро
    mod = compiler.SourceModule(cuda_code)
    assign_kernel = mod.get_function("assign_to_clusters")

    # Створимо зразкові дані та центроїди
    data = data_o
    centroids = centroids_o

    # Перетворимо дані на одномірні масиви
    data_array = np.array([list(item.values())[0] for item in data], dtype=np.float32).flatten()
    centroid_array = np.array([list(c.values())[0] for c in centroids], dtype=np.float32).flatten()

    # Виділимо пам'ять на GPU та скопіюємо дані
    data_gpu = cuda.mem_alloc(data_array.nbytes)
    centroid_gpu = cuda.mem_alloc(centroid_array.nbytes)
    cuda.memcpy_htod(data_gpu, data_array)
    cuda.memcpy_htod(centroid_gpu, centroid_array)

    # Підготуємо масив для результатів (індекси кластера)
    num_data = len(data)
    result_gpu = cuda.mem_alloc(num_data * 4)  # int32
    result_cpu = np.zeros(num_data, dtype=np.int32)

    # Визначимо параметри блоків та сітки
    block_size = 256
    grid_size = (num_data + block_size - 1) // block_size

    # Запустимо ядро
    assign_kernel(data_gpu, centroid_gpu, result_gpu, np.int32(num_data), np.int32(len(centroids)), np.int32(2),
                  block=(block_size, 1, 1), grid=(grid_size, 1))

    # Скопіюємо результат з GPU на CPU
    cuda.memcpy_dtoh(result_cpu, result_gpu)

    # Створимо словник кластера за отриманими результатами
    clusters = {}

    for idx, cluster_idx in enumerate(result_cpu):
        country_name = list(data[idx].keys())[0]
        values = np.array(list(data[idx].values())[0])

        if cluster_idx not in clusters:
            clusters[cluster_idx] = []

        clusters[cluster_idx].append({country_name: values})

    return clusters


def clusters_from_result(assignments, data):
    clusters = {}
    for idx, cluster_idx in enumerate(assignments):
        country_name = list(data[idx].keys())[0]
        values = np.array(list(data[idx].values())[0])

        if cluster_idx not in clusters:
            clusters[cluster_idx] = []

        clusters[cluster_idx].append({country_name: values})

    return clusters


def plot_clusters(clusters, centroids, dataset_name, cluster_names=None, iteration=None):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'purple', 'orange']
    markers = ['o', 's', 'D', '^', 'P', 'X', 'p']

    # Plot clusters
    for cluster_index, cluster_data in clusters.items():
        cluster_points = [list(item.values())[0] for item in cluster_data]
        cluster_points = np.array(cluster_points)

        cluster_label = f'Cluster {cluster_index}'
        if cluster_names and cluster_index < len(cluster_names):
            cluster_label = cluster_names[cluster_index]

        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[cluster_index % len(colors)],
                    marker=markers[cluster_index % len(markers)],
                    label=cluster_label)

    # Plot centroids
    if isinstance(centroids, list) and len(centroids) > 0 and isinstance(centroids[0], dict):
        for idx, centroid in enumerate(centroids):
            centroid_values = list(centroid.values())[0]
            plt.scatter(centroid_values[0], centroid_values[1],
                        c='black', marker='x', s=100, linewidths=3)
    else:
        # Assuming centroids is a numpy array
        for idx in range(0, len(centroids), 2):
            plt.scatter(centroids[idx], centroids[idx + 1],
                        c='black', marker='x', s=100, linewidths=3)

    if iteration == 0:
        plt.title(f'{dataset_name} Clusters (First iteration)')
    else:
        plt.title(f'{dataset_name} Clusters')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def assign_to_clusters_cpu(data, centroids):
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


def k_means(data, k, ind, it=None, max_iterations=100):
    centroids = random.sample(data, k)
    clusters = {}
    for i in range(max_iterations):
        if ind == 'GPU':
            clusters = cuda_kernel(data, centroids)
        else:
            clusters = assign_to_clusters_cpu(data, centroids)
        new_centroids = update_centroids(clusters)

        # Check convergence
        is_converged = all(
            np.allclose(list(centroid.values())[0], list(new_centroid.values())[0])
            for centroid, new_centroid in zip(centroids, new_centroids)
        )

        if i == 0 and it is not None:
            plot_clusters(clusters, centroids, 'Capitals', region_names, it)

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


def output_scores(X, labels):
    # Оцінюємо за допомогою різних метрик:
    silhouette_avg = silhouette_score(X, labels)
    print(f"Силуетний індекс: {silhouette_avg}")

    db_score = davies_bouldin_score(X, labels)
    print(f"Індекс Девіса-Болдіна: {db_score}")

    ch_score = calinski_harabasz_score(X, labels)
    print(f"Індекс Каліна-Харабаша: {ch_score}\n")


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


k = 7

if __name__ == "__main__":

    # Комбінована кластеризація
#
    # nex_step_dict = next_step_clustering_list(clusters_f, clusters_s)
    # print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    # print(f"Indicators and capitals(Combine): {len(grouped_with_country)}")
    # k = 7  # Кількість кластерів
    # clusters_l, centroids_l = k_means(nex_step_dict, k)
#

    # Читаємо CSV-файл

    capitals = pd.read_csv('capitals-location.csv')
    name_column = capitals.columns[0]  # Перша колонка містить назву країни
    coordinate_column = capitals.columns[2:4]  # Остання колонка містить числові дані

    grouped_with_country = make_capital_dictionary(name_column, coordinate_column)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Countries quantity(capitals): {len(grouped_with_country)}")

    # Час виконання на CPU
    start_time_f = time()
    clusters_f, centroids_f = k_means(grouped_with_country, k, 'CPU', 0)
    cpu_time_f = time() - start_time_f

    print("Capitals clusters (CPU):")
    show_clusters(clusters_f)
    print(f"CPU Time: {cpu_time_f:.5f} seconds")
    print()
    plot_clusters(clusters_f, centroids_f, 'Capitals', region_names)

    # вивід метрик
    X_f, labels_f = convert_clusters_to_arrays(clusters_f)
    output_scores(X_f, labels_f)

    # Час виконання на GPU
    start_time_f_gpu = time()
    clusters_f_gpu, centroids_f_gpu = k_means(grouped_with_country, k, 'GPU')
    gpu_time_f = time() - start_time_f_gpu

    print("Capitals clusters (GPU):")
    show_clusters(clusters_f_gpu)
    print(f"GPU Time: {gpu_time_f:.5f} seconds")
    print()
    # plot_clusters(clusters_f_gpu, centroids_f_gpu, 'Capitals')

    # вивід метрик
    X_f_gpu, labels_f_gpu = convert_clusters_to_arrays(clusters_f_gpu)
    output_scores(X_f_gpu, labels_f_gpu)

    df = pd.read_csv('Worldbank-data-all.csv')
    first_column = df.columns[0]  # Перша колонка містить назву країни
    last_column = df.columns[-1]  # Остання колонка містить числові дані
    grouped_with_country = make_indicator_dictionary(first_column, last_column)
    print(f"Countries quantity(indicators): {len(grouped_with_country)}")

    start_time_s = time()
    clusters_s, centroids_s = k_means(grouped_with_country, k, 'CPU')
    cpu_time = time() - start_time_s
    show_clusters(clusters_s)
    print(f"CPU Time: {cpu_time:.5f} seconds")
    print()

    X_s, labels_s = convert_clusters_to_arrays(clusters_f_gpu)
    output_scores(X_s, labels_s)

    start_time_s_gpu = time()
    clusters_s_gpu, centroids_s_gpu = k_means(grouped_with_country, k, 'GPU')
    elapsed_time = time() - start_time_s_gpu
    show_clusters(clusters_s_gpu)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()

    X_s_gpu, labels_s_gpu = convert_clusters_to_arrays(clusters_f_gpu)
    output_scores(X_s_gpu, labels_s_gpu)

    # Метрики для обох підходів
    # X_gpu, labels_gpu = convert_clusters_to_arrays(k_means_gpu(data, 2))
    # print("GPU Evaluation:")
    # evaluate_clustering(X_gpu, labels_gpu)

    # X_cpu, labels_cpu = convert_clusters_to_arrays(clusters_cpu)
    # print("CPU Evaluation:")
    # evaluate_clustering(X_cpu, labels_cpu)
