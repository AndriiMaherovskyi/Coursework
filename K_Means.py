import numpy as np
import random
import pycuda.driver as cuda
from pycuda import compiler

from plot import plot_clusters

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


def clusters_from_result(assignments, data):
    clusters = {}
    for idx, cluster_idx in enumerate(assignments):
        country_name = list(data[idx].keys())[0]
        values = np.array(list(data[idx].values())[0])

        if cluster_idx not in clusters:
            clusters[cluster_idx] = []

        clusters[cluster_idx].append({country_name: values})

    return clusters


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


def show_clusters(clusters):
    for cluster_index, cluster_data in clusters.items():
        country_names = [list(item.keys())[0] for item in cluster_data]
        print(f"Cluster {cluster_index}: {country_names}")