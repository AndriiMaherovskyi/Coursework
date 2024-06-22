import numpy as np
import random
import pycuda.driver as cuda
from pycuda import compiler
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from plot import plot_clusters

region_names = ["Західна Європа", "Східна Європа", "Північна Америка", "Південна Америка", "Азія", "Африка", "Океанія"]


def cuda_k_means(data, k, max_iterations=100):
    cuda_code = """
        #include <float.h>

        __global__ void assign_clusters(float *data, int *labels, float *centroids, int num_arrays, int num_samples_per_array, int k) {
            int idx = blockIdx.x;
            if (idx >= num_arrays) return;

            float min_dist, dist, diff;
            int min_index;
            int start_index = idx * num_samples_per_array;

            for (int i = 0; i < num_samples_per_array; i++) {
                min_dist = FLT_MAX;
                min_index = -1;
                for (int j = 0; j < k; j++) {
                    diff = data[start_index + i] - centroids[j];
                    dist = diff * diff;
                    if (dist < min_dist) {
                    
                        min_dist = dist;
                        min_index = j;
                    }
                }
                labels[start_index + i] = min_index;
            }
        }

        __global__ void update_centroids(float *data, float *centroids, int *labels, int k, int num_samples_per_array, float *new_centroids) {
            extern __shared__ float shared_data[];  // Подвійний розмір для суми і кількості
        
            int idx = blockIdx.x;
            int start_index = idx * num_samples_per_array;
            float sum = 0.0;
            int count = 0;
        
            for (int i = 0; i < num_samples_per_array; i++) {
                if (labels[start_index + i] == threadIdx.x) {
                    sum += data[start_index + i];
                    count++;
                }
            }
        
            // Використання двох сегментів shared_data: один для сум, інший для кількості
            shared_data[threadIdx.x] = sum;
            shared_data[threadIdx.x + blockDim.x] = count;  // Збереження кількості у другій половині масиву
        
            __syncthreads();
        
            if (threadIdx.x < k) {
                float total_sum = shared_data[threadIdx.x];
                int total_count = shared_data[threadIdx.x + blockDim.x];
                if (total_count > 0) {
                    new_centroids[threadIdx.x] = total_sum / total_count;
                } else {
                    new_centroids[threadIdx.x] = centroids[threadIdx.x];
                }
            }
        }

        """

    mod = compiler.SourceModule(cuda_code)
    assign_clusters = mod.get_function("assign_clusters")
    update_centroids = mod.get_function("update_centroids")

    # Підготовка даних і запуск алгоритму ...
    # Зверніть увагу на ці параметри
    block_size = 373  # 373 масиви в 1 блоці
    grid_size = 4  # 4 блоки на виконання всіх масивів

    # Підготовка даних для GPU
    data = data.astype(np.float32)

    centroids = np.random.rand(k * grid_size, data.shape[1]).astype(
        np.float32).flatten()  # k центроїдів для кожного з grid_size блоків

    data_gpu = gpuarray.to_gpu(data)
    centroids_gpu = gpuarray.to_gpu(centroids)
    labels_gpu = gpuarray.zeros(data.shape[0], dtype=np.int32)
    size_float32 = np.dtype(np.float32).itemsize

    # Ітераційний процес k-середніх
    for i in range(max_iterations):
        assign_clusters(data_gpu, labels_gpu, centroids_gpu, np.int32(data.shape[0]), np.int32(data.shape[1]), np.int32(k),
                        block=(block_size, 1, 1), grid=(grid_size, 1))
        new_centroids = np.zeros_like(centroids)
        new_centroids_gpu = gpuarray.to_gpu(new_centroids)
        update_centroids(data_gpu, centroids_gpu, labels_gpu, np.int32(k), np.int32(data.shape[1]), new_centroids_gpu,
                         block=(block_size, 1, 1), grid=(grid_size, 1), shared=(4 + 4) * 373 * size_float32)
        new_centroids = new_centroids_gpu.get()

        if np.allclose(centroids, new_centroids, atol=1e-5):
            break
        centroids = new_centroids
        centroids_gpu.set(centroids)

    labels = labels_gpu.get()
    centroids = centroids.reshape(-1, data.shape[1])

    # Формування кластерів
    clusters = {i: [] for i in range(k)}
    for idx, label in enumerate(labels):
        clusters[label].append(data[idx])

    return clusters, centroids


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


cluster_names = [
    "Середній",
    "Високий",
    "Низький",
    "Задовільний"
]


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

        if i == 0 and it == 0:
            plot_clusters(clusters, centroids, 'Capitals', region_names, it)
        elif i == 0 and it == 1:
            plot_clusters(clusters, centroids, '2-Indicators(First iteration)', cluster_names, it)

        if is_converged:
            break

        centroids = new_centroids

    return clusters, centroids