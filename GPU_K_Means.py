import numpy as np
import random
import pycuda.driver as cuda
from pycuda import compiler


def k_means_gpu(data, k, max_iterations=100):
    # CUDA-код для призначення кластерів і оновлення центроїдів
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

    __global__ void compute_new_centroids(float *data, int *cluster_assignments, float *centroids, int num_data, int num_centroids, int data_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_centroids) return;

        int *cluster_sizes = new int[num_centroids]();  // Ініціалізація розмірів кластерів
        for (int i = 0; i < num_data; i++) {
            if (cluster_assignments[i] == idx) {
                for (int j = 0; j < data_size; j++) {
                    centroids[idx * data_size + j] += data[i * data_size + j];
                }
                cluster_sizes[idx]++;
            }
        }

        if (cluster_sizes[idx] > 0) {
            for (int j = 0; j < data_size; j++) {
                centroids[idx * data_size + j] /= cluster_sizes[idx];
            }
        }

        delete[] cluster_sizes;
    }
    """

    # Компільоване ядро
    mod = compiler.SourceModule(cuda_code)
    assign_kernel = mod.get_function("assign_to_clusters")
    compute_centroids_kernel = mod.get_function("compute_new_centroids")

    # Перетворимо дані та центроїди на одномірні масиви
    data_array = np.array([list(item.values())[0] for item in data], dtype=np.float32).flatten()
    centroids = random.sample(data, k)
    centroid_array = np.array([list(c.values())[0] for c in centroids], dtype=np.float32).flatten()

    # Виділимо пам'ять на GPU та скопіюємо дані
    data_gpu = cuda.mem_alloc(data_array.nbytes)
    centroid_gpu = cuda.mem_alloc(centroid_array.nbytes)
    cuda.memcpy_htod(data_gpu, data_array)
    cuda.memcpy_htod(centroid_gpu, centroid_array)

    num_data = len(data)
    result_gpu = cuda.mem_alloc(num_data * 4)  # int32
    result_cpu = np.zeros(num_data, dtype=np.int32)

    block_size = 256
    grid_size = (num_data + block_size - 1) // block_size

    for _ in range(max_iterations):
        assign_kernel(data_gpu, centroid_gpu, result_gpu, np.int32(num_data), np.int32(len(centroids)), np.int32(2),
                      block=(block_size, 1, 1), grid=(grid_size, 1))

        cuda.memcpy_dtoh(result_cpu, result_gpu)

        # Оновлюємо центроїди на GPU
        compute_centroids_kernel(data_gpu, result_gpu, centroid_gpu, np.int32(num_data), np.int32(len(centroids)), np.int32(2),
                                 block=(block_size, 1, 1), grid=(len(centroids), 1))

        # Перевіряємо на конвергенцію (можна додати перевірку, якщо потрібно)

    return clusters_from_result(result_cpu, data), centroid_array


def clusters_from_result(assignments, data):
    clusters = {}
    for idx, cluster_idx in enumerate(assignments):
        country_name = list(data[idx].keys())[0]
        values = np.array(list(data[idx].values())[0])

        if cluster_idx not in clusters:
            clusters[cluster_idx] = []

        clusters[cluster_idx].append({country_name: values})

    return clusters