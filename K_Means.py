import numpy as np
import random


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


def k_means(data, k, max_iterations=100):
    centroids = random.sample(data, k)
    clusters = {}

    for i in range(max_iterations):

        clusters = assign_to_clusters_cpu(data, centroids)
        new_centroids = update_centroids(clusters)

        # Check convergence
        is_converged = all(
            np.allclose(list(centroid.values())[0], list(new_centroid.values())[0])
            for centroid, new_centroid in zip(centroids, new_centroids)
        )

        if is_converged:
            break

        centroids = new_centroids

    return clusters, centroids