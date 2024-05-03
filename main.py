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

from K_Means import k_means
from GPU_K_Means import k_means_gpu
from Dictionary_and_arrays_converts import (make_indicator_dictionary, make_capital_dictionary,
                                            convert_clusters_to_arrays, next_step_clustering_list)

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




if __name__ == "__main__":
    # Читаємо CSV-файл
    df = pd.read_csv('Worldbank-data-all.csv')

    first_column = df.columns[0]  # Перша колонка містить назву країни
    last_column = df.columns[-1]  # Остання колонка містить числові дані
    grouped_with_country = make_indicator_dictionary(first_column, last_column)
    print(f"Countries quantity(indicators): {len(grouped_with_country)}")
    k = 7  # Кількість кластерів

    # Час виконання на CPU
    start_time = time()
    clusters_i, centroids_i = k_means(grouped_with_country, k)
    cpu_time = time() - start_time

    print("Clusters (CPU):")
    show_clusters(clusters_i)
    print(f"CPU Time: {cpu_time:.5f} seconds")
    print()
    plot_clusters(clusters_i, centroids_i, 'Capitals')  # Plot відображення утворених кластерів

    # вивід метрик
    X_i, labels_i = convert_clusters_to_arrays(clusters_i)
    evaluate_clustering(X_i, labels_i)

    start_time = time()
    clusters_ig, centroids_ig = k_means_gpu(grouped_with_country, k)
    elapsed_time = time() - start_time

    print("Clusters(GPU):")
    show_clusters(clusters_ig)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()

    # вивід метрик
    X_ig, labels_ig = convert_clusters_to_arrays(clusters_ig)
    print("GPU Evaluation:")
    evaluate_clustering(X_ig, labels_ig)

    # --- ---- Столиці ---- ---

    capitals = pd.read_csv('capitals-location.csv')
    name_column = capitals.columns[0]  # Перша колонка містить назву країни
    coordinate_column = capitals.columns[2:4]  # Остання колонка містить числові дані

    grouped_with_country = make_capital_dictionary(name_column, coordinate_column)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Countries quantity(capitals): {len(grouped_with_country)}")
    k = 7  # Кількість кластерів

    # Час виконання на CPU
    start_time = time()
    clusters_f, centroids_f = k_means(grouped_with_country, k)
    cpu_time = time() - start_time

    print("Clusters (CPU):")
    show_clusters(clusters_f)
    print(f"CPU Time: {cpu_time:.5f} seconds")
    print()
    plot_clusters(clusters_f, centroids_f, 'Capitals')  # Plot відображення утворених кластерів

    # Час виконання на GPU
    start_time = time()
    clusters_s, centroids_s = k_means_gpu(grouped_with_country, k)
    elapsed_time = time() - start_time

    print("Clusters(GPU):")
    show_clusters(clusters_s)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()
    #plot_clusters(clusters_s, centroids_s, 'Capitals')  # Plot відображення утворених кластерів

    # Метрики для обох підходів
    X_gpu, labels_gpu = convert_clusters_to_arrays(clusters_s)
    print("GPU Evaluation:")
    evaluate_clustering(X_gpu, labels_gpu)


    # --- ---- Комбінована кластеризація ---- ---

    nex_step_dict = next_step_clustering_list(clusters_i, clusters_f)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Indicators and capitals(Combine): {len(grouped_with_country)}")
    k = 7  # Кількість кластерів


    # Час виконання на CPU
    start_time = time()
    clusters_l, centroids_l = k_means(nex_step_dict, k)
    cpu_time = time() - start_time

    print("Clusters (CPU):")
    show_clusters(clusters_l)
    print(f"CPU Time: {cpu_time:.5f} seconds")
    print()
    plot_clusters(clusters_l, centroids_l, 'Capitals')  # Plot відображення утворених кластерів

    X_gpu, labels_gpu = convert_clusters_to_arrays(clusters_l)
    print("GPU Evaluation:")
    evaluate_clustering(X_gpu, labels_gpu)
    print()

    print("--- --- --- Розпаралелений результат --- --- ---")
    # Час виконання на GPU
    start_time = time()
    clusters_lg, centroids_lg = k_means_gpu(nex_step_dict, k)
    elapsed_time = time() - start_time

    print("Clusters(GPU):")
    show_clusters(clusters_lg)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()

    # Метрики
    X_gpu, labels_gpu = convert_clusters_to_arrays(clusters_lg)
    print("GPU Evaluation:")
    evaluate_clustering(X_gpu, labels_gpu)