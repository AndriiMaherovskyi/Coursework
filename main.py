import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from time import time

from K_Means import k_means
from Dictionary_and_arrays_converts import (make_indicator_dictionary, make_capital_dictionary,
                                            convert_clusters_to_arrays, next_step_clustering_list)
from plot import plot_clusters



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

k = 7

if __name__ == "__main__":

    # --- --- -- Capitals --- --- ---

    # Читаємо CSV-файл
    capitals = pd.read_csv('capitals-location.csv')
    name_column = capitals.columns[0]  # Перша колонка містить назву країни
    coordinate_column = capitals.columns[2:4]  # Остання колонка містить числові дані

    grouped_with_country = make_capital_dictionary(name_column, coordinate_column)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Countries quantity(capitals): {len(grouped_with_country)}")

    # Час виконання на CPU
    start_time_f = time()
    clusters_f, centroids_f = k_means(grouped_with_country, k, 'CPU')
    cpu_time_f = time() - start_time_f

    print("Capitals clusters (CPU):")
    show_clusters(clusters_f)
    print(f"CPU Time: {cpu_time_f:.5f} seconds")
    print()
    plot_clusters(clusters_f, centroids_f, 'Capitals')

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
    plot_clusters(clusters_f_gpu, centroids_f_gpu, 'Capitals')

    # вивід метрик
    X_f_gpu, labels_f_gpu = convert_clusters_to_arrays(clusters_f_gpu)
    output_scores(X_f_gpu, labels_f_gpu)

    # --- --- -- Indicators --- --- ---

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

    X_s, labels_s = convert_clusters_to_arrays(clusters_s)
    output_scores(X_s, labels_s)

    start_time_s_gpu = time()
    clusters_s_gpu, centroids_s_gpu = k_means(grouped_with_country, k, 'GPU')
    elapsed_time = time() - start_time_s_gpu
    show_clusters(clusters_s_gpu)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()

    X_s_gpu, labels_s_gpu = convert_clusters_to_arrays(clusters_s_gpu)
    output_scores(X_s_gpu, labels_s_gpu)

    # --- --- -- Combine clusterization --- --- ---

    nex_step_dict = next_step_clustering_list(clusters_f, clusters_s)
    print("\n --- --- --- --- --- --- --- --- --- --- ---\n")
    print(f"Indicators and capitals(Combine): {len(grouped_with_country)}")

    start_time_l = time()
    clusters_l, centroids_l = k_means(nex_step_dict, k, 'CPU')
    cpu_time = time() - start_time_l
    show_clusters(clusters_l)
    print(f"CPU Time: {cpu_time:.5f} seconds")
    print()

    X_l, labels_l = convert_clusters_to_arrays(clusters_l)
    output_scores(X_l, labels_l)

    print("--- --- --- Розпаралелений обрахунок --- --- ---")
    print()

    start_time_l_gpu = time()
    clusters_l_gpu, centroids_l_gpu = k_means(nex_step_dict, k, 'GPU')
    elapsed_time = time() - start_time_l_gpu
    show_clusters(clusters_l_gpu)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()

    X_l_gpu, labels_l_gpu = convert_clusters_to_arrays(clusters_l_gpu)
    output_scores(X_l_gpu, labels_l_gpu)

    plot_clusters(clusters_l_gpu, centroids_l_gpu, 'Combine')
