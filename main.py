from time import time

from k_means import k_means, show_clusters
from plot import plot_clusters
from Dictionary_and_arrays_converts import *

region_names = ["Західна Європа", "Східна Європа", "Північна Америка", "Південна Америка", "Азія", "Африка", "Океанія"]
k = 7

if __name__ == "__main__":

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