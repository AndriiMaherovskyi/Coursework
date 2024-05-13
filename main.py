import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from time import time

from K_Means import *
from Dictionary_and_arrays_converts import *
from plot import plot_clusters


def show_clusters(clusters):
    for cluster_index, cluster_data in clusters.items():
        country_names = [list(item.keys())[0] for item in cluster_data]
        print(f"Cluster {cluster_index}: {country_names}")


# Функція для збору даних на теплову карту
def arr_for_map(grouped_with_country, grouped_with_country_s):
    for country_data in grouped_with_country:
        # Отримуємо ключ країни
        country_key = list(country_data.keys())[0]

        # Шукаємо індекс елементу з таким ключем у grouped_with_country_s
        index_country_s = next((i for i, item in enumerate(grouped_with_country_s) if country_key in item), None)

        # Якщо знайдено індекс, додаємо дані з grouped_with_country_s до відповідного масиву
        if index_country_s is not None:
            # Отримуємо значення для ключа з grouped_with_country_s
            country_s_values = grouped_with_country_s[index_country_s][country_key]

            # Отримуємо значення для ключа з grouped_with_country
            country_values = country_data[country_key]

            # Злиття значень та оновлення значення в grouped_with_country
            country_data[country_key] = np.concatenate((country_values, country_s_values))

    return grouped_with_country


def output_scores(X, labels):
    # Оцінюємо за допомогою різних метрик:
    silhouette_avg = silhouette_score(X, labels)
    print(f"Силуетний індекс: {silhouette_avg}")

    db_score = davies_bouldin_score(X, labels)
    print(f"Індекс Девіса-Болдіна: {db_score}")

    ch_score = calinski_harabasz_score(X, labels)
    print(f"Індекс Каліна-Харабаша: {ch_score}\n")


cluster_names = [
    "Середній",
    "Високий",
    "Низький",
    "Задовільний"
]


def kluster_all(last_column):
    last_elements = df[last_column].to_numpy()  # перетворюємо у масив NumPy

    # Заміна значень '..' на 0 та перетворення на float
    last_elements = np.where(last_elements == '..', 0, last_elements)  # замінюємо '..' на 0
    last_elements = last_elements.astype(float)  # перетворюємо в float

    group_size = 217
    num_groups = len(last_elements) // group_size
    grouped_array = np.array(np.array_split(last_elements[:num_groups * group_size], num_groups))

    return grouped_array


# Додаємо назву країни до кожного індикатора
def add_name(first_column, grouped_array):
    first_column = df[first_column].to_numpy()

    grouped_with_country = []
    for indicator_line in grouped_array:
        line = []
        for idx, indicator in enumerate(indicator_line):
            country_data = {first_column[idx * 1492]: np.array([indicator])}
            line.append(country_data)
        grouped_with_country.append(line)

    return grouped_with_country


if __name__ == "__main__":

    df = pd.read_csv('Worldbank-data-all.csv')
    first_column = df.columns[0]  # Перша колонка містить назву країни
    last_column = df.columns[-1]  # Остання колонка містить числові дані
    grouped_with_indicators = kluster_all(last_column)

    res = add_name(first_column, grouped_with_indicators)

    # Тест кейс
    ######################
    start_time_s = time()
    for i in res:
        clusters_cpu, centroids_cpu = k_means(i, 7, 'CPU')
    cpu_time = time() - start_time_s
    print(f"Час виконання CPU: {cpu_time}")
    ######################

    # Тест кейс GPU
    ######################
    start_time = time()
    clusters_gpu, centroids_gpu = cuda_k_means(grouped_with_indicators, 7)
    gpu_time = time() - start_time
    print(f"Час виконання GPU: {gpu_time}")
    ######################

    grouped_with_country = make_indicator_dictionary(first_column, last_column, 'i')
    print(f"Countries quantity(indicators): {len(grouped_with_country)}")

    clusters_output, centroids_output = k_means(grouped_with_country, 7, 'CPU')
    show_clusters(clusters_output)

    X_s_gpu, labels_s_gpu = convert_clusters_to_arrays(clusters_output)
    output_scores(X_s_gpu, labels_s_gpu)

'''
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
    # plot_clusters(clusters_f, centroids_f, 'Capitals', region_names)

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
    # plot_clusters(clusters_f_gpu, centroids_f_gpu, 'Capitals', region_names)

    # вивід метрик
    X_f_gpu, labels_f_gpu = convert_clusters_to_arrays(clusters_f_gpu)
    output_scores(X_f_gpu, labels_f_gpu)

    # --- --- -- Two indicators --- --- ---
    k = 4
    df = pd.read_csv('Indicators-2.csv')
    first_column = df.columns[0]  # Перша колонка містить назву країни
    last_column = df.columns[-1]  # Остання колонка містить числові дані
    grouped_with_country = make_indicator_dictionary(first_column, last_column, 'e')
    print(f"Countries quantity(two indicators): {len(grouped_with_country)}")

    start_time_e = time()
    clusters_e, centroids_e = k_means(grouped_with_country, k, 'CPU', 1)
    cpu_time = time() - start_time_e
    show_clusters(clusters_e)
    print(f"CPU Time: {cpu_time:.5f} seconds")
    print()
    plot_clusters(clusters_e, centroids_e, '2-Indicators', cluster_names)

    X_e, labels_e = convert_clusters_to_arrays(clusters_e)
    output_scores(X_e, labels_e)

    start_time_e_gpu = time()
    clusters_e_gpu, centroids_e_gpu = k_means(grouped_with_country, k, 'GPU')
    elapsed_time = time() - start_time_e_gpu
    show_clusters(clusters_e_gpu)
    print(f"GPU Time: {elapsed_time:.5f} seconds")
    print()

    X_e_gpu, labels_e_gpu = convert_clusters_to_arrays(clusters_e_gpu)
    output_scores(X_e_gpu, labels_e_gpu)


# --- --- -- Indicators --- --- ---
    # df = pd.read_csv('Worldbank-data-all.csv')
    # first_column = df.columns[0]  # Перша колонка містить назву країни
    # last_column = df.columns[-1]  # Остання колонка містить числові дані
    # grouped_with_country = make_indicator_dictionary(first_column, last_column, 'i')
    # print(f"Countries quantity(indicators): {len(grouped_with_country)}")
# 
    # start_time_s = time()
    # clusters_s, centroids_s = k_means(grouped_with_country, 7, 'CPU')
    # cpu_time = time() - start_time_s
    # show_clusters(clusters_s)
    # print(f"CPU Time: {cpu_time:.5f} seconds")
    # print()
# 
    # X_s, labels_s = convert_clusters_to_arrays(clusters_s)
    # output_scores(X_s, labels_s)
# 
    # start_time_s_gpu = time()
    # clusters_s_gpu, centroids_s_gpu = k_means(grouped_with_country, 7, 'GPU')
    # elapsed_time = time() - start_time_s_gpu
    # show_clusters(clusters_s_gpu)
    # print(f"GPU Time: {elapsed_time:.5f} seconds")
    # print()
# 
    # X_s_gpu, labels_s_gpu = convert_clusters_to_arrays(clusters_s_gpu)
    # output_scores(X_s_gpu, labels_s_gpu)
'''
