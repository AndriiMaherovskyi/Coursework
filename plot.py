import numpy as np
import matplotlib.pyplot as plt


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
