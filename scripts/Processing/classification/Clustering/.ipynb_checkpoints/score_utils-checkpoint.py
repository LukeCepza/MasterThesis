import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def read_ans(filename):
    # Read the content of the answer text file and store it as a list of lines
    with open(filename, 'r') as file:
        answer_lines = file.readlines()
    a = np.zeros((len(answer_lines), 4))
    for i, line in enumerate(answer_lines):
        a[i, :] = [int(char) for char in line.strip()]  # Convert each character to an integer
    da = np.diff(a)
    da[da > 1] = 1
    da[da < 1] = -1
    return a, da

def score(filename, a, da):
    # Read the content of the answer text file and store it as a list of lines
    with open(filename, 'r') as file:
        response = file.readlines()
    r = np.zeros((len(response), 4))
    for i, line in enumerate(response):
        # Split the line by tab characters and convert each part to an integer
        r[i, :] = [int(value) for value in line.strip().split('\t') if value]
    dr = np.diff(r)
    dr[dr > 1] = 1
    dr[dr < 1] = -1
    r_ex = np.sum(a == r, axis=1)
    r_chg = np.sum(da == dr, axis=1)

    # Initialize counters for each number (1, 2, 3, 4) in each test
    count_correct = np.zeros((3, 4))  # 3 tests, 4 numbers

    # Update counters
    for test_idx in range(3):
        start_idx = sum([17, 20, 20][:test_idx])
        end_idx = start_idx + [17, 20, 20][test_idx]
        for row_idx in range(start_idx, end_idx):
            for num in range(1, 5):
                if num in r[row_idx] and num in a[row_idx] and r[row_idx, np.where(a[row_idx] == num)] == num:
                    count_correct[test_idx, num - 1] += 1

    return r_ex, r_chg, count_correct

def summary(r_ex, r_chg, count_correct):
    output = np.zeros(20)
    output[0] = np.sum(r_ex[0:17])
    output[1] = np.sum(r_ex[17:37])
    output[2] = np.sum(r_ex[37:57])
    output[4] = np.sum(r_chg[0:17])
    output[5] = np.sum(r_chg[17:37])
    output[6] = np.sum(r_chg[37:57])
    output[3] = np.sum(r_ex)
    output[7] = np.sum(r_chg)

    # Flatten count_correct to be added to the output array
    flat_counts = count_correct.flatten()
    output[8:20] = flat_counts

    return output

def summary_porcentage(df):
    df['eAir'] = (df['eAir'] / 68) * 100
    df['eVib'] = (df['eVib'] / 80) * 100
    df['eCar'] = (df['eCar'] / 80) * 100
    df['eAll'] = (df['eAll'] / 228) * 100
    df['dAir'] = (df['dAir'] / 51) * 100
    df['dVib'] = (df['dVib'] / 60) * 100
    df['dCar'] = (df['dCar'] / 60) * 100
    df['dAll'] = (df['dAll'] / 171) * 100
    return df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

def plot_clust_metrics(df_clust_scaled, method='kmeans', cluster_range=range(2, 11)):
    # Initialize lists to store the evaluation scores
    calinski_harabasz_scores = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    if method == 'dbscan':
        # Run DBSCAN once and store the results
        model = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = model.fit_predict(df_clust_scaled)
        if len(set(cluster_labels)) > 1:
            calinski_harabasz_scores.append(calinski_harabasz_score(df_clust_scaled, cluster_labels))
            silhouette_scores.append(silhouette_score(df_clust_scaled, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(df_clust_scaled, cluster_labels))
        else:
            calinski_harabasz_scores.append(np.nan)
            silhouette_scores.append(np.nan)
            davies_bouldin_scores.append(np.nan)
    else:
        for n_clusters in cluster_range:
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            
            cluster_labels = model.fit_predict(df_clust_scaled)
            
            calinski_harabasz_scores.append(calinski_harabasz_score(df_clust_scaled, cluster_labels))
            silhouette_scores.append(silhouette_score(df_clust_scaled, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(df_clust_scaled, cluster_labels))
    
    # Plotting the scores
    plt.figure(figsize=(12, 4))
    
    if method == 'dbscan':
        # For DBSCAN, plot a single point since we only have one set of scores
        plt.subplot(1, 3, 1)
        plt.scatter([1], calinski_harabasz_scores, marker='o')
        plt.title('Calinski-Harabasz Index')
        plt.xlabel('DBSCAN')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 2)
        plt.scatter([1], silhouette_scores, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('DBSCAN')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 3)
        plt.scatter([1], davies_bouldin_scores, marker='o')
        plt.title('Davies-Bouldin Index')
        plt.xlabel('DBSCAN')
        plt.ylabel('Score')
    else:
        plt.subplot(1, 3, 1)
        plt.plot(cluster_range, calinski_harabasz_scores, marker='o')
        plt.title('Calinski-Harabasz Index')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 2)
        plt.plot(cluster_range, silhouette_scores, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 3)
        plt.plot(cluster_range, davies_bouldin_scores, marker='o')
        plt.title('Davies-Bouldin Index')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_clust_metrics(df_clust_scaled, method='kmeans', cluster_range=range(2, 11))
# plot_clust_metrics(df_clust_scaled, method='hierarchical', cluster_range=range(2, 11))
# plot_clust_metrics(df_clust_scaled, method='dbscan')


# Example usage:
# plot_clust_metrics(df_clust_scaled, method='kmeans', cluster_range=range(2, 11))
# plot_clust_metrics(df_clust_scaled, method='hierarchical', cluster_range=range(2, 11))
# plot_clust_metrics(df_clust_scaled, method='dbscan')



def plot_clust_metrics_old(df_clust_scaled, cluster_range = range(2, 11)):  
    
    # Initialize lists to store the evaluation scores
    calinski_harabasz_scores = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(df_clust_scaled)
        
        calinski_harabasz_scores.append(calinski_harabasz_score(df_clust_scaled, cluster_labels))
        silhouette_scores.append(silhouette_score(df_clust_scaled, cluster_labels))
        davies_bouldin_scores.append(davies_bouldin_score(df_clust_scaled, cluster_labels))
    
    # Plotting the scores
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(cluster_range, calinski_harabasz_scores, marker='o')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 3)
    plt.plot(cluster_range, davies_bouldin_scores, marker='o')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.show()

def plot_clusts(df_scores, df_clust_scaled, n_clusters=3):

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Adjust the number of clusters as needed
    kmeans.fit(df_clust_scaled)
    
    # Add cluster labels to the DataFrame
    df_scores['Cluster'] = kmeans.labels_
    
    # Create a mapping of the current cluster labels to the new labels
    cluster_mapping = {0: 0, 1: 3, 2: 1}  # Adjust based on your specific requirements
    
    # Apply the mapping to the 'Cluster' column
    df_scores['Cluster'] = df_scores['Cluster'].map(cluster_mapping)
    
    # If you need to swap cluster labels 1 and 2 after the initial renaming
    df_scores['Cluster'] = df_scores['Cluster'].replace({3: 2})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Plot for eAll vs dAll
    scatter1 = axes[0].scatter(df_scores['eAll'], df_scores['dAll'], c=df_scores['Cluster'], cmap='viridis')
    axes[0].set_xlabel('All stimui score')
    axes[0].set_ylabel('All difference score')
    
    # Plot for dAll vs eAir
    scatter2 = axes[1].scatter(df_scores['dAll'], df_scores['eAir'], c=df_scores['Cluster'], cmap='viridis')
    axes[1].set_xlabel('All difference score')
    axes[1].set_ylabel('Air score')
    axes[1].set_title('K-Means Clustering')
    
    # Plot for eAir vs eAll
    scatter3 = axes[2].scatter(df_scores['eAir'], df_scores['eAll'], c=df_scores['Cluster'], cmap='viridis')
    axes[2].set_xlabel('Air score')
    axes[2].set_ylabel('All stimui score')
    
    # Add a color bar to indicate clusters
    cbar = fig.colorbar(scatter3, ax=axes.ravel().tolist(), label='Cluster')
    plt.show()
    return df_scores