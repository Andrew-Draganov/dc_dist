#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:02:22 2023

@author: Ellen
"""

import numpy as np
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import minmax_scale
from DCFcluster.DCFcluster import DCFcluster


""" only Synth Data """

def loadSynthDatasets(to_load):
    
    num_points = 10000
    blobs_centers = 10
    dim = 50
    
    points = []
    labels = []
    
    for datatype in to_load:
        if datatype == 'b1':
            # make balanced blobs (b1)
            points_b1, labels_b1 = make_blobs(n_samples=num_points, centers=blobs_centers, n_features=dim, random_state=1)
            points.append(points_b1)
            labels.append(labels_b1)
        if datatype == 'b2':
            # make unbalanced blobs (b2)
            cluster_numbers_blobs_unbalanced = [882, 1577, 913, 1335, 539, 703, 1393, 744, 803, 1111]
            points_b2, labels_b2 = make_blobs(n_samples=cluster_numbers_blobs_unbalanced, centers=None, n_features=dim, random_state=1)
            points.append(points_b2)
            labels.append(labels_b2)
        if datatype == 'b3':
            # load unbalanced blobs with uniforn noise (b3)  
            dataset_blobs_unbalanced_noise = np.load("./data/synth/blobs_unbalanced_noise.npy")
            points_b3 = dataset_blobs_unbalanced_noise[:, :-1]
            labels_b3 = dataset_blobs_unbalanced_noise[:, -1]
            points.append(points_b3)
            labels.append(labels_b3)
        if datatype == 'd1':
            # load density dataset with same density (d1)
            dataset_same_dens = np.load("./data/synth/synth_data_10000_10_50_0_0.npy")
            points_d1 = dataset_same_dens[:, :-1]
            labels_d1 = dataset_same_dens[:, -1]
            points.append(points_d1)
            labels.append(labels_d1)
        if datatype == 'd2':
            # load density dataset with different density (d2)
            dataset_diff_dens = np.load("./data/synth/synth_data_10000_10_50_vardensity_0_0.npy")
            points_d2 = dataset_diff_dens[:, :-1]
            labels_d2 = dataset_diff_dens[:, -1]
            points.append(points_d2)
            labels.append(labels_d2)
        if datatype == 'd3':
            # load density dataset with different density + noise (d3)
            dataset_diff_dens_noise = np.load("./data/synth/synth_data_10000_10_50_vardensity_1000_0.npy")
            points_d3 = dataset_diff_dens_noise[:, :-1]
            labels_d3 = dataset_diff_dens_noise[:, -1]
            points.append(points_d3)
            labels.append(labels_d3)
    
    return points, labels

def loadReducedSynthDatasets(to_load, dims, reduction_types):
    all_reduced_data = []
    for datatype in to_load:
        reductiontype_data = []
        if 'pca' in reduction_types:
            embeddings_pca = []
            for i in dims:
                filename = "./reducedSynthDatasets/"+str(datatype)+"/pca"+str(i)+"_"+str(datatype)+".txt"
                pca = open(filename).read()
                pca = ast.literal_eval(pca)
                pca = np.array(pca)
                embeddings_pca.append(pca)
            reductiontype_data.append(embeddings_pca)
        
        if 'tsne' in reduction_types:
            embeddings_tsne = []
            for i in dims:
                filename = "./reducedSynthDatasets/"+str(datatype)+"/tsne"+str(i)+"_"+str(datatype)+".txt"
                tsne = open(filename).read()
                tsne = ast.literal_eval(tsne)
                tsne = np.array(tsne)
                embeddings_tsne.append(tsne)
            reductiontype_data.append(embeddings_tsne)
        
        if 'umap' in reduction_types:
            embeddings_umap = []
            for i in dims:
                filename = "./reducedSynthDatasets/"+str(datatype)+"/umap"+str(i)+"_"+str(datatype)+".txt"
                umap = open(filename).read()
                umap = ast.literal_eval(umap)
                umap = np.array(umap)
                embeddings_umap.append(umap)
            reductiontype_data.append(embeddings_umap)
            
        if 'mds1' in reduction_types:
            embeddings_mds_ours_1 = []
            for i in dims:
                filename = "./reducedSynthDatasets/"+str(datatype)+"/mds_ours_"+str(i)+"_1_"+str(datatype)+".txt"
                mds_1 = open(filename).read()
                mds_1 = ast.literal_eval(mds_1)
                mds_1 = np.array(mds_1)
                embeddings_mds_ours_1.append(mds_1)
            reductiontype_data.append(embeddings_mds_ours_1)
        
        if 'mds5' in reduction_types:
            embeddings_mds_ours_5 = []
            for i in dims:
                filename = "./reducedSynthDatasets/"+str(datatype)+"/mds_ours_"+str(i)+"_5_"+str(datatype)+".txt"
                mds_5 = open(filename).read()
                mds_5 = ast.literal_eval(mds_5)
                mds_5 = np.array(mds_5)
                embeddings_mds_ours_5.append(mds_5)
            reductiontype_data.append(embeddings_mds_ours_5)
           
        if 'mds10' in reduction_types:
            embeddings_mds_ours_10 = []
            for i in dims:
                filename = "./reducedSynthDatasets/"+str(datatype)+"/mds_ours_"+str(i)+"_10_"+str(datatype)+".txt"
                mds_10 = open(filename).read()
                mds_10 = ast.literal_eval(mds_10)
                mds_10 = np.array(mds_10)
                embeddings_mds_ours_10.append(mds_10)
            reductiontype_data.append(embeddings_mds_ours_10)
        all_reduced_data.append(reductiontype_data)

    return all_reduced_data


def correct_DBSCAN_clustering(clustering):
    clustering_max = np.max(clustering.labels_) + 1
    corrected_labels = clustering.labels_
    for i in range(len(corrected_labels)):
        if corrected_labels[i] == -1:
            corrected_labels[i] = clustering_max
            clustering_max += 1
    clustering.labels_ = corrected_labels
    return clustering

def get_good_DBSCAN_minPoints(points):
    # set minPoints to dim*2
    return points.shape[1]*2

def get_good_DBSCAN_eps_with_domain_knowledge(points, minPoints, num_clusters, labels):
    # set eps to max distance of k-nearest neighbor in all clusters
    good_eps_all = []
    n_neighbors = minPoints-1
    
    for i in range(num_clusters):
        points_in_cluster = points[np.where(labels==i)]
        neigh, neigh_indices = NearestNeighbors(n_neighbors=n_neighbors).fit(points_in_cluster).kneighbors(points_in_cluster)
        dist_desired_neigh = neigh[:, -1]
        good_eps = np.max(dist_desired_neigh)
        good_eps_all.append(good_eps)
    result = np.max(good_eps_all)
    return result


def compute_clusterings(dataset, points_original, labels_original, num_clusters, clustering_types):
    clusterings = []
    aris = []
    
    for dataset_index in tqdm(range(len(dataset))):
        dataset_aris = []
        for reductiontype_idex in range(len(dataset[dataset_index])):
            reductiontype_aris = []
            for dim_index in range(len(dataset[dataset_index][reductiontype_idex])):
                dimtype_aris = []
                points = dataset[dataset_index][reductiontype_idex][dim_index]
                labels_gt = labels_original[dataset_index]

                if 'dbscan' in clustering_types:
                    minPoints = get_good_DBSCAN_minPoints(points)
                    eps = get_good_DBSCAN_eps_with_domain_knowledge(points, minPoints, num_clusters, labels_gt)
            
                    dbscan = DBSCAN(min_samples=minPoints, eps=eps).fit(points)
                    dbscan = correct_DBSCAN_clustering(dbscan)
                    dbscan_ari_gt = adjusted_rand_score(labels_gt, dbscan.labels_)
                    clusterings.append(dbscan)
                    dimtype_aris.append(dbscan_ari_gt)
        
                if 'kmeans' in clustering_types:
                    kmeans = KMeans(n_clusters=num_clusters).fit(points)
                    kmeans_ari_gt = adjusted_rand_score(labels_gt, kmeans.labels_)
                    clusterings.append(kmeans)
                    dimtype_aris.append(kmeans_ari_gt)
        
                if 'spectral' in clustering_types:                    
                    points = minmax_scale(points, feature_range=(0, 1), axis=0, copy=True)
                    
                    dim = points.shape[1]
                    gamma = 2/np.sqrt(dim)
                    
                    affinity_matrix = rbf_kernel(points, gamma=gamma)
                    
                    spectral_object = SpectralClustering(n_clusters=num_clusters, affinity = 'precomputed')
                    spectral = spectral_object.fit(affinity_matrix)
                    
                    spectral_ari_gt = adjusted_rand_score(labels_gt, spectral.labels_)
                    print(spectral_ari_gt)
                    clusterings.append(spectral)
                    dimtype_aris.append(spectral_ari_gt)

                if 'dcf' in clustering_types:
                    k = int(np.sqrt(points.shape[0]))
                    dcf = DCFcluster.train(X=points, k=k, beta=0.8)
                    dcf_ari_gt = adjusted_rand_score(labels_gt, dcf.labels)
                    clusterings.append(dcf)
                    dimtype_aris.append(dcf_ari_gt)
                    
                    plt.scatter(points[:, 0], points[:, 1], c=dcf.labels, cmap='Spectral')
                    plt.show()
                
                reductiontype_aris.append(dimtype_aris)
            dataset_aris.append(reductiontype_aris)
        aris.append(dataset_aris)
       
    return clusterings, aris

def saveResults(aris, filename, mode):
    with open(filename, mode) as f:
        f.write(str(aris))
                
def testClustering(to_load, dims, reduction_types, clustering_types):
    points, labels = loadSynthDatasets(to_load)
    reduced_data = loadReducedSynthDatasets(to_load, dims, reduction_types)
    
    _, aris = compute_clusterings(reduced_data, points, labels, 10, clustering_types)
    
    #saveResults(aris, "../Result_txts/TESTaris_synthData_spectral.txt", 'a')
    
    return aris

def clustering_comparison_subplots(ax, aris, title):
    ax.imshow(aris, cmap='coolwarm_r', interpolation='nearest', aspect="auto", vmin=0, vmax = 1)
    
    # show numbers in cells
    for i in range(len(aris[0])):
        for j in range(len(aris)):
            ax.text(i, j, round(aris[j][i], 2), ha='center', va='center')
    
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    ax.set_title(title)
    
    return ax

def createSynthPlot(filename):
    all_aris = open(filename).read()
    all_aris = ast.literal_eval(all_aris)
    all_aris = np.array(all_aris)
    
    aris_plotting_sorted = []
    
    for reductiontype_index in range(len(all_aris[0])):
        aris_sorted_reductionType = []
        for dim_index in range(len(all_aris[0][0])):
            aris_sorted_dim = []
            for dataset_index in range(2, len(all_aris)):
                aris = all_aris[dataset_index][reductiontype_index][dim_index]
                aris_sorted_dim.append(aris)
            aris_sorted_reductionType.append(np.array(aris_sorted_dim).T)
        aris_plotting_sorted.append(aris_sorted_reductionType)
    
    plt.rcParams.update({'font.size': 11})
    fig, (ax0, ax1, ax2) = plt.subplots(3, 6, figsize=(15, 4))
    
    ax0[0] = clustering_comparison_subplots(ax0[0], aris_plotting_sorted[0][0], "euclidean")
    ax1[0] = clustering_comparison_subplots(ax1[0], aris_plotting_sorted[0][1], "")
    ax2[0] = clustering_comparison_subplots(ax2[0], aris_plotting_sorted[0][2], "")
    ax0[1] = clustering_comparison_subplots(ax0[1], aris_plotting_sorted[1][0], "TSNE")
    ax1[1] = clustering_comparison_subplots(ax1[1], aris_plotting_sorted[1][1], "")
    ax2[1] = clustering_comparison_subplots(ax2[1], aris_plotting_sorted[1][2], "")
    ax0[2] = clustering_comparison_subplots(ax0[2], aris_plotting_sorted[2][0], "UMAP")
    ax1[2] = clustering_comparison_subplots(ax1[2], aris_plotting_sorted[2][1], "")
    ax2[2] = clustering_comparison_subplots(ax2[2], aris_plotting_sorted[2][2], "")
    ax0[3] = clustering_comparison_subplots(ax0[3], aris_plotting_sorted[3][0], "cd($\mu$=1)")
    ax1[3] = clustering_comparison_subplots(ax1[3], aris_plotting_sorted[3][1], "")
    ax2[3] = clustering_comparison_subplots(ax2[3], aris_plotting_sorted[3][2], "")
    ax0[4] = clustering_comparison_subplots(ax0[4], aris_plotting_sorted[4][0], "cd($\mu$=5)")
    ax1[4] = clustering_comparison_subplots(ax1[4], aris_plotting_sorted[4][1], "")
    ax2[4] = clustering_comparison_subplots(ax2[4], aris_plotting_sorted[4][2], "")
    ax0[5] = clustering_comparison_subplots(ax0[5], aris_plotting_sorted[5][0], "cd($\mu$=10)")
    ax1[5] = clustering_comparison_subplots(ax1[5], aris_plotting_sorted[5][1], "")
    ax2[5] = clustering_comparison_subplots(ax2[5], aris_plotting_sorted[5][2], "")
    
    ax2[0].set_xticks(ticks=range(len(aris_plotting_sorted)-2), labels = ["b3", "d1", "d2", "d3"])
    ax2[1].set_xticks(ticks=range(len(aris_plotting_sorted)-2), labels = ["b3", "d1", "d2", "d3"])
    ax2[2].set_xticks(ticks=range(len(aris_plotting_sorted)-2), labels = ["b3", "d1", "d2", "d3"])
    ax2[3].set_xticks(ticks=range(len(aris_plotting_sorted)-2), labels = ["b3", "d1", "d2", "d3"])
    ax2[4].set_xticks(ticks=range(len(aris_plotting_sorted)-2), labels = ["b3", "d1", "d2", "d3"])
    ax2[5].set_xticks(ticks=range(len(aris_plotting_sorted)-2), labels = ["b3", "d1", "d2", "d3"])
  
    ax0[0].set_yticks(ticks=range(len(aris_plotting_sorted[0][0])), labels = ["DBSCAN", "kMeans", "Spectral", "DCF"])
    ax1[0].set_yticks(ticks=range(len(aris_plotting_sorted[0][0])), labels = ["DBSCAN", "kMeans", "Spectral", "DCF"])
    ax2[0].set_yticks(ticks=range(len(aris_plotting_sorted[0][0])), labels = ["DBSCAN", "kMeans", "Spectral", "DCF"])
    
    ax0[5].set_ylabel("dim=2")
    ax0[5].yaxis.set_label_position("right")
    
    ax1[5].set_ylabel("dim=10")
    ax1[5].yaxis.set_label_position("right")
    
    ax2[5].set_ylabel("dim=50")
    ax2[5].yaxis.set_label_position("right")
    
    plt.tight_layout(pad=0.2)
    plt.show()


if __name__ == '__main__':    
    data_types = ['d1', 'd2', 'd3']
    dims = [2]
    reduction_types = ['mds1','mds5', 'mds10']
    alg_types = ['spectral']
    
    res = testClustering(data_types, dims, reduction_types, alg_types)   
    print(res)
    















