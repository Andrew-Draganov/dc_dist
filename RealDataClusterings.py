#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:30:09 2023

"""

import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import glob

from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.datasets import make_blobs, fetch_olivetti_faces
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import minmax_scale
from DCFcluster.DCFcluster import DCFcluster

from experiment_utils.get_data import get_dataset

from DimReduceDatasets import loadRealDatasets

""" only Real Data """

def getReducedRealDatasets(to_load, reduction_types, dims):
    reduced_all = []
    for datatype in to_load:
        reduced_dataset = loadReducedRealDatasets(datatype, reduction_types, dims)
        reduced_all.append(reduced_dataset)
    return reduced_all
        
def loadReducedRealDatasets(datatype, reduction_types, dims):
    reduced_datasets = []
    if 'pca' in reduction_types: 
        embeddings_pca = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/pca_"+str(dim)+"_"+str(datatype)+".txt"
            pca = open(filename).read()
            pca = ast.literal_eval(pca)
            pca = np.array(pca)
            embeddings_pca.append(pca)
        reduced_datasets.append(embeddings_pca)
    
    if 'tsne' in reduction_types: 
        embeddings_tsne = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/tsne_"+str(dim)+"_"+str(datatype)+".txt"
            tsne = open(filename).read()
            tsne = ast.literal_eval(tsne)
            tsne = np.array(tsne)
            embeddings_tsne.append(tsne)
        reduced_datasets.append(embeddings_tsne)
        
    if 'umap' in reduction_types:
        embeddings_umap = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/umap_"+str(dim)+"_"+str(datatype)+".txt"
            umap = open(filename).read()
            umap = ast.literal_eval(umap)
            umap = np.array(umap)
            embeddings_umap.append(umap)
        reduced_datasets.append(embeddings_umap)

    if 'cosine' in reduction_types: 
        embeddings_cosine = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/mds_cosine_"+str(dim)+"_0_"+str(datatype)+".txt"
            cosine = open(filename).read()
            cosine = ast.literal_eval(cosine)
            cosine = np.array(cosine)
            embeddings_cosine.append(cosine)
        reduced_datasets.append(embeddings_cosine)
        
    if 'manhattan' in reduction_types: 
        embeddings_manhattan = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/mds_manhattan_"+str(dim)+"_0_"+str(datatype)+".txt"
            manhattan = open(filename).read()
            manhattan = ast.literal_eval(manhattan)
            manhattan = np.array(manhattan)
            embeddings_manhattan.append(manhattan)
        reduced_datasets.append(embeddings_manhattan)
        
    if 'mds1' in reduction_types:
        embeddings_mds1 = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/mds_ours_"+str(dim)+"_1_"+str(datatype)+".txt"
            #filename = "../reducedRealDatasets/mds_ours_2_1_coil20.txt"
            mds1 = open(filename).read()
            mds1 = ast.literal_eval(mds1)
            mds1 = np.array(mds1)
            embeddings_mds1.append(mds1)
        reduced_datasets.append(embeddings_mds1)
        
    if 'mds5' in reduction_types:
        embeddings_mds5 = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/mds_ours_"+str(dim)+"_5_"+str(datatype)+".txt"
            mds5 = open(filename).read()
            mds5 = ast.literal_eval(mds5)
            mds5 = np.array(mds5)
            embeddings_mds5.append(mds5)
        reduced_datasets.append(embeddings_mds5)
    
    if 'mds10' in reduction_types:
        embeddings_mds10 = []
        for dim in dims:
            filename = "../reducedRealDatasets/"+str(datatype)+"/mds_ours_"+str(dim)+"_10_"+str(datatype)+".txt"
            mds10 = open(filename).read()
            mds10 = ast.literal_eval(mds10)
            mds10 = np.array(mds10)
            embeddings_mds10.append(mds10)
        reduced_datasets.append(embeddings_mds10)
       
    return reduced_datasets


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
    # minPoints to dim*2
    return points.shape[1]*2

def getDBSCANeps_real(points, minPoints, num_clusters):
    # set eps to max distance of k-nearest neighbor
    n_neighbors = minPoints
    
    neigh, neigh_indices = NearestNeighbors(n_neighbors=n_neighbors).fit(points).kneighbors(points)
    dist_desired_neigh = neigh[:, -1]
    good_eps = np.mean(dist_desired_neigh)
    return good_eps


def compute_clusterings(to_load, dataset, points_original, labels_original, num_clusters_all, clustering_types):
    clusterings = []
    aris = []
    
    for dataset_index in tqdm(range(len(to_load))):
        dataset_aris = []
        num_clusters = num_clusters_all[to_load[dataset_index]]
        for reductiontype_index in tqdm(range(len(dataset[0]))):
            reductiontype_aris = []
            for dim_index in range(len(dataset[0][0])):
                dimtype_aris = []
                points = dataset[dataset_index][reductiontype_index][dim_index]
                points = minmax_scale(points, feature_range=(0, 1), axis=0, copy=False)
                labels_gt = labels_original[dataset_index]

                if 'dbscan' in clustering_types:
                    minPoints = get_good_DBSCAN_minPoints(points)
                    eps = getDBSCANeps_real(points, minPoints, num_clusters)
            
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
                    dim = points.shape[1]
                    gamma = 2/np.sqrt(dim)
                    
                    affinity_matrix = rbf_kernel(points, gamma = gamma)
                    
                    spectral_object = SpectralClustering(n_clusters=num_clusters, affinity = 'precomputed')
                    spectral = spectral_object.fit(affinity_matrix)

                    spectral_ari_gt = adjusted_rand_score(labels_gt, spectral.labels_)
                    clusterings.append(spectral)
                    dimtype_aris.append(spectral_ari_gt)
        
                if 'dcf' in clustering_types:
                    k = int(np.sqrt(points.shape[0]))
                    dcf = DCFcluster.train(X=points, k=k, beta=0.4)
                    dcf_ari_gt = adjusted_rand_score(labels_gt, dcf.labels)
                    clusterings.append(dcf)
                    dimtype_aris.append(dcf_ari_gt)
                    
                reductiontype_aris.append(dimtype_aris)
            dataset_aris.append(reductiontype_aris)
            
            print(dataset_aris)
            
        aris.append(dataset_aris)
       
    return clusterings, aris

def saveResults(aris, filename, mode):
    with open(filename, mode) as f:
        f.write(str(aris))
        
def expClusteringComparisonRealData(to_load, dims, reduction_types, clustering_types):
    points_all_datasets, labels_all_datasets = loadRealDatasets(to_load)
    reduced_datasets = getReducedRealDatasets(to_load, reduction_types, dims)
        
    num_clusters_all = {
        "olivetti":40, 
        "coil":100, 
        "coil5":5,
        "coil20":20,
        "drivface":4,
        "pendigits":10,
        "landsat":6,
        "letters":26
        }
    
    _, aris = compute_clusterings(to_load, reduced_datasets, points_all_datasets, labels_all_datasets, num_clusters_all, clustering_types)
    
    filename = "../Result_txts/new_aris_drivface_2_10.txt"
    #saveResults(aris, filename, 'w')
    

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


def createRealPlot(filename):
    all_aris = open(filename).read()
    all_aris = ast.literal_eval(all_aris)
    #all_aris = np.array(all_aris)
    
    aris_plotting_sorted = []
    
    for reductiontype_index in range(len(all_aris[0])):
        aris_sorted_reductionType = []
        for dim_index in range(len(all_aris[0][0])):
            aris_sorted_dim = []
            for dataset_index in range(len(all_aris)):
                aris = all_aris[dataset_index][reductiontype_index][dim_index]
                aris_sorted_dim.append(aris)
            aris_sorted_reductionType.append(np.array(aris_sorted_dim).T)
        aris_plotting_sorted.append(aris_sorted_reductionType)
    
    plt.rcParams.update({'font.size': 11})
    fig, (ax0, ax1, ax2) = plt.subplots(3, 6, figsize=(15, 4))
    
    ax0[0] = clustering_comparison_subplots(ax0[0], aris_plotting_sorted[0][0], "Euclidean")
    ax1[0] = clustering_comparison_subplots(ax1[0], aris_plotting_sorted[0][1], "")
    ax2[0] = clustering_comparison_subplots(ax2[0], aris_plotting_sorted[0][2], "")
    ax0[1] = clustering_comparison_subplots(ax0[1], aris_plotting_sorted[1][0], "cosine")
    ax1[1] = clustering_comparison_subplots(ax1[1], aris_plotting_sorted[1][1], "")
    ax2[1] = clustering_comparison_subplots(ax2[1], aris_plotting_sorted[1][2], "")
    ax0[2] = clustering_comparison_subplots(ax0[2], aris_plotting_sorted[2][0], "manhattan")
    ax1[2] = clustering_comparison_subplots(ax1[2], aris_plotting_sorted[2][1], "")
    ax2[2] = clustering_comparison_subplots(ax2[2], aris_plotting_sorted[2][2], "")
    ax0[3] = clustering_comparison_subplots(ax0[3], aris_plotting_sorted[3][0], "dc ($\mu$=1)")
    ax1[3] = clustering_comparison_subplots(ax1[3], aris_plotting_sorted[3][1], "")
    ax2[3] = clustering_comparison_subplots(ax2[3], aris_plotting_sorted[3][2], "")
    ax0[4] = clustering_comparison_subplots(ax0[4], aris_plotting_sorted[4][0], "dc ($\mu$=5)")
    ax1[4] = clustering_comparison_subplots(ax1[4], aris_plotting_sorted[4][1], "")
    ax2[4] = clustering_comparison_subplots(ax2[4], aris_plotting_sorted[4][2], "")
    ax0[5] = clustering_comparison_subplots(ax0[5], aris_plotting_sorted[5][0], "dc ($\mu$=10)")
    ax1[5] = clustering_comparison_subplots(ax1[5], aris_plotting_sorted[5][1], "")
    ax2[5] = clustering_comparison_subplots(ax2[5], aris_plotting_sorted[5][2], "")
    
    ax2[0].set_xticks(ticks=range(len(aris_plotting_sorted)-1), labels = ["coil5", "coil", "pend", "drivf", "oliv"])
    ax2[1].set_xticks(ticks=range(len(aris_plotting_sorted)-1), labels = ["coil5", "coil", "pend", "drivf", "oliv"])
    ax2[2].set_xticks(ticks=range(len(aris_plotting_sorted)-1), labels = ["coil5", "coil", "pend", "drivf", "oliv"])
    ax2[3].set_xticks(ticks=range(len(aris_plotting_sorted)-1), labels = ["coil5", "coil", "pend", "drivf", "oliv"])
    ax2[4].set_xticks(ticks=range(len(aris_plotting_sorted)-1), labels = ["coil5", "coil", "pend", "drivf", "oliv"])
    ax2[5].set_xticks(ticks=range(len(aris_plotting_sorted)-1), labels = ["coil5", "coil", "pend", "drivf", "oliv"])
  
    ax0[0].set_yticks(ticks=range(len(aris_plotting_sorted[0][0])), labels = ["DBSCAN", "kMeans", "Spectral", "DCF"])
    ax1[0].set_yticks(ticks=range(len(aris_plotting_sorted[0][0])), labels = ["DBSCAN", "kMeans", "Spectral", "DCF"])
    ax2[0].set_yticks(ticks=range(len(aris_plotting_sorted[0][0])), labels = ["DBSCAN", "kMeans", "Spectral", "DCF"])
    
    ax0[5].set_ylabel("dim=2")
    ax0[5].yaxis.set_label_position("right")
    
    ax1[5].set_ylabel("dim=10")
    ax1[5].yaxis.set_label_position("right")
    
    ax2[5].set_ylabel("dim=min(n, d)")
    ax2[5].yaxis.set_label_position("right")
    
    plt.tight_layout(pad=0.2)
    plt.show()
        
    
if __name__ == '__main__':
    expClusteringComparisonRealData(['coil'], [2, 10], ['mds1', 'mds5', 'mds10'], ['spectral'])
    expClusteringComparisonRealData(['pendigits'], [2], ['mds1', 'mds5', 'mds10'], ['spectral'])




    

