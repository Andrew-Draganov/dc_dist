#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import imageio
import glob

from sklearn.utils import gen_batches, get_chunk_n_rows
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, fetch_olivetti_faces

from distance_metric import get_nearest_neighbors
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
from sklearn.cluster import OPTICS

from GDR import GradientDR


""" synthetic datasets """             
def reduceSynthData(minPoints, dim, distance_metric):
    points_all_datasets = loadSynthDatasets()
    
    calcReductionSynth(minPoints, points_all_datasets, dim, distance_metric)
  
def loadSynthDatasets():
    #points_b1, labels_b1 = make_blobs(n_samples=10000, centers=10, n_features=50, random_state=1)
   
    # make unbalanced blobs (b2)
    #cluster_numbers_blobs_unbalanced = [882, 1577, 913, 1335, 539, 703, 1393, 744, 803, 1111]
    #points_b2, labels_b2 = make_blobs(n_samples=cluster_numbers_blobs_unbalanced, centers=None, n_features=50, random_state=1)

    # load unbalanced blobs with uniforn noise (b3)  
    dataset_blobs_unbalanced_noise = np.load("./data/synth/blobs_unbalanced_noise.npy")
    points_b3 = dataset_blobs_unbalanced_noise[:, :-1]
     
    # load density dataset with same density (d1)
    dataset_same_dens = np.load("./data/synth/synth_data_10000_10_50_0_0.npy")
    points_d1 = dataset_same_dens[:, :-1]
    
    # load density dataset with different density (d2)
    dataset_diff_dens = np.load("./data/synth/synth_data_10000_10_50_vardensity_0_0.npy")
    points_d2 = dataset_diff_dens[:, :-1]
    
    # load density dataset with different density + noise (d3)
    dataset_diff_dens_noise = np.load("./data/synth/synth_data_10000_10_50_vardensity_1000_0.npy")
    points_d3 = dataset_diff_dens_noise[:, :-1]
    
    #points = [points_b1, points_b2, points_b3, points_d1, points_d2, points_d3]
    points = [points_b3, points_d1, points_d2, points_d3]
    
    return points  
  
def calcReductionSynth(minPoints, points_all_datasets, dim, distance_metric):
    for i in tqdm(range(len(dim))):
        for j in tqdm(range(len(points_all_datasets))):
            points = points_all_datasets[j]
            
            if distance_metric == 'ours':
                distance_matrix = get_nearest_neighbors(points, 15, minPoints)['_all_dists']
            if distance_metric == 'cosine':
                distance_matrix = cosine_distances(points)
            if distance_metric == 'manhattan':
                distance_matrix = manhattan_distances(points)
            if distance_metric == 'mutualReachability':
                # mutual_reachability = max(core(a), core(b), dist(a, b))
                n = points.shape[0]
                euclidean = euclidean_distances(points)
                nbrs = NearestNeighbors(n_neighbors=minPoints).fit(points)
                core_distances = _compute_core_distances_(points, nbrs, minPoints, None)
                core_distances_1 = np.full((n, n), core_distances)
                core_distances_2 = core_distances_1.T
                
                core_distances = np.maximum(core_distances_1, core_distances_2)
                distance_matrix = np.maximum(core_distances, euclidean)
            
            reducedData = MDS(n_components=dim[i]).fit_transform(distance_matrix)
            
            if j < 3:
                dataType = 'b'
            else:
                dataType = 'd'
        
            filename = "../reducedSynthDatasets/"+str(distance_metric)+"_"+str(dim[i])+"_"+str(minPoints)+"_"+str(dataType)+str((j%3)+1)+".txt"
            with open(filename, 'w') as f:
                f.write(str(reducedData.tolist()))


""" real datasets """
""" quick reductions: """
def loadRealDatasets(to_load):
    points = []
    labels = []
    
    for datatype in to_load:
        if datatype=='olivetti':
            points_olivetti, labels_olivetti = loadOlivetti()
            points.append(points_olivetti)
            labels.append(labels_olivetti)
        if datatype=='coil':
            points_coil, labels_coil = loadCoil()
            points.append(points_coil)
            labels.append(labels_coil)
        if datatype=='coil5':
            points_coil5, labels_coil5 = loadCoil5()
            points.append(points_coil5)
            labels.append(labels_coil5)
        if datatype=='coil20':
            points_coil20, labels_coil20 = loadCoil20()
            points.append(points_coil20)
            labels.append(labels_coil20)
        if datatype=='skins':
            points_skins, labels_skins = loadSkins()
            points.append(points_skins)
            labels.append(labels_skins)
        if datatype=='drivface':
            points_drivface, labels_drivface = loadDrivface()
            points.append(points_drivface)
            labels.append(labels_drivface)
        if datatype=='pendigits':
            points_pendigits, labels_pendigits = loadPendigits()
            points.append(points_pendigits)
            labels.append(labels_pendigits)
        if datatype=='landsat':
            points_landsat, labels_landsat = loadLandsat()
            points.append(points_landsat)
            labels.append(labels_landsat)
        if datatype=='letters':
            points_letters, labels_letters = loadLetters()
            points.append(points_letters)
            labels.append(labels_letters)
    
    return points, labels

def loadOlivetti():
    # 400 instances, 4096 dimensions, 40 classes
    olivetti = fetch_olivetti_faces()
    points_olivetti = olivetti.data
    labels_olivetti = olivetti.target
    
    return points_olivetti, labels_olivetti
 
def loadCoil():
    # 7,200 instances, 49152 dimensions, 100 classes
    points_coil = []
    filelist = sorted(glob.glob('./data/coil-100/*.png'))
    for filename in filelist:
        im = imageio.imread(filename)
        points_coil.append(im.flatten())
    labels_coil = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    
    points_coil = np.array(points_coil)
    labels_coil = labels_coil.to_numpy(dtype = int)-1
    
    return points_coil, labels_coil
 
def loadCoil5():
    # 360 instances, 128*128 dimensions, 5 classes
    points_coil5 = []
    filelist = sorted(glob.glob('./data/coil-5/*.png'))
    for filename in filelist:
        im = imageio.imread(filename)
        points_coil5.append(im.flatten())
    labels_coil5 = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    
    points_coil5 = np.array(points_coil5)
    labels_coil5 = labels_coil5.to_numpy(dtype = int)-1
    
    return points_coil5, labels_coil5    

def loadCoil20():
    # 1,440 instances, 128*128 dimensions, 20 classes
    points_coil20 = []
    filelist = sorted(glob.glob('./data/coil-20/*.png'))
    for filename in filelist:
        im = imageio.imread(filename)
        points_coil20.append(im.flatten())
    labels_coil20 = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    
    points_coil20 = np.array(points_coil20)
    labels_coil20 = labels_coil20.to_numpy(dtype = int)-1
    
    return points_coil20, labels_coil20

def loadSkins():
    # 245,057 instances, 3 dimensions (B, G, R), 2 classes
    data = open("./data/Skin_NonSkin.txt").read()
    data = data.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split('\t') 
    data = data[:][:-1]
    data = np.array(data, dtype = int)
    
    points_skins = data[:, :-1]
    labels_skins = data[:, -1]-1
    
    return points_skins, labels_skins

def loadDrivface():
    # 606 instances, 640*480*3 dimensions, 3 classes
    points_drivface = []
    filelist = sorted(glob.glob('./data/DrivFace/DrivImages/*.jpg'))
    for filename in filelist:
        im = imageio.imread(filename)
        points_drivface.append(im.flatten())
    points_drivface = np.array(points_drivface)
    
    points_info = pd.read_table('./data/DrivFace/drivPoints.txt', delimiter=',')
    points_info = points_info.to_numpy()
    labels_drivface = points_info[:, 3]-1
    
    return points_drivface, labels_drivface

def loadPendigits():
    # 7,494 instances (only training dataset), 16 dimensions, 10 classes
    data = pd.read_csv('./data/pendigits/pendigits.tra', header=None)
    points_pendigits, labels_pendigits, _ = np.hsplit(data, np.array([16, 17]))
    labels_pendigits = (np.array(labels_pendigits)).flatten()
    points_pendigits = points_pendigits.to_numpy()
    
    return points_pendigits, labels_pendigits

def loadLandsat():
    # 4,435 instances (only training dataset), 36 dimensions, 6 classes
    data = pd.read_table('./data/Landsat/sat.trn', delimiter=' ', header = None)
    data = data.to_numpy()
    
    points_landsat = data[:, :-1]
    labels_landsat = data[:, -1]-1
    
    # there are no instances of class 6 in this dataset as stated in dataset description
    labels_landsat[np.where(labels_landsat==7)] = 6
    
    # lowest class should be 0
    labels_landsat = labels_landsat-1
       
    return points_landsat, labels_landsat

def loadLetters():
    # 20,000 instances, ?? dimensions, 26 classes
    data = np.array(pd.read_csv('./data/letter_recognition/letter-recognition.data', header=None))
    labels_letters, points_letters = np.hsplit(data, np.array([1]))
    labels_letters = np.array([ord(i) - 65 for i in labels_letters.flatten()])
    return points_letters, labels_letters
    

def reduceRealData_PCA_TSNE_UMAP(to_load, dims, reductiontype): 
    points_all_datasets, _ = loadRealDatasets(to_load)
    
    if reductiontype == 'pca':
        calcPCA(points_all_datasets, dims, to_load)
    if reductiontype == 'tsne':
        calcTSNE(points_all_datasets, dims, to_load)
    if reductiontype == 'umap': 
        calcUMAP(points_all_datasets, dims, to_load)
       
def calcPCA(points_all_datasets, dims, datatypes):
    for points_index in tqdm(range(len(points_all_datasets))):
        datatype = datatypes[points_index]
        for i in range(len(dims)):
            pca = PCA(n_components=dims[i]).fit_transform(points_all_datasets[points_index])
            
            filename = "../reducedRealDatasets/"+str(datatype)+"/pca_"+str(dims[i])+"_"+str(datatype)+".txt"
            print(filename)
            with open(filename, 'w') as f:
                f.write(str(pca.tolist()))

def calcTSNE(points_all_datasets, dims, datatypes):
    for points_index in tqdm(range(len(points_all_datasets))):
        datatype = datatypes[points_index]
        for i in range(len(dims)):
            tsne = GradientDR(normalized=True, random_init=True, dim=dims[i]).fit_transform(points_all_datasets[points_index])
        
            filename = "../reducedRealDatasets/"+str(datatype)+"/tsne_"+str(dims[i])+"_"+str(datatype)+".txt"
            with open(filename, 'w') as f:
                f.write(str(tsne.tolist()))

def calcUMAP(points_all_datasets, dims, datatypes):
    for points_index in tqdm(range(len(points_all_datasets))):
        datatype = datatypes[points_index]
        for i in range(len(dims)):
            umap = GradientDR(random_init=True, dim=dims[i]).fit_transform(points_all_datasets[points_index])
        
            filename = "../reducedRealDatasets/"+str(datatype)+"/umap_"+str(dims[i])+"_"+str(datatype)+".txt"
            with open(filename, 'w') as f:
                f.write(str(umap.tolist()))
   
                
""" time-consuming reductions: """
def reduceRealDataMDS(to_load, minPoints, dims, distance_metric):
    points_all_datasets, _ = loadRealDatasets(to_load)
    
    calcReductionRealMDS(to_load, points_all_datasets, minPoints, dims, distance_metric)
  
def calcReductionRealMDS(to_load, points_all_datasets, minPoints, dims, distance_metric):
    for dim in tqdm(dims):
        for j in tqdm(range(len(points_all_datasets))):
            dataType = to_load[j]
            points = points_all_datasets[j]
            
            if distance_metric == 'ours':
                distance_matrix = get_nearest_neighbors(points, 15, minPoints)['_all_dists']
            if distance_metric == 'cosine':
                distance_matrix = cosine_distances(points)
            if distance_metric == 'manhattan':
                distance_matrix = manhattan_distances(points)
            if distance_metric == 'mutualReachability':
                # mutual_reachability = max(core(a), core(b), dist(a, b))
                n = points.shape[0]
                euclidean = euclidean_distances(points)
                nbrs = NearestNeighbors(n_neighbors=minPoints).fit(points)
                optics_cores = _compute_core_distances_(points, nbrs, minPoints, None)
                core_distances_1 = np.full((n, n), optics_cores)
                core_distances_2 = core_distances_1.T
                
                core_distances = np.maximum(core_distances_1, core_distances_2)
                distance_matrix = np.maximum(core_distances, euclidean)
                
            reducedData = MDS(n_components=dim).fit_transform(distance_matrix)
        
            filename = "../reducedRealDatasets/"+str(dataType)+"/mds_"+str(distance_metric)+"_"+str(dim)+"_"+str(minPoints)+"_"+str(dataType)+".txt"
            #filename = "../reducedRealDatasets/mds_ours_"+str(dim)+"_"+str(minPoints)+"_"+str(dataType)+".txt"
            with open(filename, 'w') as f:
                f.write(str(reducedData.tolist()))    

def _compute_core_distances_(X, neighbors, min_samples, working_memory):
    """ COPIED FROM SKLEARN"""
    """Compute the k-th nearest neighbor of each sample.
    Equivalent to neighbors.kneighbors(X, self.min_samples)[0][:, -1]
    but with more memory efficiency.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.
    neighbors : NearestNeighbors instance
        The fitted nearest neighbors estimator.
    working_memory : int, default=None
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.
    Returns
    -------
    core_distances : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point.
        Points which will never be core have a distance of inf.
    """
    n_samples = X.shape[0]
    core_distances = np.empty(n_samples)
    core_distances.fill(np.nan)

    chunk_n_rows = get_chunk_n_rows(
        row_bytes=16 * min_samples, max_n_rows=n_samples, working_memory=working_memory
    )
    slices = gen_batches(n_samples, chunk_n_rows)
    for sl in slices:
        core_distances[sl] = neighbors.kneighbors(X[sl], min_samples)[0][:, -1]
    return core_distances
     
    
if __name__ == '__main__': 
    #reduceRealDataMDS(['pendigits', 'coil'], 5, [2, 10], 'mutualReachability')
    reduceSynthData(5, [2, 10], 'mutualReachability')
