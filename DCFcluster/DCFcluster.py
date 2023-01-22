import numpy as np
import multiprocessing as mp
from itertools import islice
import scipy.sparse
import gc
import DCFcluster.utils as utils

from sklearn.neighbors import KDTree

class DCFcluster:
  def __init__(self, peak_values = np.nan, core_sets = np.nan, labels = np.nan):
    self.peak_values = peak_values
    self.core_sets = core_sets
    self.labels = labels
  
  @classmethod
  def train(cls, X, k, beta = 0.4):
    if type(X) is not np.ndarray:
      raise ValueError("X must be an n x d numpy array.")
    
    n, d = X.shape
    if k > n:
      raise ValueError("k cannot be larger than n.")
    
    kdt = KDTree(X, metric='euclidean')
    distances, neighbors = kdt.query(X, k)
    knn_radius = distances[:, k-1]
    best_distance = np.empty((X.shape[0]))
    big_brother = np.empty((X.shape[0]))
    radius_diff = knn_radius[:, np.newaxis] - knn_radius[neighbors]
    rows, cols = np.where(radius_diff > 0)
    rows, unidx = np.unique(rows, return_index =  True)
    del radius_diff
    gc.collect()
    cols = cols[unidx]
    big_brother[rows] = neighbors[rows, cols]
    best_distance[rows] = distances[rows, cols]
    search_idx = list(np.setdiff1d(list(range(X.shape[0])), rows))
    if n*k > 10000000:
      CCmat = scipy.sparse.csr_matrix((n, n))
      for nbrs_chunk in utils.chunks(list(range(n)), 5000):
        CCmat += scipy.sparse.csr_matrix((np.ones(len(nbrs_chunk*k)), (np.repeat(nbrs_chunk, k), neighbors[nbrs_chunk, :].ravel())), shape = (n, n))
      
      del neighbors
      gc.collect()  
    else:
      CCmat = scipy.sparse.csr_matrix((np.ones(n*k), (np.repeat(range(neighbors.shape[0]), k), neighbors.ravel())), shape = (n, n))
      del neighbors
      gc.collect()
    
    CCmat = CCmat.multiply(CCmat.T) 
    for indx_chunk in utils.chunks(search_idx, 100):
      search_radius = knn_radius[indx_chunk]
      GT_radius =  knn_radius < search_radius[:, np.newaxis] 
      if any(np.sum(GT_radius, axis = 1) == 0):
        max_i = [i for i in range(GT_radius.shape[0]) if np.sum(GT_radius[i,:]) ==0]
        if len(max_i) > 1:
          for max_j in max_i[1:len(max_i)]:
            GT_radius[max_j, indx_chunk[max_i[0]]] = True
        max_i = max_i[0]
        big_brother[indx_chunk[max_i]] = indx_chunk[max_i]
        best_distance[indx_chunk[max_i]] = np.inf
        del indx_chunk[max_i]
        GT_radius = np.delete(GT_radius, max_i, 0)
      
      GT_distances = ([X[indx_chunk[i],np.newaxis], X[GT_radius[i,:],:]] for i in range(len(indx_chunk)))
      if (GT_radius.shape[0]>25):
        try:
          pool = mp.Pool(processes=20)              
          N = 25
          distances = []
          i = 0
          while True:
            distance_comp = pool.map(utils.density_broad_search_star, islice(GT_distances, N))
            if distance_comp:
              distances.append(distance_comp)
              i += 1
            else:
              break
          distances = [dis_pair for dis_list in distances for dis_pair in dis_list]
          argmin_distance = [np.argmin(l) for l in distances]
          pool.close()
          pool.terminate()
        except Exception as e:
          print("POOL ERROR: "+ e)
          pool.close()
          pool.terminate()
      else:
          distances = list(map(utils.density_broad_search_star, list(GT_distances)))
          argmin_distance = [np.argmin(l) for l in distances]
      
      for i in range(GT_radius.shape[0]):
        big_brother[indx_chunk[i]] = np.where(GT_radius[i,:] == 1)[0][argmin_distance[i]]
        best_distance[indx_chunk[i]] = distances[i][argmin_distance[i]]
  
    tested = []
    peaked = best_distance/knn_radius
    peaked[(best_distance==0)*(knn_radius==0)] = np.inf
    centers = [np.argmax(peaked)]
    tested.append(np.argmax(peaked))
    not_visited = np.ones(n, dtype = bool)
    cluster_memberships = np.repeat(-1, n)
    n_cent = 0
    if knn_radius[centers[0]] > 0:
        cutoff = knn_radius[centers[0]]/((1-beta)**(1/d))
        cut_idx = np.where(knn_radius < cutoff)[0]
    else: 
        cutoff = knn_radius[centers[0]]/((1-beta)**(1/d))
        cut_idx = np.where(knn_radius <= cutoff)[0]
    
    CCmat_cut = CCmat[cut_idx, :][:, cut_idx]
    _, cc_labels = scipy.sparse.csgraph.connected_components(CCmat_cut, directed = 'False', return_labels =True)
    center_cc_idx = np.where(cc_labels == cc_labels[np.where(cut_idx == centers[0])])[0]
    not_visited[cut_idx[center_cc_idx]] = False
    cluster_memberships[cut_idx[center_cc_idx]] = n_cent
    while True:
        if np.sum(not_visited) == 0:
            break
        subset_idx = np.argmax(peaked[not_visited])
        prop_cent = np.arange(peaked.shape[0])[not_visited][subset_idx]
        tested.append(np.arange(peaked.shape[0])[not_visited][subset_idx])
        if knn_radius[prop_cent] > max(knn_radius[~not_visited]):
            level_set = np.where(knn_radius <= knn_radius[prop_cent])[0]
            CCmat_level = CCmat[level_set, :][:, level_set]
            n_cc, _ = scipy.sparse.csgraph.connected_components(CCmat_level, directed = 'False', return_labels =True)
            if n_cc == 1:
                break
        
        if knn_radius[prop_cent] > 0:
            cutoff = knn_radius[prop_cent]/((1-beta)**(1/d))
            cut_idx = np.where(knn_radius < cutoff)[0]
        else: 
            cutoff = knn_radius[prop_cent]/((1-beta)**(1/d))
            cut_idx = np.where(knn_radius <= cutoff)[0]
        
        CCmat_cut = CCmat[cut_idx, :][:, cut_idx]
        _, cc_labels = scipy.sparse.csgraph.connected_components(CCmat_cut, directed = 'False', return_labels =True)
        center_cc = cc_labels[np.isin(cut_idx, centers)]
        prop_cent_cc = cc_labels[np.where(cut_idx == prop_cent)[0]]
        if np.isin(prop_cent_cc, center_cc):
            center_cc_idx = np.where(cc_labels == cc_labels[np.where(cut_idx == prop_cent)])[0]
            not_visited[cut_idx[center_cc_idx]] = False
        else:
            centers.append(prop_cent)
            n_cent += 1
            center_cc_idx = np.where(cc_labels == cc_labels[np.where(cut_idx == centers[n_cent])])[0]
            not_visited[cut_idx[center_cc_idx]] = False
            cluster_memberships[cut_idx[center_cc_idx]] = n_cent
    
    modal_s = np.where(cluster_memberships != -1)[0]
    BBTree = np.zeros((n, 2))
    BBTree[:, 0] = range(n)
    BBTree[:, 1] = big_brother
    BBTree[modal_s,1] = modal_s
    BBTree = BBTree.astype(int)
    Clustmat = scipy.sparse.csr_matrix((np.ones((n)), (BBTree[:,0], BBTree[:, 1])), shape = (n, n))
    n_clusts, modal_y = scipy.sparse.csgraph.connected_components(Clustmat, directed = 'True', return_labels =True) 
    y_pred = np.repeat(-1, n)
    for i in modal_s:
        label = cluster_memberships[i]
        y_pred[modal_y == modal_y[i]] = label
    
    result = cls(peaked, modal_s, y_pred)
    return result
