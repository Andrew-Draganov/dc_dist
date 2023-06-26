Code to recreate the experiments from the paper ''Connecting the Dots: density-connectivity distance unifies DBSCAN, k-center and spectral clustering.''

The distance metric is calculated in `distance_metric.py`. The k-center clustering on the dc-dist is given in `density_tree.py` and `cluster_tree.py`. We provide an
implementation of DBSCAN\* in `DBSCAN.py`. Furthermore, our implementation of Ultrametric Spectral Clustering is given in `SpectralClustering.py`.

The code to calculate the distance measure can be found in `distance_metric.py`. Experiment scripts are then located in
 - `k_vs_epsilon.py`
 - `noise_robustness.py`
 - `distances_plot.py`
 - `compare_clustering.py`

If you would like to mess around with the clusterings and assert for yourself that they are equivalent, we recommend the sandbox file `cluster_dataset.py`.

We provide an ultrametric visualization tool in the file `tree_plotting.py`. This allows you to look at the tree of dc-distances given by a specific dataset.

You will have to download the coil-100 dataset from [here](https://www.kaggle.com/datasets/jessicali9530/coil100) and unpack it
into the path `data/coil-100`.

Feel free to email me if you have any questions -- draganovandrew@cs.au.dk
