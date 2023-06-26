Code to recreate the experiments from the paper ''Connecting the Dots: density-connectivity distance unifies DBSCAN, k-center and spectral clustering.''

The code to calculate the distance measure can be found in `distance_metric.py`. Experiment scripts are then located in
 - `k_vs_epsilon.py`
 - `noise_robustness.py`
 - `distances_plot.py`
 - `compare_clustering.py`

If you would like to mess around with the clusterings and assert for yourself that they are equivalent, we recommend the sandbox file `cluster_dataset.py`.

You will have to download the coil-100 dataset from [here](https://www.kaggle.com/datasets/jessicali9530/coil100) and unpack it
into the path `data/coil-100`.

Feel free to email me if you have any questions -- draganovandrew@cs.au.dk
