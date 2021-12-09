# Theory and data fitting of the covariance spectrum of random networks

These are the codes associated with the preprint ["Hu, Y. and Sompolinsky, H., *The spectrum of covariance matrices of randomly connected recurrent neuronal networks*, bioRxiv.
"](https://doi.org/10.1101/2020.08.31.274936)

Please refer to the manuscript for more details.


**Abstract**
A key question in theoretical neuroscience is the relation between the connectivity structure and the collective dynamics of a network of neurons. Here we study the connectivity-dynamics relation as reflected in the distribution of eigenvalues of the covariance matrix of the dynamic fluctuations of the neuronal activities, which is closely related to the network's Principal Component Analysis (PCA) and the associated effective dimensionality. We consider the spontaneous fluctuations around a steady state in a randomly connected recurrent network of stochastic neurons. An exact analytical expression for the covariance eigenvalue distribution in the large-network limit can be obtained using results from random matrices. The distribution has a finitely supported smooth bulk spectrum and exhibits an approximate power-law tail for coupling matrices near the critical edge. We generalize the results to include connectivity motifs and discuss extensions to Excitatory-Inhibitory networks. The theoretical results are compared with those from finite-size networks and and the effects of temporal and spatial sampling are studied. Preliminary application to whole-brain imaging data is presented. Using simple connectivity models, our work provides theoretical predictions for the covariance spectrum, a fundamental property of recurrent neuronal dynamics, that can be compared with experimental data. 


## Usage
The code is written in Python 3.

* Before running the code, make sure there is a `/figure` and `/data` subfolder to store results.
* The code produces all figures in the preprint. Some plots take longer time to run, those parts of the code are currently commented out in the code.

### List of files
Note that the filename such as`figure3.py` may not match with the figure number in the manuscript. Default parameters in the code may be different from those used in the manuscript.

| File Name | Explanation |
|----|----|
|`cov_spectrum_randon_netw.py` | defines functions|
|`figure1.py` | plot the spectrum for iid Gaussian random connectivity |
|`rank_plot.py` | Rank plots for theory and finite-size networks |
|`figure2.py` | Examine the power-law tail of the spectrum |
|`figure3.py` | Low rank perturbations of connectivity |
|`figure_outlier.py` | Theoretical predictions of outliers due to low-rank connectivity and activity perturbations|
|`symm_antisymm.py` | Spectra for symmetric and skew-symmetric random connectivity |
|`figure4.py` | Effect of reciprocal motifs/asymmetry |
|`figure5.py` | Strong non-normal effect of connectivity|
|`figure6.py` | Sparse connectivity and E-I networks|
|`figure7.py` | Temporal and spatial sampled spectrum and fitting to empirical eigenvalues|
|`figure8.py` | Deterministic connectivity|
|`figure9.py` | Example of separating low-rank components in the covariance|
|`figure10.py` | Verifying approximations with uniform covariance diagonal and self-coupling|
|`fish_data.py` |Example application to calcium imaging data in larval zebrafish|
|`/data/select_cluster_core_F9.mat` |Exerpt calcium imaging data from  [Chen et al, Neuron, 2019](https://www.cell.com/neuron/fulltext/S0896-6273(18)30844-4) |
|`/data/`| Contains pre-computed data for longer simulations| 










## Citation
Please cite the preprint when using the code:

Hu, Y. and Sompolinsky, H., *The spectrum of covariance matrices of randomly connected recurrent neuronal networks*, bioRxiv.

