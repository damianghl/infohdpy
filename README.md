# infohdpy: Python package for mutual information estimation using Hierarchical Dirichlet Priors
Estimating the mutual information between discrete variables with limited samples.


![](https://github.com/dghernandez/info-estimation/blob/master/scheme2.jpg)

# Installation and requirements

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dghernandez/info-estimation.git
   cd info-estimation
   ```

2. **Check and Install Requirements**:
   ```bash
   python --version
   pip install -r requirements.txt
   ```

3. **Install the Package**:
   ```bash
   pip install -e .
   ```

# Basic usage

```python
# Import
from infohdp.estimators import MulticlassFullInfoHDPEstimator

# Create an instance of Estimator for multiclass case
estimator = MulticlassFullInfoHDPEstimator()

# Samples in format [(x_0, y_0), (x_1, y_1), (x_2, y_2), ...]
samples = [(15, 1), (35, 0), (2, 0), (29, 1), (35, 0), (35, 0), (21, 1), (21, 0), (29, 1), (21, 1)]

i_hdp, di_hdp = estimator.estimate_mutual_information(samples)
print(f"Ihdp full multiclass, mutual information estimation [nats]: {i_hdp:.4f} ± {di_hdp:.4f}")
```

# Core calculations and conditions
The main parts of the code are rather simple and they can be implemented in any programming language: (1) a maximization of the marginal log-likelihood (or posterior) over the hyperparameter beta, and (2) the evaluation of the posterior information in such beta. 

First, we need to obtain the ML (or MAP) estimate for the hyperparameter beta (from Eq. (14) in the paper, or Eq. (21) for the symmetric binary case). I recommend to do this maximization with the argument log(beta), and explore the interval log(0.01)<log(beta)<log(100), which is usually enough. Only terms with coincidences in the large entropy variable (n_x>1) need to be considered, as the others add a constant term in beta. In an undersampled regime, there would be many repeated terms and they can be grouped together for a more efficient evaluation (using multiplicities). Secondly, we evaluate the posterior information (see Eq. (16) in the paper, or Eq. (20) for the symmetric binary case) in the beta found previously. In this evaluation, all occupied states (n_x>0) need to be included.

Our method needs coincidences on the large entropy variable X, which starts to happen when N> exp(H(X)/2). If there are no coincidences, then the marginal likelihood is flat on beta. If there are few coincidences and no prior on beta is used, the maximum may be attained for beta tending to zero or infinity. In such cases the posterior information is still well-defined and takes the values of H(Y) or zero, respectively.

This page is maintained by Damián G. Hernández.
(email address in paper https://www.mdpi.com/1099-4300/21/6/623)
