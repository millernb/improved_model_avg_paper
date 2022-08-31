# improved_model_avg_paper

This repository contains code used for tests of Bayesian model averaging, as presented in the paper "Improved information criteria for Bayesian model averaging in lattice field theory."  In particular, the code included can be used to generate figures 1-9 of the paper (or re-generate them with different random seeds, if desired.)

## Installation

To ensure reproducibility, the repository contains an Anaconda environment specification in the file `conda-spec.yml`.  To use this specification, install the [Anaconda Python distribution](https://www.anaconda.com/), and then run the command:

```
conda env create -f conda-spec.yml
conda activate alternative_ICs
```

to download and install the required Python modules, and then activate the environment.  Then run `jupyter notebook` to access the Jupyter notebook files `test_poly_vary.ipynb` (polynomial model tests and figures 1,2) and `test_exp_tmin.ipynb` (exponential model tests and figures 3-9.)

__Note:__ there seem to be some issues with exact reproducibility of the random number generator in our tests, so the exact plots shown in the paper may not be perfectly reproducible with this repository.  However, the qualitative form of the results of course should not depend on the RNG state, so it should be possible to reproduce results that are substantially identical.
