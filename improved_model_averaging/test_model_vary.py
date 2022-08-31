#!/usr/bin/env python3

from .stats import *
from .synth_data import *
from .fitting import *


def test_vary_poly(
    test_data, Nt, mu_max=6, mu_min=1, prior_width=10, obs_name="a0", IC_list=None
):
    """
    Test a varying set of polynomial models against the given data, extracting the given
    common fit parameter using model averaging.
    Args:
      test_data: Synthetic data set to run fits against.
      Nt: Normalizing constant for the polynomial model (typically Nt=maximum value of the 't' coordinate.)
      mu_max: Max polynomial order to include (default: 6.)
      mu_min: Min polynomial order to include (default: 1, i.e. linear.)
      prior_width: Width of priors for model fit parameters (default: 10.)
      obs_name: Common fit parameter to study with model averaging (default: 'a0', the y-intercept.)
      IC_list: list of information criteria to be tested
    """
    
    if IC_list is None:
        IC_list = ['AIC', 'BAIC', 'BPIC', 'PPIC']

    # Run fits
    obs_vs_k = []
    fit_vs_k = []
    Q_vs_k = []
    
    prob_vs_k = {}
    IC_vs_k = {}
    for IC in IC_list:
        prob_vs_k.update({IC: []})
        IC_vs_k.update({IC: []})
        
    for k in range(mu_min, mu_max + 1):
        this_fit, this_design_mat = run_fit_poly(test_data, Nt, k, prior_width=prior_width)        
        fit_vs_k.append(this_fit)
        obs_vs_k.append(this_fit.p[obs_name])
        
        Q_vs_k.append(this_fit.Q)
        
        probs = {}
        ICs = {}
        for IC in IC_list:
            probs.update({IC: None})
            ICs.update({IC: None})
            ICs[IC], probs[IC] = get_model_IC(this_fit, test_data, design_mat=this_design_mat, IC=IC, return_prob=True)
            
            prob_vs_k[IC].append(probs[IC])
            IC_vs_k[IC].append(ICs[IC])
    
    # Compute model-averaged result
    obs_avg_IC = {}
    for IC in IC_list:
        obs_avg_IC.update({IC: model_avg(obs_vs_k, prob_vs_k[IC])})


    return {
        "mu": np.array(range(mu_min, mu_max + 1)),
        "data": test_data,
        "fits": fit_vs_k,
        "obs": obs_vs_k,
        "prob": prob_vs_k,
        "obs_avg_IC": obs_avg_IC,
        "Qs": Q_vs_k,
        "IC": IC_vs_k,
    }
