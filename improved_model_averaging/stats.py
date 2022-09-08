import numpy as np
import gvar as gv
import lsqfit
import jax
import jax.numpy as jnp
import vegas

jax.config.update("jax_enable_x64", True)

# Info criteria
def BAIC_fit(fr):
    prior_stats = prior_stats_fit(fr)
    chi2_prior = prior_stats['chi2_prior']
    
    BAIC = fr.chi2 - chi2_prior + 2 * len(fr.p.keys())
    return BAIC


def BPIC_linear_fit(fr, test_data, design_mat):
    tensors = tensors_linear_fit(fr, test_data, design_mat)
    chi2_prior = tensors['chi2_prior']
    hess_prior = tensors['hess_prior']
    cov_best_fit = tensors['cov_best_fit']
    
    BPIC = fr.chi2 - chi2_prior - 0.5 * np.einsum('ba,ab->', hess_prior, cov_best_fit) + 3 * len(fr.p.keys())

    return BPIC

def BPIC_nonlinear_fit(fr, test_data, model_derivs):
    tensors = tensors_nonlinear_fit(fr, test_data, model_derivs)
    cov_data_inv = tensors['cov_data_inv']
    chi2_prior = tensors['chi2_prior']
    grad_prior = tensors['grad_prior']
    hess_prior = tensors['hess_prior']
    cov_best_fit = tensors['cov_best_fit']
    cov_best_fit_2 = tensors['cov_best_fit_2']
    T = tensors['T']

    
    int_lead = -chi2_prior
    int_sub = -0.5 * np.einsum('ba,ab->', hess_prior, cov_best_fit) + 0.5 * np.einsum('d,cba,abcd->', grad_prior, T, cov_best_fit_2)

    if np.abs(int_sub) < np.abs(int_lead):
        integral = int_lead + int_sub
    else:
        integral = int_lead
        
    BPIC = fr.chi2 + integral + 3 * len(fr.p.keys())
    return BPIC


def PPIC_linear_fit(fr, test_data, design_mat):
    tensors = tensors_linear_fit(fr, test_data, design_mat)
    cov_data_inv = tensors['cov_data_inv']
    chi2_prior = tensors['chi2_prior']
    cov_best_fit = tensors['cov_best_fit']
    
    yraw = test_data['yraw']
    delta_indv = yraw - gv.mean(fr.fcn(fr.x,fr.p))
    grad_indv_data = -2 * np.einsum('ba,bc,dc->da', design_mat, cov_data_inv, delta_indv)
    hess_indv_data = 2 * np.einsum('ba,bc,cd->ad', design_mat, cov_data_inv, design_mat)
    
    log_args = 1 + 0.5 * np.einsum('cba,ab->c', 0.25 * np.einsum('ab,ac->abc', grad_indv_data, grad_indv_data) - 0.5 *  hess_indv_data, cov_best_fit)

    PPIC = fr.chi2 - chi2_prior - 2 * np.sum(np.log(log_args)) + 2 * len(fr.p.keys())
        
    return PPIC

def PPIC_nonlinear_fit(fr, test_data, model_derivs):
    tensors = tensors_nonlinear_fit(fr, test_data, model_derivs)
    cov_data_inv = tensors['cov_data_inv']
    chi2_prior = tensors['chi2_prior']
    cov_best_fit = tensors['cov_best_fit']
    cov_best_fit_2 = tensors['cov_best_fit_2']
    T = tensors['T']
    
    
    yraw = test_data['yraw']
    delta_indv = yraw - gv.mean(fr.fcn(fr.x,fr.p))
    grad_indv_data = -2 * np.einsum('ba,bc,dc->da', model_derivs['d1_model'], cov_data_inv, delta_indv)
    hess_indv_data = 2 * (-np.einsum('cba,cd,ed->eab', model_derivs['d2_model'], cov_data_inv, delta_indv) + np.einsum('ba,bc,cd->ad', model_derivs['d1_model'], cov_data_inv, model_derivs['d1_model']))
    
    int_leads = 1.0
    int_subs = 0.5 * np.einsum('cba,ab->c', 0.25 * np.einsum('ab,ac->abc', grad_indv_data, grad_indv_data) - 0.5 *  hess_indv_data, cov_best_fit) + 0.25 * np.einsum('ed,cba,abcd->e', grad_indv_data, T, cov_best_fit_2)
    
    int_subs[np.abs(int_subs)>=np.abs(int_leads)]=0
    
    log_args=int_leads+int_subs
    
    PPIC = fr.chi2 - chi2_prior - 2 * np.sum(np.log(log_args)) + 2 * len(fr.p.keys())
    return PPIC


def AIC_fit(fr):
    AIC = fr.chi2 + 2 * len(fr.p.keys())
    return AIC


def PAIC_linear_fit(fr, test_data, design_mat):
    PAIC = BPIC_linear_fit(fr, test_data, design_mat)
    return PAIC

def PAIC_from_vegas(fr):
    def chi_sq(p):
        return gv.chi2(fr.y, fr.fcn(fr.x, p))

    def pdf(p):
        return np.exp(-chi_sq(p)/2)

    exp_val = vegas.PDFIntegrator(fr.p, pdf=pdf, limit=20)

    try:
        integral = exp_val(chi_sq, neval=100, nitn=8)
    except ValueError:
        return 999.

    PAIC = gv.mean(integral + 2 * len(fr.p.keys()))

    return PAIC
    
def PAIC_nonlinear_fit(fr, test_data, model_derivs):
    tensors = tensors_nonlinear_fit(fr, test_data, model_derivs)
    cov_data_inv = tensors['cov_data_inv']
    chi2_prior = tensors['chi2_prior']
    hess_prior = tensors['hess_prior']
    cov_best_fit = tensors['cov_best_fit']
    cov_best_fit_2 = tensors['cov_best_fit_2']
    T = tensors['T']
    
    
    ND = test_data['ND']
    delta = gv.mean(fr.y) - gv.mean(fr.fcn(fr.x,fr.p))
    chi2_data = fr.chi2 - chi2_prior
    grad_data = -2 * np.einsum('ba,bc,c->a', model_derivs['d1_model'], ND * cov_data_inv, delta)
    
    
    int_lead = chi2_data
    int_sub = len(fr.p.keys())

    if np.abs(int_sub) < np.abs(int_lead):
        integral = int_lead + int_sub
    else:
        integral = int_lead
    
    PAIC = integral + 2 * len(fr.p.keys())
    return PAIC


def naive_fit(fr):
    return fr.chi2



def get_model_IC(fr, test_data, design_mat=None, model_derivs=None, IC="BAIC", return_prob=False, full_BC=False, quiet_full_BC=False):
    """
    Compute incfomration criteria from log likelihood (LL) for a given fit.
    space of all models, which should be done separately.
    Relation to info criteria:
        LL = -1/2 * IC
    Args:
      fr: Fit result object, from lsqfit module.
      test_data: dictionay of data
      design_mat: Default None. The design matrix for linear fits
      model_derivs: Default None. Dictionary of model derivatives for nonlinear fits
      IC: Which info criterion to use.  Options: BAIC (default), AIC, BPIC, PAIC, PPIC, naive.
      return_prob: Specifies if model probability is returned
      full_BC: specifies if the full bias correction is used
      quiet_full_BC: Suppresses full_BC warning in cases of data subset selection.
    Returns:
      IC: the information criteria value
      np.exp(LL): the (unnormalized) model probability
    """
    
    if design_mat is None:
        if model_derivs is None:
            assert IC == "AIC" or IC == "BAIC" or IC == "naive"
        else:
            assert 'd1_model' in model_derivs.keys()
            assert 'd2_model' in model_derivs.keys()
            assert 'd3_model' in model_derivs.keys()
            is_linear = False
    else:
        is_linear = True
        

    if IC == "BAIC":
        LL = -0.5 * BAIC_fit(fr)
    elif IC == "BPIC":
        if is_linear:
            LL = -0.5 * BPIC_linear_fit(fr, test_data, design_mat)
        else: 
            LL = -0.5 * BPIC_nonlinear_fit(fr, test_data, model_derivs)
    elif IC == "PPIC":
        if is_linear:
            LL = -0.5 * PPIC_linear_fit(fr, test_data, design_mat)
        else:
            LL = -0.5 * PPIC_nonlinear_fit(fr, test_data, model_derivs)
    elif IC == "AIC":
        LL = -0.5 * AIC_fit(fr)
    elif IC == "PAIC":
        if is_linear:
            LL = -0.5 * PAIC_linear_fit(fr, test_data, design_mat)
        else:
            LL = -0.5 * PAIC_nonlinear_fit(fr, test_data, model_derivs)
    elif IC == "naive":
        LL = -0.5 * naive_fit(fr)
    else:
        raise ValueError(f"Unrecognized choice of info criterion: {IC}")

    # Correction to IC is +2*dc - except for naive IC, which ignores this
    if 't_cut' in test_data.keys() and IC != "naive":
        dc = len(test_data['t_cut'])
        LL -= dc
    
    if IC != "naive" and IC != "BAIC" and IC != "AIC":
        # Replace +2*k with full bias correction, if option is set
        if full_BC:
            tr_B = full_bias(fr, test_data, model_derivs)
            if 't_cut' in test_data.keys() and len(test_data['t_cut'])>0 and not quiet_full_BC:
                print("Warning: Full bias correction neglects long-range correlations. Consider using +2k instead.")
            
            tr_B = full_bias(fr, model_derivs, ND, yraw)

            LL += len(fr.p.keys())
            LL -= tr_B

    IC = -2 * LL
    
    if return_prob:
        return IC, np.exp(LL)
    else:
        return IC

    

def model_avg(gv_list, pr_list):
    """
    Given a list of single-model expectation values {<f(a)>_M} as gvars,
    and a list of raw model probabilities, return the model-averaged estimate
    for <f(a)> as a gvar.
    """

    # Ensure model probabilities are normalized to 1
    pr_list /= np.sum(pr_list)

    mean_avg = np.sum(gv.mean(gv_list) * pr_list)
    var_avg = np.sum(gv.var(gv_list) * pr_list)
    var_avg += np.sum(gv.mean(gv_list) ** 2 * pr_list)
    var_avg -= (np.sum(gv.mean(gv_list) * pr_list)) ** 2

    return gv.gvar(mean_avg, np.sqrt(var_avg))



def tensors_linear_fit(fr, test_data, design_mat):
    """
    Computes relevant tensors for linear fits.
    """
        
    ND = test_data['ND']

    cov_data = ND * gv.evalcov(fr.y)
    cov_data_inv = np.linalg.inv(cov_data)
    
    prior_stats = prior_stats_fit(fr)
    cov_prior_inv = prior_stats['cov_prior_inv']
    chi2_prior = prior_stats['chi2_prior']
    grad_prior = prior_stats['grad_prior']
    hess_prior = prior_stats['hess_prior']
    
    cov_best_fit = np.linalg.inv(cov_prior_inv + ND * design_mat.T @ cov_data_inv @ design_mat)
    
    return {'cov_data_inv': cov_data_inv,
            'chi2_prior': chi2_prior,
            'hess_prior': hess_prior,
            'cov_best_fit': cov_best_fit,
           }

def tensors_nonlinear_fit(fr, test_data, model_derivs):
    """
    Computes relevant tensors for nonlinear fits.
    """
    
    ND = test_data['ND']
    
    cov_data = ND * gv.evalcov(fr.y)
    cov_data_inv = np.linalg.inv(cov_data)    
    
    prior_stats = prior_stats_fit(fr)
    cov_prior_inv = prior_stats['cov_prior_inv']
    chi2_prior = prior_stats['chi2_prior']
    grad_prior = prior_stats['grad_prior']
    hess_prior = prior_stats['hess_prior']
    
    d1_chi2, d2_chi2, d3_chi2 = chi2_derivatives(fr, model_derivs, test_data, chi2_i=False, calc_d3=True)
    
    cov_best_fit = np.linalg.inv(d2_chi2 / 2)
    cov_best_fit_2 = 3 * np.einsum('ab,cd->abcd', cov_best_fit, cov_best_fit)
    
    T = d3_chi2 / 6
    
    return {'cov_data_inv': cov_data_inv,
            'chi2_prior': chi2_prior,
            'grad_prior': grad_prior,
            'hess_prior': hess_prior,
            'cov_best_fit': cov_best_fit,
            'cov_best_fit_2': cov_best_fit_2,
            'T': T
           }



def prior_stats_fit(fr):
    """
    Computes prior statistics from fit object.
    """
        
    cov_prior = np.zeros((len(fr.prior.flat), len(fr.prior.flat)), float)
    blocks = gv.evalcov_blocks(fr.prior, compress=True)
    # uncorrelated pieces are diagonal
    idx, sdev = blocks[0]
    cov_prior[idx, idx] = sdev ** 2
    # correlated pieces
    for idx, bcov in blocks[1:]:
        cov_prior[idx[:, None], idx] = bcov
    
    cov_prior_inv = np.linalg.inv(cov_prior)
    
    p_prior = gv.mean(fr.prior.values())
    p_best_fit = gv.mean(fr.p.values())
    
    chi2_prior = (p_best_fit - p_prior).T @ cov_prior_inv @ (p_best_fit - p_prior)
    grad_prior = 2 * cov_prior_inv @ (p_best_fit - p_prior)
    hess_prior = 2 * cov_prior_inv
    
    return {'cov_prior_inv': cov_prior_inv,
            'chi2_prior': chi2_prior,
            'grad_prior': grad_prior,
            'hess_prior': hess_prior,
           }

def chi2_derivatives(fr, model_derivs, test_data, chi2_i=True, calc_d3=True):
    """
    Compute and return derivatives of the chi-squared function.

    Args:
        fr: Fit result object.
        model_derivs: Model derivatives dictionary (defined in synth_data.py.)
        test_data: Data dictionary.
        chi2_i: If "True", compute derivatives of the individual \chi_i^2 functions, 
                return as an array over the sample index.  Otherwise, compute derivatives
                of the average chi-squared.
        calc_d3: If "True", compute and return the third derivative, otherwise skip it
                to reduce computational cost.

    Returns:
        An array of derivatives of up to length 3 (if calc_d3 is True.)
    """

    ND = test_data['ND']

    cov_data = ND * gv.evalcov(fr.y)
    cov_data_inv = np.linalg.inv(cov_data)    
    
    prior_stats = prior_stats_fit(fr)
    cov_prior_inv = prior_stats['cov_prior_inv']
    
    p_prior = gv.mean(fr.prior.values())
    p_best_fit = gv.mean(fr.p.values())
    
    
    if chi2_i:
        delta = test_data['yraw'] - gv.mean(fr.fcn(fr.x, fr.p))
    else:
        delta = gv.mean(fr.y) - gv.mean(fr.fcn(fr.x,fr.p))
    
    d1_model = model_derivs['d1_model']
    d2_model = model_derivs['d2_model']
    d3_model = model_derivs['d3_model']
    
    if chi2_i:
        d1_chi2 = 2 * (np.einsum('ab,b->a', cov_prior_inv, p_best_fit-p_prior)/ND - np.einsum('ba,bc,ic->ia', d1_model, cov_data_inv, delta))
        d2_chi2 = 2 * (cov_prior_inv/ND - np.einsum('cba,cd,id->iab', d2_model, cov_data_inv,delta) + np.einsum('ba,bc,cd->ad', d1_model, cov_data_inv, d1_model))
    else:
        d1_chi2 = 2 * (np.einsum('ab,b->a', cov_prior_inv, p_best_fit-p_prior) - np.einsum('ba,bc,c->a', d1_model, ND * cov_data_inv, delta))
        d2_chi2 = 2 * (cov_prior_inv - np.einsum('cba,cd,d->ab', d2_model, ND * cov_data_inv,delta) + np.einsum('ba,bc,cd->ad', d1_model, ND * cov_data_inv, d1_model))

    if calc_d3:
        if chi2_i:
            d3_chi2 = 2 * (-np.einsum('dcba,de,ie->iabc', d3_model, cov_data_inv, delta) + (np.einsum('cba,cd,de->abe', d2_model, cov_data_inv, d1_model) + np.einsum('cba,cd,de->aeb', d2_model, cov_data_inv, d1_model) + np.einsum('cba,cd,de->eab', d2_model, cov_data_inv, d1_model)))
        else:
            d3_chi2 = 2 * (-np.einsum('dcba,de,e->abc', d3_model, ND * cov_data_inv, delta) + (np.einsum('cba,cd,de->abe', d2_model, ND * cov_data_inv, d1_model) + np.einsum('cba,cd,de->aeb', d2_model, ND * cov_data_inv, d1_model) + np.einsum('cba,cd,de->eab', d2_model, ND * cov_data_inv, d1_model)))
    
        return [d1_chi2, d2_chi2, d3_chi2]
    else:
        return [d1_chi2, d2_chi2]


def full_bias(fr, test_data, model_derivs):
    """
    Computes the full bias correction.
    """

    d1_chi2, d2_chi2 = chi2_derivatives(fr, model_derivs, test_data, chi2_i=True, calc_d3=False)

    J = np.sum(d2_chi2, axis=0) / (2*test_data['ND'])

    I = np.einsum('ia,ib->ab', d1_chi2, d1_chi2) / (4 * (test_data['ND']-1))

    tr_term = np.linalg.inv(J) @ I

    return np.trace(tr_term)
