#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Aug 02, 2020
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import scipy.stats as stats


def run_MH(MC_Prptn, MC_Seis, Ini_alpha, Seis_obs, Maxstep=25000, Jumpwidth=0.06):
    '''
    This is the function to run Metropolis-Hastings (MH) for all the real seismic observations.
    
    Input Variables - 
    MC_Prptn: Monte Carlos samples of facies proportions, 2D array, [n_MCsamples, n_facies]

    MC_Seis: Monte Carlos samples of forward modeled seismic responses corresponding to MC_c_prop, 
            1D array, [n_MCsamples]

    Ini_alpha: Dirichlet concentration parameters of the input MC_c_prop. 

    Seis_obs: Real observed seismic values, 1D array, [n_seisobs]

    Maxstep: optional, maximum steps of MH sampling, default = 10,000

    Jumpwidth: optional, jump width of MH each step, default = 6%
    
    '''
    
    post_smpl_all = []
    for i in range(len(Seis_obs)):
        start_t = time.time()
        print('Progress -> {:1.1%}'.format((i+1)/Seis_obs.shape[0]), end='\r')

        # Run MH
        post_smpl = dir_mh_sampling(MC_Prptn, MC_Seis, Ini_alpha, [Seis_obs[i]], \
                                    maxstep=Maxstep, jumpwidth=Jumpwidth)

        post_smpl_all.append(post_smpl[0][:,0,:])

        end_t = time.time()
        if i == 0:
            est_run_time = (end_t - start_t)*Seis_obs.shape[0]/60
            print ('Estimated Running Time:{:.2f}'.format(est_run_time)+' minutes')
    return post_smpl_all


def dir_mh_sampling (MC_c_prop, MC_seis, ini_alpha, seis_obs, maxstep = 10000, jumpwidth =0.06):
    
    '''
    This is Metropolis-Hastings (MH) algorithm to sample posterior facies proportion under Dirichlet Dist. 
    Input Variables - 
        MC_c_prop: Monte Carlos samples of facies proportions, 2D array, [n_MCsamples, n_facies]
        
        MC_seis: Monte Carlos samples of forward modeled seismic responses corresponding to MC_c_prop, 
                1D array, [n_MCsamples]
        
        ini_alpha: Dirichlet concentration parameters of the input MC_c_prop. 
        
        seis_obs: Real observed seismic value, 1 list item, format - [seisobs_val]
        
        maxstep: optional, maximum steps of MH sampling, default = 10,000
        
        jumpwidth: optional, jump width of MH each step, default = 6%
        
    Output 
        c_pos_all: all the posterior facies proportion samples from MH
        alpha_all: all the posterior Dirichlet concentration parameters from MH
    '''
    
    c_dim = MC_c_prop.shape[1]
    MC_c_seis = np.c_[MC_c_prop[:,:c_dim-1], MC_seis]
    
    # KDE of the joint distribution between Monte Carlo samples of seismic ('seis') and facies proportion ('c')
    endog_dim = 'c'*1
    indep_dim = 'c'*(c_dim-1)
    kde_seis_c = sm.nonparametric.KDEMultivariateConditional(endog=MC_c_seis[:,:c_dim-1], \
                                                             exog=MC_c_seis[:,c_dim-1:], \
                                                             dep_type=indep_dim, indep_type=endog_dim, bw='normal_reference')
    
    '''Initial state''' 
    alpha_pos = ini_alpha
    c_pos = np.random.dirichlet(alpha_pos, 1)
    c_pos_all = []
    c_pos_all.append(c_pos)
    alpha_all = []
    alpha_all.append(alpha_pos)
    
    '''calculate pdf of initial c_pos and the proposed c_star'''
    
    p_seis_cpos = kde_seis_c.pdf(endog_predict=c_pos[0, :c_dim-1], exog_predict=seis_obs)

    dir_cpos = stats.dirichlet.pdf(c_pos[0,:], alpha_pos)
    
    '''determine jumpwidth'''
    delta = jumpwidth * ini_alpha.mean()
    
    for i in range(maxstep):
        '''define the proposed uniform distribution J(c^*|c_i)'s lower and upper bound'''
        Jalpha_max = alpha_pos + delta
        
        Jalpha_min = alpha_pos-delta
        
        Jalpha_min[Jalpha_min<=0] = 0.001 
        
        '''ensure each element of alpha >0'''
#         while True:
        alpha_star = np.random.uniform(Jalpha_min, Jalpha_max, 4)  
#             if np.alltrue(np.greater(alpha_star,0)):
#                 break
            
        try:
            '''sample from proposed posterior'''
            c_star = np.random.dirichlet(alpha_star, 1)
            dir_cstar = stats.dirichlet.pdf(c_star[0,:], alpha_star)
        except:
            next
            
        '''calculate p(seis|c)'''
        p_seis_cstar = kde_seis_c.pdf(endog_predict=c_star[0, :c_dim-1], exog_predict=seis_obs)
        
        '''calculate acceptance ratio'''
        r = (p_seis_cstar*dir_cstar)/(p_seis_cpos*dir_cstar)
        u = np.random.uniform(0, 1, 1)

        '''Obstain posterior sample'''

#         print('Process-> {:1.2%}'.format((i+1)/maxstep), end='\r')
        if r>u: # accept
            c_pos = c_star
            c_pos_all.append(c_pos)
            dir_cpos = dir_cstar
            p_seis_cpos = p_seis_cstar
            alpha_pos = alpha_star
            alpha_all.append(alpha_pos)

    c_pos_all = np.asarray(c_pos_all)
    alpha_all = np.asarray(alpha_all)
    
    return c_pos_all, alpha_all