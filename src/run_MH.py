#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Aug 02, 2020
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import scipy.stats as stats


def run_MH(MC_Prptn, MC_Seis, Ini_alpha, Seis_obs, Maxstep=25000, delta_left=0.2, delta_right=0.2):
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
    print('Program is Running & Calculating Cost Time... Please Be Patient :)', end='\r')
    start_t_total =  time.time()
    for i in range(len(Seis_obs)):
        # Run MH
        runflag = True
        while runflag:
            start_t = time.time()
            [post_smpl, post_beta, runflag] = dir_mh_sampling(MC_Prptn, MC_Seis, Ini_alpha, Seis_obs[i:i+1], \
                                                              maxstep=Maxstep, \
                                                              delta_left=delta_left, delta_right=delta_right)
        post_smpl_all.append(post_smpl[:,0,:])

        end_t = time.time()
        est_run_time = (end_t - start_t)*(Seis_obs.shape[0]-i-1)/60
        progress = (i+1)/Seis_obs.shape[0]
        
        print(' Running in Progress -> {:1.1%}. '.format(progress) + 
              'Estimated Remaining Time:{:.2f}'.format(est_run_time)+' Minutes', end='\r')
    
    post_smpl_all = np.asarray(post_smpl_all)
    print(' \n Finished! Total Cost Time: {:.2f}'.format((time.time()-start_t_total)/60) +' Minutes')
    return post_smpl_all



def dir_mh_sampling(MC_c_prop, MC_seis, ini_alpha, seis_obs, maxstep = 1000, delta_left=0.2, delta_right=0.2):
    
    '''
    This is Metropolis-Hastings (MH) algorithm to sample posterior facies proportion under Dirichlet Dist. 
    Input Variables - 
        MC_c_prop: Monte Carlos samples of facies proportions, 2D array, [n_MCsamples, n_facies]
        
        MC_seis: Monte Carlos samples of forward modeled seismic responses corresponding to MC_c_prop, 
                1D array, [n_MCsamples]
        
        ini_alpha: Dirichlet concentration parameters of the input MC_c_prop. 
        
        seis_obs: Real observed seismic value, 1 list item, format - [seisobs_val]
        
        maxstep: optional, maximum steps of MH sampling, default = 10,000
        
        delta_left: lower bounds (delta_1) of beta, as a proportion of alpha, default = 20% of alpha
        delta_right: upper bounds (delta_2) of beta, as a proportion of alpha, default = 20% of alpha
        
    Output 
        c_pos_all: all the posterior facies proportion samples from MH
        beta_all: all the posterior Dirichlet concentration parameters from MH
    '''
    
    c_dim = MC_c_prop.shape[1]
    seis_dim = MC_seis.shape[1]
    # KDE of the joint distribution between Monte Carlo samples of seismic ('seis') and facies proportion ('c')
    ## calculate f(y|x) or here f(seis|c); 
    # dependendt var, endog: y, here is seismic; 
    # independent var, exog x, here is proportion c.
    dep_dim = 'c'*seis_dim
    indep_dim = 'c'*(c_dim-1)
    # KDE of the joint distribution between Monte Carlo samples f(seis, c)
    kde_seis_c = sm.nonparametric.KDEMultivariateConditional(endog=MC_seis, \
                                                             exog=MC_c_prop[:,:c_dim-1], \
                                                             dep_type=dep_dim, indep_type=indep_dim, 
                                                             bw='normal_reference')
    
    '''Initial state''' 
    c_pos = np.random.dirichlet(ini_alpha, 1)
    
    c_pos_all = []
    c_pos_all.append(c_pos)
    beta_all = []
    beta_all.append(ini_alpha)
    
    '''calculate pdf of initial c_pos and the proposed c_star'''
    
    p_seis_cpos = kde_seis_c.pdf(endog_predict=seis_obs, exog_predict=c_pos[0, :c_dim-1])

    dir_cpos = stats.dirichlet.pdf(c_pos[0,:], ini_alpha)
    
    dir_AlphaHat_cpos = stats.dirichlet.pdf(c_pos[0,:], ini_alpha)
    
    '''determine delta '''
    '''define the proposed uniform distribution f(beta)'s lower and upper bound'''
    Jalpha_max = ini_alpha + delta_right
    Jalpha_min = ini_alpha - delta_left
    
    '''ensure each element of alpha >0'''
    Jalpha_min[Jalpha_min<=0] = 0.1 
    
    beta = np.random.uniform(Jalpha_min, Jalpha_max)
    
    # locally perturb the beta 
    jumpwidth = 0.025
    delta = jumpwidth * ini_alpha.mean()
    
    t = 1 
    itr = 0
    while t<maxstep:
            
        # locally perturb the beta 
        Jalpha_max_local = beta + delta
        Jalpha_min_local = beta -delta
        
        Jalpha_min_local[Jalpha_min_local<=Jalpha_min] = Jalpha_min[Jalpha_min_local<=Jalpha_min]
        Jalpha_max_local[Jalpha_max_local>=Jalpha_max] = Jalpha_max[Jalpha_max_local>=Jalpha_max]
        
        beta = np.random.uniform(Jalpha_min_local, Jalpha_max_local)
        # sample c_star and calculate the probablity
        try:
            c_star = np.random.dirichlet(beta, 1)
            dir_cstar = stats.dirichlet.pdf(c_star[0,:], beta)
            dir_AlphaHat_p_star =  stats.dirichlet.pdf(c_star[0,:], ini_alpha)
        except:
            
            next
        '''calculate p(seis|c)'''

        p_seis_cstar = kde_seis_c.pdf(endog_predict=seis_obs, exog_predict=c_star[0, :c_dim-1])

        
        '''calculate acceptance ratio'''
   
        r = (p_seis_cstar*dir_AlphaHat_p_star*dir_cstar)/(p_seis_cpos*dir_AlphaHat_cpos*dir_cpos)
        
#         r = (p_seis_cstar*dir_cstar)/(p_seis_cpos*dir_cpos)
        
        u = np.random.uniform(0, 1, 1)

        '''Obtain posterior sample'''
        if r>u: # accept
            c_pos = c_star
            c_pos_all.append(c_pos)
            
            dir_cpos = dir_cstar
            dir_AlphaHat_cpos = dir_AlphaHat_p_star
            p_seis_cpos = p_seis_cstar
            beta_all.append(np.copy(beta))
            
            t=t+1
            itr = 0
#             print('Progress -> {:1.1%}'.format(t/maxstep), end='\r')
        else:
            itr = itr+1        
            # stop the run if rejected 10000 times per acceptation 
            if itr>10000:
                c_pos_all = np.asarray(c_pos_all)
                beta_all = np.asarray(beta_all)

                return c_pos_all, beta_all, True

           
    c_pos_all = np.asarray(c_pos_all)
    beta_all = np.asarray(beta_all)
    
    return c_pos_all, beta_all, False