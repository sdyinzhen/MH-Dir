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
    for i in range(len(Seis_obs)):
        start_t = time.time()
        print('Progress -> {:1.1%}'.format((i+1)/Seis_obs.shape[0]), end='\r')

        # Run MH
        post_smpl = dir_mh_sampling(MC_Prptn, MC_Seis, Ini_alpha, [Seis_obs[i]], \
                                    maxstep=Maxstep, delta_left=delta_left, delta_right=delta_right)

        post_smpl_all.append(post_smpl[0][:,0,:])

        end_t = time.time()
        if i == 0:
            est_run_time = (end_t - start_t)*Seis_obs.shape[0]/60
            print ('Estimated Running Time:{:.2f}'.format(est_run_time)+' minutes')
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
    MC_c_seis = np.c_[MC_c_prop[:,:c_dim-1], MC_seis]
    
    # KDE of the joint distribution between Monte Carlo samples of seismic ('seis') and facies proportion ('c')
    endog_dim = 'c'*1
    indep_dim = 'c'*(c_dim-1)
    kde_seis_c = sm.nonparametric.KDEMultivariateConditional(endog=MC_c_seis[:,:c_dim-1], \
                                                             exog=MC_c_seis[:,c_dim-1:], \
                                                             dep_type=indep_dim, indep_type=endog_dim, bw='normal_reference')
    
    '''Initial state''' 
    beta = np.copy(ini_alpha)
    c_pos = np.random.dirichlet(beta, 1)
    c_pos_all = []
    c_pos_all.append(c_pos)
    
    beta_all = []
    beta_all.append(beta)
    
    '''calculate pdf of initial c_pos and the proposed c_star'''
    
    p_seis_cpos = kde_seis_c.pdf(endog_predict=c_pos[0, :c_dim-1], exog_predict=seis_obs)

    dir_cpos = stats.dirichlet.pdf(c_pos[0,:], beta)
    
    dir_AlphaHat_cpos = stats.dirichlet.pdf(c_pos[0,:], ini_alpha)
    
    '''determine delta '''
#     delta = jumpwidth * ini_alpha
    
    '''define the proposed uniform distribution f(beta)'s lower and upper bound'''
    Jalpha_max = ini_alpha*(1+delta_right)
        
    Jalpha_min = ini_alpha*(1-delta_left)
    
    '''ensure each element of alpha >0'''
    Jalpha_min[Jalpha_min<=0] = 0.1 
    
    itr=0
    
    t=1
    Beta_reject = True

    reject_n = 0
    
    while t<maxstep:
#     for i in range(maxstep):
        
        try:
#             '''sample from proposed posterior'''
#         if Beta_reject:
            if reject_n > 4:
                if itr>60 :
                    pre_beta = np.copy(np.asarray(beta_all))
    #                 beta = np.mean(pre_beta,axis=0)
                    beta = np.random.uniform(Jalpha_min, Jalpha_max, 4)
    #                 beta = np.copy(beta_all[-1])
                    itr=0
                for i in range(len(beta)):
                    '''seach best beta gradually'''
                    beta[i] = np.random.uniform(0.97*beta[i], 1.03*beta[i])
                    if beta[i]<Jalpha_min[i]:
                        beta[i]=Jalpha_min[i]
                    elif beta[i]>Jalpha_max[i]:
                        beta[i]=Jalpha_max[i]
                        
                        
#             else: 
#                 beta = np.random.uniform(Jalpha_min, Jalpha_max, 4)  
            c_star = np.random.dirichlet(beta, 1)
            dir_cstar = stats.dirichlet.pdf(c_star[0,:], beta)
            dir_AlphaHat_p_star =  stats.dirichlet.pdf(c_star[0,:], ini_alpha)

        
        except:
            next
            
        '''calculate p(seis|c)'''

        p_seis_cstar = kde_seis_c.pdf(endog_predict=c_star[0, :c_dim-1], exog_predict=seis_obs)

        
        '''calculate acceptance ratio'''

        r = (p_seis_cstar*dir_AlphaHat_p_star*dir_cstar)/(p_seis_cpos*dir_AlphaHat_cpos*dir_cpos)

        u = np.random.uniform(0, 1, 1)

        '''Obstain posterior sample'''
        

        if r>u: # accept
        
            c_pos = c_star
            c_pos_all.append(c_pos)
            
            dir_cpos = dir_cstar
            dir_AlphaHat_cpos = dir_AlphaHat_p_star
            p_seis_cpos = p_seis_cstar
            beta_all.append(np.copy(beta))
            
            Beta_reject = False
            t=t+1

            #print('Progress -> {:1.1%}'.format(t/maxstep), end='\r')
            itr=0
            reject_n = 0
        else:
            Beta_reject = True
            itr = itr + 1
            reject_n = reject_n + 1
                
           
    c_pos_all = np.asarray(c_pos_all)
    beta_all = np.asarray(beta_all)
    
    return c_pos_all, beta_all