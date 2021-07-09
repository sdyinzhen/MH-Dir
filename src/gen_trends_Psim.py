
#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Aug 02, 2020
import numpy as np
from scipy.ndimage import gaussian_filter
import time
from scipy.stats import rankdata
def gen_trends_Psim(seis_obs, P_fields, fac_prptn_dist, smpl_size):
    '''
    This is the function for generating facies trends using P-field simulaiton.
    Variables - 
    seis_obs: mapped seismic respones, 2D array, [n_xdim, n_ydim]
    
    P_fields: maps of corresponding percentiles for P-filed simulation, 
                3D array, [n_realizations,n_xdim, n_ydim]
    
    fac_prptn_dist: posterior distribution of facies proportions vs seismic response, dtype-list, 
                item 1. discretized seismic repsonses, 1D array, [n_seis]. 
                        
                item 2. distribution of corresponding facies proportions , list (n_seis x items),
                        each item is a 2D array, [n_post_prptn_smpls, n_facies+1].
                            Note of "+1": is the last column containing CDF values. 
    
    smpl_size: number of trend maps to generate, int, smpl_size<=n_realizations. 
    
    '''
    
    seis_obs_dist = fac_prptn_dist[0]
    prop_cdf_dist = fac_prptn_dist[1]
    
    fac_dim = prop_cdf_dist[0].shape[1]-1
    
    facies_trends = []
    for realnum in range(smpl_size):
        start_t = time.time()
        
        P_field = P_fields[realnum]
        
        f_trend = np.zeros((seis_obs.shape[0], seis_obs.shape[1], fac_dim))
        f_trend[:, :, :] = np.nan
        
        # P-feild simulation of proportions
        for i in range(seis_obs.shape[0]):
            for j in range(seis_obs.shape[1]):
                if np.isfinite(P_field[i, j]):
                    d_obs = seis_obs[i, j]
                    P =  P_field[i,j]
                    seis_obsid = np.abs(seis_obs_dist - d_obs).argmin()
                    f_trend[i, j, :] = dir_P_field_sim(P, prop_cdf_dist[seis_obsid])
        
        # smooth results
        for i in range(f_trend.shape[2]):
            f_trend[:,:,i] = gaussian_filter(f_trend[:,:,i], sigma=0.8)
        
        end_t = time.time()
        
        if realnum == 0:
            est_run_time = (end_t - start_t)*smpl_size/60
            print ('Estimated Running Time:{:.2f}'.format(est_run_time)+' minutes')
        
        print('Progress-> {:1.1%}'.format((realnum+1)/smpl_size), end='\r')
        
        facies_trends.append(f_trend)

    facies_trends = np.asarray(facies_trends)
    
    return facies_trends


def dir_P_field_sim(P, smpls_cdf):
    '''
    This is the P-field simulation function for Dirichlet distribution. 
    variables - 
    P: percentile value to sample, float, value between 0-1. 
    smpls_cdf: distribution of facies proportions, 2D array,
                [n_post_samples, n_facies+1] (Note: the last columnn ("+1") contains the cdf values)
    '''
    
    ### Simulate the first variable
    # Rank the first variable
    i = 0
    
    smpls_cdf = smpls_cdf[smpls_cdf[:,i].argsort()]
    dim = len(smpls_cdf[0])
    
    # Find out the data samples with percertile between [P+0.02, P-0.02]
    inquiry_val = smpls_cdf[(smpls_cdf[:, dim-1]<P+0.02) & (smpls_cdf[:, dim-1]>P-0.02)]

    ### Sample the 2nd variable
    
    i = i + 1
    # Calculate the 2nd variable cdf of the selected data samples
    cdf = (rankdata(inquiry_val[:,i]))/inquiry_val.shape[0]
    # Find out the data samples with percertile between [P+0.05, P-0.05]
    new_inquiry = inquiry_val[(cdf<1-P+0.05) & (cdf>1-P-0.05)]
    
    if new_inquiry.size==0:
        final_ind = np.argmin(np.abs(cdf - P))
        Psim_result = inquiry_val[final_ind, :dim-1]
    else:
        inquiry_val = new_inquiry
        # Sample the 3rd and finish
        i = i +1 
        # Calculate the 3rd variable cdf of the selected data samples
        cdf = (rankdata(inquiry_val[:,i]))/inquiry_val.shape[0]

        # find the 3rd sample that is most close to P. 
        final_ind = np.argmin(np.abs(cdf - P))

        Psim_result = inquiry_val[final_ind, :dim-1]
    
    return Psim_result