#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Aug 02, 2020
import numpy as np
def cal_emp_cdf(insamples):
    '''
    This is  function to calcualte emperical CDF of Dirichlet distributed facies proportion samples.
    Variables:
    insamples - input samples of facies proportions, 
                3D array, [n_seis_features, n_posterior_samples, n_facies]
    '''
    smpls_cdf_libs = []
    for i in range(len(insamples)):
        samples = insamples[i]
        cdfs = [np.count_nonzero(samples[j,0]>samples[:,0])/samples.shape[0] \
                for j in range(samples.shape[0])] 
        cdfs = np.asarray(cdfs)

        smpls_cdf_libs.append(np.c_[samples, cdfs])
    return smpls_cdf_libs