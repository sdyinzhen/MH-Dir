import numpy as np
import matplotlib.pyplot as plt
def check_Genesis_MH_results(seis_obs, post_sampls, seis_obs_all):
    
    obs_id = np.argmin(np.abs(seis_obs_all - seis_obs))

    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(post_sampls[obs_id][:, 0], '.', markeredgecolor = 'k', \
             markerfacecolor = 'yellow', markeredgewidth = 0.2)
    plt.xlabel('MH steps')
    plt.ylabel('Massive sand proportion', fontsize = 12)
    plt.subplot(122)
    plt.plot(post_sampls[obs_id][:, 1], '.', markeredgecolor = 'k', \
             markerfacecolor = 'grey', markeredgewidth = 0.2)
    plt.xlabel('MH steps')
    plt.ylabel('Background shale proportion', fontsize = 12)
    
    plt.suptitle('Trace plot: Metropolis-Hastings results for Seis_obs = ' + str(seis_obs), fontsize=15)
    
    plt.show()
    
    return