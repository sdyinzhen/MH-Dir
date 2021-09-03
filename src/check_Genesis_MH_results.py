import numpy as np
import matplotlib.pyplot as plt
def check_Genesis_MH_results(seis_obs, post_sampls, seis_obs_all):
    
    obs_id = np.argmin(np.abs(seis_obs_all - seis_obs))
    names = ['Massive sand', 'Background shale', 'Thin-bedded sand','Margin silt' ]
    color = ['yellow', 'grey', 'orangered', 'dodgerblue']
    plt.figure(figsize=(12, 7))
    for i in range(post_sampls.shape[2]):
        plt.subplot(2, 2, i + 1)
        plt.plot(post_sampls[obs_id, :, i], '.', markeredgecolor = 'k', \
                 markerfacecolor = color[i], markeredgewidth = 0.2,
                 label= names[i])
#         plt.title(names[i], weight='bold')
        plt.legend(loc=2, fontsize=13)
        plt.xlabel('MH steps', fontsize = 16)
        plt.ylabel('proportion', fontsize = 16)
        plt.ylim(post_sampls[obs_id].min(),post_sampls[obs_id].max()*1.1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
    plt.tight_layout()
    print('Trace plot: Metropolis-Hastings results for ** Seis_obs = {} **'.format(seis_obs))
    
    return