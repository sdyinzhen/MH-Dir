import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
def trend_samples_vis(model_ndarray, m_name, reals = None):
    '''
    Plot the 9 samples of trend maps. 
    Args:
        model_ndarray: ndarray of monte carlo trend maps, N_realizations x Grid_dims x n_facies
        m_name: name of the trend maps, str
        reals: assign 9 realizations to visualize, 1D array, 
                defaul is None - visualize the first 9 realizations. 
    '''
    fig=plt.figure(figsize=(13,16))
    count = 1
    
    c_map =  mcolors.LinearSegmentedColormap.from_list("", ['blue','dodgerblue','white','tomato','red'])
    
    for realnum in range(10):
        if count  == 10:
            plot=fig.add_subplot(2, 5, count)
            plt.text(0.1, 0.48, '...', fontsize=50)
            plt.text(0.0, 0.6, 'Total '+str(len(model_ndarray))+' samples', fontsize=16, style='italic')
            plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
            count = count + 1
        else:
            if reals is None:
                plotnum = realnum
            else:
                plotnum = reals[realnum]
            grid_data = model_ndarray[plotnum]      
            plot=fig.add_subplot(2, 5, count)
            count = count+1
            prop_mean = format(np.mean(grid_data[np.isfinite(grid_data)]),'.2f')
            
            plot.set_xlabel('global proportion = ' + str(prop_mean), fontsize = 12)
            c_max = np.max(grid_data)
            c_min = np.min(grid_data)
            
            plt.imshow(grid_data,cmap=c_map, vmin = 0, vmax = 0.6) 
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title(m_name+ ' trend #'+str(count-1), fontsize=12, style='italic', weight='bold')
            plt.colorbar(fraction = 0.04, orientation = 'horizontal')
            
    plt.subplots_adjust(top=0.55, bottom=0.08, left=0.10, right=0.95, hspace=0.15,
                    wspace=0.35)
    return