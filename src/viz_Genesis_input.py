
import numpy as np
import matplotlib.pyplot as plt

def viz_Genesis_input(fac_prop_pri, seis_pri, SeisObs_map):
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    MS = plt.scatter(seis_pri, fac_prop_pri[:,0], c = 'yellow', linewidths=0.5, edgecolors= 'k', s = 20)
    BK = plt.scatter(seis_pri, fac_prop_pri[:,1], c='grey', linewidths=0.5, edgecolors= 'k', s = 20, alpha=0.8)
    plt.ylabel('Facies proportion', fontsize = 14)
    plt.xlabel('Seismic amplitude', fontsize = 14)
    plt.title('Prior dist.: facies proportion vs seis amplitude ')
    plt.legend((MS, BK),
               ('Massive sand', 'Background shale', ),
               scatterpoints= 1,
               loc='upper left', ncol=2, fontsize=10)
    plt.gca().invert_xaxis()
    plt.xlim(0, -1500)
    plt.subplot(122)
    plt.imshow(SeisObs_map, cmap='jet_r', vmin = -800)
    plt.colorbar(fraction = 0.04)
    plt.title('Observed seismic amplitude map')
    plt.show()
    return 