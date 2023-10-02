#heatmap_wr_ranges creates a heatmap of two welfare range distributions 


import matplotlib.pyplot as plt
import matplotlib as mpl

def heatmap_wr_ranges(data1,data2,animal1,animal2,xlims,ylims,a1_mean,a2_mean,\
                      correlation_coeff,title_str,text_loc,numBins=20,printEn=False):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('lightgray')
    h = plt.hist2d(data1, data2, density=True,bins=numBins,norm=mpl.colors.LogNorm(),cmap=mpl.cm.Reds)
    ax.set_xlabel('P(sentience)-Adjusted Welfare Range of {}'.format(animal1))
    ax.set_ylabel('P(sentience)-Adjusted Welfare Range of {}'.format(animal2))
    plt.text(text_loc[0],text_loc[1], 'Mean ({}) = {} \nMean ({}) = {} \nCorrelation = {}'.format(animal1,a1_mean,animal2,a2_mean,correlation_coeff))
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Density', rotation=270)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    plt.title('Welfare Ranges ({})'.format(title_str))
    plt.grid()
    plt.show()

    if printEn:
        name = './Plots/%s.png' % title_str
        print(name)
        fig.savefig(name)    