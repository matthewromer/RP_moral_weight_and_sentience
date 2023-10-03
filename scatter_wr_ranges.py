#scatter_wr_ranges creates a scatter plot of two welfare range distributions 


import matplotlib.pyplot as plt

def scatter_wr_ranges(data1,data2,animal1,animal2,xlims,ylims,a1_mean,a2_mean,\
                      correlation_coeff,title_str,text_loc,area=5,printEn=False):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('P(sentience)-Adjusted Welfare Range of {}'.format(animal1))
    ax.set_ylabel('P(sentience)-Adjusted Welfare Range of {}'.format(animal2))
    plt.text(text_loc[0],text_loc[1], 'Mean ({}) = {} \nMean ({}) = {} \nCorrelation = {}'.format(animal1,a1_mean,animal2,a2_mean,correlation_coeff))
    plt.scatter(data1, data2, s=area)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    plt.title('Welfare Ranges ({})'.format(title_str))
    plt.grid()
    plt.show()
    
    if printEn:
        name = './Plots/%s_Scatter.png' % title_str
        print(name)
        fig.savefig(name)    