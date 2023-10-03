#ComputeSummaryStats returns summary statistics for an array  

import numpy as np

def computeSummaryStatsArr(samplesArr,printEn=False,name=""):
    sumStats = np.percentile(samplesArr, [5, 25, 50, 75, 95]) 
    if printEn:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3e}".format(x)})
        print("Summary Statistics: {}".format(name))
        print("5th, 25th, 50th, 75th, 95th percentiles:")
        print("{}".format(sumStats))
        print("Mean:")
        print(np.mean(samplesArr))
    return sumStats