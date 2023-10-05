# ComputeSummaryStats returns summary statistics for an array

import numpy as np


def compute_summary_stats_arr(samples_arr, print_en=False, name=""):
    sum_stats = np.percentile(samples_arr, [5, 25, 50, 75, 95])
    if print_en:
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.3e}".format(x)})
        print("Summary Statistics: {}".format(name))
        print("5th, 25th, 50th, 75th, 95th percentiles:")
        print("{}".format(sum_stats))
        print("Mean:")
        print(np.mean(samples_arr))
    return sum_stats
