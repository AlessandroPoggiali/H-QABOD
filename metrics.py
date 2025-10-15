import math
from scipy import stats
import pandas as pd
import numpy as np

def precision_at_n(observed_ranking, ground_truth_ranking, n):
    
    df = pd.DataFrame(list(zip(observed_ranking,ground_truth_ranking)), columns=['observed','ground_truth'])
    new_idx = np.argsort(observed_ranking)
    new_idx2 = np.argsort(ground_truth_ranking)
    df_2 = df.reindex(new_idx)
    df_3 = df.reindex(new_idx2)

    predicted = df_2[:n].index
    actual = df_3[:n].index
    c = 0
    for p in predicted:
        if p in actual:
            c = c + 1
    return np.round((c/len(predicted)), 3)

def kendalltau(observed_ranking, ground_truth_ranking):
    new_idx = np.argsort(observed_ranking)
    new_idx2 = np.argsort(ground_truth_ranking)
    return np.round(stats.kendalltau(new_idx, new_idx2).statistic, 3)

def compute_rbo(observed_ranking, ground_truth_ranking, p):
    new_idx = np.argsort(observed_ranking)
    new_idx2 = np.argsort(ground_truth_ranking)
    return np.round(rbo(new_idx, new_idx2,p), 3)

def rbo(S,T, p= 0.9):

    """ Takes two lists S and T of any lengths and gives out the RBO Score
    Parameters
    ----------
    S, T : Lists (str, integers)
    p : Weight parameter, giving the influence of the first d
        elements on the final score. p<0<1. Default 0.9 give the top 10 
        elements 86% of the contribution in the final score.
    
    Returns
    -------
    Float of RBO score
    """
    
    # Fixed Terms
    k = max(len(S), len(T))
    x_k = len(set(S).intersection(set(T)))
    
    summation_term = 0

    # Loop for summation
    # k+1 for the loop to reach the last element (at k) in the bigger list    
    for d in range (1, k+1): 
            # Create sets from the lists
            set1 = set(S[:d]) if d < len(S) else set(S)
            set2 = set(T[:d]) if d < len(T) else set(T)
            
            # Intersection at depth d
            x_d = len(set1.intersection(set2))

            # Agreement at depth d
            a_d = x_d/d   
            
            # Summation
            summation_term = summation_term + math.pow(p, d) * a_d

    # Rank Biased Overlap - extrapolated
    rbo_ext = (x_k/k) * math.pow(p, k) + ((1-p)/p * summation_term)

    return rbo_ext

def weightage_calculator(p,d):
    """ Takes values of p and d
    ----------
    p : Weight parameter, giving the influence of the first d
        elements on the final score. p<0<1.
    d : depth at which the weight has to be calculated
    
    Returns
    -------
    Float of Weightage Wrbo at depth d
    """

    summation_term = 0

    for i in range (1, d): # taking d here will loop upto the value d-1 
        summation_term = summation_term + math.pow(p,i)/i


    Wrbo_1_d = 1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - summation_term))

    return Wrbo_1_d

def compute_discrepancy(trials, var_tutta, var_triu):
    c = 0
    k = 0
    for i in range(trials):
        for j in range(trials):
            if i != j:
                if (var_tutta[i] > var_tutta[j] and var_triu[i] < var_triu[j]) or (var_tutta[i] < var_tutta[j] and var_triu[i] > var_triu[j]):
                    c = c + 1
                else:
                    k = k + 1
    #print(str(round(c/(c+k) * 100, 2)) + "%")
    return round(c/(c+k) * 100, 2)

