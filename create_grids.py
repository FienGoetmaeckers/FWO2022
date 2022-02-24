# -*- coding: utf-8 -*-
"""
This module creates the reward distribution of the spatial grid
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def bivariate(W, L, lam = 2, max_r = 100, min_r = 0, show = True):
    kernel_matrix = k_matrix(lam, W, L)

    bandit = np.random.multivariate_normal([0]*L*W, kernel_matrix)
    bandit = (bandit - min(bandit))/(max(bandit)-min(bandit)) #to normalize
    bandit = bandit * (max_r-min_r) #to let it vary between min_r and max_r
   
    if show:
        plt.rcParams['font.size'] = str(W*4)
        plt.figure(figsize=(2*W, 2*L))  
        ax = sns.heatmap(bandit.reshape((L, W)), linewidth=0.5, cmap = 'Reds', annot=True, square = True, cbar = True)
        plt.show()
        
    return bandit


def kRBF(v1, v2, lam): #v1 en #v2 are vectors (2D), saved as np.arrays
    return np.exp(-(np.linalg.norm(v1-v2))**2/lam)


def k_matrix(lam, W, L = 11):
    
    spatial_vector_list = []
    for i in range(0, L):
        for j in range(0, W):
            spatial_vector_list.append(np.array([i, j]))
    return [[kRBF(spatial_vector_list[i], spatial_vector_list[j], lam) 
             for i in range(0, W*L)] for j in range(0, W*L)]