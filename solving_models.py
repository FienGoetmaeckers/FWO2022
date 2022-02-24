# -*- coding: utf-8 -*-
"""
different models to find rewards on the bandit
"""
import random
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns
from statistics import mean

def opened_cell(true_bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number, W, L, max_r=100, min_r = 0, follow = False):
    """
    function called after every sample
    the bandit is shown (if follow = True)
    the lists are updated

    Parameters
    ----------
    true_bandit : the generated spatial grid, the underlying reward distribution
    observed_bandit : remembers which rewards have been observed where
    mean_bandit : remembers the mean of all the observed rewards per cell
                    defines the color of the cell in the shown grid
    tile_number : the sampled cell number

    Returns
    -------
    the updated lists

    """
    reward = true_bandit[tile_number]
    
    #step 1: check if this cell has been opened or not
    if hidden[tile_number] == True: #then it's the first time
        mean_bandit[tile_number] = reward
        observed_bandit[tile_number] = [reward]
    else: 
        #add it to the observed_bandit list 
        observed_bandit[tile_number].append(reward)
        mean_bandit[tile_number] = mean(observed_bandit[tile_number])
        
    
    latest_bandit[tile_number] = reward
    hidden[tile_number] = False
    
    if follow:
        
        plt.rcParams['font.size'] = str(W*4)
        plt.figure(figsize=(2*W, 2*L))
        #ax = sns.heatmap(latest_bandit.reshape((L, W)), linewidth=0.5, linecolor='k', cmap = mean_bandit, vmax=max_r , vmin=min_r , annot=True, square = True, cbar = False, mask = hidden.reshape((L, W)))
        ax = sns.heatmap(mean_bandit.reshape((L, W)), linewidth=0.5, linecolor='k', cmap = 'Reds', vmax=max_r , vmin=min_r , annot=latest_bandit.reshape((L, W)), square = True, cbar = False, mask = hidden.reshape((L, W)))
        plt.show()
    
    return (observed_bandit, latest_bandit, mean_bandit, hidden, reward)
    

def comp_random(bandit, W, L, trials, max_r, min_r, follow = False):
    """
    computer solves a grid randomly (no learning behaviour)
 
    Parameters
    ----------
    bandit :    The bandit, which is a list with the rewards.
    W :         width of the bandit.
    L :         length of the bandit.
    trials :    size of the search horizon, number of tiles that can be chosen.
    max_r :     maximum reward
    min_r:      minimum reward

    Returns
    -------
    Learning curve

    """
    
    total_r = 0
    total_r_list = []
    
    hidden = np.array([True]*L*W)  
    observed_bandit = [[value] for value in bandit]
    latest_bandit = bandit.copy()
    mean_bandit = bandit.copy()
        
    
    tile_number = random.randint(0, (W*L-1))
    row = math.floor(tile_number/W)
    column = tile_number%W
    
    
    (observed_bandit, latest_bandit, mean_bandit, hidden, reward) = opened_cell(bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number, W, L, max_r, min_r, follow)

    
    for picks in range(0, trials):
        tile_number = random.randint(0, (W*L-1))
        row = math.floor(tile_number/W)
        column = tile_number%W
        
        (observed_bandit, latest_bandit, mean_bandit, hidden, reward) = opened_cell(bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number,  W, L, max_r, min_r, follow)
        
        total_r += reward
        total_r_list.append(total_r)
        
        
        if follow:
            print("The computer chose cell ({}, {}).".format(row, column))
            
            #show_bandit(bandit, W, L, hidden, max_r, min_r)
            print("Average accumulated reward = {}.".format(total_r/(picks+1)))
    
    return total_r_list
    
   
def me(bandit, W, L, trials, max_r, min_r, follow=True):
    """
    Participant solves this
 
    Parameters
    ----------
    bandit : The bandit, which is a list with the rewards.
    W : width of the bandit.
    L : length of the bandit.
    trials : size of the search horizon, number of tiles that can be chosen.
    max_r :     maximum reward
    min_r:      minimum reward

    Returns
    -------
    Learning curve

    """
    
    
    '''
    #step 1: one random cell is revealed
    '''
    
    total_r = 0
    total_r_list = []
    
    hidden = np.array([True]*L*W)  
    observed_bandit = [[value] for value in bandit]
    latest_bandit = bandit.copy()
    mean_bandit = bandit.copy()
    
    tile_number = random.randint(0, (W*L-1))
    row = math.floor(tile_number/W)
    column = tile_number%W
    
    
    (observed_bandit, latest_bandit, mean_bandit, hidden, reward) = opened_cell(bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number, W, L, max_r, min_r, follow)

  
    for picks in range(0, trials):
        print("\n\nChoose a tile by giving its coordinates.")
        
        row = int(input("chosen row = "))
        while row >= 11:
            print("This row is not an option. Try a row number between 0 and 10.")
            row = int(input("chosen row = "))
        column = int(input("chosen column = "))
        while column >= 11:
            print("This columnis not an option. Try a column number between 0 and 10.")
            column = int(input("chosen column = "))
        tile_number = row*W + column
        (observed_bandit, latest_bandit, mean_bandit, hidden, reward) = opened_cell(bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number,  W, L, max_r, min_r, follow)
        total_r += reward
        total_r_list.append(total_r)
        print("\n\nYou chose cell ({}, {}).".format(row, column))
        print("Average accumulated reward = {}.".format(total_r/(picks+1)))
        
    return total_r_list



def GP(true_bandit, observed_bandit, mean_bandit, latest_bandit, W, L, l_fit, hidden, follow = True, epsilon = 0.01):
    '''

    Parameters
    ----------
    bandit : the bandit with the observed reward distribution.
    W : width of the bandit.
    L : lenght of the bandit.
    hidden : list of which cells are hidden. True if hidden, False if the reward is known.
    follow: to show figures of the process
    epsilon: the expected noise

    Returns
    -------
    Returns the mean function m(x) and the uncertainty function s(x) per tile.

    '''
    hidden2 = hidden.reshape(L, W)
    xlist = [] #list with the coordinates of the data points (list of doubles)
    rewardlist = []
    for i in range(0, len(hidden2)):
        
        for j in range(0, len(hidden2[0])):
            if hidden2[i][j] == False:
                for observation in observed_bandit[i*W + j]:
                    xlist.append([i, j])
                    rewardlist.append(observation)
    cells =[[i,j] for i in range(0, W) for j in range(0, L)]    
    
    kernel = RBF(l_fit, "fixed") + WhiteKernel(epsilon, "fixed")
    gp = GaussianProcessRegressor(kernel=kernel)
    
    gp.fit(xlist, [(reward-50)/100 for reward in rewardlist]) #GP defined by gp will work on the training data
    #the reward is rescaled such that mean = 0 and variance is 1
    #this has found the best fitting function, now we match these with the wanted grid list

    mlist, sigmalist = gp.predict(cells, return_std=True)
    
    #after the fitting, we can redo this rescaling to get it back to the original scale
    mlistplot = np.array([value*100 + 50 for value in mlist])
    sigmalistplot = np.array([value*100 for value in sigmalist])
    
    if follow:            
        plt.rcParams['font.size'] = str(W*4)
        non_hidden = np.array([False]*L*W)
        fig = plt.figure(figsize = (2*W, 8*L))
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412, sharex=ax1)
        ax3 = fig.add_subplot(413, sharex=ax1)
        ax1.set_title('observed rewards')
        #sns.heatmap(mean_bandit.reshape((L, W)), ax=ax1, linewidth=0.5, linecolor='k', cmap = 'Reds', vmax=100 , vmin=0 , annot=mean_bandit.reshape((L, W)), square = True, cbar = False, mask = hidden.reshape((L, W)))
        sns.heatmap(mean_bandit.reshape((L, W)), ax=ax1, linewidth=0.5, linecolor='k', cmap = 'Reds', vmax=100 , vmin=0 , annot=False, square = True, cbar = False, mask = hidden.reshape((L, W)))
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_title(r'm$(\bf{x}$)')
        sns.heatmap(mlistplot.reshape((L, W)), ax=ax2, linewidth=0.5, linecolor='k', cmap = 'Reds' , vmax=100, vmin=0, annot=False, square = True, cbar = False, mask = non_hidden.reshape((L, W)))
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3.set_title(r's$(\bf{x})$')
        sns.heatmap(sigmalistplot.reshape((L, W)), ax=ax3, linewidth=0.5, linecolor='k', cmap = 'Reds', annot=False, square = True, cbar = False, mask = non_hidden.reshape((L, W)))
        plt.setp(ax3.get_xticklabels(), visible=True)
        plt.show()
                 
    
    return mlist, np.sqrt(sigmalist)


def comp_learn(bandit, W, L, trials, max_r, min_r, l_fit, beta, tau, follow):
    """
    agent solves while learning!

    Parameters
    bandit :    The bandit, which is a list with the rewards.
    W :         width of the bandit.
    L :         length of the bandit.
    trials :    size of the search horizon, number of tiles that can be chosen.
    max_r :     maximum reward
    min_r:      minimum reward
    beta :      strenght of directed exploration
                0 for PureExploit.
    tau :       strenght of random exploration, temperature.
    
    follow :    True when we want to follow the agent's steps.

    Returns
    -------
    total_r_list : average accumulated reward list (per trail).

    """
    total_r = 0
    total_r_list = []
    
    hidden = np.array([True]*L*W)  
    observed_bandit = [[value] for value in bandit]
    latest_bandit = bandit.copy()
    mean_bandit = bandit.copy()
        
    '''
    step 1: one random cells is revealed
    '''
    tile_number = random.randint(0, (W*L-1))
    row = math.floor(tile_number/W)
    column = tile_number%W
    
    (observed_bandit, latest_bandit, mean_bandit, hidden, reward) = opened_cell(bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number, W, L, max_r, min_r, follow)

    
    for picks in range(0, trials):
        '''
        first we have to learn from the already available data
        '''
       
        m, s = GP(bandit, observed_bandit, mean_bandit, latest_bandit, W, L, l_fit, hidden, follow)
        
       
        """
        to show the expected reward on the grid
        """
                
        UCB = [m[i] + beta * s[i] for i in range(0, W*L)]
        P = softmax(UCB, W, L, tau)
    
        if follow:
            non_hidden = np.array([False]*L*W)
            
            fig = plt.figure(figsize = (8*W, 8*L))
            plt.rcParams['font.size'] = str(W*4)
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            
            plt.rcParams['font.size'] = str(W*15)
            ax1.set_title('expected rewards')
            sns.heatmap(m.reshape((L, W)), ax=ax1, linewidth=0.5, linecolor='k', cmap = 'Reds' , annot=False, square = True, cbar = False, mask = non_hidden.reshape((L, W)))
            plt.setp(ax1.get_xticklabels(), visible=True)
            ax2.set_title('uncertainty')
            sns.heatmap(s.reshape((L, W)), ax=ax2, linewidth=0.5, linecolor='k', cmap = 'Reds', annot=False, square = True, cbar = False, mask = non_hidden.reshape((L, W)))
            plt.setp(ax2.get_xticklabels(), visible=True)
            ax3.set_title(r'UCB$(\bf{x})$')
            sns.heatmap(np.array(UCB).reshape((L, W)), ax=ax3, linewidth=0.5, linecolor='k', cmap = 'Reds' , annot=False, square = True, cbar = False, mask = non_hidden.reshape((L, W)))
            plt.setp(ax3.get_xticklabels(), visible=True)
            ax4.set_title(r'P$(\bf{x})$')
            sns.heatmap(np.array(P).reshape((L, W)), ax=ax4, linewidth=0.5, linecolor='k', cmap = 'Reds' , annot=False, square = True, cbar = False, mask = non_hidden.reshape((L, W)))
            
            plt.show()
        #the agent will choose from the tiles, for which the probabilites are given by P
        tile_number = random.choices(np.arange(0, W*L), weights=P)[0]
          
        row = math.floor(tile_number/W)
        column = tile_number%W
        
        
        (observed_bandit, latest_bandit, mean_bandit, hidden, reward) = opened_cell(bandit, observed_bandit, latest_bandit, mean_bandit, hidden, tile_number,  W, L, max_r, min_r, follow)
        
        total_r += reward
        total_r_list.append(total_r)
        
        if follow:
            print("The computer chose cell ({}, {}).".format(row, column))
            print("Average accumulated reward = {}\n.".format(total_r/(picks+1)))
            
    
    return total_r_list


def softmax(UCB, W, L, tau):
    """
    translates the UCB (the inflated expectation) to probabilities to sample tiles

    Parameters
    ----------
    UCB : the inflated expectations per tile.
    W : width.
    L : length.
    tau : softmax temperature, the level of undirected (random) exploration.

    Returns
    -------
    the probability to sample a tile, per tile.

    """
    
    UCB = [ucb - max(UCB) for ucb in UCB] #rescaling to avoid overflows
    exp_list = [np.exp(ucb/tau) for ucb in UCB]
    Z = sum(exp_list)
    returnlijst = [value/Z for value in exp_list]

    return returnlijst