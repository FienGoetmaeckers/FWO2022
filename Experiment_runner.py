# -*- coding: utf-8 -*-
"""
main code
runs the spatial search paradigm
"""
import matplotlib.pyplot as plt
import numpy as np
from create_grids import bivariate
from solving_models import me, comp_random, comp_learn

"""
welcome the reseracher and define experiment parameters
"""
print("Welcome researcher!")

#grid parameters
W = 11 #width of grid
L = W  #length of grid
lam = 8 #the smoothness of the grid

trials = 20
twoD = True


#max_r = random.randint(65, 85) #per round, maximum should change so that participants don't know 
max_r = 100 #for computational comparison
min_r = 0 #always zero   


print("Which model do you want to solve the grid?")
model_nr = int(input("(1) I want to solve it (2) Let the computer solve it randomly (3) Let the computer learn.  "))

print("Great! You're ready to go!")


"""
step 1: generate the grid
"""
if model_nr == 1:
    print("Do you want to see the generated reward distribution? (This is not recommended for these settings.)")
    show_grid = (input("y/n: ") == 'y')

else:
    show_grid = True
    
bandit = bivariate(W, L, lam, max_r, min_r, show_grid)

print("\n\nPress any key to continue and start the experiment.")
input()

"""
step 2: welcome the agent and first random choice
"""

hidden = np.array([True]*L*W)

total_r = 0
total_r_list = []

print("\n\n\nWelcome to the experiment!")
print("Your goal is to accumulate the highest average reward.\n")


if model_nr == 1:
    total_r_list = me(bandit, W, L, trials, max_r, min_r) 

elif model_nr == 2:
    total_r_list = comp_random(bandit, W, L, trials, max_r, min_r, follow = True) 
    
elif model_nr == 3:
    '''
    define computational model's parameters
    '''
    
    tau = 0.1 #softmax temperature
    beta = 0.5 #uncertainty-guided exploration parameter
    l_fit = 8 #generalization parameter
    
    total_r_list = comp_learn(bandit, W, L, trials, max_r, min_r, l_fit, beta, tau, follow = True) 

'''
showing the learning curve
'''

plt.rcParams['font.size'] = '8'
  
for i in range(0, trials):
    total_r_list[i] = total_r_list[i]/(i+1)
plt.plot([i for i in range(0,trials)], total_r_list, '.')
plt.xlabel('Trial')
plt.ylabel("Average accumulated reward")
plt.show()