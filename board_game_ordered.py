#!/usr/bin/env python
# coding: utf-8

# # Snakes and ladders 

# written by: Sander De Smedt

# ## General functions

# In[1]:


import random as rand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# In[2]:


snakes = {
    16: 6,
    47: 26,
    49: 11,
    56: 53,
    64: 60,
    62: 19,
    87: 24,
    93: 73,
    95: 75,
    98: 78
}

snakes_position = {
    6:1,
    26:2,
    11:3,
    53:4,
    60:5,
    19:6,
    24:7,
    73:8,
    75:9,
    78:10
}

ladders = {
    1:38,
    4:14,
    9:31,
    28:84,
    21:42,
    36:44,
    51:67,
    71:91,
    80:100
}

ladders_position = {
    38:1,
    14:2,
    31:3,
    84:4,
    42:5,
    44:6,
    67:7,
    91:8,
    100:9
}

trans = {
    16: 6,
    47: 26,
    49: 11,
    56: 53,
    64: 60,
    62: 19,
    87: 24,
    93: 73,
    95: 75,
    98: 78,
    1:38,
    4:14,
    9:31,
    28:84,
    21:42,
    36:44,
    51:67,
    71:91,
    80:100
}
max_value = 100
max_value_dice = 6


# In[3]:


snakes.values()


# In[4]:


def dice_value():
    value = rand.randint(1,max_value_dice)
    return value


# In[5]:


def check_win(position):
    win = False
    if position >= max_value:
        win = True
    return win


# In[6]:


def snake_ladder(position, dice_value):
    old_position = position
    current_position = position+dice_value
    if current_position > max_value:
        return old_position
    elif current_position in snakes:
        final_position = snakes.get(current_position)
    elif current_position in ladders:
        final_position = ladders.get(current_position)
    else:
        final_position = current_position
    return final_position


# In[ ]:





# ## Roll simulation

# In[7]:


def roll_simulation():
    trajectory = [0]
    position = 0
    while position <= max_value:
        steps = dice_value()
        position = snake_ladder(position, steps)
        trajectory.append(position)
        if check_win(position):
            break
    return trajectory


# In[ ]:





# In[8]:


def homogenize_length(data, fill):
    max_length = max(map(len, data))
    for i in range(len(data)):
        data[i] = data[i] + [fill]*(max_length-len(data[i]))


# In[9]:


def count_entries(trajectories):
    """Count the entries per timestep in a set of board game trajectories."""
    trajectories = np.asarray(trajectories)
    minlength = trajectories.max()+1
    count = [np.bincount(trajectories[:,i], minlength=minlength) for i in range(trajectories.shape[1])]
    return np.array(count)







# In[10]:


def used_snakes(data):
    used_snakes = np.zeros(11)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in snakes_position:
                used_snakes[snakes_position[data[i][j]]] += 1
    dic_snakes = {}
    for i in range(len(used_snakes)):
        dic_snakes[i] = used_snakes[i]
    return dic_snakes


# In[11]:


def used_ladders(data):
    used_ladders = np.zeros(11)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in ladders_position:
                used_ladders[ladders_position[data[i][j]]] += 1
                if data[i][j-1] > 80 and data[i][j] == 100:
                    used_ladders[9] -= 1
    dic_ladders = {}
    for i in range(len(used_ladders)):
        dic_ladders[i] = used_ladders[i]
    return dic_ladders


# In[12]:


roll_sim = [roll_simulation() for i in range(100000)]


# In[13]:


homogenize_length(roll_sim, max_value+1)
roll_count = count_entries(roll_sim)/len(roll_sim)


# In[14]:


'''used_snakes = used_snakes(roll_sim)
dic_snakes = {}
for i in range(len(used_snakes)):
    dic_snakes[i] = used_snakes[i]
dic_snakes
'''


# In[15]:


'''used_ladders = used_ladders(roll_sim)
dic_ladders = {}
for i in range(len(used_ladders)):
    dic_ladders[i] = used_ladders[i]
dic_ladders
'''


# In[16]:


dic_snakes = used_snakes(roll_sim)


# In[17]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.title("Number of times snakes were used after 100.000 simulations")
plt.xlim(0.5,10.5)
plt.ylabel("number of times used")
plt.xlabel("snakes")
plt.bar(dic_snakes.keys(), dic_snakes.values() , color = 'r')
plt.xticks(np.arange(11, step=1))
fig.savefig('used_snakes.png', bbox_inches = 'tight')


# In[18]:


dic_ladders = used_ladders(roll_sim)


# In[19]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 4))
plt.title("Number of times ladders were used after 100.000 simulations")
plt.xlim(0.5,9.5)
plt.ylabel("number of times used")
plt.xlabel("ladders")
plt.bar(dic_ladders.keys(), dic_ladders.values() , color = 'g')
fig.savefig('used_ladders.png', bbox_inches = 'tight')


# ### Graphs

# In[20]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.title("Cumulative probability of being on the last square")
plt.plot(roll_count[:,-1], color = 'r')
plt.xlabel('number of steps')
plt.ylabel('Probability')
fig.savefig('roll_simulation_cum.png', bbox_inches = 'tight')


# In[ ]:





# ## Stochastic simulation

# def transition_matrix():
#     transition = np.zeros((max_value+1,max_value+1))
#     for x in range(max_value + 2):
#         for k in range(1,7):
#             if x+k < max_value:
#                 if x+k in snakes:
#                     transition[x,x+k] = 0
#                     transition[x,snakes.get(x+k)] += 1/6
#                     #print("x:",x)
#                     #print("x+k",x+k)
#                     #print("value:",snakes.get(x+k))
#                 if x+k in ladders:
#                     transition[x,x+k]=0
#                     transition[x,ladders.get(x+k)] += 1/6
#                 else:
#                     transition[x,x+k] += 1/6
#             else:
#                 break
#     return transition

# In[ ]:





# In[21]:


def transition_matrix():
    T = np.zeros((max_value+1, max_value+1))
    for i in range(1,max_value+1):
        T[i-1,i:i+6] = 1/6

    for element in trans:
        iw = np.where(T[:,element] > 0)
        T[:,element] = 0
        T[iw,trans.get(element)] += 1/6
    
    #extra rule to make things easier
    T[95:100,100] += np.linspace(1/6, 5/6, 5)
    for snake in snakes:
        T[snake,100] = 0

    for i in range(11):
        T[i,:]/= T[i,:].sum()
    return T


# In[22]:


def stochastic_matrix(steps):
    x0 = np.array([1, ] + max_value*[0,])
    x = [x0]
    y = x0
    matrix = transition_matrix()
    for i in range(steps):
        y = matrix @ y
        x.append(y)
    return np.array(x)


# In[23]:


def matrix_simulation():
    trajectory = [0]
    x = 0
    indices = np.arange(max_value+1)
    matrix = transition_matrix()
    while True:
        possibilities = matrix[x,:]
        x = np.random.choice(indices, p=possibilities)
        trajectory.append(x)
        #print(trajectory)
        if x==max_value:
            break
        
    return trajectory


# In[24]:


def winning_n_moves():
    v = np.zeros(max_value+1)
    v[0] = 1
    n, P = 0, []
    cumulative_prob = 0
    T = transition_matrix()
    # Update the state vector v until the cumulative probability of winning
    # is "effectively" 1
    while cumulative_prob < 0.99999:
        n += 1
        v = v.dot(T)
        P.append(v[100])
        cumulative_prob += P[-1]
    mode = np.argmax(P)+1
    #print('modal number of moves:', mode)
    return [n,P]


# ## graph

# In[25]:


roll_mat = [matrix_simulation() for i in range(100000)]


# In[ ]:





# In[26]:


homogenize_length(roll_mat, max_value+1)
roll_count_mat = count_entries(roll_mat)/len(roll_mat)


# In[27]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.title("Cumulative probability of being on the last square")
plt.plot(roll_count_mat[:,-1], color = 'black', label = 'matrix simulation')
plt.xlabel('number of steps')
ax.legend()
plt.ylabel('Probability')
fig.savefig('mat_simulation_cum.png', bbox_inches = 'tight')


# In[28]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.title("comparison cumulative probability ")
plt.plot(roll_count_mat[:,-1], color = 'black', label = 'matrix simulation')
plt.plot(roll_count[:,-1], color = 'red', label = 'roll simulation')
plt.xlabel('number of steps')
ax.legend()
plt.ylabel('Probability')
fig.savefig('mat_and_roll_cum.png', bbox_inches = 'tight')


# In[29]:


n = winning_n_moves()[0]
P = winning_n_moves()[1]
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.plot(np.linspace(1,n,n), P, color = 'black')
ax.set_xlabel('Number of moves')
ax.set_ylabel('Probability of winning')
#ax.axvline(x= 20, color ='r')
#ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Chance of winning in n moves')
fig.savefig('chance_winning_n.png', bbox_inches = 'tight')


# In[30]:


dic_snakes_mat = used_snakes(roll_mat)


# In[31]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.title("Number of times snakes were used after 100.000 simulations")
plt.xlim(0.5,10.5)
plt.ylabel("number of times used")
plt.xlabel("snakes")
plt.bar(dic_snakes_mat.keys(), dic_snakes_mat.values() , color = 'r')
plt.xticks(np.arange(11, step=1))
fig.savefig('used_snakes_mat.png', bbox_inches = 'tight')


# In[32]:


dic_ladders_mat = used_ladders(roll_mat)


# In[33]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 3))
plt.title("Number of times ladders were used after 100.000 simulations")
plt.xlim(0.5,9.5)
plt.ylabel("number of times used")
plt.xlabel("ladders")
plt.bar(dic_ladders_mat.keys(), dic_ladders_mat.values() , color = 'g')
fig.savefig('used_ladders_mat.png', bbox_inches = 'tight')


# In[36]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 4))
plt.title("Number of times ladders were used after 100.000 simulations")
plt.xlim(0.5,9.5)
plt.ylabel("number of times used")
plt.xlabel("ladders")
plt.bar(dic_ladders.keys(), dic_ladders.values(),label='roll simulation' ,alpha = 0.5, color = 'r')
plt.bar(dic_ladders_mat.keys(), dic_ladders_mat.values(), label = 'matrix simulation' ,alpha = 1, color='g')
plt.legend()
fig.savefig('ladders_comp.png', bbox_inches = 'tight')


# In[35]:


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(7, 4))
plt.title("Number of times snakes were used after 100.000 simulations")
plt.xlim(0.5,10.5)
plt.ylabel("number of times used")
plt.xlabel("ladders")
plt.bar(dic_snakes.keys(), dic_snakes.values(),label='roll simulation' ,alpha = 0.5, color = 'r')
plt.bar(dic_snakes_mat.keys(), dic_snakes_mat.values(), label = 'matrix simulation' ,alpha = 1, color = 'g')
plt.xticks(np.arange(11, step=1))
plt.legend()
fig.savefig('snakes_comp.png', bbox_inches = 'tight')


# In[ ]:




