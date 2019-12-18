## Claim: Copied from Ali's repository. Modified its FD PDE solver and plot.

# SCRIPT FOR SOLVING THE BLACK-SCHOLES EQUATION FOR A EUROPEAN CALL OPTION 

#%% import needed packages
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")
from blackscholes.pde.American import Amer1d
from utils.Domain import Domain1d
import utils.Pickle as pickle
import DGM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

tf.random.set_random_seed(1)
np.random.seed(1)
#%% Parameters 

# Option parameters
r = 0.05           # Interest rate
sigma = 0.5        # Volatility
K = 50             # Strike
T = 1              # Terminal time
S0 = 50            # Initial price

# Solution parameters (domain on which to solve PDE)
t_low = 0 + 1e-10    # time lower bound
S_low = 0.0 + 1e-10  # spot price lower bound
S_high = 2*K         # spot price upper bound
fd_S_high = 4*K
domain = Domain1d(0, fd_S_high, T)
pde_solver = Amer1d(domain, sigma, r, 0, K, -1)

# Finite difference parameters
Ngrids = 1600
Nsteps = 400

# neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training parameters
sampling_stages  = 3000   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 2000
nSim_terminal = 100
S_multiplier  = 1.1   # multiplier for oversampling i.e. draw S from [S_low, S_high * S_multiplier]

# Plot options
n_plot = 200  # Points on plot grid for each dimension

# Save options
saveOutput = False
saveName   = 'BlackScholes_AmericanPut'
saveFigure = True
figureName = 'BlackScholes_AmericanPut'

#%% Sampling function - randomly sample time-space pairs 

def sampler(nSim_interior, nSim_terminal):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler #1: domain interior
    t_interior = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    S_interior = np.random.uniform(low=S_low, high=S_high*S_multiplier, size=[nSim_interior, 1])

    # Sampler #2: spatial boundary
        # unknown spatial boundar - will be determined indirectly via loss function
    
    # Sampler #3: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    S_terminal = np.random.uniform(low=S_low, high=S_high*S_multiplier, size = [nSim_terminal, 1])
    
    return t_interior, S_interior, t_terminal, S_terminal

#%% Loss function for Fokker-Planck equation

def loss(model, t_interior, S_interior, t_terminal, S_terminal):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        S_interior: sampled space points in the interior of the function's domain
        t_terminal: sampled time points at terminal point (vector of terminal times)
        S_terminal: sampled space points at terminal time
    '''  

    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    V = model(t_interior, S_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_S = tf.gradients(V, S_interior)[0]
    V_SS = tf.gradients(V_S, S_interior)[0]
    diff_V = V_t + 0.5 * sigma**2 * S_interior**2 * V_SS + r * S_interior * V_S - r*V

    # compute average L2-norm of differential operator (with inequality constraint)
    payoff = tf.nn.relu(K - S_interior)
    value = model(t_interior, S_interior)
    L1 = tf.reduce_mean(tf.square( diff_V * (value - payoff) )) 
    
    temp = tf.nn.relu(diff_V)                   # Minimizes -min(-f,0) = max(f,0)
    L2 = tf.reduce_mean(tf.square(temp))
     
    # Loss term #2: boundary condition
    V_ineq = tf.nn.relu(-(value-payoff))      # Minimizes -min(-f,0) = max(f,0)
    L3 = tf.reduce_mean(tf.square(V_ineq))
    
    # Loss term #3: initial/terminal condition
    target_payoff = tf.nn.relu(K - S_terminal)
    fitted_payoff = model(t_terminal, S_terminal)
    
    L4 = tf.reduce_mean( tf.square(fitted_payoff - target_payoff) )

    return L1, L2, L3, L4

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, 1)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
S_interior_tnsr = tf.placeholder(tf.float32, [None,1])
t_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
S_terminal_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L2_tnsr, L3_tnsr, L4_tnsr = loss(model, t_interior_tnsr, S_interior_tnsr, t_terminal_tnsr, S_terminal_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr + L4_tnsr

# option value function
V = model(t_interior_tnsr, S_interior_tnsr)

# set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%% Train network
# for each sampling stage
loss_vec, L1_vec, L2_vec, L3_vec, L4_vec = [], [], [], [], []
start = timer()
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    t_interior, S_interior, t_terminal, S_terminal = sampler(nSim_interior, nSim_terminal)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss, L1, L2, L3, L4, _ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, L4_tnsr, optimizer],
                                feed_dict = {t_interior_tnsr:t_interior, S_interior_tnsr:S_interior, t_terminal_tnsr:t_terminal, S_terminal_tnsr:S_terminal})
    loss_vec.append(loss); L1_vec.append(L1); L2_vec.append(L2); L3_vec.append(L3); L4_vec.append(L4)
    print(loss, L1, L2, L3, L4, i)
end = timer()
print("consumed time: " + str(end - start))
pickle.dump(loss_vec, figureName+"_lossvec.pickle")
pickle.dump(L1_vec, figureName+"_l1vec.pickle")
pickle.dump(L2_vec, figureName+"_l2vec.pickle")
pickle.dump(L3_vec, figureName+"_l3vec.pickle")
pickle.dump(L4_vec, figureName+"_l4vec.pickle")

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)

#%% Plot results

# LaTeX rendering for text in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# figure options
plt.figure(figsize = (12,10))

# time values at which to examine density
# valueTimes = [t_low, T/3, 2*T/3, T]
valueTimes = [0, 0.333, 0.667, 1]

# vector of t and S values for plotting
S_plot = np.linspace(S_low, S_high, n_plot)

# solution using finite differences
price = pde_solver.solve(Ngrids, Nsteps)
t_idx = [0, Nsteps//3, 2*Nsteps//3, Nsteps]

d = np.linspace(0, fd_S_high, Ngrids+1)
legal_idx = d <= S_high
for i, curr_t in enumerate(valueTimes):
    
    # specify subplot
    plt.subplot(2, 2, i+1)
    
    # simulate process at current t 
    americanOptionValue = price[t_idx[3 - i]]
    
    # compute normalized density at all x values to plot and current t value
    t_plot = curr_t * np.ones_like(S_plot.reshape(-1,1))
    fitted_optionValue = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, S_interior_tnsr:S_plot.reshape(-1,1)})
    
    # plot histogram of simulated process values and overlay estimated density
    plt.plot(d[legal_idx], americanOptionValue[legal_idx], color = 'b', label='Finite differences', linewidth = 3, linestyle=':')
    plt.plot(S_plot, fitted_optionValue[0], color = 'r', label='DGM estimate')    
    
    # subplot options
    plt.ylim(ymin=0.0, ymax=K)
    plt.xlim(xmin=0.0, xmax=S_high)
    plt.xlabel(r"Spot Price", fontsize=15, labelpad=10)
    plt.ylabel(r"Option Price", fontsize=15, labelpad=20)
    plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(linestyle=':')
    
    if i == 0:
        plt.legend(loc='upper right', prop={'size': 16})
    
# adjust space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

if saveFigure:
    plt.savefig(figureName + '.png', dpi=1000)
    
#%% Exercise boundary heatmap plot 
# vector of t and S values for plotting
S_plot = np.linspace(S_low, S_high, n_plot)
t_plot = np.linspace(t_low, T, n_plot)

# compute European put option value for eact (t,S) pair
americanOptionValue_mesh = np.zeros([n_plot, n_plot])

for i in range(n_plot):
    for j in range(n_plot):
        americanOptionValue_mesh[j, i] = pde_solver.evaluate(S_plot[j], T - t_plot[i], 'linear')
    
# compute model-implied American put option value for eact (t,S) pair
t_mesh, S_mesh = np.meshgrid(t_plot, S_plot)

t_plot = np.reshape(t_mesh, [n_plot**2,1])
S_plot = np.reshape(S_mesh, [n_plot**2,1])

optionValue = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, S_interior_tnsr:S_plot})
optionValue_mesh = np.reshape(optionValue, [n_plot, n_plot])

# plot difference between American and European options in heatmap and overlay 
# exercise boundarycomputed using finite differences
plt.figure(figsize = (8,6))

plt.pcolormesh(t_mesh, S_mesh, np.abs(optionValue_mesh - americanOptionValue_mesh), cmap = "rainbow")
# plt.plot(t, exer_bd, color = 'r', linewidth = 3)

# plot options
plt.colorbar()
plt.ylabel("Spot Price", fontsize=15, labelpad=10)
plt.xlabel("Time", fontsize=15, labelpad=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if saveFigure:
    plt.savefig(figureName + '_exerciseBoundary.png', dpi=1000)
