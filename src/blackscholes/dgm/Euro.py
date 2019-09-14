import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
from blackscholes.dgm.DGMNet import DGMNet
import tensorflow as tf

class Euro1d:

    def __init__(self, domain, vol, ir, dividend, strike, cp_type):
        """
        cp_type (call/put type): 1 if call, -1 if put
        """
        self.p = lambda S, t: vol**2*S**2/2
        self.q = lambda S, t: (ir-dividend)*S
        self.ir = ir
        self.strike = strike
        self.cp_type = cp_type
        # domain.bc = lambda S, t: strike*np.exp(-ir*t) if abs(S) < 7/3-4/3-1 else 0
        self.domain = domain

    def run(self, n_layers=3, layer_width=50):
        model = DGMNet(n_layers, layer_width, input_dim=1)
        S_interior_tnsr = tf.placeholder(tf.float32, [None,1])
        t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
        S_boundary_tnsr = tf.placeholder(tf.float32, [None,1])
        t_boundary_tnsr = tf.placeholder(tf.float32, [None,1])
        S_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
        t_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
        L1_tnsr, L2_tnsr, L3_tnsr = self.loss(model, S_interior_tnsr, t_interior_tnsr,\
            S_boundary_tnsr, t_boundary_tnsr, S_terminal_tnsr, t_terminal_tnsr)
        loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr

        global_step = tf.Variable(0, trainable=False)
        boundaries = [5000, 10000, 20000, 30000, 40000, 45000]
        values = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())



    def loss(self, model, S_interior, t_interior, S_boundary, t_boundary, S_terminal, t_terminal):
        ''' Compute total loss for training.
        
        Args:
            model:      DGMNet model object
            t_interior: sampled time points in the interior of the function's domain
            S_interior: sampled space points in the interior of the function's domain
            t_terminal: sampled time points at terminal point (vector of terminal times)
            S_terminal: sampled space points at terminal time
        ''' 
        # Loss term #1: PDE
        # compute function value and derivatives at current sampled points
        fitted_V = model(S_interior, t_interior)
        V_t = tf.gradients(fitted_V, t_interior)[0]
        V_s = tf.gradients(fitted_V, S_interior)[0]
        V_ss = tf.gradients(V_s, S_interior)[0]
        diff_V = V_t + self.p(S_interior, t_interior)*V_ss + self.q(S_interior, t_interior)*V_s + self.ir*fitted_V

        # compute average L2-norm of differential operator
        L1 = tf.reduce_mean(tf.square(diff_V)) 
        
        # Loss term #2: boundary condition
        fitted_bc_val = model(S_boundary, t_boundary)
        valuable_index = tf.where(S_boundary >= self.domain.b) if self.cp_type == 1 else tf.where(S_boundary <= self.domain.a)
        target_bc_val = tf.zeros_like(fitted_bc_val)
        print(fitted_bc_val, valuable_index, target_bc_val[valuable_index], S_boundary[valuable_index])
        #target_bc_val[valuable_index] = 
        # L2 = tf.reduce_mean(tf.square())
        L2 = tf.constant(0)
        
        # Loss term #3: initial/terminal condition
        target_payoff = tf.nn.relu(self.cp_type*(S_terminal - self.strike))
        fitted_payoff = model(S_terminal, t_terminal)
        
        L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

        return L1, L2, L3