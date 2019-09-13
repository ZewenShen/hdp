from DGMNet import DGMNet

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

    def loss(model, S_interior, t_interior, S_boundary, t_boundary, S_terminal, t_terminal):
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
        V_t = tf.gradients(V, t_interior)[0]
        V_s = tf.gradients(V, S_interior)[0]
        V_ss = tf.gradients(V_s, S_interior)[0]
        diff_V = V_t + self.p(S_interior, t_interior)*V_ss + self.q(S_interior, t_interior)*V_s + self.ir*fitted_V

        # compute average L2-norm of differential operator
        L1 = tf.reduce_mean(tf.square(diff_V)) 
        
        # Loss term #2: boundary condition
        # target_bc_val = tf.cond(S_boundary <= self.domain.a, tf.zeros_like())
        # L2 = tf.reduce_mean(tf.square())
        
        # Loss term #3: initial/terminal condition
        target_payoff = tf.nn.relu(self.cp_type*(S_terminal - self.strike))
        fitted_payoff = model(S_terminal, t_terminal)
        
        L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

        return L1, L3