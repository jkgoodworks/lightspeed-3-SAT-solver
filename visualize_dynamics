#the same (?), but makes plots (see README)
import os
import numpy as np
import numba
from numba import cuda, njit, float32
numba.config.NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS = 0  # Add this
import math, time
import matplotlib.pyplot as plt  # Add matplotlib for plotting

# silence NumbaPerformanceWarning (don't deploy)
#import warnings
#from numba.core.errors import NumbaPerformanceWarning
#warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# --------------------------
# GPU Configuration
# --------------------------
os.environ['NUMBA_ENABLE_CUDASIM'] = '0'
os.environ['NUMBA_CUDA_DRIVER'] = 'cuda'

# RTX 3060 Specifics (3584 CUDA cores, 28 SMs)
MAX_THREADS_PER_SM = 1536
THREADS_PER_BLOCK=128 #even better performance
#THREADS_PER_BLOCK = 256 #better performance
#THREADS_PER_BLOCK = 1024

try:
    from numba.cuda.cudadrv import devices
    device = devices.get_context().device
    cc = device.compute_capability
    print(f"[GPU Active] {device.name}, Compute Capability: {cc[0]}.{cc[1]}")
    USE_CUDA = True
except Exception as e:
    print(f"[CPU Fallback] {e}")
    USE_CUDA = False

# --------------------------
# Optimized CUDA Kernels
# --------------------------
MAX_THREADS_PER_SM = 1536
THREADS_PER_BLOCK=128
@cuda.jit(fastmath=True)
def compute_Cm_gpu(clauses, v, C, min_terms, min_indices):
    """
    Mathematical formula:
    C_m = 0.5 * min_{i∈{0,1,2}} (1 - q_i * v_{var_i})
    where:
    - q_i is the sign of literal i in clause m
    - v_{var_i} is the current value of variable in position i
    - min_idx tracks which literal achieves the minimum
    """
    m = cuda.grid(1)
    if m < clauses.shape[0]:
        var0, q0 = clauses[m, 0, 0], clauses[m, 0, 1]
        var1, q1 = clauses[m, 1, 0], clauses[m, 1, 1]
        var2, q2 = clauses[m, 2, 0], clauses[m, 2, 1]
        
        # Calculate term_i = (1 - q_i * v_{var_i}) for each literal
        term0 = 1.0 - q0 * v[var0]
        term1 = 1.0 - q1 * v[var1]
        term2 = 1.0 - q2 * v[var2]
        
        # Find minimum term and its index
        if term0 < term1:
            min_val, min_idx = (term0, 0) if term0 < term2 else (term2, 2)
        else:
            min_val, min_idx = (term1, 1) if term1 < term2 else (term2, 2)
            
        min_terms[m] = min_val
        C[m] = 0.5 * min_val  # C_m = 0.5 * min_val
        min_indices[m] = min_idx

@cuda.jit(fastmath=True)
def compute_derivatives_kernel(clauses, v, x_lm, x_sm, 
                              min_terms, min_indices, dv, 
                              beta, epsilon, gamma, alpha, delta,
                              dx_sm, dx_lm, zeta):
    """
    Mathematical formulas:
    1. dx_sm/dt = β(x_sm + ε)(0.5*min_val - γ)
    2. dx_lm/dt = α(0.5*min_val - δ)
    3. dv_i/dt = ∑_m [x_lm^m * x_sm^m * G_i^m + (1 + ζ*x_lm^m)(1 - x_sm^m)*R_i^m]
    where:
    - G_i^m = 0.5*q_i^m*min_val^m (gradient term for all variables)
    - R_i^m = 0.5*(q_i^m - v_i) if i is minimizer of clause m, else 0 (residual term)
    """
    m = cuda.grid(1)
    if m < clauses.shape[0]:
        # Auxiliary variable derivatives 
        dx_sm[m] = beta * (x_sm[m] + epsilon) * math.sin(0.5 * min_terms[m] - gamma)**3 #added **3 wtffffffffffffffff it improves...#to sin^3 fainetai na bohthaei ligo kltr
        dx_lm[m] = alpha * (0.5 * min_terms[m] - delta) 
        
        min_idx = min_indices[m]
        min_val = min_terms[m]
        x_lm_m = x_lm[m]
        x_sm_m = x_sm[m]
        
        # Compute contribution to variable derivatives
        for i in 0, 1, 2:
            var = clauses[m, i, 0]
            q = clauses[m, i, 1]
            
            # G term: affects all variables in the clause
            G = 0.5 * q * min_val
            
            # R term: affects only the variable corresponding to the minimum term
            R = 0.5 * (q - v[var]) if i == min_idx else 0.0  #delete **3
            
            # Combined influence term with auxiliary variable modulation
            term = x_lm_m * x_sm_m * G + (1.0 + zeta * x_lm_m) * (1.0 - x_sm_m) * R
            cuda.atomic.add(dv, var, term)

@cuda.jit(fastmath=True)
def update_v_kernel_momentum(v, v_prev, dv, dt, mu, v_new): #####change###########################################################################
    """
    Mathematical formula:
    v_i(t+dt) = v_i(t) + dv_i*dt + μ[v_i(t) - v_i(t-dt)]
    Constrained to: -1 ≤ v_i ≤ 1
    where:
    - μ is the momentum coefficient
    - dv_i is the derivative of v_i
    """
    i = cuda.grid(1)
    if i < v.size:
        # Momentum term: μ(v(t) - v(t-dt))
        momentum_term = mu * (v[i] - v_prev[i])
        
        # Update rule with momentum
        #new_v = v[i] + (dv[i] +0.1*math.tanh((v[i]**3)*(1-v[i]**2)) ) * dt + momentum_term ###################$$$( dv[i] +0.38*v[i]*(1-v[i]**2) )
        new_v = v[i] + (dv[i] ) * dt + momentum_term
        # Constrain to [-1, 1]
        new_v = max(-1.0, min(1.0, new_v))
        v_new[i] = new_v

@cuda.jit(fastmath=True)
def update_x_kernel(x_sm, x_lm, dx_sm, dx_lm, dt):
    """
    Mathematical formulas:
    1. x_sm(t+dt) = clip(x_sm(t) + dx_sm*dt, 0, 1)
    2. x_lm(t+dt) = clip(x_lm(t) + dx_lm*dt, 1, 1e4*M)
    
    Where M is the number of clauses and clip constrains values
    to the specified range.
    """
    i = cuda.grid(1)
    if i < x_sm.size:
        # Update x_sm with constraint 0 ≤ x_sm ≤ 1
        new_x_sm = x_sm[i] + dx_sm[i] * dt
        x_sm[i] = max(0.0, min(1.0, new_x_sm))
        
        # Update x_lm with constraint 1 ≤ x_lm ≤ 1e4*M
        new_x_lm = x_lm[i] + dx_lm[i] * dt
        x_lm[i] = max(1.0, min(1e4 * x_lm.size, new_x_lm))

@cuda.jit(fastmath=True)
def max_reduce_kernel(arr, result):
    """
    Mathematical operation:
    result = max(|arr_i|) for all i
    
    Implemented using parallel reduction pattern
    with shared memory for performance.
    """
    shared = cuda.shared.array(1024, dtype=float32)
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * cuda.blockDim.x + tid
    
    local_max = -math.inf
    if idx < arr.size:
        local_max = abs(arr[idx])
    
    shared[tid] = local_max
    cuda.syncthreads()
    
    i = cuda.blockDim.x // 2
    while i > 0:
        if tid < i:
            shared[tid] = max(shared[tid], shared[tid + i])
        cuda.syncthreads()
        i //= 2
    
    if tid == 0:
        result[bid] = shared[0]

@cuda.jit(fastmath=True)
def copy_kernel(src, dest):
    i = cuda.grid(1)
    if i < src.size:
        dest[i] = src[i]

# Add these new kernels for performance
@cuda.jit(fastmath=True)
def compute_stats_kernel(C, stats_out):
    """Compute min, max, mean, non-zero count efficiently on GPU"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * cuda.blockDim.x + tid
    
    # Use shared memory for reduction
    shared_sum = cuda.shared.array(shape=(256), dtype=float32)
    shared_max = cuda.shared.array(shape=(256), dtype=float32)
    shared_count = cuda.shared.array(shape=(256), dtype=float32)
    
    # Initialize
    shared_sum[tid] = 0.0
    shared_max[tid] = 0.0
    shared_count[tid] = 0.0
    
    # Load and process data
    if idx < C.size:
        val = C[idx]
        shared_sum[tid] = val
        shared_max[tid] = val
        shared_count[tid] = 1.0 if val > 0.05 else 0.0
    
    cuda.syncthreads()
    
    # Reduce
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s])
            shared_count[tid] += shared_count[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write output: [sum, max, count]
    if tid == 0:
        stats_out[bid*3] = shared_sum[0]      # Sum
        stats_out[bid*3+1] = shared_max[0]    # Max
        stats_out[bid*3+2] = shared_count[0]  # Count of non-zero

# --------------------------
# DMM Solver (Class)
# --------------------------
class DMM3SATOptimized:
    def __init__(self, clauses, num_vars, 
                 alpha=5.0, beta=20.0, gamma=0.25, 
                 delta=0.05, epsilon=1e-3, zeta=0.1, mu=0.9):
        
        self.M = clauses.shape[0]
        self.N = num_vars
        self.alpha = np.float32(alpha)
        self.beta = np.float32(beta)
        self.gamma = np.float32(gamma)
        self.delta = np.float32(delta)
        self.epsilon = np.float32(epsilon)
        self.zeta = np.float32(zeta)
        self.mu = np.float32(mu)

        # Initialize base parameter values
        self.alpha0 = np.float32(alpha)
        self.beta0 = np.float32(beta)
        self.gamma0 = np.float32(gamma)
        self.delta0 = np.float32(delta)
        self.zeta0 = np.float32(zeta)
        
        # Step counter
        self.step_count = 0

        # New parameters for faster convergence
        self.adaptive_dt_factor = .5#!!0.25  # More aggressive time stepping (was 0.1)
        self.check_interval = 3 # Only check solution periodically
        
        # Stats buffer for efficient GPU calculations
        if USE_CUDA:
            self.stats_buffer = None  # Will initialize in _init_gpu

        # Add tracking for visualization
        self.tracked_vars = None
        self.tracked_clauses = None
        self.var_history = None
        self.clause_history = None

        if USE_CUDA:
            self._init_gpu(clauses)
        else:
            self._init_cpu(clauses)
    
    def _init_gpu(self, clauses):
        self.clauses_gpu = cuda.to_device(np.ascontiguousarray(clauses.astype(np.int32)))
        self.v = cuda.to_device(np.random.uniform(-1, 1, self.N).astype(np.float32))
        self.v_prev = cuda.device_array_like(self.v)
        self.v_new = cuda.device_array_like(self.v)
        copy_kernel[(self.N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK](self.v, self.v_prev)
        self.x_sm = cuda.to_device(np.full(self.M, 0.5, dtype=np.float32))
        self.x_lm = cuda.to_device(np.ones(self.M, dtype=np.float32))
        
        self.dv = cuda.device_array(self.N, dtype=np.float32)
        self.dx_sm = cuda.device_array(self.M, dtype=np.float32)
        self.dx_lm = cuda.device_array(self.M, dtype=np.float32)
        self.C = cuda.device_array(self.M, dtype=np.float32)
        self.min_terms = cuda.device_array(self.M, dtype=np.float32)
        self.min_indices = cuda.device_array(self.M, dtype=np.int32)
        
        self._init_reduction_buffers()
        self.threads = THREADS_PER_BLOCK
        self.blocks = (self.M + self.threads - 1) // self.threads

        # Initialize stats buffer for GPU-side statistics
        max_blocks = min(1024, (self.M + 255) // 256)
        self.stats_buffer = cuda.device_array(max_blocks * 3, dtype=np.float32)

    def _init_reduction_buffers(self):
        self.max_temp = cuda.device_array(1024, dtype=np.float32)
        self.max_result = cuda.device_array(1, dtype=np.float32)
        
    def _gpu_max(self, arr):
        blocks_needed = (arr.size + self.threads - 1) // self.threads
        blocks = min(blocks_needed, 1024)
        max_reduce_kernel[blocks, self.threads](arr, self.max_temp)
        max_reduce_kernel[1, min(blocks, self.threads)](self.max_temp[:blocks], self.max_result)
        return self.max_result.copy_to_host()[0]

    def step(self, dt):
        if USE_CUDA:
            return self._gpu_step(dt)
        return self._cpu_step(dt)

    def setup_tracking(self, num_to_track=2):
        """Set up tracking for visualization of random variables and their clauses"""
        # Select random variables to track
        self.tracked_vars = np.random.choice(self.N, num_to_track, replace=False)
        print(f"Tracking variables: {self.tracked_vars}")
        
        # Find clauses containing these variables
        self.tracked_clauses = []
        clauses_host = self.clauses_gpu.copy_to_host() if USE_CUDA else self.clauses
        
        # Store related variables - other variables that appear in clauses with our tracked variables
        self.related_vars = [set() for _ in range(num_to_track)]
        
        # For each tracked variable, find clauses that contain it
        for i, var_idx in enumerate(self.tracked_vars):
            # Search for clauses containing this variable
            clauses_with_var = []
            for m in range(self.M):
                if var_idx in clauses_host[m, :, 0]:
                    clauses_with_var.append(m)
                    
                    # Get all variables in this clause and add to related variables
                    for var in clauses_host[m, :, 0]:
                        if var != var_idx:  # Don't include the tracked variable itself
                            self.related_vars[i].add(var)
                    
                    # Limit to 5 clauses per variable to avoid too much data
                    if len(clauses_with_var) >= 5:
                        break
            self.tracked_clauses.append(clauses_with_var)
        
        # Convert related variables to lists for easier indexing
        self.related_vars = [list(vars_set) for vars_set in self.related_vars]
        for i, vars_list in enumerate(self.related_vars):
            print(f"Variable {self.tracked_vars[i]} appears with variables: {vars_list}")
        
        # Initialize history arrays
        max_steps = 1000  # Pre-allocate for efficiency
        self.var_history = np.zeros((num_to_track, max_steps))
        
        # Also track related variables
        self.related_var_history = [np.zeros((len(vars_list), max_steps)) for vars_list in self.related_vars]
        
        # Flatten clause list for easier tracking
        all_tracked_clauses = []
        for clauses in self.tracked_clauses:
            all_tracked_clauses.extend(clauses)
        self.all_tracked_clauses = list(set(all_tracked_clauses))  # Remove duplicates
        
        self.clause_history = np.zeros((len(self.all_tracked_clauses), max_steps))
        self.step_count_hist = 0
    
    def track_step(self):
        """Record current values of tracked variables and clauses"""
        if self.tracked_vars is None or self.step_count_hist >= self.var_history.shape[1]:
            return
        
        # Extend arrays if needed
        if self.step_count_hist >= self.var_history.shape[1]:
            self.var_history = np.hstack([self.var_history, np.zeros_like(self.var_history)])
            self.clause_history = np.hstack([self.clause_history, np.zeros_like(self.clause_history)])
            
            # Also extend related variable histories
            for i in range(len(self.related_var_history)):
                self.related_var_history[i] = np.hstack([self.related_var_history[i], 
                                                       np.zeros_like(self.related_var_history[i])])
        
        # Get current values
        if USE_CUDA:
            v_current = self.v.copy_to_host()
            c_current = self.C.copy_to_host()
        else:
            v_current = self.v
            c_current = self.C
        
        # Record variable values
        for i, var_idx in enumerate(self.tracked_vars):
            self.var_history[i, self.step_count_hist] = v_current[var_idx]
        
        # Record related variable values
        for i, var_list in enumerate(self.related_vars):
            for j, var_idx in enumerate(var_list):
                self.related_var_history[i][j, self.step_count_hist] = v_current[var_idx]
        
        # Record clause values
        for i, clause_idx in enumerate(self.all_tracked_clauses):
            self.clause_history[i, self.step_count_hist] = c_current[clause_idx]
        
        self.step_count_hist += 1

    def plot_dynamics(self):
        """Plot the tracked variables and clauses over time"""
        if self.tracked_vars is None or self.step_count_hist == 0:
            print("No data to plot. Call setup_tracking() before solving.")
            return
        
        # Trim history to actual steps
        var_hist = self.var_history[:, :self.step_count_hist]
        clause_hist = self.clause_history[:, :self.step_count_hist]
        related_var_hist = [hist[:, :self.step_count_hist] for hist in self.related_var_history]
        steps = np.arange(self.step_count_hist)
        
        # Calculate how many subplots we need
        n_tracked = len(self.tracked_vars)
        
        # Create figure with subplots - more subplots to include related variables
        fig = plt.figure(figsize=(16, 4 * (n_tracked + 2)))
        
        # Main grid: tracked vars, clauses, convergence metrics, then related vars for each tracked var
        gs = fig.add_gridspec(n_tracked + 2, 2)
        
        # Plot tracked variable trajectories
        ax_vars = fig.add_subplot(gs[0, 0])
        for i, var_idx in enumerate(self.tracked_vars):
            ax_vars.plot(steps, var_hist[i], label=f'Variable {i+1}', linewidth=2)
        ax_vars.set_title('Primary Tracked Variables')
        ax_vars.set_xlabel('Steps')
        ax_vars.set_ylabel('v[k]')
        ax_vars.legend()
        ax_vars.grid(True)
        
        # Plot convergence metrics
        ax_metrics = fig.add_subplot(gs[1, 0])
        ax_metrics.plot(steps, np.mean(clause_hist, axis=0), 'g-', label='Mean C')
        ax_metrics.plot(steps, np.max(clause_hist, axis=0), 'r-', label='Max C')
        ax_metrics.set_title('Convergence Metrics')
        ax_metrics.set_xlabel('Steps')
        ax_metrics.set_ylabel('C Value')
        ax_metrics.legend()
        ax_metrics.grid(True)
        
        # Plot clause values for each tracked variable
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        
        for var_i, var_idx in enumerate(self.tracked_vars):
            # Plot clauses for this variable
            ax_clauses = fig.add_subplot(gs[var_i, 1])
            ax_clauses.set_title(f'Clauses Containing Variable {var_i+1}')
            ax_clauses.set_xlabel('Steps')
            ax_clauses.set_ylabel('C[m]')
            
            # Find indices in all_tracked_clauses that correspond to clauses with this variable
            for i, clause_idx in enumerate(self.tracked_clauses[var_i]):
                # Get the position in all_tracked_clauses
                pos = self.all_tracked_clauses.index(clause_idx)
                ax_clauses.plot(steps, clause_hist[pos], 
                        label=f'Clause {i+1}', 
                        color=colors[i % len(colors)])
            ax_clauses.legend()
            ax_clauses.grid(True)
            
            # Plot related variables for this tracked variable
            ax_related = fig.add_subplot(gs[var_i + 2, var_i % 2])
            ax_related.set_title(f'Variables Related to Variable {var_i+1}')
            ax_related.set_xlabel('Steps')
            ax_related.set_ylabel('v[k]')
            
            # Plot the tracked variable itself as reference
            ax_related.plot(steps, var_hist[var_i], 'k-', 
                           linewidth=2, label=f'Var {var_i+1} (tracked)')
            
            # Plot all the related variables
            for j, rel_var in enumerate(self.related_vars[var_i]):
                ax_related.plot(steps, related_var_hist[var_i][j], 
                               label=f'Var R{j+1}',
                               alpha=0.7, linewidth=1,
                               color=colors[(j+1) % len(colors)])
            
            # Only show legend if not too many variables
            if len(self.related_vars[var_i]) < 15:
                ax_related.legend()
            else:
                ax_related.legend([f'Var {var_i+1} (tracked)', f'+ {len(self.related_vars[var_i])} related vars'])
            
            ax_related.grid(True)
        
        plt.tight_layout()
        plt.savefig('dmm_sat_dynamics.png', dpi=150)
        
        # Create a second, more detailed figure showing how each clause's variables evolve together
        self._plot_clause_variable_relationships()
        
        plt.show()
        
    def _plot_clause_variable_relationships(self):
        """Create an additional plot showing how variables within each clause evolve together"""
        if self.tracked_vars is None or self.step_count_hist == 0:
            return
            
        clauses_host = self.clauses_gpu.copy_to_host() if USE_CUDA else self.clauses
        v_history = self.v.copy_to_host() if USE_CUDA else self.v
        steps = np.arange(self.step_count_hist)
        
        # Create a figure with one subplot per tracked clause
        n_clauses = sum(len(clauses) for clauses in self.tracked_clauses)
        fig = plt.figure(figsize=(15, 4 * ((n_clauses + 1) // 2)))
        
        # Use GridSpec for flexible subplot layout
        gs = fig.add_gridspec((n_clauses + 1) // 2, 2)
        
        plot_idx = 0
        colors = ['b', 'r', 'g', 'c', 'm']
        var_map = {} # Map real variable indices to simplified numbers
        var_counter = 1
        
        # For each tracked variable
        for tracked_i, tracked_var in enumerate(self.tracked_vars):
            # Map tracked variables first
            var_map[tracked_var] = tracked_i + 1
            
            # Get its tracked clauses
            for clause_i, clause_idx in enumerate(self.tracked_clauses[tracked_i]):
                # Create subplot
                ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
                
                # Get variables in this clause
                clause_vars = clauses_host[clause_idx, :, 0]
                clause_signs = clauses_host[clause_idx, :, 1]
                
                # Plot each variable in the clause
                for i, (var, sign) in enumerate(zip(clause_vars, clause_signs)):
                    # Find this variable in our history
                    var_data = None
                    linestyle = '-'
                    linewidth = 1.5
                    
                    # Check if it's a tracked variable
                    if var in self.tracked_vars:
                        idx = np.where(self.tracked_vars == var)[0][0]
                        var_data = self.var_history[idx, :self.step_count_hist]
                        linewidth = 2.5
                        var_label = var_map[var]  # Use the mapped number
                    else:
                        # Check related variables - map them if not already mapped
                        for rel_i, rel_vars in enumerate(self.related_vars):
                            if var in rel_vars:
                                if var not in var_map:
                                    var_map[var] = f"R{var_counter}"
                                    var_counter += 1
                                rel_idx = rel_vars.index(var)
                                var_data = self.related_var_history[rel_i][rel_idx, :self.step_count_hist]
                                var_label = var_map[var]
                                break
                    
                    # Plot if we found data
                    if var_data is not None:
                        # Use dashed lines for negative literals
                        linestyle = '-' if sign > 0 else '--'
                        ax.plot(steps, var_data, 
                               color=colors[i % len(colors)],
                               linestyle=linestyle, 
                               linewidth=linewidth,
                               label=f'Var {var_label} ({"+" if sign > 0 else "-"})')
                
                # Plot C value for this clause
                clause_pos = self.all_tracked_clauses.index(clause_idx)
                c_data = self.clause_history[clause_pos, :self.step_count_hist]
                ax.plot(steps, c_data, 'k:', linewidth=1, label='C value')
                
                # Add horizontal lines at -1, 0, 1 for reference
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=-1, color='gray', linestyle='-', alpha=0.3)
                
                ax.set_title(f'Clause {clause_i+1} (with Var {tracked_i+1})')
                ax.set_xlabel('Steps')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('dmm_sat_clause_variables.png', dpi=150)

    def _cpu_step(self, dt):
        C = np.zeros(self.M, dtype=np.float32)
        min_terms = np.zeros(self.M, dtype=np.float32)
        for m in range(self.M):
            terms = [1 - q * self.v[var] for var, q in self.clauses[m]]
            min_val = min(terms)
            C[m] = 0.5 * min_val
            min_terms[m] = min_val
        
        dx_sm = self.beta * (self.x_sm + self.epsilon) * (C - self.gamma)
        dx_lm = self.alpha * (C - self.delta)
        dv = np.zeros(self.N, dtype=np.float32)
        
        for m in range(self.M):
            min_idx = np.argmin([1 - q * self.v[var] for var, q in self.clauses[m]])
            for i in range(3):
                var, q = self.clauses[m, i]
                G = 0.5 * q * min_terms[m]
                R = 0.5 * (q - self.v[var]) if i == min_idx else 0.0
                term = self.x_lm[m] * self.x_sm[m] * G + (1 + self.zeta * self.x_lm[m]) * (1 - self.x_sm[m]) * R
                dv[var] += term
        
        max_deriv = max(np.max(np.abs(dv)), np.max(np.abs(dx_sm)), np.max(np.abs(dx_lm)))
        dt = np.clip(0.1 / max_deriv, 2**-7, 1e5) if max_deriv > 0 else 1e5
        
        # Apply momentum update for v ################################################################################change
        new_v = self.v + dv * dt + self.mu * (self.v - self.v_prev)
        self.v_prev = self.v.copy()
        self.v = np.clip(new_v, -1, 1)
        
        self.x_sm = np.clip(self.x_sm + dx_sm * dt, 0, 1)
        self.x_lm = np.clip(self.x_lm + dx_lm * dt, 1, 1e4 * self.M)
        return C, dt

    def _gpu_step(self, dt):
        """
        Main update equations for one time step with faster optimization
        """
        # Zero out derivative accumulator
        cuda.to_device(np.zeros(self.N, dtype=np.float32), to=self.dv)
        
        # Compute clause satisfaction measures
        compute_Cm_gpu[self.blocks, self.threads](self.clauses_gpu, self.v, self.C, 
                                               self.min_terms, self.min_indices)
        
        # OPTIMIZATION: Compute stats on GPU directly - no need for expensive host transfers
        blocks_for_stats = min(1024, (self.M + 255) // 256)
        compute_stats_kernel[blocks_for_stats, 256](self.C, self.stats_buffer)
        
        # Only pull summary stats to host instead of entire array
        stats = self.stats_buffer[:blocks_for_stats*3].copy_to_host()
        C_sum = np.sum(stats[::3])
        C_max = np.max(stats[1::3]) 
        active_count = np.sum(stats[2::3])
        C_mean = C_sum / self.M
        
        # OPTIMIZATION: Simplified adaptive parameters
        # No complex updates, just use quick rules based on key metrics
        step_factor = min(1.0, self.step_count / 500.0) #(was) /5000 is insignificantly small  # Early vs late stage behavior
        avg_scale = 1.0 + max(0, min(2.0, 5.0 * (C_mean - 0.1)))
        
        # Fast heuristic parameter updates that don't require costly computation
        alpha_dynamic = self.alpha0 * avg_scale
        beta_dynamic = self.beta0 * avg_scale
        gamma_dynamic = self.gamma0 * (1.0 - 0.3 * step_factor)  # Decrease threshold over time
        delta_dynamic = self.delta0
        zeta_dynamic = self.zeta0 * (1.0 + step_factor)  # Increase residual influence over time
        mu_dynamic = min(0.95, 0.8 + 0.4 * step_factor)  # (was) 0.1  # Increase momentum over time

        # Compute derivatives with simplified parameters
        compute_derivatives_kernel[self.blocks, self.threads](
            self.clauses_gpu, self.v, self.x_lm, self.x_sm,
            self.min_terms, self.min_indices, self.dv,
            beta_dynamic, self.epsilon, gamma_dynamic,
            alpha_dynamic, delta_dynamic, self.dx_sm, self.dx_lm,
            zeta_dynamic
        )
        
        # Fast approximation for adaptive time step - skip expensive max reduction
        if self.step_count % 10 == 0:  # Only recompute every 10 steps
            # Compute max derivatives (expensive)
            max_dv = self._gpu_max(self.dv)
            max_dx_sm = self._gpu_max(self.dx_sm)
            max_dx_lm = self._gpu_max(self.dx_lm)
            max_deriv = max(max_dv, max_dx_sm, max_dx_lm)
            
            # OPTIMIZATION: More aggressive timestep
            if max_deriv > 0:
                base_dt = self.adaptive_dt_factor / max_deriv                  
                dt = np.float32(np.clip(base_dt, 2**-7, 1e5))
            else:
                dt = np.float32(1e5)
        
        # Update v with adaptive momentum
        blocks_v = (self.N + self.threads - 1) // self.threads
        update_v_kernel_momentum[blocks_v, self.threads](
            self.v, self.v_prev, self.dv, dt, mu_dynamic, self.v_new
        )
        self.v_prev.copy_to_device(self.v)
        self.v.copy_to_device(self.v_new)
        
        # Update auxiliary variables
        update_x_kernel[(self.M + self.threads - 1) // self.threads, self.threads](
            self.x_sm, self.x_lm, self.dx_sm, self.dx_lm, dt)

        # Track variables and clauses for visualization
        self.track_step()

        # Increment step count
        self.step_count += 1
        
        # Only when needed: Find out how many clauses are satisfied for reporting
        if self.step_count % 1000 == 0:
            satisfied = np.sum(C_mean * self.M < 0.1)
            print(f"Step {self.step_count}: C_mean={C_mean:.4f}, C_max={C_max:.4f}, Satisfied≈{satisfied}/{self.M}")
        
        # Return just what's needed without transferring full arrays
        return C_mean, C_max, dt

    def solve(self, max_steps=100000, tol=0.49, track_dynamics=False): #* even satisfied with tol=0.89 
        """Solve SAT problem with optional dynamics tracking"""
        dt = 1.0
        C_mean, C_max = 1.0, 1.0  # Initial values
        patience = 5000
        last_best = 1.0
        last_best_step = 0
        
        # Setup tracking if requested
        if track_dynamics and self.tracked_vars is None:
            self.setup_tracking()
        
        for step in range(int(max_steps)):
            # Only do full verification periodically to save time
            if step % self.check_interval == 0 and C_max < 0.2:
                print("_ _ _ _ _")
                print("step ", step, ".")
                print("C_max- ", C_max)
                print("C_mean- ", C_mean)
                try:
                    print("max(C_host)=",max(C_host))
                except:
                    pass
                C_host = self.C.copy_to_host()
                if  np.all(C_host < tol):  #* even satisfied with tol=0.99
                    print("yes")
                    print("max(C_host)=",max(C_host))
                    print(f"Solution found at step {step}!")
                    return (self.v.copy_to_host() > 0).astype(np.uint8)
            
            # Fast step without copying full arrays
            C_mean, C_max, dt = self._gpu_step(dt)
            
            # Early stopping with patience
            if C_max < last_best:
                last_best = C_max
                last_best_step = step
            elif step - last_best_step > patience:
                print(f"No improvement for {patience} steps, stopping early")
                break
        
        # Final check
        C_host = self.C.copy_to_host()
        if np.all(C_host < tol):
            return (self.v.copy_to_host() > 0).astype(np.uint8)
        return None

# --------------------------
# SAT Instance Generation
# --------------------------

def generate_hard_clauses(num_vars, num_clauses, seed=None, noise=0.35):
    rng = np.random.default_rng(seed)
    
    # 1. Generate unique variables per clause
    var1 = rng.integers(0, num_vars, size=num_clauses, dtype=np.int32)
    var2 = rng.integers(0, num_vars - 1, size=num_clauses, dtype=np.int32)
    var2 = np.where(var2 >= var1, var2 + 1, var2)
    
    var3 = rng.integers(0, num_vars - 2, size=num_clauses, dtype=np.int32)
    sorted_vars = np.sort(np.stack([var1, var2], axis=1), axis=1)
    a, b = sorted_vars[:, 0], sorted_vars[:, 1]
    var3 = np.where(var3 < a, var3, np.where(var3 < (b - 1), var3 + 1, var3 + 2))
    
    # 2. Plant hidden solution
    solution = rng.choice([-1, 1], size=num_vars)
    
    # 3. Generate clause signs with guaranteed satisfaction
    vars = np.stack([var1, var2, var3], axis=1)
    base_signs = solution[vars]  # Signs that would perfectly satisfy
    
    # 4. Apply controlled noise while maintaining ≥1 true literal
    noise_mask = rng.random(size=(num_clauses, 3)) < noise
    
    # Fix clauses where all literals would be flipped
    all_flipped = np.all(noise_mask, axis=1)
    fix_pos = rng.integers(0, 3, size=np.sum(all_flipped))
    noise_mask[all_flipped, fix_pos] = False
    
    # Apply flips
    final_signs = base_signs * np.where(noise_mask, -1, 1)
    
    # 5. Build clauses array
    clauses = np.empty((num_clauses, 3, 2), dtype=np.int32)
    clauses[:, :, 0] = vars
    clauses[:, :, 1] = final_signs
    
    return clauses


if __name__ == "__main__":
    # For quick visualization with smaller problem size
    NUM_VARS = 10_000_000  # Smaller problem for faster visualization
    CRITICAL_RATIO = 4.25
    NUM_CLAUSES = int(NUM_VARS * CRITICAL_RATIO)
    
    print(f"Generating {NUM_CLAUSES:,} clauses...")
    clauses = generate_hard_clauses(NUM_VARS, NUM_CLAUSES, seed=4612)
    
    print("Initializing solver...")
    solver = DMM3SATOptimized(clauses, NUM_VARS)
    
    # Setup tracking before solving
    print("Setting up dynamics tracking...")
    solver.setup_tracking(num_to_track=2)
    
    print("Solving...")
    start=time.time()
    solution = solver.solve(track_dynamics=True)
    end=time.time()
    print(f"Elapsed time: {end-start:.2f} seconds")
    
    # Generate and show the dynamics plots
    print("Generating dynamics plots...")
    solver.plot_dynamics()
    
    if solution is not None:
        print("Solution found! Verifying satisfiability...")
        # Convert solution to integers (0 or 1)
        solution_int = solution.astype(np.int32)
        # Extract variables and signs from clauses
        vars = clauses[:, :, 0]
        signs = clauses[:, :, 1]
        # Compute literal values (1 or -1 based on solution)
        literal_values = 2 * solution_int[vars] - 1
        # Check if each literal is satisfied
        satisfied_literals = signs * literal_values > 0
        # Check each clause has at least one satisfied literal
        clause_satisfied = np.any(satisfied_literals, axis=1)
        all_clauses_satisfied = np.all(clause_satisfied)
        if all_clauses_satisfied:
            print("Solution is verified to satisfy all clauses.")
        else:
            num_violated = np.sum(~clause_satisfied)
            print(f"Solution does NOT satisfy all clauses. Violated clauses: {num_violated}")
    else:
        print("No solution found")
