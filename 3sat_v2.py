#(hopefully) same as version 1 but faster 

import os
import numpy as np
import numba
from numba import cuda, njit, float32
import math, time
import sys

# Set global random seed for deterministic behavior
GLOBAL_SEED = 4
np.random.seed(GLOBAL_SEED)

# Configure environment variables for deterministic CUDA operation
os.environ['NUMBA_ENABLE_CUDASIM'] = '0'
os.environ['NUMBA_CUDA_DRIVER'] = 'cuda'

numba.config.NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS = 0

# RTX 3060 Specifics (3584 CUDA cores, 28 SMs)
MAX_THREADS_PER_SM = 1536
THREADS_PER_BLOCK = 128

# Ensure CUDA is available, otherwise fall back to CPU
try:
    from numba.cuda.cudadrv import devices
    device = devices.get_context().device
    cc = device.compute_capability
    print(f"[GPU Active] {device.name}, Compute Capability: {cc[0]}.{cc[1]}")
    USE_CUDA = True
    
    # Force deterministic operations on GPU if possible
    cuda.select_device(0)  # Ensure we're always using the same GPU
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
    Optimized implementation with reduced redundant calculations and better memory access patterns.
    """
    m = cuda.grid(1)
    if m < clauses.shape[0]:
        # Cache frequently used values in registers for faster access
        min_val = min_terms[m]
        min_idx = min_indices[m]
        x_lm_m = x_lm[m]
        x_sm_m = x_sm[m]
        
        # Pre-compute shared factors to avoid duplicate calculations
        half_min_val = 0.5 * min_val
        x_lm_factor = 1.0 + zeta * x_lm_m
        g_scale = x_lm_m * x_sm_m
        r_scale = x_lm_factor * (1.0 - x_sm_m)
        
        # Compute auxiliary variable derivatives more efficiently
        sin_term = math.sin(half_min_val - gamma)
        dx_sm[m] = beta * (x_sm_m + epsilon) * sin_term * sin_term * sin_term  # sin³ optimization
        dx_lm[m] = alpha * (half_min_val - delta)
        
        # Process all three literals in one batch with fewer conditionals
        for i in range(3):
            var = clauses[m, i, 0]
            q = clauses[m, i, 1]
            
            # G term is always calculated
            G = q * half_min_val  # Factored out the 0.5
            
            # R term equals G only if this is the minimizer
            term = g_scale * G
            if i == min_idx:
                term += r_scale * G
                
            # Use atomic add only once per literal
            cuda.atomic.add(dv, var, term)

@cuda.jit(fastmath=True)
def update_v_kernel_momentum(v, v_prev, dv, dt, mu, v_new):
    """Optimized version with improved memory access pattern"""
    i = cuda.grid(1)
    if i < v.size:
        # Load values once to registers
        v_i = v[i]
        v_prev_i = v_prev[i]
        
        # Momentum term pre-calculation
        momentum_term = mu * (v_i - v_prev_i)
        
        # Single update with minimal operations
        new_v = v_i + dv[i] * dt + momentum_term
        
        # Optimized constraint using min/max
        v_new[i] = max(-1.0, min(1.0, new_v))

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
    block_size = cuda.blockDim.x
    idx = bid * block_size + tid
    
    # Use dynamic shared memory size based on block dimensions
    shared_sum = cuda.shared.array(shape=(THREADS_PER_BLOCK), dtype=float32)
    shared_max = cuda.shared.array(shape=(THREADS_PER_BLOCK), dtype=float32)
    shared_count = cuda.shared.array(shape=(THREADS_PER_BLOCK), dtype=float32)
    
    # Initialize
    shared_sum[tid] = 0.0
    shared_max[tid] = -math.inf  # Initialize to -infinity to find true max
    shared_count[tid] = 0.0
    
    # Load and process data
    if idx < C.size:
        val = C[idx]
        shared_sum[tid] = val
        shared_max[tid] = val
        shared_count[tid] = 1.0 if val > 0.05 else 0.0
    
    cuda.syncthreads()
    
    # Reduce
    s = block_size // 2
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

# Add optimized kernels from fvisualize_dynamics.py
@cuda.jit(fastmath=True)
def compute_mean_max_optimized(C, result):
    """Ultra-optimized kernel for computing mean, max using warp shuffle operations
    
    Args:
        C: Input array
        result: Output array with [sum, max]
    """
    # Shared memory for block reduction
    shared_sum = cuda.shared.array(shape=(THREADS_PER_BLOCK), dtype=float32)
    shared_max = cuda.shared.array(shape=(THREADS_PER_BLOCK), dtype=float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    idx = bid * block_size + tid
    grid_stride = block_size * cuda.gridDim.x
    
    # Initialize 
    local_sum = 0.0
    local_max = -math.inf
    
    # Grid-stride loop for processing multiple elements per thread
    while idx < C.size:
        val = C[idx]
        local_sum += val
        local_max = max(local_max, val)
        idx += grid_stride
    
    # Initialize shared memory
    shared_sum[tid] = local_sum
    shared_max[tid] = local_max
    cuda.syncthreads()
    
    # Optimized warp-level reduction
    # First combine within warps using shuffle operations
    mask = 0xffffffff  # All threads participate
    warp_size = 32
    lane = tid % warp_size
    warp_id = tid // warp_size
    
    # Warp-level reduction using shuffle operations (much faster than shared memory)
    warp_sum = local_sum
    warp_max = local_max
    
    # This is the most efficient way to do reduction on modern GPUs
    for offset in [16, 8, 4, 2, 1]:
        # Use __shfl_down_sync for warp-level reduction
        other_sum = cuda.shfl_down_sync(mask, warp_sum, offset)
        other_max = cuda.shfl_down_sync(mask, warp_max, offset)
        
        if lane < offset and lane + offset < warp_size:
            warp_sum += other_sum
            warp_max = max(warp_max, other_max)
    
    # First thread in each warp writes result to shared memory
    if lane == 0:
        shared_sum[warp_id] = warp_sum
        shared_max[warp_id] = warp_max
    
    cuda.syncthreads()
    
    # Final reduction across warps (typically just a few iterations)
    num_warps = min(block_size // warp_size, 32)  # Ensure we don't exceed 32 warps
    if tid < num_warps:
        warp_sum = shared_sum[tid]
        warp_max = shared_max[tid]
        
        # Small reduction across warps
        for i in range(1, num_warps):
            if tid + i < num_warps:
                warp_sum += shared_sum[tid + i]
                warp_max = max(warp_max, shared_max[tid + i])
    
    # First thread writes block results
    if tid == 0:
        cuda.atomic.add(result, 0, warp_sum)  # Sum
        cuda.atomic.max(result, 1, warp_max)  # Max

# --------------------------
# DMM Solver (Class)
# --------------------------
class DMM3SATOptimized:
    def __init__(self, clauses, num_vars, 
                 alpha=5.0, beta=20.0, gamma=0.25, 
                 delta=0.05, epsilon=1e-3, zeta=0.1, mu=0.9,
                 seed=GLOBAL_SEED):
        
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
        
        # Use consistent random seed
        self.rng = np.random.RandomState(seed)
        
        # Step counter
        self.step_count = 0

        # New parameters for faster convergence
        self.adaptive_dt_factor = .5  # More aggressive time stepping
        self.check_interval = 3       # Only check solution periodically
        
        # Stats buffer for efficient GPU calculations
        if USE_CUDA:
            self.stats_buffer = None  # Will initialize in _init_gpu

        if USE_CUDA:
            self._init_gpu(clauses)
        else:
            self._init_cpu(clauses)
    
    def _init_gpu(self, clauses):
        # Use deterministic variable initialization with fixed seed
        v_init = self.rng.uniform(-1, 1, self.N).astype(np.float32)
        
        self.clauses_gpu = cuda.to_device(np.ascontiguousarray(clauses.astype(np.int32)))
        self.v = cuda.to_device(v_init)
        self.v_prev = cuda.device_array_like(self.v)
        self.v_new = cuda.device_array_like(self.v)
        
        # Always copy initial values deterministically
        copy_kernel[(self.N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK](self.v, self.v_prev)
        cuda.synchronize()  # Force synchronization to ensure deterministic copy
        
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
        max_blocks = min(1024, (self.M + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK)
        self.stats_buffer = cuda.device_array(max_blocks * 3, dtype=np.float32)
        
        # Add buffer for optimized mean/max calculation
        self.mean_max_result = cuda.device_array(2, dtype=np.float32)

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


    def _gpu_step(self, dt):
        """
        Optimized main update function with reduced synchronization points
        """
        # Zero out derivative accumulator
        cuda.to_device(np.zeros(self.N, dtype=np.float32), to=self.dv)
        
        # Execute kernels with fewer synchronization points
        compute_Cm_gpu[self.blocks, self.threads](
            self.clauses_gpu, self.v, self.C, 
            self.min_terms, self.min_indices)
        
        # Get statistics with a single kernel call
        cuda.to_device(np.array([0.0, -np.inf], dtype=np.float32), to=self.mean_max_result)
        blocks_needed = min(1024, (self.M + self.threads - 1) // self.threads)
        compute_mean_max_optimized[blocks_needed, self.threads](self.C, self.mean_max_result)
        
        # Get results directly - reduces host/device transfers
        results = self.mean_max_result.copy_to_host()
        C_sum = results[0]
        C_max = results[1]  
        C_mean = C_sum / self.M
        
        # Calculate parameters once and reuse
        step_factor = min(1.0, self.step_count / 5000.0)
        avg_scale = 1.0 + max(0, min(2.0, 5.0 * (C_mean - 0.06)))
        
        # Pre-compute all parameters in one batch
        alpha_dynamic = self.alpha0 * math.sqrt(avg_scale)
        beta_dynamic = self.beta0 * avg_scale
        gamma_dynamic = self.gamma0 * (1.0 - 0.3 * step_factor)
        delta_dynamic = self.delta0
        zeta_dynamic = self.zeta0 * (1.0 + step_factor)
        mu_dynamic = min(0.95, 0.8 + 0.1 * step_factor)
        
        # Log only every N steps to reduce I/O overhead
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: C_mean={C_mean:.4f} avg_scale={avg_scale:.4f}")
        
        # Compute derivatives with all parameters sent at once
        compute_derivatives_kernel[self.blocks, self.threads](
            self.clauses_gpu, self.v, self.x_lm, self.x_sm,
            self.min_terms, self.min_indices, self.dv,
            beta_dynamic, self.epsilon, gamma_dynamic,
            alpha_dynamic, delta_dynamic, self.dx_sm, self.dx_lm,
            zeta_dynamic
        )
        
        # Calculate dt less frequently - reuse previous value more often
        if self.step_count % 10 == 0:
            # Use batched derivative calculation for better performance
            cuda.to_device(np.array([0.0], dtype=np.float32), to=self.max_result)
            max_reduce_kernel[min(1024, (self.N + self.threads - 1) // self.threads), 
                               self.threads](self.dv, self.max_temp)
            max_reduce_kernel[1, self.threads](self.max_temp[:min(1024, (self.N + self.threads - 1) // self.threads)], 
                                             self.max_result)
            max_dv = self.max_result.copy_to_host()[0]
            
            # Repeat for other arrays (condensed for brevity)
            # Similar operations for dx_sm and dx_lm
            max_dx_sm = self._gpu_max(self.dx_sm)
            max_dx_lm = self._gpu_max(self.dx_lm)
            
            max_deriv = max(max_dv, max_dx_sm, max_dx_lm)
            if max_deriv > 0:
                dt = np.float32(np.clip(self.adaptive_dt_factor / max_deriv, 2**-7, 1e5))
            else:
                dt = np.float32(1e5)
        
        # Update v with optimized momentum kernel
        blocks_v = (self.N + self.threads - 1) // self.threads
        update_v_kernel_momentum[blocks_v, self.threads](
            self.v, self.v_prev, self.dv, dt, mu_dynamic, self.v_new
        )
        
        # Batch copy operations for better memory throughput
        self.v_prev, self.v = self.v, self.v_new
        self.v, self.v_new = self.v_new, self.v
        
        # Update auxiliary variables
        update_x_kernel[(self.M + self.threads - 1) // self.threads, self.threads](
            self.x_sm, self.x_lm, self.dx_sm, self.dx_lm, dt)
        
        # Increment step count
        self.step_count += 1
        
        # Return just what's needed without transferring full arrays
        return C_mean, C_max, dt

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
                R = G if i == min_idx else 0.0
                term = self.x_lm[m] * self.x_sm[m] * G + (1 + self.zeta * self.x_lm[m]) * (1 - self.x_sm[m]) * R
                dv[var] += term
        
        max_deriv = max(np.max(np.abs(dv)), np.max(np.abs(dx_sm)), np.max(np.abs(dx_lm)))
        dt = np.clip(0.1 / max_deriv, 2**-7, 1e5) if max_deriv > 0 else 1e5
        
        # Apply momentum update for v
        new_v = self.v + dv * dt + self.mu * (self.v - self.v_prev)
        self.v_prev = self.v.copy()
        self.v = np.clip(new_v, -1, 1)
        
        self.x_sm = np.clip(self.x_sm + dx_sm * dt, 0, 1)
        self.x_lm = np.clip(self.x_lm + dx_lm * dt, 1, 1e4 * self.M)
        return C, dt

    def solve(self, max_steps=100000, tol=0.49):
        dt = 1.0  # Always start with the same dt
        C_mean, C_max = 1.0, 1.0
        patience = 5000
        last_best = 1.0
        last_best_step = 0
        
        # Force consistent execution pattern
        for step in range(int(max_steps)):
            # Fixed verification schedule
            if step % self.check_interval == 0 and C_max < 0.2:
                C_host = self.C.copy_to_host()
                if np.all(C_host < tol):
                    print(f"Solution found at step {step}!")
                    return (self.v.copy_to_host() > 0).astype(np.uint8)
            
            # Fast step without copying full arrays
            C_mean, C_max, dt = self._gpu_step(dt)
            
            # Always use exact deterministic early stopping logic
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
    """Generate SAT clauses deterministically with fixed seed"""
    # Always use provided seed or fall back to global seed
    seed = seed if seed is not None else GLOBAL_SEED
    rng = np.random.RandomState(seed)  # Use RandomState for isolated random generation
    
    # 1. Generate unique variables per clause - using randint instead of integers
    var1 = rng.randint(0, num_vars, size=num_clauses, dtype=np.int32)
    var2 = rng.randint(0, num_vars - 1, size=num_clauses, dtype=np.int32)
    var2 = np.where(var2 >= var1, var2 + 1, var2)
    
    var3 = rng.randint(0, num_vars - 2, size=num_clauses, dtype=np.int32)
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
    
    # Fix clauses where all literals would be flipped - also using randint here
    all_flipped = np.all(noise_mask, axis=1)
    fix_pos = rng.randint(0, 3, size=np.sum(all_flipped))
    noise_mask[all_flipped, fix_pos] = False
    
    # Apply flips
    final_signs = base_signs * np.where(noise_mask, -1, 1)
    
    # 5. Build clauses array
    clauses = np.empty((num_clauses, 3, 2), dtype=np.int32)
    clauses[:, :, 0] = vars
    clauses[:, :, 1] = final_signs
    
    return clauses


if __name__ == "__main__":
    # Allow command-line override of seed for testing
    if len(sys.argv) > 1 and sys.argv[1].startswith("--seed="):
        try:
            custom_seed = int(sys.argv[1].split("=")[1])
            GLOBAL_SEED = custom_seed
            np.random.seed(GLOBAL_SEED)
            print(f"Using custom seed: {GLOBAL_SEED}")
        except:
            print(f"Invalid seed argument, using default: {GLOBAL_SEED}")
    
    NUM_VARS = 23_000_000
    """
    #only works good for <5 , works very good inside the critical 4.25+-[0.2]
    so the problem of not being as good for approximately 5 and above;
    that's probably fixable algorithmically""" 
    CRITICAL_RATIO = 4.262 
    NUM_CLAUSES = int(NUM_VARS * CRITICAL_RATIO)
    GENERATOR_SEED = 135 #1899999
    
    print(f"Generating {NUM_CLAUSES:,} clauses with seed {GENERATOR_SEED}...")
    clauses = generate_hard_clauses(NUM_VARS, NUM_CLAUSES, seed=GENERATOR_SEED)
    
    print(f"Initializing solver with deterministic seed {GLOBAL_SEED}...")
    solver = DMM3SATOptimized(clauses, NUM_VARS, seed=GLOBAL_SEED)
    
    print("Solving...")
    start=time.time()
    solution = solver.solve()
    end=time.time()
    print(f"Elapsed time: {end-start:.2f} seconds") 
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
