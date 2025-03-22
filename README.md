<img src="https://latex.codecogs.com/gif.latex?O({n\over{\text{gpu\thinspace{size}}}}\log{n})"/> 3-SAT solver.

obtained by tweeking, a bit, the system presented in https://arxiv.org/abs/2011.06551 of MemComputing Inc.

Related code: https://github.com/yuanhangzhang98/MemComputing_3SAT_demo 




### Problem Formulation

The 3-SAT problem consists of:

- **Variables**: We have <img src="https://latex.codecogs.com/gif.latex?N"/> Boolean variables, indexed as <img src="https://latex.codecogs.com/gif.latex?k=0,1,\dots,N-1"/>. Each variable needs to be assigned either "true" or "false."
- **Clauses**: There are <img src="https://latex.codecogs.com/gif.latex?M"/> clauses, indexed as <img src="https://latex.codecogs.com/gif.latex?m=0,1,\dots,M-1"/>. Each clause is a disjunction (OR) of three literals, and all clauses must be satisfied simultaneously.

For each clause <img src="https://latex.codecogs.com/gif.latex?m"/>, we define three literals by:
- <img src="https://latex.codecogs.com/gif.latex?\text{var}_{m,i}\in\{0,1,\dots,N-1\}"/> (for <img src="https://latex.codecogs.com/gif.latex?i=0,1,2"/>), indices the variable in clause <img src="https://latex.codecogs.com/gif.latex?m"/> literal <img src="https://latex.codecogs.com/gif.latex?i"/>.
- <img src="https://latex.codecogs.com/gif.latex?q_{m,i}\in\{-1,1\}"/>, where 1 indicates the variable is positive and -1 indicates it is negated.

The objective is to assign truth values to all <img src="https://latex.codecogs.com/gif.latex?N"/> variables such that all <img src="https://latex.codecogs.com/gif.latex?M"/> clauses are true.

### State Variables

The system uses three types of continuous state variables:
- **Variable States**: For each variable <img src="https://latex.codecogs.com/gif.latex?k"/>, we have <img src="https://latex.codecogs.com/gif.latex?v_k\in[-1,1]"/>. A value near 1 represents "true," and a value near -1 represents "false."
- **Short-Term Auxiliary Variables**: For each clause <img src="https://latex.codecogs.com/gif.latex?m"/>, <img src="https://latex.codecogs.com/gif.latex?x_{sm}^{(m)}\in[0,1]"/> helps modulate short-term dynamics related to clause satisfaction.
- **Long-Term Auxiliary Variables**: Also per clause <img src="https://latex.codecogs.com/gif.latex?m"/>, <img src="https://latex.codecogs.com/gif.latex?x_{lm}^{(m)}\in[1,10^4M]"/> amplifies the effect of unsatisfied clauses over time.

These variables evolve over time, with their updates parallelized on the GPU.

### Clause Satisfaction Measure

For each clause <img src="https://latex.codecogs.com/gif.latex?m"/> and literal <img src="https://latex.codecogs.com/gif.latex?i"/>, the dissatisfaction is:
<img src="https://latex.codecogs.com/gif.latex?t_{m,i}=1-q_{m,i}v_{\text{var}_{m,i}}"/>

- If the literal is satisfied (e.g., <img src="https://latex.codecogs.com/gif.latex?q_{m,i}=1"/> and <img src="https://latex.codecogs.com/gif.latex?v_{\text{var}_{m,i}}>0"/>), then <img src="https://latex.codecogs.com/gif.latex?t_{m,i}\leq0"/>.
- If unsatisfied, <img src="https://latex.codecogs.com/gif.latex?t_{m,i}>0"/>.

The clause satisfaction measure is:
<img src="https://latex.codecogs.com/gif.latex?C_m=\frac{1}{2}\min_{i=0,1,2}t_{m,i}"/>
where <img src="https://latex.codecogs.com/gif.latex?min_{i=0,1,2}t_{m,i}"/> is the smallest dissatisfaction term among the three literals of the clause. Here, <img src="https://latex.codecogs.com/gif.latex?C_m=0"/> means the clause is satisfied, and the range is <img src="https://latex.codecogs.com/gif.latex?C_m\in[0,1]"/>.

Additionally, we track:
- <img src="https://latex.codecogs.com/gif.latex?i_m^*=\arg\min_{i=0,1,2}t_{m,i}"/>, the index of the least satisfied literal.

### Dynamical Equations

The system evolves through these differential equations:

1. **Short-Term Auxiliary Dynamics**  
   For each clause <img src="https://latex.codecogs.com/gif.latex?m"/>:  
   <img src="https://latex.codecogs.com/gif.latex?\frac{dx_{sm}^{(m)}}{dt}=\beta\left(x_{sm}^{(m)}+\epsilon\right)\sin^3\left(C_m-\gamma\right)"/>  
   - <img src="https://latex.codecogs.com/gif.latex?\beta"/> (e.g., 20.0): adjustment rate.  
   - <img src="https://latex.codecogs.com/gif.latex?\epsilon=0.001"/>: prevents singularity.  
   - <img src="https://latex.codecogs.com/gif.latex?\gamma"/> (e.g., 0.25): threshold.

2. **Long-Term Auxiliary Dynamics**  
   For each clause <img src="https://latex.codecogs.com/gif.latex?m"/>:  
   <img src="https://latex.codecogs.com/gif.latex?\frac{dx_{lm}^{(m)}}{dt}=\alpha\left(C_m-\delta\right)"/>  
   - <img src="https://latex.codecogs.com/gif.latex?\alpha"/> (e.g., 5.0): adjustment rate.  
   - <img src="https://latex.codecogs.com/gif.latex?\delta"/> (e.g., 0.05): threshold.

3. **Variable Dynamics**  
   For each variable <img src="https://latex.codecogs.com/gif.latex?k"/>:  
   <img src="https://latex.codecogs.com/gif.latex?\frac{dv_k}{dt}=\sum_{\overset{m=0}{\textbf{clauses}}}^{M-1}\sum_{\overset{i=0}{\quad\textbf{literals}}}^{2}\mathbb{I}[\text{var}_{m,i}=k]\left[x_{lm}^{(m)}x_{sm}^{(m)}G_{m,i}+(1+\zeta{x}_{lm}^{(m)})(1-x_{sm}^{(m)})R_{m,i}\right]"/>  
   Where:  
   - <img src="https://latex.codecogs.com/gif.latex?\mathbb{I}[\text{var}_{m,i}=k]"/>: 1 if variable <img src="https://latex.codecogs.com/gif.latex?k"/> is at <img src="https://latex.codecogs.com/gif.latex?\text{var}_{m,i}"/>, 0 otherwise.  
   - Gradient: <img src="https://latex.codecogs.com/gif.latex?G_{m,i}=q_{m,i}C_m"/>.  
   - Residual: <img src="https://latex.codecogs.com/gif.latex?R_{m,i}=\begin{cases}\frac{1}{2}(q_{m,i}-v_k)&\text{if}\quad{i}=i_m^*\\0&\text{otherwise}\end{cases}\quad=\begin{cases}q_{m,i}C_m&\text{if}\quad{i}=i_m^*\\0&\text{otherwise}\end{cases}=\begin{cases}G_{m,i}&\text{if}\quad{i}=i_m^*\\0&\text{otherwise}\end{cases}"/>.  
   - <img src="https://latex.codecogs.com/gif.latex?\zeta"/> (e.g., 0.1): weighting factor.

### Numerical Integration

- **Variable Update with Momentum**:  
  <img src="https://latex.codecogs.com/gif.latex?v_k(t+\Delta{t})=v_k(t)+\frac{dv_k}{dt}\Delta{t}+\mu\left(v_k(t)-v_k(t-\Delta{t})\right)"/>  
  - <img src="https://latex.codecogs.com/gif.latex?\mu"/> (e.g., 0.9): momentum coefficient.  
  - Clamped to <img src="https://latex.codecogs.com/gif.latex?[-1,1]"/>.

- **Auxiliary Updates**:  
  <img src="https://latex.codecogs.com/gif.latex?x_{sm}^{(m)}(t+\Delta{t})=x_{sm}^{(m)}(t)+\frac{dx_{sm}^{(m)}}{dt}\Delta{t}"/>  
  <img src="https://latex.codecogs.com/gif.latex?x_{lm}^{(m)}(t+\Delta{t})=x_{lm}^{(m)}(t)+\frac{dx_{lm}^{(m)}}{dt}\Delta{t}"/>  
  - <img src="https://latex.codecogs.com/gif.latex?x_{sm}^{(m)}"/> in <img src="https://latex.codecogs.com/gif.latex?[0,1]"/>.  
  - <img src="https://latex.codecogs.com/gif.latex?x_{lm}^{(m)}"/> in <img src="https://latex.codecogs.com/gif.latex?[1,10^4M]"/>.

### Adaptive Time Step

The time step <img src="https://latex.codecogs.com/gif.latex?\Delta{t}"/> adjusts based on:  
<img src="https://latex.codecogs.com/gif.latex?\text{max\_deriv}=\max\left(\max_k\left|\frac{dv_k}{dt}\right|,\max_m\left|\frac{dx_{sm}^{(m)}}{dt}\right|,\max_m\left|\frac{dx_{lm}^{(m)}}{dt}\right|\right)"/>  
Then:  
<img src="https://latex.codecogs.com/gif.latex?\Delta{t}=\begin{cases}\frac{0.5}{\text{max\_deriv}}&\text{if}\quad\text{max\_deriv}>0\\10^5&\text{if}\quad\text{max\_deriv}=0\end{cases}"/>  
Clamped to <img src="https://latex.codecogs.com/gif.latex?[2^{-7},10^5]"/>.

### Parameter Adaptation

Initial values:
- <img src="https://latex.codecogs.com/gif.latex?\alpha_0=5.0"/>, <img src="https://latex.codecogs.com/gif.latex?\beta_0=20.0"/>, <img src="https://latex.codecogs.com/gif.latex?\gamma_0=0.25"/>, <img src="https://latex.codecogs.com/gif.latex?\delta_0=0.05"/>, <img src="https://latex.codecogs.com/gif.latex?\zeta_0=0.1"/>, <img src="https://latex.codecogs.com/gif.latex?\mu_0=0.9"/>.

Adapted using average satisfaction <img src="https://latex.codecogs.com/gif.latex?\bar{C}=\frac{1}{M}\sum_m{C_m}"/>:
- <img src="https://latex.codecogs.com/gif.latex?\text{avg\_scale}=1+\max(0,\min(2.0,5.0\cdot(\bar{C}-0.1)))"/>
- <img src="https://latex.codecogs.com/gif.latex?\text{step\_factor}=\min(1.0,\frac{\text{step\_count}}{500})"/> (not needed ; equals 1, as the timesteps are much less than 500)

Updates:
- <img src="https://latex.codecogs.com/gif.latex?\alpha(t)=\alpha_0\cdot\text{avg\_scale}"/>
- <img src="https://latex.codecogs.com/gif.latex?\beta(t)=\beta_0\cdot\text{avg\_scale}"/>
- <img src="https://latex.codecogs.com/gif.latex?\gamma(t)=\gamma_0\cdot(1-0.3\cdot\text{step\_factor})"/>
- <img src="https://latex.codecogs.com/gif.latex?\zeta(t)=\zeta_0\cdot(1+\text{step\_factor})"/>
- <img src="https://latex.codecogs.com/gif.latex?\mu(t)=\min(0.95,0.8+0.1\cdot\text{step\_factor})"/>

### Solution Extraction

The solver runs until <img src="https://latex.codecogs.com/gif.latex?C_m<0.49"/> for all clauses. The assignment is:
- <img src="https://latex.codecogs.com/gif.latex?v_k>0"/>: "true."
- <img src="https://latex.codecogs.com/gif.latex?v_k\leq0"/>: "false."

### **Results**
Here the instance is 10 million variables and 4.25*10 million clauses (4.25 ratio was chosen because it is the hardest to solve)<img src="https://latex.codecogs.com/gif.latex?v_k"/>

I chose one variable at random

Fig.1 here the variables <img src="https://latex.codecogs.com/gif.latex?v"/> that share clauses with the variable (x-axis are timesteps)
![Fig.1](https://github.com/user-attachments/assets/6e90015d-2803-4b6d-b52b-181b64f30f80)

Fig.2 <img src="https://latex.codecogs.com/gif.latex?C_m"/> of the clauses it appears in
![Fig2](https://github.com/user-attachments/assets/ef6ecfc6-ebc7-4bf8-97a3-4a23a6730b6f)

Giving the system random 3-sat instances to solve; the solution is found typically in approximaly 87 timesteps to  solve (after running it manually many times didn't find an exception).
it takes about 13-17 seconds on my computer that has an nvidia rtx geforce 3060 12GB gpu (better gpus can handle bigger instances and faster). I think there is still a lot **room for improvement** even for this implementation, and there can be a much better more efficient algorithm, based on the ideas of https://arxiv.org/abs/2011.06551.

##**Epilogue and Apologies**

The gradient term <img src="https://latex.codecogs.com/gif.latex?G"/> is different than in the paper, originally due to mistake but correcting it in the current code increases the solving time. Concepts of 'adaptivity' greatly improved the algorithm, same as simple momentum term (I also tried nesterov momentum and adam like optimizations, but I think that didn't help), the <img src="https://latex.codecogs.com/gif.latex?\sin^3\left(C_m-\gamma\right)"/>  vs just <img src="https://latex.codecogs.com/gif.latex?C_m-\gamma"/> helped consistently solve in the fewest timesteps (ie. reduce deviations in solving time).

(typical long memory plot)
![image](https://github.com/user-attachments/assets/b63e8f92-b3b4-4286-b8da-effa72362282)


(typical short memory plot)
![image](https://github.com/user-attachments/assets/6ce7ab5d-355f-48c9-943d-91a9abc041e1)



The GPU maybe isn't fully used and there are many other **ineffiencies** (algorithmic and programming wise ie. data transfers, unnecessary computation, precision etc.). Also better memory management is necessary;  as with together 40 million variables and 4.25*40 million clauses gpu gets overwhelmed and computers crashes (Note: an  H100 gpu (with plenty of RAM) can handle succesfully 350 million variables and 1,49 billion clauses (tried on a server) ). Also although the code is very effiecient up to even further than the critical ratio, for ratios more close to 5 it start getting inefficient (but that must be  algorithmically fixable).

The timesteps <img src="https://latex.codecogs.com/gif.latex?\Delta{t}"/>.
 seems to scale logarithmically with the number of variables <img src="https://latex.codecogs.com/gif.latex?N"/> . The actual time per <img src="https://latex.codecogs.com/gif.latex?\Delta{t}"/> is is linear (but fully parallelizable ie. doubling the gpu halves the time) function of <img src="https://latex.codecogs.com/gif.latex?N"/>. 
![image](https://github.com/user-attachments/assets/f4fc0fac-0256-46ba-93c1-5ba2da5c027a)

![steps_heatmap](https://github.com/user-attachments/assets/b59649ae-f766-4a1a-9e55-9e5d112591df)

also the average clause dissatisfaction <img src="https://latex.codecogs.com/gif.latex?\bar{C_m}"/> goes fast towards <img src="https://latex.codecogs.com/gif.latex?0"/> in timesteps independent of the number of variables <img src="https://latex.codecogs.com/gif.latex?N"/>
![image](https://github.com/user-attachments/assets/abdcfe93-6cf9-4127-8598-08941cbfec08) 

Few experiments ,of instances picked at random, on an H100 gpu.

![image](https://github.com/user-attachments/assets/cea76b03-96dd-431f-9299-20dd64ac524e)
typical output of running an instance of 23 million variables and at the critical ratio (4.256) 

![Animation1](https://github.com/user-attachments/assets/8bd9137b-adf1-4b77-bdbf-0697f136d58a)
83 million variables at the critical ratio (4.256)

![image](https://github.com/user-attachments/assets/48dce0f1-6559-44db-936a-a060b5c66e18)
330 million variables at the critical ratio (4.256)

