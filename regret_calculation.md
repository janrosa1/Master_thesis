#Life-cycle model
Before presenting a calculation of the saving regret I set the life cycle model with idiosyncratic and aggregate risk and hyperbolic preferences. I assume that consumer live maximally T years, and retire after R years. There are two levels of the education of consumers (collage or lower).
##Idiosyncratic wage process
Before consumer retire, she obtains a wage $w_t$ for each productivity unit which she provides. The consumer's productivity process is given by idiosyncratic and aggregate component. Idiosyncratic productivity component of consumer with education level $i\in \lbrace 1,2 \rbrace$ in age j in time t: $p_{j,t,i}$ is:
$$
\begin{aligned}
\log p_{j,t,i} &= \kappa_i + \eta_{j,t,i} + \gamma_1 j -\gamma_2 j^{2}\\
eta_{j+1,t+1} &=\rho_i eta_{j,t} + \epsilon_{j,t,i}\\
\kappa_i &\sim N(0, \sigma_i)
\end{aligned}
$$
Where $\rho<1$. The aggregate productivity component is common to all agents at period t, and is given by transitory process:
$$
P_{t+1} = \theta_{t+1}P_t
$$  
Where $\theta$ is a random variable with positive values.
Therefore the wage of consumer with education level i in age j at period t is given as:
$$
W_{j,t} = p_{j,t,i} P_t w
$$
Where w is a wage for one production unit.
## Sticky expectations
Consumer do not always observe each realization of ($P_t$), insead she may estimate the ($\tilde{P}_t$,) using the last observation. Thus, if consumer do not observe real state in the period t, and the last observation was in the period t-n, then $\tilde{P}_t$ is given by the formula:
$$
\begin{aligned}
\tilde{P}_t &= E[P_{t}|P_{t-n}]
\end{aligned}
$$
The probability of updating expectations is given by the geometric distribution.
##Consumer problem
I assume that consumers have CRRA consumption function with parameter $\sigma$ and hyperbolic discounted parameter $\delta\leq 1$, conditional surviving probability $\pi_{j,t}$, discounting parameter $\beta$. The consumer problem for $\delta =1$ is given by:
$$
\begin{aligned}
\max_{a_{j+1,t+1},c_{j,t}}& u(c_{j,t})+E\sum_{k=0}\pi_{j+i,t+i} \beta^k u(c_{j+k,t+k})\\
& c_{j,t} +a_{j+1,t+1} \leq w_t p_{j,t,i} \tilde{P}_t   + (1 +r_t)a_{j,t}  
\end{aligned}      
$$

Where $r_t$ is an interest rate, $c_{j,t}$ is a consumption, $a_{j,t}$ is the asset holdings.  
For calculating policy functions, it is possible to write the problem using Bellman equations.
$$
\begin{aligned}
V_{j,t,i}(a_{j,t})&= \max_{a_{j+1,t+1},c_{j,t}}  u(c_{j,t}) + \beta \pi_{j,t}\textbf{E}(V_{j+1,t+1} (\hat{a}_{j+1,t+1}, \tilde{\Phi}_{t+1})\\
& \hat{c}_{j,t} + a_{j+1,t+1} \leq w_t p_{j,t,i} + (1 +r_t)a_{j,t}/\theta_t   
\end{aligned}
$$
Then the problem for the consumer with ($\delta \leq 1$) is given, using previous value function V, as:
$$
\begin{aligned}
\max_{a_{j+1,t+1},c_{j,t}}  u(c_{j,t}) + \delta \beta \pi_{j,t}\textbf{E}(V_{j+1,t+1} (\hat{a}_{j+1,t+1}, \tilde{\Phi}_{t+1})\\
\hat{c}_{j,t} + a_{j+1,t+1} \leq w_t p_{j,t,i} + (1 +r_t)a_{j,t}/\theta_t   
\end{aligned}
$$
Using homogeneity of the value function, the bellman equation may be given using new variables $\hat{x_t} = \frac{x_t}{\tilde{P_t}}$:
$$
\begin{aligned}
V_{j,t,i}(\hat{a}_{j,t})&= \max_{\hat{a}_{j+1,t+1},\hat{c}_{j,t}}  u(\hat{c}_{j,t}) + \beta \pi_{j,t}\textbf{E}(\tilde{\theta}_{t+1} )^{1-\sigma} V_{j+1,t+1} (\hat{a}_{j+1,t+1}, \tilde{\Phi}_{t+1})\\
& \hat{c}_{j,t} + a_{j+1,t+1} \leq \hat{w}_t p_{j,t,i} + (1 +r_t)\hat{a}_{j,t}/\theta_t   
\end{aligned}
$$
Similarly the maximization problem is:
$$
\begin{aligned}
\max_{\hat{a}_{j+1,t+1},\hat{c}_{j,t}}  u(\hat{c}_{j,t}) + \delta \beta \pi_{j,t}\textbf{E}(\tilde{\theta}_{t+1} )^{1-\sigma} V_{j+1,t+1} (\hat{a}_{j+1,t+1}, \tilde{\Phi}_{t+1})\\
 \hat{c}_{j,t} + a_{j+1,t+1} \leq \hat{w}_t p_{j,t,i} + (1 +r_t)\hat{a}_{j,t}/\theta_t   
\end{aligned}
$$
#Saving regret
In some period of her life (let us denote it by K), consumer has knowledge about all the values and distributions of the past shocks. She now can evaluate her previous decisions - calculate optimal savings and consumption paths, and decide if (and how much) she was wrong.   
The saving regret can be viewed in a few dimensions which we elaborate here:
- timing of the regretted decisions
- time horizon of regret       
- sources

##Timing of the regretted decisions
Consumer may regret the decision from every periods of hers life (from entering to the labor market), but also from the particular period of time. For example, consumer may regret only the decisions after her age-depended productivity achieved a particular level (denote this period by L). Thus, when she evaluate her decision, she calculate the optimal values of savings and consumption only from this particular period L.   
##Sources of regret
As it was mentioned before, the consumer recalculate optimal consumption and savings knowing all the values and distributions of the previous shocks. However, the optimal values can be defined in a few ways.
Consider the case without hyperbolic discounting. Consumer may evaluate her decision in two ways. Firstly, she may regret that she would have made other decision if she had known the exact values of the all shocks which have taken place until period K. Assuming that she evaluate decisions form period L, she than calculates optimal values $c^{1}_{j,t}, a^{1}_{j+k,t}$, solving the problem:
$$ \begin{aligned}
\max_{a^1_{L+k+1,t_L+1},c^1_{L+k,t_L+k}}&  \sum_{k=0}^{K-L}\pi_{L+k,t_L+k} \beta^k u(c_{L+k,t_L+k}) + \mathbf{E}_{K}\sum_{k=K}^{T-K}\pi_{L+k,t_L+k} \beta^k u(c_{L+k,t_L+k}) \\
if \; k\leq R:\;& c_{L+k,t_L+k} +a_{j+k+1,t+k+1} \leq w_{t+k} p_{j+k,t+k,i} P_{t_L+k}   + (1 +r_{t_L+k})a_{L+k,t_L+k}\\
if \; k> R:\;& c_{L+k,t_L+k} +a_{L+k+1,t_L+k_1} \leq pens + (1 +r_{t_L+k})a_{L+k,t_L+k}
\end{aligned}   $$
Consumer may also regret only the decisions which were not optimal knowing only the exact distribution of the shocks (which she wrongly estimate due to sticky expectations), not the values. In this case, she calculates optimal values $c^{2}_{L+s,t_L+s}, a^{2}_{L+s,t+L +s}$, solving series of problems:

$$ \begin{aligned}
\max_{a^2_{L+s+1,t_L+s},c^2_{L+s,t_L+s}}&  \mathbf{E}_{L+s} sum_{k=s}^{T-L-s}\pi_{L+k,t_L+k} \beta^k u(c_{L+k,t_L+k}) +  \\
if \; k\leq R:\;& c_{L+k,t_L+k} +a_{j+k+1,t+k+1} \leq w_{t+k} p_{j+k,t+k,i} P_{t_L+k}   + (1 +r_{t_L+k})a_{L+k,t_L+k}\\
if \; k> R:\;& c_{L+k,t_L+k} +a_{L+k+1,t_L+k_1} \leq pens + (1 +r_{t_L+k})a_{L+k,t_L+k}
\end{aligned}   $$           

##Time Horizon of the regret
Lastly, consumer may regret the utility loss for the her lifetime, thus difference i the utility level from period L, or from period K. In the first case consumer evaluate her decsion from her point of view of herself from period L. To find the scale of utility decrease, she find consumption equivalent $\lambda$ for which the actual utility (discounted from period L). In case without hyperbolic discounting, with regret concerning knowledge about values of the shocks, it is:

$$ \begin{aligned}
&  \sum_{k=0}^{K-L}\pi_{L+k,t_L+k} \beta^k u(c^1_{L+k,t_L+k}) + \mathbf{E}_{K}\sum_{k=K}^{T-K}\pi_{L+k,t_L+k} \beta^k u(c^1_{L+k,t_L+k})\\
   = &\sum_{k=0}^{K-L}\pi_{L+k,t_L+k} \beta^k u(\lambda c^0_{L+k,t_L+k}) + \mathbf{E}_{K}\sum_{k=K}^{T-K}\pi_{L+k,t_L+k} \beta^k u(\lambda c^0_{L+k,t_L+k})
\end{aligned}   $$
In a second case consumer evaluate her taking into account the past utility loss. For instance, in case without hyperbolic discounting, with regret concerning knowledge about values of the shocks, it is:   
$$ \begin{aligned}
  \mathbf{E}_{K}\sum_{k=0}^{T-K}\pi_{K+k,t_K+k} \beta^k u(c^1_{K+k,t_K+k}) & = \ \mathbf{E}_{K}\sum_{k=K}^{T-K}\pi_{K+k,t_K+k} \beta^k u(\lambda c^0_{K+k,t_K+k})
\end{aligned}   $$


##Regret level and decomposition
Having different values of lambda for different consumers it is then possible to calculate a "regeret" level of lambda - to match the regret percenitles from Borsh-Suspan et all (2018) for different groups. Then, I can derive regret decomposition:
starting from the most general case, calsulate the difference in the
