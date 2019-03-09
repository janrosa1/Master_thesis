# Code overview
Presented here code is used to finding policy functions and simulate life cycle. In order to do so, a class life_cycle is defined and a few additional functions. The code will be presented as follows: firstly I describe the model which is simulated using life_cycle class, then I show implementation, finally some results.

## Model
I used simple life cycle model following the formulation of (among others) Kindermann & Kruger (2017), Kaplan (2012). Consumers live T periods, work R, then retire. There re two types f consumers: educated and uneducated. When consumer work her wage is determined by the price of the unit of labor $w$, idiosyncratic productivity $p_{j,i}$ (j-age, i-education type) which follows process:
$$
\begin{aligned}
\log p_{j,i} &= \kappa_i + \eta_{j,i} + \gamma_1 j -\gamma_2 j^{2}-\gamma_2 j^{3}\\
\eta_{j+1} &=\rho_i \eta_{j} + \epsilon_{i}\\
\kappa_i &\sim N(0, \sigma_i)\\
\epsilon_i &\sim N(0, \sigma_{\rho})
\end{aligned}
$$      
Lastly, wage is determined by aggregate productivity $P_j$ which follow random walk:
$$
\begin{aligned}
P_{j+1} &= \theta_{j+1}P_j\\
\theta_{j+1} &\sim N(0, \sigma_{ag})
\end{aligned}
$$      
When consumer retires, she obtain the same minimal pension.
Therefore, assuming CRRA utility, after dividing by $P_j$, consumers solve a problem defined by a Bellman equation (denoting by $a_j$ assets, $c_{j}$, consumption, $\pi_{j}$ probability of surviving to the next period, r- interst rate, pens- pension):
$$
\begin{aligned}
V_{j,i}(a_{j})&= \max_{a_{j+1},c_{j}}  u(c_{j}) + \beta \pi_{j}\textbf{E}(\Phi_{j+1}^{1-\sigma}V_{j+1} (\hat{a}_{j+1,t+1}, \Phi_{j+1})\\
if \; j\leq R:\;  & c_{j} + a_{j+1} \leq w p_{j,i} + (1 +r)a_{j}/\theta_j \\
if \; j> R  & c_{j} + a_{j+1} \leq pens + (1 +r)a_{j}\\
&a_{j+1}>0
\end{aligned}
$$
FOC is:
$$
\begin{aligned}
c_j^{-\sigma} &= \beta \pi_j (1+r) E_{j}(\Phi_{j+1}c_{j+1} )^{-\sigma}
\end{aligned}
$$

##Code structure
The main part of the code is a class life_cycle. Before I define two auxiliary functions:

- approx_markov(...) - tauchen apoximation of the AR, from QunatEcon toolbooks,
- normal_discrete_1(...) - discretization of the normal distribution from Kindermann toolbocks, converted from Fortran

### Life_cycle class
The main part of  code is
