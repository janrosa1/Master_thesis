# Code overview
Presented here code is used to finding policy functions and simulate life cycle. In order to do so, a class life_cycle is defined and a few additional functions. The code will be presented as follows: firstly I describe the model which is simulated using life_cycle class, then I show implementation, finally some results.

## Model
I used simple life cycle model following the formulation of (among others) Kindermann & Kruger (2017), Kaplan (2012). Consumers live T periods, work R, then retire. There re two types f consumers: educated and uneducated. When consumer work her wage is determined by the price of the unit of labor $w$, idiosyncratic productivity $p_{j,k}$ (j-age, k-education type) which follows process:
$$
\begin{aligned}
\log p_{j,k} &= \kappa_i + \eta_{j,k} + \gamma_1 j -\gamma_2 j^{2}-\gamma_2 j^{3}\\
\eta_{j+1} &=\rho_k \eta_{j} + \epsilon_{k}\\
\kappa_i &\sim N(0, \sigma_k)\\
\epsilon_k &\sim N(0, \sigma_{\rho})
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
V_{j,i}(a_{j})&= \max_{a_{j+1},c_{j}}  u(c_{j}) + \beta \pi_{j}\textbf{E}(\theta_{j+1}^{1-\sigma}V_{j+1} (\hat{a}_{j+1}, \theta_{j+1})\\
if \; j\leq R:\;  & c_{j} + a_{j+1} \leq w p_{j,k} + (1 +r)a_{j}/\theta_j \\
if \; j> R  & c_{j} + a_{j+1} \leq pens + (1 +r)a_{j}\\
&a_{j+1}>0
\end{aligned}
$$
FOC is:
$$
\begin{aligned}
c_j^{-\sigma} &= \beta \pi_j (1+r) E_{j}(\theta_{j+1}c_{j+1} )^{-\sigma}
\end{aligned}
$$

##Code structure
The main part of the code is a class life_cycle. Before I define two auxiliary functions:

- approx_markov(...) - tauchen apoximation of the AR, from QunatEcon toolbooks,
- normal_discrete_1(...) - discretization of the normal distribution from Kindermann toolbocks, converted from Fortran

### life_cycle class
The main part of the code is class life_cycle. It contains constructor, two main methods and some additional functions to make plots.
#### Constructor
Constructor is defined as function of all parameters.
- T=79, R = 45 what correspond to the maximal age 100 years, a retirement age is 65
- Different $\rho, \epsilon_i$ for educated and uneducated (in conde indexed by p), discretized by Tauchen method, values from Kindremann & Kruger (2017)
- Fixed effect is represented is discretized into two values (indexed by f)
- Aggregate shock $\theta$ discretized into $aq$ points, using normal distribution discretization (indexed by q)
- Exponential assets' grid minimal asset is set from $10^{-4}$ to 60, gird points are indexed by i

 #### Policy_functions()    
Method to find policy functions for consumption and savings using endogenous grid method. Firstly, decision in the last period of life is given, then assets annd consumptions are calculated using endogenous grid algorithm. Firstly, the right hand side of FOC is calculated (here is a part for educated workers):
```python
for j in range(self.R,-1,-1):
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for q in range(self.n_ag):
                       w = self.edu_premium[1]*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)

                       for i in range(self.a_n+1):
                           m_cons_sum = 0
                           for i_p in range(self.a_r):
                               for i_q in range(self.n_ag):

                                   m_cons_sum = m_cons_sum+ self.P_edu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])

```
Next consumption is found using FOC and endogenous grid is established:
```python
cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption

a = 1/(1+self.r)*(self.grid[i]+cons-w)*self.val_ag[q] #compute endogenous grid values
a = np.maximum(self.grid_min,a)

end_grid[i] = a
```
Then the saving policy function for the exogenous grid points is found using linear interpolation:
```python
pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

pf_c[j,:,p,f,q] = (1+self.r)*self.grid/self.val_ag[q]+ w - pf_a[j,:,p,f,q] #find consumption policy function
```

Then, the same is redone for uneducated consumers.

 #### simulate_life_cycle()
 Due to (possibly) aggregate shocks with no initial distribution, many possible shocks paths are simulated instead of finding general distribution of savings and consumption.
 Firstly the aggregate values of shocks are set:
 ```python
 for j in range(self.R+1):
           rand = np.random.uniform(low=0.0, high=1.0)
           for q in range(self.n_ag):
                 if ( rand <=np.sum(self.Prob_ag[0:q+1]) ):
                      z_ag_hist[aq,j] = q
                      prod_history[aq,j] = self.val_ag[q]
                      break
 ```
Where aq index aggregate shocks, j year.
Then simulations for each consumer are done: the idiosyncratic productivity path, savings and consumption are simulated. Finally, the savings and consumptions which are nt divided by aggregate productivity are calculated:
```python
s_path_true[s,aq,j] = s_path[s,aq,j]*np.prod(prod_history[aq,0:j])
c_path_true[s,aq,j] = c_path[s,aq,j]*np.prod(prod_history[aq,0:j+1])       
```          
## Results
Here I present a few plots of policy functiions, consumption paths and savings paths which were found using life_cycle class methods

![alt text](https://github.com/janrosa1/Master_thesis/blob/master/Figure_1.png)
![alt text](https://github.com/janrosa1/Master_thesis/blob/master/Figure_2.png)
![alt text](https://github.com/janrosa1/Master_thesis/blob/master/Figure_3.png)
![alt text](https://github.com/janrosa1/Master_thesis/blob/master/Figure_4.png)
