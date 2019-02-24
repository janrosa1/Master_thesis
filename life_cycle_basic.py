import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math as m
"""
This is first version of the basic code
"""

#Tauchen method from QuantEcon
"""
Filename: tauchen.py

Authors: Thomas Sargent, John Stachurski

Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

from scipy.stats import norm


def approx_markov(rho, sigma_u, m=3, n=7):
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian
    process with zero mean.

    Parameters
    ----------
    rho : scalar(float)
        The autocorrelation coefficient
    sigma_u : scalar(float)
        The standard deviation of the random process
    m : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to
    n : scalar(int), optional(default=7)
        The number of states to use in the approximation

    Returns
    -------

    x : array_like(float, ndim=1)
        The state space of the discretized process
    P : array_like(float, ndim=2)
        The Markov transition matrix where P[i, j] is the probability
        of transitioning from x[i] to x[j]

    """
    F = norm(loc=0, scale=sigma_u).cdf

    # standard deviation of y_t
    std_y = np.sqrt(sigma_u**2 / (1-rho**2))

    # top of discrete state space
    x_max = m * std_y

    # bottom of discrete state space
    x_min = - x_max

    # discretized state space
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    for i in range(n):
        P[i, 0] = F(x[0]-rho * x[i] + half_step)
        P[i, n-1] = 1 - F(x[n-1] - rho * x[i] - half_step)
        for j in range(1, n-1):
            z = x[j] - rho * x[i]
            P[i, j] = F(z + half_step) - F(z - half_step)

    return x, P

#def Rouwen(rho, sigma_u, m, n):
    ## TO Do

class life_cycle:
    def __init__(self,
     T = 79, # number of periods of life
     R = 45, # retirement age
     alpha=0.4, # capital elasticity - used to to have suitable guess of w (optional, assuming closed econonmy what contradicts partial equilibirum)
     beta=0.98, #discount factor
     r = 0.04, # interst rate- since it is partial equilibrium
     sigma  = 2.0, #CRRA function parameter
     rho_edu = 0.985, # autororelation parameter in AR(1) productivity process
     sigma_AR_edu = 0.0180, #conditional variance  in AR(1) productivity process
     rho_unedu = 0.985, # autororelation parameter in AR(1) productivity process
     sigma_AR_unedu = 0.0346, #conditional variance  in AR(1) productivity process
     a_n =60, #number of points
     a_r = 5, #number of grid points at Rouwenhorst discretization
     n_sim = 1000, #nuber of simulations to get distribution of consumers
     grid_min =1e-4, #minimal assets d
     grid_max = 100.0, #maximal assets
     edu_premium = [1,1.34],
     edu_prop = [0.7,0.3],
     ind_fe = [0.2061,  0.1517],
     n_fe =2,
     age_eff_coef_1 = 0.109/2,
     age_eff_coef_2 = -0.001/2
     ):
        self.T, self.R, self.alpha, self.beta, self.r, self.sigma, self.rho_edu, self.sigma_AR_edu, self.rho_unedu, self.sigma_AR_unedu, self.a_n,\
        self.a_r,self.n_sim, self.grid_min, self.grid_max, self.edu_premium, self.edu_prop, self.ind_fe, self.n_fe, self.age_eff_coef_1, self.age_eff_coef_2 \
        = T, R, alpha, beta, r, sigma, rho_edu, sigma_AR_edu, rho_unedu, sigma_AR_unedu,\
        a_n, a_r, n_sim, grid_min, grid_max, edu_premium, edu_prop, ind_fe, n_fe, age_eff_coef_1, age_eff_coef_2
        self.inv_sigma = 1/sigma
        self.w = 1 #(1-alpha)*((1+r)/alpha)**(alpha/(alpha-1)) #wage guess
        [val,P] =  approx_markov(rho_edu, sigma_AR_edu, m=2, n=a_r) #take values of shocks and transition matrix
        self.val_edu =  val
        self.P_edu =  P #np.full((a_r,a_r),1/7.0)  #tranistion matrix
        self.ind_fe_edu = [-ind_fe[1], ind_fe[1] ]
        self.P_ind_fe_edu = [0.5,0.5]
        print(self.P_edu)
        print(self.val_edu)
        [val,P] =  approx_markov(rho_unedu, sigma_AR_unedu, m=2, n=a_r) #take values of shocks and transition matrix
        self.P_unedu =  P #np.full((a_r,a_r),1/7.0)  #tranistion matrix
        self.val_unedu =  val #AR discretizion vector values
        self.ind_fe_unedu = [-ind_fe[0], ind_fe[0] ]
        self.P_ind_fe_unedu = [0.5,0.5]
        #print(m.exp(val[0]), m.exp(val[3]), m.exp(val[6]))
        self.pens = self.w #pension
        self.pf_a_edu = np.zeros((self.T+1, self.a_n+1, self.a_r, n_fe)) #saving policy function
        self.pf_c_edu = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe)) #consumption policy unction
        self.pf_a_unedu = np.zeros((self.T+1, self.a_n+1, self.a_r,n_fe)) #saving policy function
        self.pf_c_unedu = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe)) #consumption policy unction
        self.grid = np.geomspace(self.grid_min, self.grid_max, num=self.a_n+1) #grid definition
        self.edu_premium = edu_premium
        self.edu_prop = edu_prop

        self.initial_prod = 2 #initial productivity
        self.initial_asset = self.grid[0] #initial assets
        self.sav_distr =  np.zeros((2*n_sim, T+1)) #distribution of savings from simulation
        self.cons_distr = np.zeros((2*n_sim, T+1)) #distribution of consumption from simulation
        prob_dead = np.genfromtxt('life_table.csv', delimiter=',')
        self.prob_surv = 1 - prob_dead
        self.zsim1 = np.zeros((n_sim, T+1))


    def utility(self,x): #calculate utility
         return x**(1-self.sigma)/(1-self.sigma)

    def marginal_u(self,x): #marginal utility
         if(x<1e-6):
              print("error")
         return x**(-self.sigma)
    def inv_marginal_u(self,x): #inverse of marginal utility
         if(x**(-self.inv_sigma)<1e-6):
              print("error",x)
         return x**(-self.inv_sigma)

    def policy_functions(self):
          """
          Find policy functions using endogenous grid method
          """
          #grid definition
          end_grid = np.zeros(self.a_n+1) #endogenous grid definition
          pf_a = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe)) #asset policy function
          pf_c = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe)) #cnsumption policy function
          #iteration for last year
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for i in range(self.a_n+1):
                       pf_c[self.T,i,p,f] = (1+self.r)*self.grid[i] + self.pens
                       pf_a[self.T,i,p,f] = 0
          #start iterations
          for j in range(self.T-1,-1,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):

                        w = self.edu_premium[1]*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*j**2 )

                        for i in range(self.a_n+1):
                            m_cons_sum = 0
                            for i_p in range(self.a_r):
                                 #compute marginal consumptions sum
                               m_cons_sum = m_cons_sum + self.P_edu[p,i_p]*self.marginal_u(pf_c[j+1,i,i_p,f])

                            cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption

                            a = 1/(1+self.r)*(self.grid[i]+cons-w) #compute endogenous grid values

                            if j >self.R: #for retiree, it made the same iteration p times (it will be corrected) TO Do
                                a = 1/(1+self.r)*(self.grid[i]+cons-self.pens)

                            a = np.maximum(0,a)

                            end_grid[i] = a
                        pf_a[j,:,p,f] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

                        pf_c[j,:,p,f] = (1+self.r)*self.grid+ w - pf_a[j,:,p,f] #find consumption policy function
                        if j >self.R:
                           pf_c[j,:,p,f] = (1+self.r)*self.grid + self.pens - pf_a[j,:,p,f]

          self.pf_a_edu = pf_a
          self.pf_c_edu = pf_c

          pf_a = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe)) #asset policy function
          pf_c = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe)) #cnsumption policy function
          #iteration for last year
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for i in range(self.a_n+1):
                       pf_c[self.T,i,p,f] = (1+self.r)*self.grid[i] + self.pens
                       pf_a[self.T,i,p,f] = 0
          #start iterations

          for j in range(self.T-1,-1,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        w = self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*j**2)
                        for i in range(self.a_n+1):
                            m_cons_sum = 0
                            for i_p in range(self.a_r):
                                 #compute marginal consumptions sum
                               m_cons_sum = m_cons_sum + self.P_unedu[p,i_p]*self.marginal_u(pf_c[j+1,i,i_p,f])
                            cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption

                            a = 1/(1+self.r)*(self.grid[i]+cons-w) #compute endogenous grid values

                            if j >self.R: #for retiree, it made the same iteration p times (it will be corrected) TO Do
                                a = 1/(1+self.r)*(self.grid[i]+cons-self.pens)
                            end_grid[i] = a
                        pf_a[j,:,p,f] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

                        pf_c[j,:,p,f] = (1+self.r)*self.grid+w - pf_a[j,:,p,f] #find consumption policy function
                        if j >self.R:
                           pf_c[j,:,p,f] = (1+self.r)*self.grid + self.pens - pf_a[j,:,p,f]

          self.pf_a_unedu = pf_a
          self.pf_c_unedu = pf_c
          return self.pf_a_edu, self.pf_c_edu, self.pf_a_unedu, self.pf_c_unedu

    def simulate_life_cycle(self):
          '''
          Due to (possibly) aggregate shocks with no initial distribution,
          we simulate many possible shocks paths and saving and consumption paths
          instead of finding general distribution
          '''
          initial_prod = self.initial_prod
          initial_sav = self.initial_asset
          s_path = np.zeros((self.n_sim, self.T+1))
          c_path = np.zeros((self.n_sim, self.T+1))
          zsim1 = np.zeros((self.n_sim, self.T+1)) #table of shocks, used for
          zsim = initial_prod
          zsim1[:,0] = zsim
          s_path[:,0] = initial_sav #initial productivity
           #initial conusumption
          for s in range(self.n_sim):
              rand1 = np.random.uniform(low=0.0, high=1.0)
              f=1
              if ( rand1 <=self.P_ind_fe_edu[0] ):
                   f = 0
              c_path[s,0] = self.pf_c_edu[0,0,zsim,f]
              zsim = initial_prod
              for j in range(1,self.T+1,1):
                  s_path[s,j] = np.interp(s_path[s,j-1], self.grid, self.pf_a_edu[j-1,:,zsim,f])
                  c_path[s,j] = np.interp(s_path[s,j], self.grid, self.pf_c_edu[j,:,zsim,f])
                  rand = np.random.uniform(low=0.0, high=1.0)
                  for p in range(self.a_r):
                       temp = np.sum(self.P_unedu[zsim, 0:p+1])
                       # if(p==0):
                       #      temp = self.P_edu[sim,0]
                       # else:
                       #      temp = np.sum(self.P_edu[sim, 0:p+1])
                       if ( rand <=temp ):
                            zsim =p
                            break
                  zsim1[s,j] = zsim

          self.sav_distr[0:self.n_sim ,:] = self.edu_prop[1]*s_path
          self.cons_distr[0:self.n_sim ,:] = self.edu_prop[1]*c_path

          zsim = initial_prod
          zsim1[:,0] = zsim
          s_path[:,0] = initial_sav #initial productivity
          for s in range(self.n_sim):
              zsim = initial_prod
              rand1 = np.random.uniform(low=0.0, high=1.0)
              f=1
              if ( rand1 <=self.P_ind_fe_unedu[0] ):
                   f = 0
              c_path[s,0] = self.pf_c_unedu[0,0,zsim,f]
              for j in range(1,self.T+1,1):
                  s_path[s,j] = np.interp(s_path[s,j-1], self.grid, self.pf_a_unedu[j-1,:,zsim,f])
                  c_path[s,j] = np.interp(s_path[s,j], self.grid, self.pf_c_unedu[j,:,zsim,f])
                  rand = np.random.uniform(low=0.0, high=1.0)
                  for p in range(self.a_r):
                       temp = np.sum(self.P_unedu[zsim, 0:p+1])
                            # if(p==0):
                            #     temp = self.P_unedu[sim,0]
                            # else:
                            # temp = np.sum(self.P_unedu[sim, 0:p+1])
                       if ( rand <=temp ):
                                 zsim =p
                                 break
                  zsim1[s,j] = zsim


          self.sav_distr[self.n_sim: 2*self.n_sim,:] = self.edu_prop[0]*s_path
          self.cons_distr[self.n_sim: 2*self.n_sim,:] = self.edu_prop[0]*c_path
          self.zsim1 = zsim1
          return self.sav_distr, self.cons_distr, self.zsim1

    def plot_life_cycle(self):

         s_mean = np.zeros(self.T+1)
         s_max = np.zeros(self.T+1)
         s_min = np.zeros(self.T+1)

         c_mean = np.zeros(self.T+1)
         c_max = np.zeros(self.T+1)
         c_min = np.zeros(self.T+1)

         z_mean = np.zeros(self.T+1)
         z_max = np.zeros(self.T+1)
         z_min = np.zeros(self.T+1)

         for j in range(1,self.T+1,1):
             s_mean[j] = np.mean(self.sav_distr[:,j])
             s_max[j] = np.percentile(self.sav_distr[:,j],90)
             s_min[j] = np.percentile(self.sav_distr[:,j],10)
         plt.plot(range(self.T+1), s_mean, label = "mean savings")
         plt.plot(range(self.T+1), s_max, label = "90th percentile of savings")
         plt.plot(range(self.T+1), s_min, label = "90th percentile of savings")
         plt.ylabel('savings profile')
         plt.legend(loc='best')
         plt.show()
         for j in range(1,self.T+1,1):
             z_mean[j] = np.mean(self.zsim1[:,j])
             z_max[j] = np.max(self.zsim1[:,j])
             z_min[j] = np.min(self.zsim1[:,j])
         plt.plot(range(self.T+1), z_mean, label = "mean shocks")
         plt.plot(range(self.T+1), z_max, label = "max shocks")
         plt.plot(range(self.T+1), z_min, label = "min shocks")
         plt.ylabel('shocks')
         plt.legend(loc='best')
         plt.show()

         for j in range(1,self.T+1,1):
             c_mean[j] = np.mean(self.cons_distr[:,j])
             c_max[j] = np.percentile(self.cons_distr[:,j],90)
             c_min[j] = np.percentile(self.cons_distr[:,j],10)
         plt.plot(range(self.T+1), c_mean, label = "mean consumption")
         plt.plot(range(self.T+1), c_max, label = "90th percentile of consumption")
         plt.plot(range(self.T+1), c_min, label = "10th percentile consumption")
         plt.ylabel('savings')
         plt.legend(loc='best')
         plt.show()


         plt.plot(self.grid[0:50], self.pf_a_edu[44,0:50,0,1], label = "savings for worst group")
         plt.plot(self.grid[0:50], self.pf_a_edu[44,0:50,1,1], label = "savings for second worst group")
         plt.plot(self.grid[0:50], self.pf_a_edu[44,0:50,2,1], label = "savings for mednian group")
         plt.plot(self.grid[0:50], self.pf_a_edu[44,0:50,3,1], label = "savings for second best group")
         plt.plot(self.grid[0:50], self.pf_a_edu[44,0:50,4,1], label = "savings for best group")
         plt.ylabel('saving policy function for eductated')
         plt.legend(loc='best')
         plt.show()


    def execute_life_cycle(self):
          self.policy_functions()
          self.simulate_life_cycle()
          self.plot_life_cycle() #some basic plots

#### test ######
test_1 = life_cycle()
test_1.execute_life_cycle()
