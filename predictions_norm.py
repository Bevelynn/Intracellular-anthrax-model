# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:02:08 2020

@author: 44794
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.integrate import quad
from scipy.stats import norm, truncnorm
from scipy.integrate import odeint


plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'axes.labelsize': 'large'})

#spore and bacteria data for MOI 1:1, 1:2, 1:10, and 1:20

spdata1 = np.array([200000, 121500, 138500, 255000])
bacdata1 = np.array([205000, 173500, 70000, 25000])

spdata2 = np.array([105000, 11400, 10250, 29500])
bacdata2 = np.array([23000, 67100, 52250, 20000])

spdata10 = np.array([27000,12000,9100,2750])
bacdata10 = np.array([9500,14000,7650,2500])

spdata20 = np.array([7900, 6450, 3100, 300])
bacdata20 = np.array([6000, 2250, 1750, 1000])

#data time points
data_times=np.array([0.5,2.5,4.5,23.5])

#time points for our curves
sim_times = np.linspace(0, 24, 49)

########################################################
'''functions to create curves for the mean number of spores and bacteria'''

#pdf of time to get from state 1_S to state 1, calculated at time t
def f1S1(t,g,mut):
    return g**2/mut*(np.exp(-g*t)-np.exp(-(g+mut)*t))

#pdf of time to get from state 1 to state R, calculated at time t
def f1R(t,a,b,lamb,gamma):
    return (gamma*(b-a)**2*np.exp(-lamb*(b-a)*t)/(b+(1-a)*np.exp(-lamb*(b-a)*t)-1)**2)

#pdf of germination rate
def fg(g,mean,sd):
    return truncnorm.pdf(g,(0 - mean) / sd, np.inf, loc=mean, scale=sd)

#mean number of newly germinated bacteria in a single cell at time t
def ngb(t,mug,sdg,mut,z):
    return (1/mut)*(1-np.exp(-mut*t))*(1/z)*(sdg/np.sqrt(2*np.pi)*np.exp(-mug**2/(2*sdg**2))+(mug-sdg**2*t)*np.exp(sdg**2*t**2/2-mug*t)*norm.cdf((mug-sdg**2*t)/sdg))

#mean number of spores in a single cell at time t
def s(t,mug,sdg,z):
    return (1/z)*np.exp(sdg**2*t**2/2-mug*t)*norm.cdf((mug-sdg**2*t)/sdg)

#pdf of time to get from state 1_S to state R, calculated at time t, given germination rate g
#calculates the value of f_{T_{1_S}^R}(t;g) for t, g
def f1SRg(t,g,mut,a,b,lamb,gamma):
    #step size for convolution
    h = 0.05
    tt = np.linspace(0,t,int(t/h)+1)
    ff1R = [f1R(t,a,b,lamb,gamma) for t in tt]
    ff1S1g = [f1S1(t,g,mut) for t in tt]
    return (fftconvolve(ff1S1g,ff1R)*h)[int(t/h)]

#integrand for the integral to get f_{T_{1_S}^R}(t)
def integrand(g,t,mean,sd,mut,a,b,lamb,gamma):
    f = f1SRg(t,g,mut,a,b,lamb,gamma)
    if f<0:
        f = 0
    return fg(g,mean,sd)*f
    
#simulation to give the mean number of intracellular spores and bacteria inside a single cell over time
def simulation(mean,sd,mut,lamb,mu,gamma,times):
    #transform back the parameters that have been log-transformed
    mean = pow(10,mean)
    sd = pow(10,sd)
    mut = pow(10,mut)
    lamb = pow(10,lamb)
    mu = pow(10,mu)
    gamma = pow(10,gamma)
    a = ((lamb+mu+gamma)-np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    b = ((lamb+mu+gamma)+np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    fR = np.array([quad(lambda g: integrand(g,t,mean,sd,mut,a,b,lamb,gamma),0,np.inf)[0] for t in times]) #density of rupture times for t in times
    #mean number of vegetative bacteria
    B = fR/gamma
    #initiate arrays for the time-courses of spores and bacteria
    spstar = np.zeros(len(times))
    bacstar = np.zeros(len(times))
    #normalisation factor for the germination rate pdf
    z = norm.cdf(mean/sd)
    for i,t in enumerate(times):
        #expected number of NGB at each time point
        NGB = ngb(t,mean,sd,mut,z)
        #expected number of total bacteria at each time point
        bacstar[i] = (NGB+B[i])
        #expected number of spores at each time point
        spstar[i] = s(t,mean,sd,z)
    return spstar,bacstar

###################################################################################
#load the posterior sample, sorted in order of distance from largest to smallest
sorted_params = np.loadtxt('posterior_norm.txt')

#number of time points in the prediction curves
l = len(sim_times)

#size of posterior sample
sample_size = 1000

#simulate the model with each of the posterior parameter sets
Sims_spores1 = np.zeros((sample_size,l))
Sims_bac1 =np.zeros((sample_size,l))
Sims_spores2 = np.zeros((sample_size,l))
Sims_bac2 =np.zeros((sample_size,l))
Sims_spores10 = np.zeros((sample_size,l))
Sims_bac10 = np.zeros((sample_size,l))
Sims_spores20 = np.zeros((sample_size,l))
Sims_bac20 = np.zeros((sample_size,l))
for i in range(sample_size):
   mean = sorted_params[i,0]
   sd = sorted_params[i,1]
   mut = sorted_params[i,2]
   lamb = sorted_params[i,3]
   mu = sorted_params[i,4]
   gamma = sorted_params[i,5]
   sim = simulation(mean,sd,mut,lamb,mu,gamma, sim_times)
   Sims_spores1[i,] = 377500*sim[0]
   Sims_bac1[i,] = 377500*sim[1]
   Sims_spores2[i,] = 139000*sim[0]
   Sims_bac2[i,] = 139000*sim[1]
   Sims_spores10[i,] = 30500*sim[0]
   Sims_bac10[i,] = 30500*sim[1]
   Sims_spores20[i,] = 13925*sim[0]
   Sims_bac20[i,] = 13925*sim[1]

lowws1 = []
highhs1 = []
lowwb1 = []
highhb1 = []

lowws2 = []
highhs2 = []
lowwb2 = []
highhb2 = []

lowws10 = []
highhs10 = []
lowwb10 = []
highhb10 = []

lowws20 = []
highhs20 = []
lowwb20 = []
highhb20 = []

#compute the 95% credible intervals of spores and bacteria for each of the 4 MOIs
for i in range(l):
   lows1 = np.percentile(Sims_spores1[:,i],2.5)
   highs1 = np.percentile(Sims_spores1[:,i],97.5)
   lowws1.append(lows1)
   highhs1.append(highs1)
   lowb1 = np.percentile(Sims_bac1[:,i],2.5)
   highb1 = np.percentile(Sims_bac1[:,i],97.5)
   lowwb1.append(lowb1)
   highhb1.append(highb1)
   
   lows2 = np.percentile(Sims_spores2[:,i],2.5)
   highs2 = np.percentile(Sims_spores2[:,i],97.5)
   lowws2.append(lows2)
   highhs2.append(highs2)
   lowb2 = np.percentile(Sims_bac2[:,i],2.5)
   highb2 = np.percentile(Sims_bac2[:,i],97.5)
   lowwb2.append(lowb2)
   highhb2.append(highb2)
   
   lows10 = np.percentile(Sims_spores10[:,i],2.5)
   highs10 = np.percentile(Sims_spores10[:,i],97.5)
   lowws10.append(lows10)
   highhs10.append(highs10)
   lowb10 = np.percentile(Sims_bac10[:,i],2.5)
   highb10 = np.percentile(Sims_bac10[:,i],97.5)
   lowwb10.append(lowb10)
   highhb10.append(highb10)
   
   lows20 = np.percentile(Sims_spores20[:,i],2.5)
   highs20 = np.percentile(Sims_spores20[:,i],97.5)
   lowws20.append(lows20)
   highhs20.append(highs20)
   lowb20 = np.percentile(Sims_bac20[:,i],2.5)
   highb20 = np.percentile(Sims_bac20[:,i],97.5)
   lowwb20.append(lowb20)
   highhb20.append(highb20)

low_s1 = np.hstack(lowws1)
high_s1 = np.hstack(highhs1)
low_b1 = np.hstack(lowwb1)
high_b1 = np.hstack(highhb1)

low_s2 = np.hstack(lowws2)
high_s2 = np.hstack(highhs2)
low_b2 = np.hstack(lowwb2)
high_b2 = np.hstack(highhb2)
   
low_s10 = np.hstack(lowws10)
high_s10 = np.hstack(highhs10)
low_b10 = np.hstack(lowwb10)
high_b10 = np.hstack(highhb10)

low_s20 = np.hstack(lowws20)
high_s20 = np.hstack(highhs20)
low_b20 = np.hstack(lowwb20)
high_b20 = np.hstack(highhb20)

####################################################################################
#prediction of the model using the parameter set with the smallest distance

#the parameter set with the smallest distance
mean = sorted_params[-1,0]
sd = sorted_params[-1,1]
mut = sorted_params[-1,2]
lamb = sorted_params[-1,3]
mu = sorted_params[-1,4]
gamma = sorted_params[-1,5]

#simulate the model with this parameter set
sim = simulation(mean,sd,mut,lamb,mu,gamma,sim_times)

#multiply by the intial condition for each MOI
spores1 = 377500*sim[0]
bac1 = 377500*sim[1]
spores2 = 139000*sim[0]
bac2 = 139000*sim[1]
spores10 = 30500*sim[0]
bac10 = 30500*sim[1]
spores20 = 13925*sim[0]
bac20 = 13925*sim[1]

##############################################################################
#calculate the predictions from pantha et al.

#define the differential equations, for spores, ngb, and vegetative bacteria
def diff_eqns(R,t=0):
   return np.array([-g*R[0],
                    g*R[0]-(mu*R[1]*cells/(cells+0.2*(R[0]+R[1]+R[2])))-m*R[1],
                    m*R[1]+lamb*R[2]-mu*R[2]*cells/(cells+0.2*(R[0]+R[1]+R[2]))])

times = np.linspace(0.0, 24, 1000) #Timecourse for integration

''''simulating the model for each MOI'''

'''For MOI 1:20'''

#values of the parameters
m = 0.05
g = 0.9547 
mu = 1.1727 
lamb = 1.173

#initial conditions
R0 = [13925,0,0]
cells = 13828

#get the output of the model
R, infodict = odeint(diff_eqns, R0, times, full_output = 1)

#take the outputs of interest
pantha_spores20 = R[:,0] 
pantha_bac20 = R[:,1]+R[:,2]

'''For MOI 1:10'''

#values of the parameters
m = 0.05
g = 0.6622
mu = 0.4787
lamb = 0.4477

#initial conditions
R0 = [30500,0,0]
cells = 24203

#get the output of the model
R, infodict = odeint(diff_eqns, R0, times, full_output = 1)

#take the outputs of interest
pantha_spores10 = R[:,0]
pantha_bac10 = R[:,1]+R[:,2]

'''For MOI 1:2'''

#values of the parameters
m = 0.05
g = 0.6636
mu = 0.4195
lamb = 0.4125
 
#initial conditions
R0 = [139000,0,0]
cells = 129772

#get the output of the model
R, infodict = odeint(diff_eqns, R0, times, full_output = 1)

#take the outputs of interest
pantha_spores2 = R[:,0]
pantha_bac2 = R[:,1]+R[:,2]

'''For MOI 1:1'''

#values of the parameters
m = 0.05
g = 1.2108
mu = 0.5605
lamb = 0.5246

#initial conditions
R0 = [377500,0,0]
cells = 314406

#get the output of the model
R, infodict = odeint(diff_eqns, R0, times, full_output = 1)

#Take the outputs of interest
pantha_spores1 = R[:,0]
pantha_bac1 = R[:,1]+R[:,2]

##############################################################################
#plotting the predictions

f1, ax1 = plt.subplots(2,2,figsize=(15,10))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.24, wspace=0.3, hspace=0.3)

plt.subplot(221)
plt.plot(sim_times, spores1, label='Best prediction of spores', color='blue')
plt.fill_between(sim_times, high_s1, low_s1, alpha=0.3, label='95% CI', color='blue')
plt.plot(sim_times, bac1, label='Best prediction of bacteria', color='orange')
plt.fill_between(sim_times, high_b1, low_b1, alpha=0.3, label='95% CI', color='orange')
plt.scatter(data_times, spdata1, label='Spore data', color='blue')
plt.scatter(data_times, bacdata1, label='Bacteria data', color='orange')
plt.plot(times,pantha_spores1,label='Prediction of spores from Pantha et al.', color='blue', linestyle='--')
plt.plot(times,pantha_bac1,label='Prediction of bacteria from Pantha et al.', color='orange', linestyle='--')
plt.ylabel('Number of intracellular components')
plt.xlabel('Time (hours)')
plt.title('MOI 1:1')

plt.subplot(222)
plt.plot(sim_times, spores2, label='Best prediction of spores', color='blue')
plt.fill_between(sim_times, high_s2, low_s2, alpha=0.3, label='95% CI', color='blue')
plt.plot(sim_times, bac2, label='Best prediction of bacteria', color='orange')
plt.fill_between(sim_times, high_b2, low_b2, alpha=0.3, label='95% CI', color='orange')
plt.scatter(data_times, spdata2, label='Spore data', color='blue')
plt.scatter(data_times, bacdata2, label='Bacteria data', color='orange')
plt.plot(times,pantha_spores2,label='Prediction of spores from Pantha et al.', color='blue', linestyle='--')
plt.plot(times,pantha_bac2,label='Prediction of bacteria from Pantha et al.', color='orange', linestyle='--')
plt.legend()
plt.ylabel('Number of intracellular components')
plt.xlabel('Time (hours)')
plt.title('MOI 1:2')

plt.subplot(223)
plt.plot(sim_times, spores10, label='Best prediction of spores', color='blue')
plt.fill_between(sim_times, high_s10, low_s10, alpha=0.3, label='95% CI', color='blue')
plt.plot(sim_times, bac10, label='Best prediction of bacteria', color='orange')
plt.fill_between(sim_times, high_b10, low_b10, alpha=0.3, label='95% CI', color='orange')
plt.scatter(data_times, spdata10, label='Spore data', color='blue')
plt.scatter(data_times, bacdata10, label='Bacteria data', color='orange')
plt.plot(times,pantha_spores10,label='Prediction of spores from Pantha et al.', color='blue', linestyle='--')
plt.plot(times,pantha_bac10,label='Prediction of bacteria from Pantha et al.', color='orange', linestyle='--')
plt.ylabel('Number of intracellular components')
plt.xlabel('Time (hours)')
plt.title('MOI 1:10')

plt.subplot(224)
plt.plot(sim_times, spores20, label='Best prediction of spores', color='blue')
plt.fill_between(sim_times, high_s20, low_s20, alpha=0.3, label='95% CI', color='blue')
plt.plot(sim_times, bac20, label='Best prediction of bacteria', color='orange')
plt.fill_between(sim_times, high_b20, low_b20, alpha=0.3, label='95% CI', color='orange')
plt.scatter(data_times, spdata20, label='Spore data', color='blue')
plt.scatter(data_times, bacdata20, label='Bacteria data', color='orange')
plt.plot(times,pantha_spores20,label='Prediction of spores from Pantha et al.', color='blue', linestyle='--')
plt.plot(times,pantha_bac20,label='Prediction of bacteria from Pantha et al.', color='orange', linestyle='--')
plt.ylabel('Number of intracellular components')
plt.xlabel('Time (hours)')
plt.title('MOI 1:20')

plt.suptitle('Model with continuous heterogeneity of germination rate', y=1.3, fontsize=15)
plt.savefig('prediction_norm.png', bbox_inches='tight')