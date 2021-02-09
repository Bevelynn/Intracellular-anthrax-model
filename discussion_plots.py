# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:10:14 2020

@author: 44794
"""

import numpy as np   
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'axes.labelsize': 'large'})

#########################################################
'''functions to calculate the rupture size 
distribution and expected rupture size 
for a given parameter set'''

#probability of rupture given germination rate g
def prob_rupt_g(g,mut,lamb,mu,gamma):
    a = ((lamb+mu+gamma)-np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    return g*(1-a)/(g+mut)

#probability of rupture for the parameter set 'params'
def prob_rupt(params):
    e = params[0]
    gA = params[1]
    gB = params[2]
    mut = params[3]
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    return e*prob_rupt_g(gA,mut,lamb,mu,gamma)+(1-e)*prob_rupt_g(gB,mut,lamb,mu,gamma)

#probability of rupture size n>=1, starting from state 1
def R1n(n,lamb,mu,gamma):
    a = (lamb+mu+gamma-np.sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    b = (lamb+mu+gamma+np.sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return (b-1)*(1-a)/b**n

#probability of rupture size n>=1, starting from a spore with germination rate 'germ'
def R1Sn(germ,n,mut,lamb,mu,gamma):
    return (germ/(germ+mut))*R1n(n,lamb,mu,gamma)
    
#rupture size distribution for the parameter set 'params'
def rupt_dist(params):
    e = params[0]
    gA = params[1]
    gB = params[2]
    mut = params[3]
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    n = 0
    ruptdist = [1-prob_rupt(params)]
    while 1-sum(ruptdist)>0.00000001:
        n+=1
        prob = e*R1Sn(gA,n,mut,lamb,mu,gamma)+(1-e)*R1Sn(gB,n,mut,lamb,mu,gamma)
        ruptdist.append(prob)
    return ruptdist

#expected rupture size for the parameter set 'params'
def expected_size(params):
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    b = (lamb+mu+gamma+np.sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return prob_rupt(params)*b/(b-1)

###################################################################
'''creating the scatter plots for the combinations of
parameters that lead to the desired value of r'''

#function giving r/phi where r is
#the probability that a single spore will establish
#infection (i.e lead to a population of M bacteria)
def a(p,M,ruptdist):
    prob = 0
    if p!=0.5:
        if M<len(ruptdist):
            prob = sum([ruptdist[i]*(1-(p/(1-p))**i)/(1-(p/(1-p))**M) for i in range(M)])
            for i in range(M,len(ruptdist)):
                prob+=ruptdist[i]
        else:
            prob = sum([ruptdist[i]*(1-(p/(1-p))**i)/(1-(p/(1-p))**M) for i in range(1, len(ruptdist))])
    else:
        if M<len(ruptdist):
            prob = sum([ruptdist[i]*i/M for i in range(M)])
            for i in range(M,len(ruptdist)):
                prob+=ruptdist[i]
        else:
            prob = sum([ruptdist[i]*i/M for i in range(1, len(ruptdist))])
    return prob

#the value of r that we want
rstar = 3.31*10**(-5)

n = 100 #number of posterior parameter sets to use
n2 = 100 #number of p values to use
max_p = 0.99 #maximum value of p to use
p_vals = np.linspace(0,max_p,n2) #vector of p values to use
#load an array of n parameter sets from the posterior
parameter_sets = np.loadtxt('posterior_2spore.txt')[::int(1000/n),:]
#transform back the parameters that have been log-transformed
parameter_sets[:,1:] = pow(10,parameter_sets[:,1:])
#make lists of the rupture size distributions
#and expected rupture size for these parameters
ruptdists = []
averages = []
for i in range(n):
        #take a parameter set from the posterior sample
        params = parameter_sets[i,:]
        #calculate the rupture size distribution
        ruptdists.append(rupt_dist(params))
        #calculate the expected rupture size
        averages.append(expected_size(params))

#function that takes a value of M and returns vectors for
#p values, phi values, and the corresponding average rupture size
#to create the scatter plot with
def parameter_combinations(M):
    all_phi_vals = []
    all_p_vals = []
    average_rupture_sizes = []
    #add the possible combinations of p and phi to the lists
    #for each parameter set
    for i in range(n):
        #take the correct rupture size distribution
        #and expected rupture size from the lists
        ruptdist = ruptdists[i]
        average = averages[i]
        #find a corresponding value of phi for each p value
        for p in p_vals:
            phi = rstar/(a(p,M,ruptdist))
            #only add this parameter combination to the list
            #if phi is in a realistic range
            if 0.0001<phi<0.5:
                all_p_vals.append(p)
                all_phi_vals.append(phi)
                average_rupture_sizes.append(average)
    #turn the lists into arrays so they can be used in the scatter plot
    all_phi_vals = np.array(all_phi_vals)
    all_p_vals = np.array(all_p_vals)
    average_rupture_sizes = np.array(average_rupture_sizes)
    return all_phi_vals, all_p_vals, average_rupture_sizes

f1, ax1 = plt.subplots(1,3,figsize=(20,4))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.24, wspace=0.2, hspace=0.4)

comb = parameter_combinations(10)
plt.subplot(1,3,1)
cm = plt.cm.get_cmap('RdYlBu_r')
z = np.log10(comb[2])
plt.scatter(comb[0], comb[1], c=z, vmin=min(z), vmax=max(z), s=35, cmap=cm)
plt.xlim(10**(-4)*0.8,1)
plt.gca().set_xscale("log")
plt.ylabel('$p$')
plt.xlabel('$\phi$')
plt.title('M=10')

comb = parameter_combinations(15)
plt.subplot(1,3,2)
z = np.log10(comb[2])
plt.scatter(comb[0], comb[1], c=z, vmin=min(z), vmax=max(z), s=35, cmap=cm)
plt.xlim(10**(-4)*0.8,1)
plt.gca().set_xscale("log")
plt.xlabel('$\phi$')
plt.title('M=15')

comb = parameter_combinations(50)
plt.subplot(1,3,3)
z = np.log10(comb[2])
plt.scatter(comb[0], comb[1], c=z, vmin=min(z), vmax=max(z), s=35, cmap=cm)
plt.ylim(0,1.05)
plt.xlim(10**(-4)*0.8,1)
plt.gca().set_xscale("log")
cbar = plt.colorbar()
cbar.set_label('$log_{10}$(Average rupture size)', rotation=270, labelpad=25)
plt.xlabel('$\phi$')
plt.title('M=50')
plt.savefig("dose_response_params.png",bbox_inches='tight')

#################################################################
'''creating the plot of the exponential dose-response curve
fitted to the Altboum data'''

#Altboum dose-response data
alt_doses = np.array([2*10**2,2*10**3,2*10**4,2*10**5,2*10**6,2*10**7])
alt_probs = np.array([0,0,0.5,10/12,0.875,1])

#doses for curve
doses = np.logspace(2,8,50)

#plotting exponential dose response curve for the best r value
#compared to Altboum data
labels = ['$3x10^{-6}$','$3x10^{-5}$','$3.31x10^{-5}$','$3x10^{-4}$']
plt.figure()
plt.scatter(alt_doses,alt_probs)
plt.plot(doses,1-np.exp(-rstar*doses))
plt.xticks([100,1000,10**4,10**5,10**6,10**7],labels=['$10^2$','$10^3$','$10^4$','$10^5$','$10^6$','$10^7$'])
plt.gca().set_xscale("log")
plt.xlabel('Inhaled dose of spores')
plt.ylabel('Probability of infection')
plt.savefig('r_curve.png', bbox_inches='tight')