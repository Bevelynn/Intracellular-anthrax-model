# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:10:14 2020

@author: 44794
"""

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'axes.labelsize': 'large'})

#define best parameter values
e = 0.7788462481219255        #propbability that a given spore is type A
gA = 0.8942735777721504       #germination rate of spores of type A
gB = 0.04679394310233008      #germination rate of spores of type B
mut = 0.003502002100322627    #death rate of newly germinated bacteria
lamb = 0.6431108733101802     #replication rate of vegetative bacteria
mu = 1.6379885584258536       #death rate of vegetative bacteria
gamma = 0.04379217045983538   #rupture rate

#load the posterior sample, ordered from largest distance to smallest distance
parameters = np.loadtxt('posterior_2spore.txt')
#transform back the parameters that have been log-transformed
parameters[:,1:] = pow(10,parameters[:,1:])

#############################################################
#probabilities of rupture and recovery

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

#calculate the probability of rupture/recovery for all posterior samples
rupt_probs = [prob_rupt(parameters[i,:]) for i in range(1000)]
recover_probs = np.ones(1000)-np.array(rupt_probs)

#the probability of rupture for the best parameter set
rupt_prob = prob_rupt(parameters[-1,:])
print('probability of rupture =', rupt_prob)

########################################################
#mean time until rupture

#restricted mean time until rupture, given germination rate g
def tau1SR(g,mut,lamb,mu,gamma):
    a = (lamb+mu+gamma-sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    b = (lamb+mu+gamma+sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return g/(g+mut)*(1/lamb*np.log((b-a)/(b-1))+(mut+2*g)*(1-a)/(g*(g+mut)))

#conditional mean time until rupture for the parameter set 'params'
def cond_rupt_time(params):
    e = params[0]
    gA = params[1]
    gB = params[2]
    mut = params[3]
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    return (e*tau1SR(gA,mut,lamb,mu,gamma)+(1-e)*tau1SR(gB,mut,lamb,mu,gamma))/prob_rupt(params)

#calculate the conditional mean time until rupture for all posterior samples
rupt_times = [cond_rupt_time(parameters[i,:]) for i in range(1000)]

#the conditional mean rupture time for the best parameter set
rupt_time = cond_rupt_time(parameters[-1,:])
print('conditional mean time until rupture =', rupt_time)

########################################################
#mean time until recovery

#restricted mean time until recovery, given germination rate g
def tau1S0(g,mut,lamb,mu,gamma):
    a = (lamb+mu+gamma-sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    b = (lamb+mu+gamma+sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return g/(g+mut)*(1/lamb*np.log((b)/(b-a))+(mut+2*g)*(g*a+mut)/(g**2*(g+mut)))

#conditional mean time until recovery for the parameter set 'params'
def cond_rec_time(params):
    e = params[0]
    gA = params[1]
    gB = params[2]
    mut = params[3]
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    return (e*tau1S0(gA,mut,lamb,mu,gamma)+(1-e)*tau1S0(gB,mut,lamb,mu,gamma))/(1-prob_rupt(params))

#the conditional mean recovery time for the best parameter set
rec_time = cond_rec_time(parameters[-1,:])
print('conditional mean time until recovery=',rec_time)

########################################################
#rupture size distrubution

#probability of rupture size n>=1, starting from state 1
def R1n(n,lamb,mu,gamma):
    a = (lamb+mu+gamma-sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    b = (lamb+mu+gamma+sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return (b-1)*(1-a)/b**n

#probability of rupture size n>=1, starting from a spore with germination rate 'germ'
def R1Sn(germ,n,mut,lamb,mu,gamma):
    return (germ/(germ+mut))*R1n(n,lamb,mu,gamma)
    
#rupture size distribution for the parameter set 'params'
def ruptdist(params):
    e = params[0]
    gA = params[1]
    gB = params[2]
    mut = params[3]
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    ruptdist = []
    n = 0
    prob = 1-prob_rupt(params)
    while prob>0.0001:
        ruptdist.append(prob)
        n+=1
        prob = e*R1Sn(gA,n,mut,lamb,mu,gamma)+(1-e)*R1Sn(gB,n,mut,lamb,mu,gamma)
    return ruptdist

#expected rupture size for the parameter set 'params'
def expected_size(params):
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    b = (lamb+mu+gamma+sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return prob_rupt(params)*b/(b-1)

#conditional expected rupture size for the parameter set 'params'
def expected_size_conditional(params):
    lamb = params[4]
    mu = params[5]
    gamma = params[6]
    b = (lamb+mu+gamma+sqrt((lamb+mu+gamma)**2-(4*lamb*mu)))/(2*lamb)
    return b/(b-1)

#calculate the conditional mean rupture size for all posterior samples
average_sizes = [expected_size_conditional(parameters[i,:]) for i in range(1000)]

#expected rupture size for the best parameter set
average_size = expected_size(parameters[-1,:])
print('average rupture size =', average_size)

#conditional expected rupture size for the best parameter set
average_size = expected_size_conditional(parameters[-1,:])
print('conditional average rupture size =', average_size)

#rupture size distribution for the best parameter set
rupt_dist = ruptdist(parameters[-1,:])

#conditioned rupture size distribution for the best parameter set
cond_ruptdist = np.array(rupt_dist[1:])/prob_rupt(parameters[-1,:])

###############################################################
#plots

max_n = len(rupt_dist)-1

plt.figure()
plt.bar([0],rupt_dist[0], width=0.85, color='lightsteelblue')
plt.bar(np.arange(1,max_n+1,1),rupt_dist[1:max_n+1], width=0.85, color='tomato')
plt.ylabel('Probability')
plt.xlabel('Number of bacteria released')
plt.title('Rupture size distribution')
plt.savefig("rupturesizedistribution.png",bbox_inches='tight')

ticks = range(1,6)
max_n = 5 #choose what you want the bars to go up to
plt.figure(figsize=(3.3,2))
plt.bar(np.arange(1,max_n+1,1),cond_ruptdist[:max_n], width=0.85, color='tomato')
plt.title('Conditioned on rupture')
plt.xticks(ticks)
plt.savefig("rupturesizedistribution_conditional.png",bbox_inches='tight')


plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
z = rupt_times
plt.scatter(rupt_probs, average_sizes, c=z, vmin=min(z), vmax=max(z), s=35, cmap=cm)
ine1 = plt.axvline(rupt_prob,color="firebrick")
line2 = plt.axhline(average_size,color="firebrick")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.ylim(min(average_sizes)*0.8,max(average_sizes)*1.5)
plt.xlim(min(rupt_probs)*0.8,1)
cbar = plt.colorbar()
cbar.set_label('Conditional mean time \n until rupture (hours)', rotation=270, labelpad=40)
plt.ylabel('Conditional average rupture size')
plt.xlabel('Probability of rupture')
plt.savefig("prob_against_mean.png",bbox_inches='tight')