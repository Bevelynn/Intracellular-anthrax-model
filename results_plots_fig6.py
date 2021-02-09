# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:10:14 2020

@author: 44794
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
import random
from numpy import searchsorted as mychoose
import math
from statistics import mean

plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'axes.labelsize': 'large'})

##########################################################
'''running many stochastic simulations of
the two types of spores model'''

#define events that can occur to the state
#of the process, x

def germination(x):
    #germination of spore to newly germinated bacteria
    return '$1_{NGB}$'
    
def maturation(x):
    #maturation of newly germinated bacteria to vegetative bacteria
    return 1

def death_ngb(x):
    #death of newly germinated bacteria
    return 0

def birth(x):
    #replication of vegetative bacteria
    return x+1

def death_bac(x):
    #death of vegetative bacteria
    return x-1

def rupture(x):
    #rupture of the cell
    return 'R'

#define best parameter values
e = 0.7788462481219255        #propbability that a given spore is type A
gA = 0.8942735777721504       #germination rate of spores of type A
gB = 0.04679394310233008      #germination rate of spores of type B
mut = 0.003502002100322627    #death rate of newly germinated bacteria
lamb = 0.6431108733101802     #replication rate of vegetative bacteria
mu = 1.6379885584258536       #death rate of vegetative bacteria
gamma = 0.04379217045983538   #rupture rate


#set the number of simulations to do
numruns = 10**6
events = {0:germination,1:maturation, 2:death_ngb, 3:birth, 4:death_bac, 5:rupture}
#initial counters for the number of ruptures and recoveries, and the number of spores of type A/B
number_of_ruptures = 0
number_of_recoveries = 0
number_of_type_A = 0
number_of_type_B = 0
#initiate lists for the time until rupture or recovery and rupture size
time_until_rupture = []
time_until_recovery = []
rupture_size = []
maxstates = []
#run the simulation many times
for _ in range(numruns):
    #the spore has germination rate gA with probability e, and germination rate gB otherwise
    u=random.random()
    if u < e:
        g = gA
        number_of_type_A+=1
    else:
        g = gB
        number_of_type_B+=1
    #time is set to 0
    t = 0
    #initial state of the process is 1 spore
    x = '$1_S$'
    #keep track of the maximum state reached, m
    m = 1
    #initiate lists for the states and transition times
    X = []
    T = []
    X.append(0)
    T.append(t)
    #run the gillespie simulation until the process reaches one of the absorbing states 
    while x!=0 and x!='R':
        X.append(x)
        #calculate the transition rates out of state x
        if x=='$1_S$':
            grate = g #germination rate
            mrate = 0 #maturation rate
            drate = 0 #deathrate of ngb
            uprate = 0 #replication rate of vegetative bacteria
            downrate = 0 #death rate of vegetative bacteria
            ruptrate = 0 #rupture rate
        elif x=='$1_{NGB}$':
            grate = 0
            mrate = g
            drate = mut
            uprate = 0
            downrate = 0
            ruptrate = 0
        else:
            if x>m:
                m = x
            grate = 0
            mrate = 0
            drate = 0
            uprate = lamb*x
            downrate = mu*x
            ruptrate = gamma*x
        rates = [grate,mrate,drate,uprate,downrate,ruptrate]
        totalrate = sum(rates)
        urv = random.random()
        #choose which event will occur
        thisevent = mychoose(np.cumsum(rates),urv*totalrate)
        #update the state of the process
        x = events[thisevent](x)
        #calculate the time step that the transition took
        t+=-math.log(random.random())/totalrate
        T.append(t)
    X.append(x)
    T.append(t)
    maxstates.append(m)
    if x=='R':
        number_of_ruptures+=1
        time_until_rupture.append(t)
        rupture_size.append(X[-2])
    elif x==0:
        number_of_recoveries+=1
        time_until_recovery.append(t)

#probability of rupture or recovery
print('proportion of ruptures from simulations =',number_of_ruptures/numruns)
print('proportion of recoveries from simulations =',number_of_recoveries/numruns)

#probability of type A or type B should match with value of e
print('proportion of type A from simulations =',number_of_type_A/numruns)
print('proportion of type B from simulations =',number_of_type_B/numruns)
        
#conditional mean times until absorption
print('mean time until rupture from simulations =',mean(time_until_rupture))
print('mean time until recovery from simulations =',mean(time_until_recovery))

#expected rupture size, conditioned on rupture
print('conditional mean rupture size from simulations =',mean(rupture_size))

np.savetxt('sim_rupture_times.txt', np.array(time_until_rupture))
np.savetxt('sim_recovery_times.txt', np.array(time_until_recovery))

#################################################################
'''functions to analytically calculate the densities
of rupture and recovery times'''

#probability that the process is in state of newly germinated bacteria at time t
#given germination rate g
def p1NGB(t,g,mut):
    return g/mut*(np.exp(-g*t)-np.exp(-(g+mut)*t))

#pdf of time to get from state 1_S to state 1, calculated at time t
#given germination rate g
def f1S1(t,g,mut):
    return g*p1NGB(t,g,mut)

#pdf of time to get from state 1 to state R, calculated at time t
def f1R(t,a,b,lamb,gamma):
    return (gamma*(b-a)**2*np.exp(-lamb*(b-a)*t)/(b+(1-a)*np.exp(-lamb*(b-a)*t)-1)**2)

def f1SR(times,g,mut,lamb,mu,gamma):
    #gives a vector of [f_{T_{1_S}^R}(t;g) for t in times]
    h = 0.01 #step size for convolution
    tmax=max(times)
    tt = np.linspace(0,tmax,int(tmax/h)+1)
    a = ((lamb+mu+gamma)-np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    b = ((lamb+mu+gamma)+np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    ff1S1 = [f1S1(t,g,mut) for t in tt]
    ff1R = [f1R(t,a,b,lamb,gamma) for t in tt]
    ff1SR = (fftconvolve(ff1S1,ff1R)*h)[:len(tt)]
    f1SR = np.zeros(len(times))
    for i,t in enumerate(times): 
        f1SR[i] = ff1SR[int(t/h)]
    return f1SR

#pdf of time to get from state 1 to state 0, calculated at time t
def f10(t,a,b,lamb,gamma):
    return (mu*(b-a)**2*np.exp(-lamb*(b-a)*t))/((a*np.exp(-lamb*(b-a)*t)-b)**2)

def f1S0(times,g,mut,lamb,mu,gamma):
    #gives a vector of [f_{T_{1_S}^0}(t;g) for t in times]
    #step size for convolution
    h = 0.01
    tmax = max(times)
    tt = np.linspace(0,tmax,int(tmax/h)+1)
    a = ((lamb+mu+gamma)-np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    b = ((lamb+mu+gamma)+np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    ff1S1 = [f1S1(t,g,mut) for t in tt]
    ff10 = [f10(t,a,b,lamb,gamma) for t in tt]
    ff1S0=(fftconvolve(ff1S1,ff10)*h)[:len(tt)]
    f=np.zeros(len(times))
    for i,t in enumerate(times): 
        f[i] = ff1S0[int(t/h)]+mut*p1NGB(t,g,mut)
    return f
    
#probability of rupture given germination rate g
def prob_rupt_g(g,mut,lamb,mu,gamma):
    a = ((lamb+mu+gamma)-np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    return g*(1-a)/(g+mut)

#time-course to calculate the functions for
times = np.linspace(0,100,501)

#rupture time density if germination rate is gA
fA = f1SR(times,gA,mut,lamb,mu,gamma)

#rupture time density if germination rate is gB
fB = f1SR(times,gB,mut,lamb,mu,gamma)

#overall rupture time density
rupture_time_density = e*fA+(1-e)*fB

#overall probability of rupture
prob_rupt = e*prob_rupt_g(gA,mut,lamb,mu,gamma)+(1-e)*prob_rupt_g(gB,mut,lamb,mu,gamma)

#to get the conditional pdf of rupture time, divide by the rupture probability
cond_rupture_time_density = rupture_time_density/prob_rupt

#recovery time density if germination rate is gA
f0A = f1S0(times,gA,mut,lamb,mu,gamma)

#recovery time density if germination rate is gB
f0B = f1S0(times,gB,mut,lamb,mu,gamma)

#overall recovery time density
recovery_time_density = e*f0A+(1-e)*f0B

#overall probability of recovery
prob_rec=1-prob_rupt

#to get the conditional pdf of recovery time, divide by the recovery probability
cond_recovery_time_density = recovery_time_density/prob_rec

#loading the times until rupture and recovery from 10^6 simulations
time_until_rupture = np.loadtxt('sim_rupture_times.txt')
time_until_recovery = np.loadtxt('sim_recovery_times.txt')

#plots of the different functions
f1, ax1 = plt.subplots(2,4,figsize=(27,8))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.24, wspace=None, hspace=0.2)

plt.subplot(241)
plt.fill_between(times, [0]*len(times), fA, alpha=0.5, label='Rupture time for type A', color='tomato')
plt.xlabel('Time (hours)')
plt.ylabel('Density')
plt.legend()

plt.subplot(242)
plt.fill_between(times, [0]*len(times), fB, alpha=0.5, label='Rupture time for type B', color='darkred')
plt.xlabel('Time (hours)')
plt.legend()

plt.subplot(243)
plt.fill_between(times,e*fA, [0]*len(times), alpha=0.5,color='tomato')
plt.fill_between(times,(1-e)*fB,[0]*len(times), alpha=0.5, color='darkred')
plt.xlabel('Time (hours)')


plt.subplot(244)
plt.hist(x=time_until_rupture,bins=np.linspace(0,50,51), density=bool, alpha=0.3, color='tomato', edgecolor='red')
plt.xlabel('Conditional time until rupture (hours)')
plt.plot(times[:251],cond_rupture_time_density[:251], color='firebrick')

plt.subplot(245)
plt.fill_between(times, [0]*len(times), f0A, alpha=0.5, label='Recovery time for type A', color='cornflowerblue')
plt.xlabel('Time (hours)')
plt.ylabel('Density')
plt.legend()

plt.subplot(246)
plt.fill_between(times, [0]*len(times), f0B, alpha=0.5, label='Recovery time for type B', color='midnightblue')
plt.xlabel('Time (hours)')
plt.legend()

plt.subplot(247)
plt.fill_between(times,e*f0A, [0]*len(times),alpha=0.5,color='cornflowerblue')
plt.fill_between(times,(1-e)*f0B,[0]*len(times), alpha=0.5, color='midnightblue')
plt.xlabel('Time (hours)')

plt.subplot(248)
plt.hist(x=time_until_recovery,bins=np.linspace(0,50,51), density=bool, alpha=0.3, color='cornflowerblue', edgecolor='blue')
plt.xlabel('Conditional time until recovery (hours)')
plt.plot(times[:251],cond_recovery_time_density[:251])

plt.savefig('rupture_recovery_time_densities.png', bbox_inches='tight')
