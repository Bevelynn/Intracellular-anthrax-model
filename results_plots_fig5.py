# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:10:14 2020

@author: 44794
"""

import matplotlib.pyplot as plt
import random
import math
import numpy as np
from numpy import searchsorted as mychoose
from scipy.signal import fftconvolve

plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'axes.labelsize': 'large'})

##########################################################
'''a stochastic simulation of many infected cells at once
for the two types of spores model'''

#define the events that can occur to a cell
#s = number of intracellular spores
#n = number of intracellular newly germinated bacteria
#b = number of intracellular vegetative bacteria
#c = 1 if the cell is alive, and c = 0 otherwise

def germination(s,n,b,c):
    return (s-1,n+1,b,c)  

def maturation(s,n,b,c):
    return (s,n-1,b+1,c)

def death_ngb(s,n,b,c):
    return (s,n-1,b,c)

def birth(s,n,b,c):
    return (s,n,b+1,c)

def death(s,n,b,c):
    return (s,n,b-1,c)

def rupture(s,n,b,c):
    return (0,0,0,0)

#time-course for the simulation
sim_times = np.array([0,2,4,6,8,10,15,20,25,30,35,40,45,50])

#function to run a simulation for an intial number of infected cells/ intracellular spores given by 'spores'
#outputs arrays for the total number of spores, newly germinated bacteria, and vegetative bacteria inside cells at each of the time points in 'sim_times'
#arising from each of the two types of spores.
#also outputs the list of times when a cell ruptures
def simulation(e,gA,gB,mut,lamb,mu,gamma,spores):
    #initial number of cells, each containing one spore
    x0 = spores
    #initialise time-courses for number of spores, ngb, and vegetative bacteria
    #for each of the two types of spores
    spAstar = np.zeros(len(sim_times))
    spBstar = np.zeros(len(sim_times))
    ngbAstar = np.zeros(len(sim_times))
    ngbBstar = np.zeros(len(sim_times))
    bacAstar = np.zeros(len(sim_times))
    bacBstar = np.zeros(len(sim_times))
    #initialise a list for the rupture times
    rupture_times = [0]
    events = {0:germination, 1:maturation, 2:death_ngb, 3:birth, 4:death, 5:rupture}
    #simulate each of the initial x0 cells
    for _ in range(x0):
        #germination rate is gA with probability e and gB otherwise
        g = random.choices(population=[gA,gB],weights=[e,1-e],k=1)[0]
        #set initial time to 0
        t = 0
        #initial number of intracellular spores
        s = 1
        #initial number of intracellular newly germinated bacteria
        n = 0
        #initial number of intracellular vegetative bacteria
        b = 0
        #cell is initially alive
        c = 1
        #initialise lists for the number in each population
        #and the transition times
        S = []
        N = []
        B = []
        T = []
        S.append(s)
        N.append(n)
        B.append(b)
        T.append(t)
        #maximum number of hours to run the simulation for
        tmax = max(sim_times)
        #run the simulation until the cell has either ruptured, recovered, or the time is over tmax
        while (s+n+b)!=0 and t<tmax:
            S.append(s)
            N.append(n)
            B.append(b)
            #calculate the rates
            grate = g*s
            mrate = g*n
            drate = mut*n
            uprate = lamb*b
            downrate = mu*b
            rupturerate = gamma*b
            rates = [grate,mrate,drate,uprate,downrate,rupturerate]
            totalrate = sum(rates)
            urv = random.random()
            #choose which event occurs
            thisevent = mychoose(np.cumsum(rates),urv*totalrate)
            #update the number in each population
            X = events[thisevent](s,n,b,c)
            s = X[0]
            n = X[1]
            b = X[2]
            c = X[3]
            t+=-math.log(random.random())/totalrate
            if c==0:
                rupture_times.append(t)
            T.append(t)            
        S.append(s)
        N.append(n)
        B.append(b)
        T.append(t)
        #find the number in each population in the cell at each of the specified time points and add it on to the total at that time point
        for i in range(len(sim_times)):    
            position = mychoose(np.array(T), sim_times[i])
            if position<len(T):
                if g==gA:
                    spAstar[i]+=S[position]
                    ngbAstar[i]+=N[position]
                    bacAstar[i]+=B[position]
                if g==gB:
                    spBstar[i]+=S[position]
                    ngbBstar[i]+=N[position]
                    bacBstar[i]+=B[position]
                
    return spAstar,spBstar,ngbAstar,ngbBstar,bacAstar,bacBstar,rupture_times

#define the best parameter values
e = 0.7788462481219255        #propbability that a given spore is type A
gA = 0.8942735777721504       #germination rate of spores of type A
gB = 0.04679394310233008      #germination rate of spores of type B
mut = 0.003502002100322627    #death rate of newly germinated bacteria
lamb = 0.6431108733101802     #replication rate of vegetative bacteria
mu = 1.6379885584258536       #death rate of vegetative bacteria
gamma = 0.04379217045983538   #rupture rate

#simulation for 30500 initial spores
sim = simulation(e,gA,gB,mut,lamb,mu,gamma,30500)

spA = sim[0]
spB = sim[1]
ngbA = sim[2]
ngbB = sim[3]
bacA = sim[4]
bacB = sim[5]
sp = spA+spB
ngb = ngbA+ngbB
bac = bacA+bacB

np.savetxt('simsA.txt',spA)
np.savetxt('simsB.txt',spB)
np.savetxt('simnA.txt',ngbA)
np.savetxt('simnB.txt',ngbB)
np.savetxt('simbA.txt',bacA)
np.savetxt('simbB.txt',bacB)
np.savetxt('sims.txt',sp)
np.savetxt('simn.txt',ngb)
np.savetxt('simb.txt',bac)

#simulation for 100 initial spores
sim = simulation(e,gA,gB,mut,lamb,mu,gamma,100)

spA = sim[0]
spB = sim[1]
ngbA = sim[2]
ngbB = sim[3]
bacA = sim[4]
bacB = sim[5]
sp = spA+spB
ngb = ngbA+ngbB
bac = bacA+bacB

np.savetxt('simsA2.txt',spA)
np.savetxt('simsB2.txt',spB)
np.savetxt('simnA2.txt',ngbA)
np.savetxt('simnB2.txt',ngbB)
np.savetxt('simbA2.txt',bacA)
np.savetxt('simbB2.txt',bacB)
np.savetxt('sims2.txt',sp)
np.savetxt('simn2.txt',ngb)
np.savetxt('simb2.txt',bac)

##############################################################
'''functions to analytically calculate the mean population sizes'''

#probability that the process is in state 1_S at time t
#given germination rate g
def p1S(t,g):
    return np.exp(-g*t)

#probability that the process is in state 1_NGB at time t
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

#mean number of vegetative bacteria in a cell over time, given germination rate g
def Bg(times,g,mut,lamb,mu,gamma):
    #gives a vector of [B(t;g) for t in times]
    #stepsize for convolution
    h = 0.01
    tmax = max(times)
    tt = np.linspace(0,tmax,int(tmax/h)+1)
    a = ((lamb+mu+gamma)-np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    b = ((lamb+mu+gamma)+np.sqrt((lamb+mu+gamma)**2-4*mu*lamb))/(2*lamb)
    ff1S1 = [f1S1(t,g,mut) for t in tt]
    ff1R = [f1R(t,a,b,lamb,gamma) for t in tt]
    f1SR = (fftconvolve(ff1S1,ff1R)*h)[:len(tt)]
    B = np.zeros(len(times))
    for i,t in enumerate(times): 
        B[i]=f1SR[int(t/h)]/gamma
    return B

###############################################################
#define time points to calculate the above functions for
CItimes = np.linspace(0,50,101)
times = np.linspace(0,50,101)

##############################################################
#finding 95% CIs of the predictions

posterior = np.loadtxt('posterior_2spore.txt')

l = len(times) # l is number of time points you have
SpA = np.zeros((1000,l))
SpB = np.zeros((1000,l))
NgbA = np.zeros((1000,l))
NgbB = np.zeros((1000,l))
BacA = np.zeros((1000,l))
BacB = np.zeros((1000,l))
Sp = np.zeros((1000,l))
Ngb = np.zeros((1000,l))
Bac = np.zeros((1000,l))
for i in range(1000):
   e = posterior[i,0]
   gA = pow(10,posterior[i,1])
   gB = pow(10,posterior[i,2])
   mut = pow(10,posterior[i,3])
   lamb = pow(10,posterior[i,4])
   mu = pow(10,posterior[i,5])
   gamma = pow(10,posterior[i,6])
   spA=e*p1S(times,gA)
   spB=(1-e)*p1S(times,gB)
   ngbA=e*p1NGB(times,gA,mut)
   ngbB=(1-e)*p1NGB(times,gB,mut)
   bacA=e*Bg(times,gA,mut,lamb,mu,gamma)
   bacB=(1-e)*Bg(times,gB,mut,lamb,mu,gamma)
   SpA[i,]=spA
   SpB[i,]=spB
   NgbA[i,]=ngbA
   NgbB[i,]=ngbB
   BacA[i,]=bacA
   BacB[i,]=bacB
   Sp[i,]=spA+spB
   Ngb[i,]=ngbA+ngbB
   Bac[i,]=bacA+bacB

#Pointwise credible intervals
lowwsA = []
highhsA = []
lowwsB = []
highhsB = []
lowwnA = []
highhnA = []
lowwnB = []
highhnB = []
lowwbA = []
highhbA = []
lowwbB = []
highhbB = []
lowws = []
highhs = []
lowwn = []
highhn = []
lowwb = []
highhb = []
for i in range(l):
   lowsA = np.percentile(SpA[:,i],2.5)
   highsA = np.percentile(SpA[:,i],97.5)
   lowsB = np.percentile(SpB[:,i],2.5)
   highsB = np.percentile(SpB[:,i],97.5)
   lownA = np.percentile(NgbA[:,i],2.5)
   highnA = np.percentile(NgbA[:,i],97.5)
   lownB = np.percentile(NgbB[:,i],2.5)
   highnB = np.percentile(NgbB[:,i],97.5)
   lowbA = np.percentile(BacA[:,i],2.5)
   highbA = np.percentile(BacA[:,i],97.5)
   lowbB = np.percentile(BacB[:,i],2.5)
   highbB = np.percentile(BacB[:,i],97.5)
   lows = np.percentile(Sp[:,i],2.5)
   highs = np.percentile(Sp[:,i],97.5)
   lown = np.percentile(Ngb[:,i],2.5)
   highn = np.percentile(Ngb[:,i],97.5)
   lowb = np.percentile(Bac[:,i],2.5)
   highb = np.percentile(Bac[:,i],97.5)
   lowwsA.append(lowsA)
   highhsA.append(highsA)
   lowwsB.append(lowsB)
   highhsB.append(highsB)
   lowwnA.append(lownA)
   highhnA.append(highnA)
   lowwnB.append(lownB)
   highhnB.append(highnB)
   lowwbA.append(lowbA)
   highhbA.append(highbA)
   lowwbB.append(lowbB)
   highhbB.append(highbB)
   lowws.append(lows)
   highhs.append(highs)
   lowwn.append(lown)
   highhn.append(highn)
   lowwb.append(lowb)
   highhb.append(highb)

lowsA_final = np.hstack(lowwsA)
highsA_final = np.hstack(highhsA)
lowsB_final = np.hstack(lowwsB)
highsB_final = np.hstack(highhsB)
lownA_final = np.hstack(lowwnA)
highnA_final = np.hstack(highhnA)
lownB_final = np.hstack(lowwnB)
highnB_final = np.hstack(highhnB)
lowbA_final = np.hstack(lowwbA)
highbA_final = np.hstack(highhbA)
lowbB_final = np.hstack(lowwbB)
highbB_final = np.hstack(highhbB)
lows_final = np.hstack(lowws)
highs_final = np.hstack(highhs)
lown_final = np.hstack(lowwn)
highn_final = np.hstack(highhn)
lowb_final = np.hstack(lowwb)
highb_final = np.hstack(highhb)

np.savetxt('lowsA.txt',lowsA_final)
np.savetxt('highsA.txt',highsA_final)
np.savetxt('lowsB.txt',lowsB_final)
np.savetxt('highsB.txt',highsB_final)
np.savetxt('lownA.txt',lownA_final)
np.savetxt('highnA.txt',highnA_final)
np.savetxt('lownB.txt',lownB_final)
np.savetxt('highnB.txt',highnB_final)
np.savetxt('lowbA.txt',lowbA_final)
np.savetxt('highbA.txt',highbA_final)
np.savetxt('lowbB.txt',lowbB_final)
np.savetxt('highbB.txt',highbB_final)
np.savetxt('lows.txt',lows_final)
np.savetxt('highs.txt',highs_final)
np.savetxt('lown.txt',lown_final)
np.savetxt('highn.txt',highn_final)
np.savetxt('lowb.txt',lowb_final)
np.savetxt('highb.txt',highb_final)

###############################################################
#calculating and plotting the best predictions, 95% CIs, and simulations for S_0=30500

initial_cells = 30500

#best predictions
spA = e*initial_cells*p1S(times,gA)
spB = (1-e)*initial_cells*p1S(times,gB)
ngbA = e*initial_cells*p1NGB(times,gA,mut)
ngbB = (1-e)*initial_cells*p1NGB(times,gB,mut)
bacA = e*initial_cells*Bg(times,gA,mut,lamb,mu,gamma)
bacB = (1-e)*initial_cells*Bg(times,gB,mut,lamb,mu,gamma)

#95% CIs
lowsA = initial_cells*np.loadtxt('lowsA.txt')
highsA = initial_cells*np.loadtxt('highsA.txt')
lowsB = initial_cells*np.loadtxt('lowsB.txt')
highsB = initial_cells*np.loadtxt('highsB.txt')
lownA = initial_cells*np.loadtxt('lownA.txt')
highnA = initial_cells*np.loadtxt('highnA.txt')
lownB = initial_cells*np.loadtxt('lownB.txt')
highnB = initial_cells*np.loadtxt('highnB.txt')
lowbA = initial_cells*np.loadtxt('lowbA.txt')
highbA = initial_cells*np.loadtxt('highbA.txt')
lowbB = initial_cells*np.loadtxt('lowbB.txt')
highbB = initial_cells*np.loadtxt('highbB.txt')
lows = initial_cells*np.loadtxt('lows.txt')
highs = initial_cells*np.loadtxt('highs.txt')
lown = initial_cells*np.loadtxt('lown.txt')
highn = initial_cells*np.loadtxt('highn.txt')
lowb = initial_cells*np.loadtxt('lowb.txt')
highb = initial_cells*np.loadtxt('highb.txt')

#simulations
simsA = np.loadtxt('simsA.txt')
simsB = np.loadtxt('simsB.txt')
simnA = np.loadtxt('simnA.txt')
simnB = np.loadtxt('simnB.txt')
simbA = np.loadtxt('simbA.txt')
simbB = np.loadtxt('simbB.txt')
sims = np.loadtxt('sims.txt')
simn = np.loadtxt('simn.txt')
simb = np.loadtxt('simb.txt')


f1, ax1 = plt.subplots(2,3,figsize=(20,8))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.24, wspace=None, hspace=0.2)

plt.subplot(231)
plt.plot(times, spA, label='Intracellular spores of type A')
plt.plot(times, ngbA, label='Intracellular NGB of type A')
plt.plot(times, bacA, label='Intracellular vegetative bacteria \n arising from type A')
plt.fill_between(CItimes, highsA, lowsA, alpha=0.3)
plt.fill_between(CItimes, highnA, lownA, alpha=0.3)
plt.fill_between(CItimes, highbA, lowbA, alpha=0.3)
plt.scatter(sim_times, simsA, color='blue')
plt.scatter(sim_times, simnA, color='orange')
plt.scatter(sim_times, simbA, color='green')
plt.xlabel('Time (hours)')
plt.ylabel('Number of intracellular components')
plt.legend()

plt.subplot(232)
plt.plot(times,spB,label='Intracellular spores of type B')
plt.plot(times,ngbB,label='Intracellular NGB of type B')
plt.plot(times,bacB, label='Intracellular vegetative bacteria \n arising from type B')
plt.fill_between(CItimes, highsB, lowsB, alpha=0.3)
plt.fill_between(CItimes, highnB, lownB, alpha=0.3)
plt.fill_between(CItimes, highbB, lowbB, alpha=0.3)
plt.scatter(sim_times,simsB, color='blue')
plt.scatter(sim_times,simnB, color='orange')
plt.scatter(sim_times,simbB, color='green')
plt.xlabel('Time (hours)')
plt.legend()

plt.subplot(233)
plt.plot(times,spA+spB, label='Total intracellular spores')
plt.plot(times,ngbA+ngbB, label='Total intracellular NGB')
plt.plot(times,bacA+bacB, label='Total intracellular vegetative bacteria')
plt.fill_between(CItimes, highs, lows, alpha=0.3)
plt.fill_between(CItimes, highn, lown, alpha=0.3)
plt.fill_between(CItimes, highb, lowb, alpha=0.3)
plt.scatter(sim_times,sims, color='blue')
plt.scatter(sim_times,simn, color='orange')
plt.scatter(sim_times,simb, color='green')
plt.xlabel('Time (hours)')
plt.legend()

###############################################################
#calculating and plotting the best predictions, 95% CIs, and simulations for S_0=30500

initial_cells = 100

#best predictions
spA = e*initial_cells*p1S(times,gA)
spB = (1-e)*initial_cells*p1S(times,gB)
ngbA = e*initial_cells*p1NGB(times,gA,mut)
ngbB = (1-e)*initial_cells*p1NGB(times,gB,mut)
bacA = e*initial_cells*Bg(times,gA,mut,lamb,mu,gamma)
bacB = (1-e)*initial_cells*Bg(times,gB,mut,lamb,mu,gamma)

#95% CIs
lowsA = initial_cells*np.loadtxt('lowsA.txt')
highsA = initial_cells*np.loadtxt('highsA.txt')
lowsB = initial_cells*np.loadtxt('lowsB.txt')
highsB = initial_cells*np.loadtxt('highsB.txt')
lownA = initial_cells*np.loadtxt('lownA.txt')
highnA = initial_cells*np.loadtxt('highnA.txt')
lownB = initial_cells*np.loadtxt('lownB.txt')
highnB = initial_cells*np.loadtxt('highnB.txt')
lowbA = initial_cells*np.loadtxt('lowbA.txt')
highbA = initial_cells*np.loadtxt('highbA.txt')
lowbB = initial_cells*np.loadtxt('lowbB.txt')
highbB = initial_cells*np.loadtxt('highbB.txt')
lows = initial_cells*np.loadtxt('lows.txt')
highs = initial_cells*np.loadtxt('highs.txt')
lown = initial_cells*np.loadtxt('lown.txt')
highn = initial_cells*np.loadtxt('highn.txt')
lowb = initial_cells*np.loadtxt('lowb.txt')
highb = initial_cells*np.loadtxt('highb.txt')

simsA = np.loadtxt('simsA2.txt')
simsB = np.loadtxt('simsB2.txt')
simnA = np.loadtxt('simnA2.txt')
simnB = np.loadtxt('simnB2.txt')
simbA = np.loadtxt('simbA2.txt')
simbB = np.loadtxt('simbB2.txt')
sims = np.loadtxt('sims2.txt')
simn = np.loadtxt('simn2.txt')
simb = np.loadtxt('simb2.txt')

plt.subplot(234)
plt.plot(times,spA, label='Intracellular spores of type A')
plt.plot(times,ngbA,label='Intracellular NGB of type A')
plt.plot(times,bacA, label='Intracellular vegetative bacteria \n arising from type A')
plt.fill_between(CItimes, highsA, lowsA, alpha=0.3)
plt.fill_between(CItimes, highnA, lownA, alpha=0.3)
plt.fill_between(CItimes, highbA, lowbA, alpha=0.3)
plt.scatter(sim_times,simsA, color='blue')
plt.scatter(sim_times,simnA, color='orange')
plt.scatter(sim_times,simbA, color='green')
plt.ylabel('Number of intracellular components')
plt.xlabel('Time (hours)')

plt.subplot(235)
plt.plot(times,spB,label='Intracellular spores of type B')
plt.plot(times,ngbB,label='Intracellular NGB of type B')
plt.plot(times,bacB, label='Intracellular vegetative bacteria \n arising from type B')
plt.fill_between(CItimes, highsB, lowsB, alpha=0.3)
plt.fill_between(CItimes, highnB, lownB, alpha=0.3)
plt.fill_between(CItimes, highbB, lowbB, alpha=0.3)
plt.scatter(sim_times,simsB, color='blue')
plt.scatter(sim_times,simnB, color='orange')
plt.scatter(sim_times,simbB, color='green')
plt.xlabel('Time (hours)')

plt.subplot(236)
plt.plot(times,spA+spB, label='Total intracellular spores')
plt.plot(times,ngbA+ngbB, label='Total intracellular NGB')
plt.plot(times,bacA+bacB, label='Total intracellular vegetative bacteria')
plt.fill_between(CItimes, highs, lows, alpha=0.3)
plt.fill_between(CItimes, highn, lown, alpha=0.3)
plt.fill_between(CItimes, highb, lowb, alpha=0.3)
plt.scatter(sim_times,sims, color='blue')
plt.scatter(sim_times,simn, color='orange')
plt.scatter(sim_times,simb, color='green')
plt.xlabel('Time (hours)')
plt.savefig('population_plots.png', bbox_inches='tight')