## Description of .txt files: 
 
### "posterior_norm.txt":  
Required for the file "predictions_norm.py". Contains lists of parameter values from the posterior distribution for the model with truncated normal distribution of germination rate, ordered from largest to smallest distance.  
Column 1: A list of values for log_{10}\mu_g.  
Column 2: A list of values for log_{10}\sigma_g.  
Column 3: A list of values for log_{10}\tilde\mu.  
Column 4: A list of values for log_{10}\lambda.  
Column 5: A list of values for log_{10}\mu.  
Column 6: A list of values for log_{10}\gamma.  

### "posterior_2spore.txt":  
Required for the files "predictions_2spore.py", "results_plots_fig7.py", and "discussion_plots.py". Contains lists of parameter values from the posterior distribution for the model with two types of spores, ordered from largest to smallest distance.  
Column 1: A list of values for epsilon.  
Column 2: A list of values for log_{10}g_A.  
Column 3: A list of values for log_{10}g_B.  
Column 4: A list of values for log_{10}\tilde\mu.  
Column 5: A list of values for log_{10}\lambda.  
Column 6: A list of values for log_{10}\mu.  
Column 7: A list of values for log_{10}\gamma.  
          
## Description of .py files

### "predictions_norm.py":  
Uses the posterior distribution in "posterior_norm.txt" to calculate the prediction of the model with truncated normal distribution of the germination rate for the parameter set with the smallest distance, and the pointwise 95% credible intervals of the predictions using all posterior parameter sets. These are used to reproduce the top plot of Figure 4.

### "predictions_2spore.py":  
Uses the posterior distribution in "posterior_2spore.txt" to calculate the prediction of the model with two types of spores for the parameter set with the smallest distance, and the pointwise 95% credible intervals of the predictions using all posterior parameter sets. These are used to reproduce the bottom plot of Figure 4.

### "results_plots_fig5.py":  
Simulates realisations of the model with two types of spores for multiple infected cells, using the parameter set with the smallest distance from the data. The mean of the different populations in the model over time are also calculated. The mean population sizes and the output of a stochastic simulation is plotted for initial conditions of 30500 and 100 infected cells, creating Figure 5.

### "results_plots_fig6.py":  
Simulates realisations of the model with two types of spores, using the parameter set with the smallest distance from the data. Calculates the probability density functions of time until rupture and recovery. Creates the plot in Figure 6.

### "results_plots_fig7.py":  
Calculates the probability of rupture, conditional mean time until rupture, rupture size distribution, and average rupture size, for each parameter set in the posterior distribution in "posterior_2spore.txt". Reproduces Figure 7.

### "discussion_plots.py":  
Reproduces Figures 8 and 10.
