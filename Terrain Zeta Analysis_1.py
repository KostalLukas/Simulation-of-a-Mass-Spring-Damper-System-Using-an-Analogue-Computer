'''
Analogue Computer Terrain Zeta Analysis

Lukas Kostal, 18.4.2022, ICL
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy as sp
import scipy.constants as sc
import scipy.optimize as op
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import numba
import time

#record the time at which the code starts
starttim = time.time()


#function to implement driving signal as a driving function in the DE
@numba.jit(nopython = True, parallel = False)
def func(t_sim, t, f, diff, s):
    delta =  np.absolute(t - t_sim)
    i = delta.argmin()
    return(s * np.array([f[i], diff[i]]))
        
#define the differential equation (DE) to be solved
@numba.jit
def dSdt(t_sim, S, gma, omg_0, t, f, diff, s):    
    y, v = S
    dvdt = - gma*(v - func(t_sim, t, f, diff, s)[1]) - omg_0**2 * (y - func(t_sim, t, f, diff, s)[0])
    print('t_sim = %.6fs (6dp)' % (t_sim))
    return(v, dvdt)

#define function to calucalte value of gamma from resistance values by averaging
def gamma(R, scale):
    gma = (3.24/R * 4.59e-6 * 5.49e3)*scale/2 + (3.24/(6*R) * 1/(94.57e-6 * 984) * 2.15e3/983)/2
    return(gma)


#specify whether to save output or not
saveoutput = False

#specify the colormap to be used for plotting
cmap = cm.plasma_r

#define physical varibales
freq_0 = 1.18               #resonant frequency in Hz
m = 360                     #mass in kg
h = 0.1                     #peak height of terrain in m
s_internal = 7.209629e-1    #overall scaling of the circuit

#specify the min and max resistance used to set zeta in kOhm
R_min = 0.7
R_max = 4.2

#define paths to the experimental data and LTspice simulations
path_ex = 'Data/Terrain Zeta/'
path_lts = 'LTspice/Terrain Zeta/'

#calcualte the value of omega in rads^-1
omg_0 = 2 * sc.pi * freq_0

#calculate the scaling factor for the input signal
scale = omg_0**2 / (2.14/2 * 3.24/9.86)

#use the min and max resistances to calcualte the min and max values for zeta
#careful R_min --> zeta_max (lowest damping) and R_max --> zeta_min (highest damping)
zeta_max = gamma(R_min, scale) / (2*omg_0)
zeta_min = gamma(R_max, scale) / (2*omg_0) * 0 #set to 0 to rescale and avoid bright colors

#normalization for plotting with the colormap
norm = mcolors.Normalize(vmin=zeta_min, vmax=zeta_max)

#prepare arrays to hold values for zeta in kgs^-1 and amplitudes in m
zeta_arr = np.empty(0)
amp_ex = np.empty(0)
amp_lts = np.empty(0)
amp_num = np.empty(0)

#loop over all of the measured frequencies values of R from 0.7 to 4.2 Ohm
for i in range (0, 14):
    
    #calculate the value of R to be analyzed in Ohm
    R = 0.7 + 0.3*i

    #calculate the value of gamma using R
    gma =  gamma(R, scale)

    #calculate the value of zeta from gamma
    zeta = gma/(2*omg_0)
    
    #determine the filename of the data to be analyzed
    if i<9:
        filename = f'ZTA_0{i+1}'
    else:
        filename = f'ZTA_{i+1}'
    
    #load the experimental data and LTspice simulation
    #units are s, V, V respectively
    t_ex, f_ex, V_ex = np.loadtxt(f'{path_ex}{filename}.CSV', skiprows=(1),delimiter=',', unpack=True)
    t_lts, V_lts, f_lts = np.loadtxt(f'{path_lts}{filename}.txt', skiprows=(1), unpack=True)

    #calculate and normalize the time derivative of the driving signal
    diff = np.diff(f_ex)
    diff = diff/np.amax(diff)
    
    #find the edges in the driving signal as the peaks in the time derivative
    maxima, _ = find_peaks(diff, height=0.1, distance=5000)
    minima, _ = find_peaks(-diff, height=0.1, distance=5000)

    #specify start and end indices based on the found peaks
    start = maxima[0]
    end = maxima[1]

    #crop the experimental data
    t_ex = t_ex[start:end]
    f_ex = f_ex[start:end]
    V_ex = V_ex[start:end]

    #shift the timescale to start at 0s
    t_ex = t_ex - np.amin(t_ex)
    
    #crop the LTspice simulated data
    t_lts = t_lts[t_lts<np.amax(t_ex)]
    f_lts = f_lts[:len(t_lts)]
    V_lts = V_lts[:len(t_lts)]
    
    #calculate the input scaling in mV^-1
    s_input = h / np.amax(f_ex)
    
    #average the experimental data to minimize noise when calcualting derivative
    upto = (len(t_ex)//100) * 100
    t_avg = t_ex[:upto]
    f_avg = f_ex[:upto]
    t_avg = np.mean(t_avg.reshape(-1, 100), axis=1)
    f_avg = np.mean(f_avg.reshape(-1, 100), axis=1)
    
    #calcuate time derivative of driving signal
    diff = np.diff(f_avg)/np.diff(t_avg)
    
    #calculate and scale the initial conditions based on the experimental data
    y_0, v_0 = V_ex[0], (V_ex[80] - V_ex[0]) / (t_ex[80] - t_ex[0])
    S_0 = np.array([y_0, v_0]) * s_input / s_internal

    #array of values to be passed into the DE function
    vals = np.array([gma, omg_0, t_avg, f_avg, diff, s_input], dtype=object)
    
    #numerically solve the differential equation
    print('Starting to solve DE using IVP')
    print()
    S = solve_ivp(dSdt, t_span=[0, t_ex[-1]], y0=S_0, t_eval=t_ex, args=vals)
    print()
    print('------------------------------------------------------------------------------------------')
    print(S)
    print('------------------------------------------------------------------------------------------')
    print('DE solved for zeta = %.3f in %.3fs (3dp)' % (zeta, (time.time() - starttim)))
    print('------------------------------------------------------------------------------------------')

    #obtain numerical solution for the DE in m
    y_num = S.y[0]
    
    #scale the experimental data and LTspice simulation to obtain final displacement in m
    y_ex = V_ex * s_input / s_internal
    y_lts = V_lts * s_input / s_internal

    #append the determined values onto the arrays for further plotting
    #zeta_arr in kgs^-1 and amp_ex, amp_ltsm, amp_num all in m
    zeta_arr = np.append(zeta_arr, zeta)
    amp_ex = np.append(amp_ex, np.amax(y_ex))
    amp_lts = np.append(amp_lts, np.amax(y_lts))
    amp_num = np.append(amp_num, np.amax(y_num))

    #use the colormap to determine the color corresponding to the zeta being plotted
    color = cmap(norm(zeta))

    #plot the experimental data for the calculated zeta
    plt.plot(t_ex, y_ex, color=color, linewidth=1.5)
    

#set fonts and font size for plotting
plt.rcParams['font.family'] = 'baskerville'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11

#set title and axis lables for plotting
plt.title('Experimental Response to Varying $\zeta$')
plt.ylabel('$y \, (t)$ / m')
plt.xlabel('$t$ / s')

#plot the LTspice and numerical lines for the lowest damping
#replot the last experimental curve to make a legend
plt.plot(t_ex, y_ex, color=color, linewidth=1.3, label='Experimental')
plt.plot(t_lts, y_lts, color='limegreen', linewidth=1.3, label='LTspice')
plt.plot(t_ex, y_num, color='royalblue', linewidth=1.3, label='Numerical')
plt.legend(loc = 1)

#create a colorbar to show how zeta was varied
cbar = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar.set_array([])
plt.colorbar(cbar, label='$\zeta$ / $\mathrm{kg} \; \mathrm{s}^{-1}$')

#save the plot
if saveoutput == True:
    plt.savefig(f'Output/Terrain Zeta/Terrain Zeta Color.png', bbox_inches='tight', dpi=300)
plt.show(block=False)

#set title and axis lables for plotting
plt.title('Peak Amplituide Against $\zeta$')
plt.ylabel('$A$ / m')
plt.xlabel('$\zeta$ / $\mathrm{kg} \; \mathrm{s}^{-1}$')

#plot the amplitude against zeta for experimenta, LTspice and numerical data
plt.plot(zeta_arr, amp_ex, 'x',color='red', label='Experimental')
plt.plot(zeta_arr, amp_lts, linewidth=1.3, color='limegreen', label='LTspice')
plt.plot(zeta_arr, amp_num, linewidth=1.3, color='royalblue', label='Numerical')
plt.legend(loc = 7)

#save the plot
if saveoutput == True:
    plt.savefig(f'Output/Terrain Zeta/Terrain Zeta Amplitude.png', bbox_inches='tight', dpi=300)
plt.show(block=False)

print(np.amin(zeta_arr))
print(np.amax(zeta_arr))
















