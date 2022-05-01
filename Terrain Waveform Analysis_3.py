'''
Analogue Computer Terrain Waveform Analysis

Lukas Kostal, 11.4.2022, ICL
'''

import numpy as np
import matplotlib.pyplot as plt
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


#specify whether to save output or not
saveoutput = False

#define physical varibales
freq_0 = 1.18               #resonant frequency in Hz
m = 360                     #mass in kg
R = 2.5                     #zeta given as resistance value in Ohm
h = 0.1                     #peak height of terrain in m
s_internal = 7.209629e-1    #overall scaling of the circuit

#define filename to be analysed
filename = 'RCT_1200'

#define paths to the experimental data and LTspice simulations
path_lts = 'LTspice/Terrain Waveforms/'
path_ex = 'Data/Terrain Waveforms/'

#calcualte the value of omega in rads^-1
omg_0 = 2 * sc.pi * freq_0

#calculate the scaling factor for the input signal
scale = omg_0**2 / (2.14/2 * 3.24/9.86)

print(scale)

#calculate the value of gamma by averaging
gma =  (3.24/R * 4.59e-6 * 5.49e3)*scale/2 + (3.24/(6*R) * 1/(94.57e-6 * 984) * 2.15e3/983)/2

#calculate the value of zeta from gamma
zeta = gma/(2*omg_0)


#load the experimental data and LTspice simulation
#units are s, V, V respectively
t_ex, f_ex, V_ex = np.loadtxt(f'{path_ex}{filename}.csv', skiprows=(1),delimiter=',', unpack=True)
t_lts, V_lts, f_lts = np.loadtxt(f'{path_lts}{filename}.txt', skiprows=(1), unpack=True)


#determine type of waveform from name
waveform = filename[:3]
#determine driving frequency in mHz from name
frequency = filename[4:]

#define parameters for processing driving signal based on frequency
if frequency == '1200':
    cycles = 5
    spread = 1000
    avgover = 100
    first, last = 0, 5

if frequency == '500':
    cycles = 3
    spread = 5000
    avgover = 600
    first, last = 0, 2
    
if frequency == '200':
    cycles = 2
    spread = 5000
    avgover = 1000 #600
    first, last = 0, 1


#allign and process the driving signal based on the type of waveform and frequency
if waveform == 'SQR':
    
    avgover = 100 #50
    diff = np.diff(f_ex)
    diff = diff/np.amax(diff)
    maxima, _ = find_peaks(diff, height=0.1, distance=spread)
    minima, _ = find_peaks(-diff, height=0.1, distance=spread)

    start = maxima[0]
    end = maxima[cycles]

if waveform == 'TRG':
    minima, _ = find_peaks(-f_ex, height=-0.1, distance=spread)
    
    start = minima[0]
    end = minima[cycles]

if waveform == 'SIN':
    maxima, _ = find_peaks(f_ex, height=2.3, distance=spread)
    minima, _ = find_peaks(-f_ex, height=2.3, distance=spread)
    
    start = maxima[first] - np.absolute(minima[first] - maxima[first])//2
    end = minima[last] + np.absolute(minima[last] - maxima[last])//2
    
if waveform == 'RCT':
    spread = spread * 5
    if frequency == '1200':
        avgover = avgover * 2
    maxima, _ = find_peaks(f_ex, height=2.8, distance=spread)
    
    start = (maxima[1] + maxima[0])//2
    end = (maxima[cycles+1] + maxima[cycles])//2


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
upto = (len(t_ex)//avgover) * avgover
t_avg = t_ex[:upto]
f_avg = f_ex[:upto]
t_avg = np.mean(t_avg.reshape(-1, avgover), axis=1)
f_avg = np.mean(f_avg.reshape(-1, avgover), axis=1)

#calcuate time derivative of driving signal
diff = np.diff(f_avg)/np.diff(t_avg)

#calculate and scale the initial conditions based on the experimental data
y_0, v_0 = V_ex[0], (V_ex[avgover] - V_ex[0]) / (t_ex[avgover] - t_ex[0])
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
print('DE solved for ', filename, ' in %.3fs (3dp)' % (time.time() - starttim))
print('------------------------------------------------------------------------------------------')

#obtain numerical solution for the DE in m
y_num = S.y[0]

#scale the experimental data and LTspice simulation to obtain final displacement in m
y_ex = V_ex * s_input / s_internal
y_lts = V_lts * s_input / s_internal

#set fonts and font size for plotting
plt.rcParams['font.family'] = 'baskerville'
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.size'] = 11

#set title and axis lables for plotting
plt.title(f'Terrain Displacement for {filename}')
plt.ylabel('$x \, (t)$ / m')
plt.xlabel('$t$ / s')

#plot the terrain displacement and legend
plt.plot(t_ex, f_ex * s_input, linewidth=1.2, color='red', label='Experimental')
plt.plot(t_lts, f_lts * s_input, linewidth=1.6, color='limegreen', label='LTspice')
plt.legend(loc = 7)

#save the plot
if saveoutput == True:
    plt.savefig(f'Output/Terrain Waveforms/{filename} Terrain.png', bbox_inches='tight', dpi=300)
plt.show(block=False)

#set title and axis lables for plotting
plt.title(f'System Displacement for {filename}')
plt.ylabel('$y \, (t)$ / m')
plt.xlabel('$t$ / s')

#plot the system displacement and legend
plt.plot(t_ex, y_ex, linewidth=1.2, color='red', label='Experimental')
plt.plot(t_lts, y_lts, linewidth=1.6, color='limegreen', label='LTspice')
plt.plot(t_ex, y_num, linewidth=1.6, color='royalblue', label='Numerical')
plt.legend(loc = 7)

#save the plot
if saveoutput == True:
    plt.savefig(f'Output/Terrain Waveforms/{filename} System.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


#set up the sub plots for a double plot
fig, (ax1, ax2) = plt.subplots(2, sharex = True)

#set titles for double plot
ax1.title.set_text(f'Terrain Displacement for {filename}')
ax2.title.set_text(f'System Displacement for {filename}')

#label the axis for double plot
ax1.set_ylabel('$x \, (t)$ / m')
ax2.set_ylabel('$y \, (t)$ / m')
ax2.set_xlabel('$t$ / s')

#plot the terrain displacement
ax1.plot(t_ex, f_ex * s_input, linewidth=1.2, color='red')
ax1.plot(t_lts, f_lts * s_input, linewidth=1.6, color='limegreen')

#plot the system displacement
ax2.plot(t_ex, y_ex, linewidth=1.2, color='red', label='Experimental')
ax2.plot(t_lts, y_lts, linewidth=1.6, color='limegreen', label='LTspice')
ax2.plot(t_ex, y_num, linewidth=1.6, color='royalblue', label='Numerical')

#create plot legend
fig.legend(bbox_to_anchor=(0.91, 0.82))

#save the double plot
if saveoutput == True:
    plt.savefig(f'Output/Terrain Waveforms/{filename} Double.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


#note at higher driving frequencies (1.2Hz) program sometimes returns error message
#usually after re-running the program up to 2 times it runs fine









