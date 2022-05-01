'''
Analogue Computer Terrain Bode Analysis

Lukas Kostal, 17.4.2022, ICL
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as sc
import scipy.optimize as op
import scipy.signal as sg
import time

#record the time at which the code starts
starttim = time.time()

#enable waveform plots at each frequency
plots = False

#define physical varibales
freq_0 = 1.18               #resonant frequency in Hz
m = 360                     #mass in kg
R = 2.5                     #zeta given as resistance value in Ohm
h = 0.1                     #peak height of terrain in m
s_internal = 7.209629e-1    #overall scaling of the circuit

#define path to the experimental data
path_ex = 'Data/Terrain Bode/'

#calcualte the value of omega in rads^-1
omg_0 = 2 * sc.pi * freq_0

#calculate the scaling factor for the input signal
scale = omg_0**2 / (2.14/2 * 3.24/9.86)

#calculate the value of gamma by averaging
gma =  (3.24/R * 4.59e-6 * 5.49e3)*scale/2 + (3.24/(6*R) * 1/(94.57e-6 * 984) * 2.15e3/983)/2

#define a sine function for curve fitting
def sine(x, A, omg, phi):
    y = A * np.sin(omg*x + phi)
    return(y)

#prepare arrays to hold experimental frequency, magnitude and phase difference respectively
f_ex = np.empty(0)
amp_ex = np.empty(0)
phi_ex = np.empty(0)

#loop over all of the measured frequencies 0.1 Hz to 2.5 Hz
for i in range (0, 25):
    
    #determine the filepath based on the frequency being analyzed
    if i<9:
        filepath = f'{path_ex}BODE_0{i+1}.CSV'
    else:
        filepath = f'{path_ex}BODE_{i+1}.CSV'   
    
    #load the experimental waveforms for the given frequency
    #units are s, V, V respectively
    t, f, V = np.loadtxt(filepath, skiprows=(1),delimiter=',', unpack=True)
    
    #shift the timescale to start at 0s
    t = t - np.amin(t)

    #initial guess for amplitude in V for terrain waveform
    A_ig = np.amax(f)
    #initial guess for frequency in Hz
    omg_ig = sc.pi * (i+1) * 0.2
    #initial guess for phase in rad
    phi_ig = 0
    #combine all initial guesses into an array
    ig = np.array([A_ig, omg_ig, phi_ig])
    
    #curve fit for the terrain displacement waveform
    opt_f, cov_f = op.curve_fit(sine, t, f, ig)
    
    #initial guess for amplitude in V for system waveform
    A_ig = np.amax(V)
    #modify the initial guesses array with new amplitude
    ig = np.array([A_ig, omg_ig, phi_ig])

    #curve fit for the system displacement waveform
    opt_V, cov_V = op.curve_fit(sine, t, V, ig)
    
    #this section calculates the phase difference using cross correlation of the two waveforms
    
    #calcualte the cross correlation of the terrain and response waveforms
    corr = sg.correlate(f, V)
    #prepare an array containing the lag between the two waveforms
    lag = sg.correlation_lags(len(f), len(V))
    #calcualte the average resolution in time t in s between elements
    deltat = np.sum(np.diff(t))/(len(t)-1)
    #calculate the phase difference in radians
    phase = 0.2 * sc.pi * (i+1) * deltat* lag[np.argmax(corr)]
    
    #if enabled plot the experimental and curve fit waveforms for the terrain and response
    if plots == True:
        plt.plot(t, f, color='forestgreen')
        plt.plot(t, sine(t, *opt_f), color='lime')
        plt.plot(t, V, color='red')
        plt.plot(t, sine(t, *opt_V), color='tomato')
        plt.show()

    #append the determined values onto the arrays for further plotting
    #f_ex in s, amp_ex is uniteless, phi_ex in rad
    f_ex = np.append(f_ex, [(i+1) * 0.1])
    amp_ex = np.append(amp_ex, [opt_V[0]/opt_f[0]])
    phi_ex = np.append(phi_ex, phase)
    
#calculate absolute value of amp_ex since some values are -ve
#scale up to account for the internal scaling of the circuit
amp_ex = np.absolute(amp_ex)/s_internal

#load the LTspice ac analysis data
#units are s, V, V respectively
f_lts, Re, Im = np.loadtxt('LTspice/Terrain Bode/Terrain Bode.csv', skiprows=(1),delimiter=',', unpack=True)

#calcualte the absolute value of the voltage from the real and imaginary components
#account for the 2.5V amplitude of driving signal and the internal scaling of the circuit
amp_lts = np.sqrt(Re**2 + Im**2)/2.5 /s_internal

#calcualte the phase difference in rad from the real and imaginary components
phi_lts = np.arctan2(Im, Re)

#prepare array of frequencies in Hz for calcualting numerical data
f_num = np.linspace(0.1, 2.5, 500)
#convert the frequencies to rad
omg = 2 * sc.pi * f_num

#calcuate the complex amplitude of the system response based on theoretical model
#the amplitude is divided by the amplitude of the driving signal
cA = (gma*omg*1j + omg_0**2)/(-1* omg**2 + gma*omg*1j + omg_0**2)

#calculate the absolute value of the magnitude for all frequencies
amp_num = np.absolute(cA)

#calcualte the phase in rad of the complex amplitude for all frequencuies
phi_num = np.angle(cA)

#determine the frequency in Hz at which the peak occours
f_res_num = f_num[np.argmax(amp_num)]
f_res_lts = f_lts[np.argmax(amp_lts)]
f_res_ex = (f_ex[np.argmax(amp_ex)-1] + f_ex[np.argmax(amp_ex)]) /2

#determine the magnitude of the peak which is uniteless
amp_res_num = np.amax(amp_num)
amp_res_lts = np.amax(amp_lts)
amp_res_ex = np.amax(amp_ex)


#print the time taken to complete the calcualtions
print('Calculations finished in %.3fs (3dp)' % (time.time() - starttim))

#print the numerical values
print('------------------------------------------------------------------------------------------')
print('Natural frequency           =', freq_0, 'Hz')
print()
print('Numerical peak frequency    = %.3f Hz (3dp)' %  (f_res_num))
print('Numerical peak magnitude    = %.3f (3dp)' %  (amp_res_num))
print()
print('LTspice peak frequency      = %.3f Hz (3dp)' %  (f_res_lts))
print('LTspice peak magnitude      = %.3f (3dp)' %  (amp_res_lts))
print()
print('Experimental peak frequency = %.3f Hz (3dp)' % (f_res_ex))
print('Experimenta peak magnitude  = %.3f (3dp)' % (amp_res_ex))
print('------------------------------------------------------------------------------------------')


#set fonts and font size for plotting
plt.rcParams['font.family'] = 'baskerville'
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.size'] = 11

#set up the sub plots
fig, (ax1, ax2) = plt.subplots(2, sharex = True)

#set title for plotting
fig.suptitle('Terrain Bode Plot')

#label the axis for ploting
ax1.set_ylabel('$\\frac{A}{A_0}$ / uniteless')
ax2.set_ylabel('$\phi$ / rad')
ax2.set_xlabel('$f$ / Hz')

#plot the amplitude curves
ax1.plot(f_ex, amp_ex, linewidth=1.3, color='red', label='Experimental')
ax1.plot(f_lts, amp_lts, linewidth=1.3, color='limegreen', label='LTspice')
ax1.plot(f_num, amp_num, linewidth=1.3, color='royalblue', label='Numerical')
#plot lines representing frequencies of peaks
ax1.axvline(x=f_res_ex, linewidth=1.1, ls='--', color='red')
ax1.axvline(x=f_res_lts, linewidth=1.1, ls='--', color='limegreen')
ax1.axvline(x=f_res_num, linewidth=1.1, ls='--', color='royalblue')
ax1.axvline(x=freq_0, linewidth=1.1, ls='--', color='black')

#plot the phase difference curves
ax2.plot(f_ex, phi_ex, linewidth=1.3, color='red')
ax2.plot(f_lts, phi_lts, linewidth=1.3, color='limegreen')
ax2.plot(f_num, phi_num, linewidth=1.3, color='royalblue')
#plot lines representing frequencies of peaks
ax2.axvline(x=f_res_ex, linewidth=1.1, ls='--', color='red')
ax2.axvline(x=f_res_lts, linewidth=1.1, ls='--', color='limegreen')
ax2.axvline(x=f_res_num, linewidth=1.1, ls='--', color='royalblue')
ax2.axvline(x=freq_0, linewidth=1.1, ls='--', color='black')
#plot the line representing a phase difference of 90 deg
ax2.axhline(y=-sc.pi/4, linewidth=1.1, ls='--', color='black')

#plot frequency lables for the lines
fig.text(0.35, 0.49, '%.2f' % (f_res_ex), color='red')
fig.text(0.45, 0.49, '%.2f' % (f_res_lts), color='limegreen')
fig.text(0.4, 0.49, '%.2f' % (f_res_num), color='royalblue')
fig.text(0.5, 0.49, '%.2f' % (freq_0), color='black')
#plot legend
fig.legend(bbox_to_anchor=(0.9, 0.45))

print(np.amin(zeta_arr))
print(np.amax(zeta_arr))

#save the plot
plt.savefig(f'Output/Terrain Bode/Terrain Bode Plot.png', bbox_inches='tight', dpi=300)
plt.show(block=False)