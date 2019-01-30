
# coding: utf-8

# # The basics of making a sound... 
# ### (1) a sine wave: what does it mean ?  
# ### (2) time and frequency
# ### (3) making a sound
# ### (4) adding and concatenating
# ### (5) LOOKING at a sound: the Spectrogram
# ### (6) data: modulating amplitude with an envelope. 

# In[14]:

import numpy as np
from matplotlib import pyplot as plt # for plots

import librosa


# In[15]:

# (1) a sine wave: 
# way 1: numpy sin, in radians: a cycle is 2*pi-- this is GEOMETRY, not time ! 
# time is how fast it sweeps through that geometry ! 
# remember that the motion of a wave is not the motion of the particle ! 

pi  = np.pi 
npts_cycle = 40 # number of points per cycle, or samples per cycle
n_cycles = 3 
x = np.linspace(0,2*pi*n_cycles,npts_cycle*n_cycles)
y = np.sin(x)

plt.plot(x,y,'k') 
plt.xlabel('radians')
plt.ylabel('amplitude of wave')
plt.grid()

# notice that we did not use frequency here. 
# to convert to a sound, we need to assign time: how face each cycle takes. 


# ## (2) time and frequency 

# In[16]:

n_cycles = 440 
npts_cycle = 40

# note that this x does not have frequency in it ! 
x = np.linspace(0,2*pi*n_cycles,npts_cycle*n_cycles)
y = np.sin(x)


time_per_cycle = period = p = 1/440 # [seconds]
frequency = f = 1/p
print('frequency = ', f)

dt = 1/f/npts_cycle 
print('dt = ', dt)
# sampling frequency ! not the same as the wave frequency ! 
fs = int(1/dt) # must be an integer ! 
print('sampling frequency, fs = ', fs)
time1 = np.linspace(0,period*n_cycles,len(x))
print(len(time1))
time2 = np.arange(0,(period*n_cycles),dt)
print(len(time2))

# Plot
plt.figure(figsize=(9,5))
plt.subplot(2,1,1)
plt.plot(time2,y,'k') 
plt.xlabel('time [s]')
plt.ylabel('amplitude of wave')
plt.grid()
plt.subplot(2,1,2)
plt.plot(time2,y,'k') 
plt.xlabel('time [s]')
plt.ylabel('amplitude of wave')
plt.grid()
plt.xlim(0, time_per_cycle*3)


# ## (3) MAKE A SOUND ! ! 

# In[17]:

# name the file: 
outfile = 'sound_0.wav'
# call the wave file writing module in librosa: 
# y is the data, fs is the sampling frequency, 
# norm=False means the signal will not be normalized (default is True). 
# the a_scale scales the amplitude 
a_scale = 0.2
librosa.output.write_wav(outfile, y*a_scale, fs, norm=False)


# ## (4) adding and concatenating ! 

# In[18]:

# adding waveforms/signals to make two frequencies
# make vectors of two frequencies, that have the same number of points, long enough to hear

n_cycles = 440
npts_cycle = 40
x1 = np.linspace(0,2*pi*n_cycles,npts_cycle*n_cycles)
y1 = np.sin(x1)

f1 = 440.
p1 = 1/f1
dt = p1/npts_cycle
# duration
dur1 = n_cycles*p1
print('sound duration = ', dur1)
f2 = 2*f1
p2 = 1/f2
n_cycles2 = dur1/p2
print('number of cycles of f1 = ', n_cycles)
print('number of cycles of f2 = ', n_cycles2)
print('ratio of f2/f1 = ', f2/f1)

x2 = np.linspace(0,2*pi*n_cycles2,len(x1))
y2 = np.sin(x2)
# add them together

y = y1+y2
# plot a slice
time1 = np.linspace(0,p1*n_cycles,len(x1))
print(len(time1))

# plot the wave
plt.figure(figsize=(9,5))
plt.subplot(2,1,1)
plt.plot(time1,y,'k') 
plt.xlabel('time [s]')
plt.ylabel('amplitude of wave')
plt.grid()
plt.subplot(2,1,2)
plt.plot(time1,y,'k') 
plt.xlabel('time [s]')
plt.ylabel('amplitude of wave')
plt.grid()
plt.xlim(0, p1*3)

# write the sound: 
outfile = 'sound_1.wav'
fs = int(1/dt) # must be an integer ! 
a_scale = 0.2
librosa.output.write_wav(outfile, y*a_scale, fs, norm=False)


# In[19]:

# concatenating

print(y2.shape)
yc = np.concatenate((y1,y2),axis=0)
outfile = 'sound_conc.wav'
fs = int(1/dt) # must be an integer ! 
a_scale = 0.6
librosa.output.write_wav(outfile, yc*a_scale, fs, norm=False)


# ## (5) Looking at the sound: The Fourier Transform ! 

# In[20]:

from scipy import fftpack as spfft # for computing spectra
#from scipy import signal as spsig

########## First do the Fourier transform = look at the frequency content of the entire signal
# We do a fast Fourier transform (FFT)

Nfft = 1000 # Number of points (discrete frequencies) on which to compute the FFT 
# (actually twice the number of frequency point we need, but that's another story). Better if it's a power of 2
sr = fs
f = np.arange(-Nfft/2,Nfft/2,1)*sr/Nfft 
# the discretized vector of frequencies: contains negative and positive frequencies 
# (see mathematical definition of the Fourier transform)

# FFT computation -> gives the spectrum S
S = spfft.fft(y, n=Nfft)
# S is an array of complex numbers, containing the amplitude for each frequency bin in f, but in another order... 
# First positive frequency bins, then negative frequency bins...

# Let's consider only positive frequencies
f = f[int(len(f)/2):] 
S = S[:int(len(S)/2)] 
# Now f covers the (discretized) frequency range from 0 to sr/2 Hz (see Nyquist theorem, Nyquist frequency)

########## Finally plot the spectrum
plt.figure(figsize=(9,5))
plt.plot(f,np.absolute(S),'k') # Since S is complex-valued, we have to choose between the absolute value and phase
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (linear)')
plt.xlim([0,sr/6])

plt.figure(figsize=(9,5))
plt.plot(f,20*np.log10(np.absolute(S)),'k') # Take the log10 with numpy, and have a factor 20 because we're dealing with acoustic pressure (L = 20log10(p/pref)) and not intensity (L = 10log10(p/pref))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.xlim([0,sr/6])


# ## (5) Looking at the sound: The Spectrogram ! 

# The "problem" with FFT is that it averages the frequency content over the whole signal. We completely lose the 
# information of time.
# 
# so... Short Term Fourier Transform (STFT).
# 
# The idea is to cut the signal into successive time frames and perform an FFT for each frame
# Involves quite a bit of mathematical stuff which we will ingeniously skip here...
# 
# See here for a graphical explanation https://www.researchgate.net/figure/231828310_fig7_Figure-7-Short-time-Fourier-transform-STFT-with-envelope-and-two-sample-overlap
# 
# And the well-known "spectrogram" is only a graphical representation of the STFT. 

# In[21]:

from scipy import signal as spsig

# Let's compute the spectrogram
NfftSTFT = 4096 # The number of frequency points for the FFT of each frame
SliceLength = int(0.05*fs) # The length of each frame (should be expressed in samples)
Overlap = int(SliceLength/4) # The overlapping between successive frames (should be expressed in samples)
[fSTFT, tSTFT, STFT] = spsig.spectrogram(y, fs=sr, nperseg=SliceLength, noverlap=Overlap, nfft=NfftSTFT) 
# also provides associated f and t vectors!

# Let's plot the spectrogram
plt.figure(figsize=(10,6))
plt.pcolormesh(tSTFT,fSTFT,20*np.log10(np.absolute(STFT)))
plt.ylim([0,sr/8])
#plt.xlim([0,dur1-SliceLength/sr])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.ax.set_ylabel('dB', rotation=270)


# ## (6) modulate the amplitude with an envelope ! 

# In[22]:

# 1) make a curve that will be the envelope, of the same length as the original signal
N = len(y)
envelope = np.zeros(N)
print(envelope.shape)
peak = int(N/6)
print(peak)

# from 0 to the peak index (peak):
up = np.linspace(0,1,peak)
print(up.shape)
envelope[:peak] = up

# and fill in the rest: 
down = np.linspace(1,0,(N-peak))
print(down.shape)
envelope[peak:] = down

# plot the envelope
plt.figure(figsize=(9,5))
plt.plot(envelope,'k') 
plt.xlabel('time [s]')
plt.ylabel('amplitude of wave')

# modulate the signal by multiplication
ym = y*envelope

# plot the wave
plt.figure(figsize=(9,5))
plt.plot(ym,'k') 
plt.xlabel('time [s]')
plt.ylabel('amplitude of wave')
plt.grid()

# make the sound
outfile = 'sound_env.wav'
fs = int(1/dt) # must be an integer ! 
a_scale = 0.9
librosa.output.write_wav(outfile, ym*a_scale, fs, norm=False)


# ## (7) PLOT THE SPECTROGRAM OF THE envelope modulated signal

# In[23]:

from scipy import signal as spsig

# Let's compute the spectrogram
NfftSTFT = 4096 # The number of frequency points for the FFT of each frame
SliceLength = int(0.05*fs) # The length of each frame (should be expressed in samples)
Overlap = int(SliceLength/4) # The overlapping between successive frames (should be expressed in samples)
[fSTFT, tSTFT, STFT] = spsig.spectrogram(ym, fs=sr, nperseg=SliceLength, noverlap=Overlap, nfft=NfftSTFT) 
# also provides associated f and t vectors!

# Let's plot the spectrogram
plt.figure(figsize=(10,6))
plt.pcolormesh(tSTFT,fSTFT,20*np.log10(np.absolute(STFT)))
plt.ylim([0,sr/8])
#plt.xlim([0,dur1-SliceLength/sr])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.ax.set_ylabel('dB', rotation=270)
plt.show()


# In[ ]:



