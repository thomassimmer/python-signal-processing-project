import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

#Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
#Modulating wave m(t)=A_m*cos(2*pi*f_m*t)
#Modulated wave s(t)=A_c[1+mu*cos(2*pi*f_m*t)]cos(2*pi*f_c*t)

# Variables for signal characterization
A_c = 1
f_c = 20 * (10 ** 3)
A_m = 1
f_m = 104
modulation_index = 2

# Variables for sampling
samplingRate = 44100
beginTime = 0
endTime = 1
samplingInterval = 1 / samplingRate
t = np.arange(beginTime, endTime, samplingInterval)

tpCount = len(t)
values = np.arange(int(tpCount))
timePeriod = tpCount/samplingRate
f = values/timePeriod
 
# Generates 3 signals
carrier = A_c*np.cos(2*np.pi*f_c*t)
modulator = A_m*np.cos(2*np.pi*f_m*t)
product = A_c*(1+modulation_index*np.cos(2*np.pi*f_m*t))*np.cos(2*np.pi*f_c*t)

# Generate spectra
carrier_spectrum = np.fft.fft(carrier)/len(carrier)
modulator_spectrum = np.fft.fft(modulator)/len(modulator)
product_spectrum = np.fft.fft(product)/len(product)

# Write wav files
write("results/AM/message.wav", samplingRate, modulator)
write("results/AM/modulatedSignal.wav", samplingRate, product)

# Show plots
plt.subplot(3,2,1)
plt.title('Amplitude Modulation')
plt.plot(t, modulator,'g')
plt.ylabel('Amplitude')
plt.xlabel('Message signal')

plt.subplot(3,2, 2)
plt.plot(f[0:120], modulator_spectrum[0:120])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplot(3,2,3)
plt.plot(t, carrier, 'r')
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')

plt.subplot(3,2,4)
plt.plot(f[19950:20050], carrier_spectrum[19950:20050])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplot(3,2,5)
plt.plot(t, product, color="purple")
plt.ylabel('Amplitude')
plt.xlabel('AM signal')

plt.subplot(3,2,6)
plt.plot(f[19800:20200], product_spectrum[19800:20200])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()

# Save plot
# plt.show()
fig.savefig('results/AM/modulation.png', dpi=100)