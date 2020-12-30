import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write

from core.utils import filters

# Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
# Modulating wave m(t)=A_m*cos(2*pi*f_m*t)
# Modulated wave s(t)=A_c[1+mu*cos(2*pi*f_m*t)]cos(2*pi*f_c*t)

# Convert modulated signal to array
samplingRate, signal = read("results/AM/modulatedSignal.wav")

# Variables for signal characterization
A_c = 1
f_c = 20 * (10 ** 3)
A_m = 1
f_m = 104
modulation_index = 2

# Variables for sampling
beginTime = 0
endTime = len(signal)/samplingRate
samplingInterval = 1 / samplingRate
t = np.arange(beginTime, endTime, samplingInterval)

tpCount = len(t)
values = np.arange(int(tpCount))
timePeriod = tpCount/samplingRate
f = values/timePeriod

# Apply pass band-filter
signal_band_pass = filters.butter_bandpass_filter(signal, 19800, 20200, 44100)

# We multiply the signal after the passband filter by the carrier to center it around 0Hz
signal_carrier = filters.butter_bandpass_filter(signal, 19950, 20050, 44100)
multiplied_signal = signal_band_pass * signal_carrier

# We apply highpass filter to the carrier and keep only the modulator signal
signal_demodulated = filters.butter_highpass_filter(multiplied_signal, 50, 44100)

# Generate spectra
signal_carrier_spectrum = np.fft.fft(signal_carrier)/len(signal_carrier)
signal_demodulated_spectrum = np.fft.fft(signal_demodulated)/len(signal_demodulated)
signal_spectrum = np.fft.fft(signal)/len(signal)

# Write wav files
write("results/AM/demodulatedSignal.wav", samplingRate, signal_demodulated)

# Show plots
plt.subplot(3, 2, 1)
plt.title('Amplitude Demodulation')
plt.plot(t, signal_demodulated, 'g')
plt.ylabel('Amplitude')
plt.xlabel('Message signal')

plt.subplot(3, 2, 2)
plt.plot(f[0:120], signal_demodulated_spectrum[0:120])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplot(3, 2, 3)
plt.plot(t, signal_carrier, 'r')
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')

plt.subplot(3, 2, 4)
plt.plot(f[19800:20200], signal_carrier_spectrum[19800:20200])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplot(3, 2, 5)
plt.plot(t, signal, color="purple")
plt.ylabel('Amplitude')
plt.xlabel('AM signal')

plt.subplot(3, 2, 6)
plt.plot(f[19800:20200], signal_spectrum[19800:20200])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')


plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()

# Save plot
# plt.show()
fig.savefig('results/AM/demodulation.png', dpi=100)
