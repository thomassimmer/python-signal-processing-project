import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write

from core.utils import filters, helpers, symbols

# Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
# Modulating wave m(t)=A_m *[ I(t)*cos(2*pi*f_m*t) + Q(t)*sin(2*pi*f_m*t) ]
# Modulated wave s(t)=A_c * cos(2*pi*f_c*t) * [1+m*m(t)]

# Convert modulated signal to array
samplingRate, signal = read("results/32APSK/modulatedSignal.wav")

# Variables for signal characterization
A_c = 1
f_c = 10 * (10 ** 3)
A_m = 1
f_m = 200
T_m = 1 / f_m
modulation_index = 2

# Variables for sampling
beginTime = 0
endTime = len(signal) / samplingRate
samplingInterval = 1 / samplingRate
t = np.arange(beginTime, endTime, samplingInterval)

tpCount = len(t)
values = np.arange(int(tpCount))
timePeriod = tpCount / samplingRate
f = values / timePeriod

# Apply pass band-filter
signal_band_pass = filters.butter_bandpass_filter(signal, 9000, 11000, samplingRate)

# We multiply the signal after the passband filter by the carrier to center it around 0Hz
signal_carrier = 2 * A_c * np.cos(2 * np.pi * f_c * t)
multiplied_signal = signal_band_pass * signal_carrier

# We apply highpass filter to the carrier and keep only the modulator signal
signal_demodulated = filters.butter_lowpass_filter(
    multiplied_signal, 1000, samplingRate
)
signal_demodulated = filters.butter_highpass_filter(
    signal_demodulated, 10, samplingRate
)

# Get I and Q
# 1. Multiply demodulated signal by sin(f_m) and cos(f_m) for Q and I respectively
Q = signal_demodulated * A_m * np.sin(2 * np.pi * f_m * t) * 1.1
I = signal_demodulated * A_m * np.cos(2 * np.pi * f_m * t) * 1.1
# 2. Apply lowpass filter to keep continous part
Q = filters.butter_lowpass_filter(Q, f_m, samplingRate)
I = filters.butter_lowpass_filter(I, f_m, samplingRate)


# Symbol detection
k = 0
symbols_received, paths = [], []
I_values, Q_values = [], []
bits_received = []
threshold_radius = 0.725
for i, time in enumerate(t):
    # every instant check value for I and Q
    if time < (k + 1) * T_m:
        paths.append(np.complex(real=I[i], imag=Q[i]))
        I_values.append(I[i])
        Q_values.append(Q[i])

    # every T, make an average for I and Q to get symbol
    else:
        paths.append(np.complex(real=I[i], imag=Q[i]))
        average_I = sum(I_values) / len(I_values)
        average_Q = sum(Q_values) / len(Q_values)
        symbols_received.append(np.complex(real=average_I, imag=average_Q))
        k = k + 1
        I_values, Q_values = [], []
        # Check distance to symbols
        for symbol in symbols.symbols_32APSK:
            if np.linalg.norm(symbol.number - symbols_received[-1]) < threshold_radius:
                bits_received.extend([i for i in symbol.string])


# Generate spectra
signal_carrier_spectrum = np.fft.fft(signal_carrier) / len(signal_carrier)
signal_demodulated_spectrum = np.fft.fft(signal_demodulated) / len(signal_demodulated)
signal_spectrum = np.fft.fft(signal_band_pass) / len(signal_band_pass)

# Write wav files
write("results/32APSK/demodulatedSignal.wav", samplingRate, signal_demodulated)

# Show plots
plt.subplot(3, 2, 1)
plt.title("32APSK Demodulation")
plt.plot(t, signal_demodulated, "g")
# plt.plot(t, I,'b')
# plt.plot(t, Q,'r')
plt.ylabel("Amplitude")
plt.xlabel("Message signal")

plt.subplot(3, 2, 2)
# plt.plot(f, signal_demodulated_spectrum)
plt.plot(f[0:1000], signal_demodulated_spectrum[0:1000])
plt.ylabel("Amplitude")
plt.xlabel("Frequency")

plt.subplot(3, 2, 3)
plt.plot(t, signal_carrier, "r")
plt.ylabel("Amplitude")
plt.xlabel("Carrier signal")

plt.subplot(3, 2, 4)
plt.plot(f[9800:10200], signal_carrier_spectrum[9800:10200])
plt.ylabel("Amplitude")
plt.xlabel("Frequency")

plt.subplot(3, 2, 5)
plt.plot(t, signal, color="purple")
plt.ylabel("Amplitude")
plt.xlabel("AM signal")

plt.subplot(3, 2, 6)
plt.plot(f[9800:15200], signal_spectrum[9800:15200])
plt.ylabel("Amplitude")
plt.xlabel("Frequency")

plt.subplots_adjust(hspace=1)
plt.rc("font", size=15)
fig = plt.gcf()

# Save plot
# plt.show()
fig.savefig("results/32APSK/demodulation.png", dpi=100)

# SYMBOLS
fig2 = plt.figure(2)
axes = plt.gca()
axes.set_xlim([-symbols.r3 - threshold_radius, symbols.r3 + threshold_radius])
axes.set_ylim([-symbols.r3 - threshold_radius, symbols.r3 + threshold_radius])
for symbol in symbols_received:
    plt.scatter(symbol.real, symbol.imag, s=3, c="red")
# paths = np.array(paths)
# plt.plot(paths.real, paths.imag, markersize=1, color="black")
# Detection circles
theta = np.linspace(0, 2 * np.pi, 100)
r = threshold_radius
for symbol in symbols.symbols_32APSK:
    plt.plot(
        symbol.number.real + r * np.cos(theta),
        symbol.number.imag + r * np.sin(theta),
        "black",
    )

plt.ylabel("Imaginary")
plt.xlabel("Real")

# plt.show()
fig2.savefig("results/32APSK/received_symbols.png", dpi=100)

# print("Number of bits received : ", len(bits_received))
# print("Bits received : ", ''.join([str(i) for i in bits_received]))
decoded_message = helpers.frombits(bits_received)
# print("The message emitted was : ", decoded_message)
