import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from core.utils import helpers, filters

# Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
# Modulating wave m(t)=A_m *[ I(t)*cos(2*pi*f_m*t) + Q(t)*sin(2*pi*f_m*t) ]
# Modulated wave s(t)=A_c * cos(2*pi*f_c*t) * [1+m*m(t)]

# NOTE : Be careful, when changing the message.
# if longer, you might need to increase the endTime variable as the message needs more time to be transmitted.

# Variables for signal characterization
A_c = 1
f_c = 10 * (10 ** 3)
A_m = 1
f_m = 200
T_m = 1 / f_m
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
 
# Generate carrier
carrier = A_c*np.cos(2*np.pi*f_c*t)

# Generate bits from message
message_to_transmit = "Hello The World ! QPSK is great. "
bits_to_transmit =  helpers.tobits(message_to_transmit)
# print("Message to transmit : ", message_to_transmit)
# print("Number of bits to transmit : ", len(bits_to_transmit))
# print("Bits to transmit : ", "".join([str(i) for i in bits_to_transmit] ))

# Generate I and Q
I, Q, I_channel, Q_channel = [], [], [], []

# Generate symbols
symbol_for_11 = np.exp(1j * (np.pi / 4))
symbol_for_01 = np.exp(1j * (3 * np.pi / 4))
symbol_for_00 = np.exp(1j * (-3 * np.pi / 4))
symbol_for_10 = np.exp(1j * (- np.pi / 4))
symbol_for_1 = np.exp(1j * 0)
symbol_for_0 = np.exp(1j * (- np.pi))
k = 0
for i in t:
    # apply phase shift of PI every T if bit == 0
    if (i < (k + 1) * T_m):
        if ( int(len(bits_to_transmit) / 2) + 1 > k):
            # Catch 2 bits from the list of bits to transmit
            if (len(bits_to_transmit) > 2 * k + 1):
                bits_to_encode = ''.join([str(i) for i in bits_to_transmit[2 * k : 2 * k + 2]])
            # If number of bits is even, we use BPSK for the last one
            else:
                bits_to_encode = str(bits_to_transmit[-1])

            if (bits_to_encode == '11'):                
                I.append(symbol_for_11.real)
                Q.append(symbol_for_11.imag)
            elif (bits_to_encode == '01'):                
                I.append(symbol_for_01.real)
                Q.append(symbol_for_01.imag)
            elif (bits_to_encode == '00'):                
                I.append(symbol_for_00.real)
                Q.append(symbol_for_00.imag)
            elif (bits_to_encode == '10'):                
                I.append(symbol_for_10.real)
                Q.append(symbol_for_10.imag)
            elif (bits_to_encode == '0'): 
                I.append(symbol_for_0.real)
                Q.append(symbol_for_0.imag)
            elif (bits_to_encode == '1'):                
                I.append(symbol_for_1.real)
                Q.append(symbol_for_1.imag)
        else:
            I.append(0)
            Q.append(0)
    else:
        k = k + 1
        I.append(I[-1])
        Q.append(Q[-1])

# Apply lowpass filter to I and Q
I = filters.butter_lowpass_filter(I, 500, samplingRate)
Q = filters.butter_lowpass_filter(Q, 500, samplingRate)

# Create I & Q channels, product with cos/sin at frequence f_m
I_channel = I * A_m * np.cos(2 * np.pi * f_m * t)
Q_channel = Q * A_m * np.sin(2 * np.pi * f_m * t)

# Generate modulator signal and product with carrier
modulator = np.sum([I_channel, Q_channel], axis=0)
modulator = filters.butter_lowpass_filter(modulator, 1000, 44100)
product = A_c * np.cos(2 * np.pi * f_c * t) * (1 + modulation_index * modulator)

# Apply passband filter on the product (which we emit)
product = filters.butter_bandpass_filter(product, 9000, 11000, samplingRate)

# Generate spectra
carrier_spectrum = np.fft.fft(carrier)/len(carrier)
modulator_spectrum = np.fft.fft(modulator)/len(modulator)
product_spectrum = np.fft.fft(product)/len(product)

# Write wav files
write("results/QPSK/message.wav", samplingRate, modulator)
write("results/QPSK/modulatedSignal.wav", samplingRate, product)

# Show plots
plt.subplot(3,2,1)
plt.title('QPSK Modulation')
plt.plot(t[0:3000], modulator[0:3000],'g')
# plt.plot(t[0:3000], I[0:3000],'b')
# plt.plot(t[0:3000], Q[0:3000],'r')
# plt.plot(t[0:3000], I_channel[0:3000],'b')
# plt.plot(t[0:3000], Q_channel[0:3000],'r')
plt.ylabel('Amplitude')
plt.xlabel('Message signal')

plt.subplot(3,2, 2)
plt.plot(f[0:1500], modulator_spectrum[0:1500])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplot(3,2,3)
plt.plot(t, carrier, 'r')
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')

plt.subplot(3,2,4)
plt.plot(f[9950:10050], carrier_spectrum[9950:10050])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplot(3,2,5)
plt.plot(t, product, color="purple")
plt.ylabel('Amplitude')
plt.xlabel('AM signal')

plt.subplot(3,2,6)
plt.plot(f[9800:15200], product_spectrum[9800:15200])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')

plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()

# Save plot
# plt.show()
fig.savefig('results/QPSK/modulation.png', dpi=100)


# # --------------- Symbol detection before filters... ------------------
# k = 0
# symbols, paths = [], []
# I_values, Q_values = [], []
# bits_received = []
# threshold_radius = 0.4
# symbol_for_0 = np.complex(real=-1, imag=0)
# symbol_for_1 = np.complex(real=1, imag=0)

# I_channel = modulator * 2 * A_m * np.cos(2 * np.pi * f_m * t)
# Q_channel = modulator * 2 * A_m * np.sin(2 * np.pi * f_m * t)

# I_channel = filters.butter_lowpass_filter(I_channel, f_m , 44100)
# Q_channel = filters.butter_lowpass_filter(Q_channel, f_m , 44100)

# I_spectrum = np.fft.fft(I)/len(I)
# Q_spectrum = np.fft.fft(Q_channel)/len(Q)
# I_channel_spectrum = np.fft.fft(I_channel)/len(I)

# fig2 = plt.figure(2)
# # plt.plot(f[0:1500], I_channel_spectrum[0:1500], 'g')
# # plt.plot(f[0:1500], I_spectrum[0:1500], 'c')
# # plt.plot(f[0:1500], Q_spectrum[0:1500], 'r')
# # plt.plot(t[0:3000], modulator[0:3000],'g')
# plt.plot(t[0:3000], I_channel[0:3000],'b')
# plt.plot(t[0:3000], Q_channel[0:3000],'r')
# # plt.plot(f, modulator_spectrum)
# plt.ylabel('Amplitude')
# plt.xlabel('Frequency')

# for i, time in enumerate(t):
#     # every instant check value for I and Q
#     if (time < (k + 1) * T_m):
#         paths.append(np.complex(real=I_channel[i], imag=Q_channel[i]))
#         I_values.append(I_channel[i])
#         Q_values.append(Q_channel[i])

#     # every T, make an average for I and Q to get symbol
#     else:
#         average_I = sum(I_values) / len(I_values)
#         average_Q = sum(Q_values) / len(Q_values)
#         symbols.append(np.complex(real=average_I, imag=average_Q))
#         k = k + 1
#         I_values, Q_values = [], []
#         # Check distance to symbols
#         if (np.linalg.norm(symbol_for_0 - symbols[-1]) < threshold_radius):
#             bits_received.append(0)
#         elif (np.linalg.norm(symbol_for_1 - symbols[-1]) < threshold_radius):
#             bits_received.append(1)
            
# fig3 = plt.figure(3)
# axes = plt.gca()
# axes.set_xlim([-2,2])
# axes.set_ylim([-2,2])
# for symbol in symbols:
#     plt.scatter(symbol.real, symbol.imag, s=3, c='red')
# paths = np.array(paths)
# plt.plot(paths.real, paths.imag, markersize=1, color="black")
# # Detection circles
# theta = np.linspace(0, 2*np.pi, 100)
# r = threshold_radius
# x1 = symbol_for_0.real + r*np.cos(theta)
# x2 = symbol_for_0.imag + r*np.sin(theta)
# x3 = symbol_for_1.real + r*np.cos(theta)
# x4 = symbol_for_1.imag + r*np.sin(theta)
# plt.plot(x1, x2, 'black')
# plt.plot(x3, x4, 'black')
# plt.ylabel('Imaginary')
# plt.xlabel('Real')

# # plt.show()
# fig3.savefig('results/QPSK/emitted_symbols.png', dpi=100)

# print("Number of bits received : ", len(bits_received))
# print("Bits received : ", "".join([str(i) for i in bits_received] ))
# decoded_message = helpers.frombits(bits_received)
# print("The message emitted was : ", decoded_message)