import numpy as np

class Symbol:
  def __init__(self, string, number):
    self.string = string
    self.number = number

# BPSK
symbol_for_1 = Symbol('1', np.exp(1j * 0))
symbol_for_0 = Symbol('0', np.exp(1j * (- np.pi)))

# QPSK
symbol_for_11 = Symbol('11', np.exp(1j * (np.pi / 4)))
symbol_for_01 = Symbol('01', np.exp(1j * (3 * np.pi / 4)))
symbol_for_00 = Symbol('00', np.exp(1j * (-3 * np.pi / 4)))
symbol_for_10 = Symbol('10', np.exp(1j * (- np.pi / 4)))

# 32APSK (3/4)
r1 = 1
gamma_1 = 2.84
gamma_2 = 5.27
r2 = r1 * gamma_1
r3 = r1 * gamma_2
symbol_for_10001 = Symbol('10001', r1 * np.exp(1j * (np.pi / 4)))
symbol_for_10101 = Symbol('10101', r1 * np.exp(1j * (3 * np.pi / 4)))
symbol_for_10111 = Symbol('10111', r1 * np.exp(1j * (-3 * np.pi / 4)))
symbol_for_10011 = Symbol('10011', r1 * np.exp(1j * (- np.pi / 4)))
symbol_for_10000 = Symbol('10000', r2 * np.exp(1j * (np.pi / 12)))
symbol_for_00000 = Symbol('00000', r2 * np.exp(1j * (np.pi / 4)))
symbol_for_00001 = Symbol('00001', r2 * np.exp(1j * (5 * np.pi / 12)))
symbol_for_00101 = Symbol('00101', r2 * np.exp(1j * (7 * np.pi / 12)))
symbol_for_00100 = Symbol('00100', r2 * np.exp(1j * (3 * np.pi / 4)))
symbol_for_10100 = Symbol('10100', r2 * np.exp(1j * (11 * np.pi / 12)))
symbol_for_10110 = Symbol('10110', r2 * np.exp(1j * (- 11 * np.pi / 12)))
symbol_for_00110 = Symbol('00110', r2 * np.exp(1j * (- 3 * np.pi / 4)))
symbol_for_00111 = Symbol('00111', r2 * np.exp(1j * (- 7 * np.pi / 12)))
symbol_for_00011 = Symbol('00011', r2 * np.exp(1j * (- 5 * np.pi / 12)))
symbol_for_00010 = Symbol('00010', r2 * np.exp(1j * (- np.pi / 4)))
symbol_for_10010 = Symbol('10010', r2 * np.exp(1j * (- np.pi / 12)))
symbol_for_11000 = Symbol('11000', r3 * np.exp(1j * 0))
symbol_for_01000 = Symbol('01000', r3 * np.exp(1j * (np.pi / 8)))
symbol_for_11001 = Symbol('11001', r3 * np.exp(1j * (np.pi / 4)))
symbol_for_01001 = Symbol('01001', r3 * np.exp(1j * (3 * np.pi / 8)))
symbol_for_01101 = Symbol('01101', r3 * np.exp(1j * (np.pi / 2)))
symbol_for_11101 = Symbol('11101', r3 * np.exp(1j * (5 * np.pi / 8)))
symbol_for_01100 = Symbol('01100', r3 * np.exp(1j * (3 * np.pi / 4)))
symbol_for_11100 = Symbol('11100', r3 * np.exp(1j * (7 * np.pi / 8)))
symbol_for_11110 = Symbol('11110', r3 * np.exp(1j * (np.pi)))
symbol_for_01110 = Symbol('01110', r3 * np.exp(1j * (- 7 * np.pi / 8)))
symbol_for_11111 = Symbol('11111', r3 * np.exp(1j * (- 3 * np.pi / 4)))
symbol_for_01111 = Symbol('01111', r3 * np.exp(1j * (- 5 * np.pi / 8)))
symbol_for_01011 = Symbol('01011', r3 * np.exp(1j * (- np.pi / 2)))
symbol_for_11011 = Symbol('11011', r3 * np.exp(1j * (- 3 * np.pi / 8)))
symbol_for_01010 = Symbol('01010', r3 * np.exp(1j * (- np.pi / 4)))
symbol_for_11010 = Symbol('11010', r3 * np.exp(1j * (- np.pi / 8)))

symbols_32APSK = [

    symbol_for_10001,
    symbol_for_10101,
    symbol_for_10111,
    symbol_for_10011,
    symbol_for_10000,
    symbol_for_00000,
    symbol_for_00001,
    symbol_for_00101,
    symbol_for_00100,
    symbol_for_10100,
    symbol_for_10110,
    symbol_for_00110,
    symbol_for_00111,
    symbol_for_00011,
    symbol_for_00010,
    symbol_for_10010,
    symbol_for_11000,
    symbol_for_01000,
    symbol_for_11001,
    symbol_for_01001,
    symbol_for_01101,
    symbol_for_11101,
    symbol_for_01100,
    symbol_for_11100,
    symbol_for_11110,
    symbol_for_01110,
    symbol_for_11111,
    symbol_for_01111,
    symbol_for_01011,
    symbol_for_11011,
    symbol_for_01010,
    symbol_for_11010
]