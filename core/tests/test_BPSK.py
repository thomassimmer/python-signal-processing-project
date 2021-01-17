import unittest
import os

from core.BPSK import modulation, demodulation


class TestBPSK(unittest.TestCase):
    def test_signal_emission_reception(self):
        print("Message to transmit : ", modulation.message_to_transmit)
        print("Message decoded : ", demodulation.decoded_message)
        self.assertEqual(modulation.message_to_transmit, demodulation.decoded_message)


if __name__ == "__main__":
    unittest.main()
