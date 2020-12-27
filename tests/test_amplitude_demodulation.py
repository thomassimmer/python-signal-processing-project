
import unittest
import os

class TestAmplitudeDemodulation(unittest.TestCase):

    def test_check_existance_files(self):
        CWD = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.assertEqual( os.path.exists(CWD + '/results/demodulatedSignal.wav'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/demodulation.png'), 1)

if __name__ == '__main__':
    unittest.main()