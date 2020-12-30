
import unittest
import os

from core.AM import modulation, demodulation

class TestAM(unittest.TestCase):

    def test_check_existance_files(self):
        CWD = os.getcwd()
        self.assertEqual( os.path.exists(CWD + '/results/AM/message.wav'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/AM/modulatedSignal.wav'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/AM/modulation.png'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/AM/demodulatedSignal.wav'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/AM/demodulation.png'), 1)

if __name__ == '__main__':
    unittest.main()