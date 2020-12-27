
import unittest
import os


class TestAmplitudeModulation(unittest.TestCase):

    def test_check_existance_files(self):
        CWD = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.assertEqual( os.path.exists(CWD + '/results/message.wav'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/modulatedSignal.wav'), 1)
        self.assertEqual( os.path.exists(CWD + '/results/modulation.png'), 1)

if __name__ == '__main__':
    unittest.main()