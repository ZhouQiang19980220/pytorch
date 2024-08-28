import unittest

class TestPytorchAPI(unittest.TestCase):
    def test_torch(self):
        import torch
        print(torch.__version__)

    def test_test(self):
        self.assertEqual(1, 1)