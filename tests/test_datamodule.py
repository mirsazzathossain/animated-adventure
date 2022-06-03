import pathlib
import sys
import unittest

root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from utils.cifar10 import Cifar10DataModule

class TestCifar10DataModule(unittest.TestCase):
    EXPECTED_DIME = (3, 32, 32)
    EXPECTED_NUM_TRAIN_SAMPLES = 40000
    EXPECTED_NUM_VAL_SAMPLES = 10000
    EXPECTED_NUM_TEST_SAMPLES = 10000

    def setUp(self) -> None:
        self.data_module = Cifar10DataModule()
        self.data_module.prepare_data()
        self.data_module.setup()

    def test_datamodule(self):
        self.assertEqual(len(self.data_module.train_dataloader().dataset), self.EXPECTED_NUM_TRAIN_SAMPLES)
        self.assertEqual(len(self.data_module.val_dataloader().dataset), self.EXPECTED_NUM_VAL_SAMPLES)
        self.assertEqual(len(self.data_module.test_dataloader().dataset), self.EXPECTED_NUM_TEST_SAMPLES)

        batch= iter(self.data_module.train_dataloader())
        images, labels = batch.next()
        self.assertEqual(images.shape[1:], self.EXPECTED_DIME)

    def tearDown(self) -> None:
        self.data_module.teardown()
        return super().tearDown()

if __name__ == '__main__':
    test = TestCifar10DataModule()
    test.setUp()
    test.test_datamodule()
    test.tearDown()
