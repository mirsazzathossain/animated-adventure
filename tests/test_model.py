import pathlib
import sys
import unittest
import torch

root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from models.lit_resnet import LitResnet
from configs.setup_configs import load_config


class TestModel(unittest.TestCase):
    INPUT_CHANNELS = 3
    NUM_CLASSES = 10

    def setUp(self) -> None:
        self.configs = load_config()
        self.model = LitResnet(
            lr = self.configs[1]['exp_params']['lr'],
            momentum = self.configs[1]['exp_params']['momentum'],
            weight_decay = self.configs[1]['exp_params']['weight_decay'],
            batch_size = self.configs[1]['exp_params']['batch_size'],
            max_lr = self.configs[1]['exp_params']['max_lr'],
        )

    def test_model(self):
        x = torch.randn(self.configs[1]['exp_params']['batch_size'], self.INPUT_CHANNELS, self.configs[0]['model_params']['in_size'], self.configs[0]['model_params']['in_size']).float()

        logits = self.model.forward(x)
        self.assertEqual(logits.shape[0], self.configs[1]['exp_params']['batch_size'])
        self.assertEqual(logits.shape[-1], self.NUM_CLASSES)

    def tearDown(self) -> None:
        self.data_module.teardown()
        return super().tearDown()

if __name__ == '__main__':
    test = TestModel()
    test.setUp()
    test.test_model()