import unittest
from unittest.mock import patch, MagicMock
from src.model_deployer import ModelDeployer
from src.nim_client import NIMClient

class TestModelDeployer(unittest.TestCase):
    def setUp(self):
        self.nim_client = MagicMock(spec=NIMClient)
        self.model_deployer = ModelDeployer(self.nim_client)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_prepare_and_deploy_model(self, mock_tokenizer, mock_model):
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        self.nim_client.deploy_model.return_value = {"status": "success"}

        result = self.model_deployer.prepare_and_deploy_model("test_model", "path/to/model")
        self.assertEqual(result, {"status": "success"})
        mock_model.assert_called_once_with("path/to/model")
        mock_tokenizer.assert_called_once_with("path/to/model")
        self.nim_client.deploy_model.assert_called_once_with("test_model", "path/to/model")

    def test_inference(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        self.model_deployer.inference("test_model", "test input")
        self.nim_client.inference.assert_called_once_with("test_model", {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})

    def test_undeploy_model(self):
        self.model_deployer.undeploy_model("test_model")
        self.nim_client.undeploy_model.assert_called_once_with("test_model")

if __name__ == '__main__':
    unittest.main()