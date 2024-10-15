import unittest
from unittest.mock import patch, MagicMock
from src.nim_client import NIMClient

class TestNIMClient(unittest.TestCase):
    def setUp(self):
        self.nim_client = NIMClient("http://localhost:8000")

    @patch('requests.post')
    def test_deploy_model(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response

        result = self.nim_client.deploy_model("test_model", "path/to/model")
        self.assertEqual(result, {"status": "success"})
        mock_post.assert_called_once_with("http://localhost:8000/models/test_model", json={"model_path": "path/to/model_onnx_optimized"})

    @patch('requests.post')
    def test_inference(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "test output"}
        mock_post.return_value = mock_response

        result = self.nim_client.inference("test_model", {"input": "test input"})
        self.assertEqual(result, {"output": "test output"})
        mock_post.assert_called_once_with("http://localhost:8000/models/test_model/infer", json={"input": "test input"})

    @patch('requests.get')
    def test_get_model_status(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "running"}
        mock_get.return_value = mock_response

        result = self.nim_client.get_model_status("test_model")
        self.assertEqual(result, {"status": "running"})
        mock_get.assert_called_once_with("http://localhost:8000/models/test_model/status")

    @patch('requests.delete')
    def test_undeploy_model(self, mock_delete):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_delete.return_value = mock_response

        result = self.nim_client.undeploy_model("test_model")
        self.assertEqual(result, {"status": "success"})
        mock_delete.assert_called_once_with("http://localhost:8000/models/test_model")

if __name__ == '__main__':
    unittest.main()