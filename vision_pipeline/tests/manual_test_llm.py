import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pipelines.verify_pipeline import VerifyPipeline
from modules.llm.verifier import LLMVerifier
from domain.label import LabelResult

class TestVerifyPipeline(unittest.TestCase):
    
    @patch('modules.llm.verifier.OpenAI')
    @patch('builtins.open') # Mock file opening for base64 encoding
    def test_pipeline_flow(self, mock_open, mock_openai):
        # Setup Mock API response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"verified": true, "reason": "Looks good", "confidence": 0.95}'
        mock_client.chat.completions.create.return_value = mock_response
        
        # Setup pipeline
        pipeline = VerifyPipeline(config_path="configs/llm.yaml")
        # We also need to avoid file open in base64 encoding inside Verifier
        # The patch 'builtins.open' handles the context manager
        
        # Mock detection results
        detection_results = [{
            "image_id": "test_img_1",
            "bboxes": [{"label": "cat"}],
            "crop_paths": ["/tmp/crop1.jpg"]
        }]
        
        # Run
        results = pipeline.run(detection_results)
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].verified)
        self.assertEqual(results[0].label, "cat")
        print("SUCCESS: Pipeline flow verified with mocked OpenAI.")

if __name__ == "__main__":
    unittest.main()
