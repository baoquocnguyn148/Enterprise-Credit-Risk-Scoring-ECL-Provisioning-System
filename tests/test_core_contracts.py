import importlib
import os
import sys
import tempfile
import unittest

import pandas as pd


class CoreContractTests(unittest.TestCase):
    def test_lgd_estimate_is_available_and_deterministic(self):
        from src.lgd_ead_model import estimate_lgd

        df = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": ["Cash loans", "Consumer loans"],
                "AMT_CREDIT": [100_000.0, 100_000.0],
                "AMT_GOODS_PRICE": [0.0, 90_000.0],
            }
        )

        first = estimate_lgd(df)
        second = estimate_lgd(df)

        self.assertEqual(first.tolist(), second.tolist())
        self.assertTrue(first.between(0.20, 0.95).all())

    def test_api_can_import_without_loading_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_skip = os.environ.get("CREDIT_RISK_SKIP_AUTOLOAD")
            old_models = os.environ.get("CREDIT_RISK_MODELS_DIR")
            os.environ["CREDIT_RISK_SKIP_AUTOLOAD"] = "1"
            os.environ["CREDIT_RISK_MODELS_DIR"] = tmp_dir

            try:
                sys.modules.pop("app.api", None)
                api = importlib.import_module("app.api")
                self.assertEqual(api._available_model_paths(), [])
                self.assertTrue(api.ALLOWED_ORIGINS)
            finally:
                sys.modules.pop("app.api", None)
                if old_skip is None:
                    os.environ.pop("CREDIT_RISK_SKIP_AUTOLOAD", None)
                else:
                    os.environ["CREDIT_RISK_SKIP_AUTOLOAD"] = old_skip
                if old_models is None:
                    os.environ.pop("CREDIT_RISK_MODELS_DIR", None)
                else:
                    os.environ["CREDIT_RISK_MODELS_DIR"] = old_models


if __name__ == "__main__":
    unittest.main()
