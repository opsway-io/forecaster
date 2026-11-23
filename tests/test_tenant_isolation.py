import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data
import train
import predict


class TestForecasterTenantIsolation(unittest.TestCase):
    def test_get_monitor_data_includes_team_id_parameterized(self):
        client = MagicMock()
        data.get_monitor_data(client, monitor_id=10, team_id=42)

        self.assertEqual(client.query_df.call_count, 1)
        query, kwargs = client.query_df.call_args.args[0], client.query_df.call_args.kwargs
        self.assertIn("WHERE team_id = {team_id:UInt64} AND monitor_id = {monitor_id:UInt64}", query)
        self.assertEqual(kwargs["parameters"], {"team_id": 42, "monitor_id": 10})

    def test_get_monitor_data_without_team_id_still_parameterized(self):
        client = MagicMock()
        data.get_monitor_data(client, monitor_id=10)

        query, kwargs = client.query_df.call_args.args[0], client.query_df.call_args.kwargs
        self.assertIn("WHERE monitor_id = {monitor_id:UInt64}", query)
        self.assertNotIn("team_id", query)
        self.assertEqual(kwargs["parameters"], {"monitor_id": 10})

    def test_model_registry_names_scoped_by_team(self):
        self.assertEqual(train.model_registry_name(5, team_id=7), "forecaster_team_7_monitor_5")
        self.assertEqual(predict._model_registry_name(5, team_id=7), "forecaster_team_7_monitor_5")
        self.assertEqual(train.model_registry_name(5), "forecaster_monitor_5")

    def test_profile_cache_keys_are_tenant_isolated(self):
        predict._profile_cache.clear()
        with patch("data.get_clickhouse_client") as mock_ch, patch("data.get_monitor_data") as mock_get:
            mock_ch.return_value = MagicMock()
            mock_get.return_value = predict.pd.DataFrame()

            predict.get_monitor_profile(monitor_id=1, team_id=10)
            predict.get_monitor_profile(monitor_id=1, team_id=20)

            self.assertIn((10, 1), predict._profile_cache)
            self.assertIn((20, 1), predict._profile_cache)
            self.assertEqual(mock_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
