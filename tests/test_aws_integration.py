"""Comprehensive unit and integration test suite for Role 2 Cloud Infrastructure.

Tests:
1. 83-dim observation state vector generation and range constraints.
2. State Builder Lambda execution and SageMaker/fallback decision logic.
3. Microservice stages and empirical latency pipeline simulation.
4. Action Executor execution, realized reward calculation, and audit trail.
5. DynamoDB logging, querying, and system metrics caching.
6. CloudWatch metrics publishing and feedback loop state refresh.
7. End-to-end ClosedLoopController simulation and performance comparison.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Set environment variables for tests
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["DYNAMODB_TRANSACTION_TABLE"] = "TestTransactionLogs"
os.environ["DYNAMODB_METRICS_TABLE"] = "TestSystemMetricsCache"
os.environ["SLA_MS"] = "374.19"
os.environ["TIMEOUT_MS"] = "748.38"

from src.aws.dynamodb_helpers import (
    decimal_to_float,
    float_to_decimal,
)
from src.aws.lambda_action_executor import (
    compute_realized_reward,
    lambda_handler as action_executor_handler,
)
from src.aws.lambda_metrics_publisher import (
    lambda_handler as metrics_publisher_handler,
    publish_cloudwatch_metrics,
)
from src.aws.lambda_microservices import (
    STAGES,
    microservice_pipeline_handler,
    simulate_stage_latency,
)
from src.aws.lambda_state_builder import (
    build_observation_vector,
    invoke_routing_agent,
    lambda_handler as state_builder_handler,
)


class TestRole2CloudInfrastructure(unittest.TestCase):
    """Test suite for Role 2 components."""

    def setUp(self):
        self.sample_tx = {
            "transaction_id": "test_tx_001",
            "type": "PAYMENT",
            "amount": 1250.50,
            "fraud_risk": 0.03,
            "slot": 15,
            "window_id": 2,
        }
        self.sample_metrics_cache = {}
        for stage in STAGES:
            for cand in range(3):
                self.sample_metrics_cache[f"{stage}#{cand}"] = {
                    "stage": stage,
                    "candidate": cand,
                    "p50": 35.0 + cand * 5.0,
                    "p95": 70.0 + cand * 10.0,
                    "p99": 110.0 + cand * 15.0,
                    "n_calls": 120,
                    "sla_violation": 0,
                }

    def test_01_observation_vector_shape_and_ranges(self):
        """Verify the observation vector contains exactly 83 normalized floats in [0, 1]."""
        obs = build_observation_vector(
            tx_event=self.sample_tx,
            metrics_cache=self.sample_metrics_cache,
            timeout_ms=748.38,
        )
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (83,))
        self.assertEqual(obs.dtype, np.float32)

        # Check all values are within normalized bounds [0, 1]
        self.assertTrue(np.all(obs >= 0.0), "Observation has values < 0.0")
        self.assertTrue(np.all(obs <= 1.0), "Observation has values > 1.0")

        # Check transaction type one-hot encoding for PAYMENT
        # PAYMENT is index 3 in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
        type_slice = obs[75:80]
        self.assertEqual(list(type_slice), [0.0, 0.0, 0.0, 1.0, 0.0])

    def test_02_observation_transfer_type(self):
        """Verify TRANSFER flag and one-hot encoding for TRANSFER transactions."""
        tx_transfer = dict(self.sample_tx)
        tx_transfer["type"] = "TRANSFER"
        obs = build_observation_vector(
            tx_event=tx_transfer,
            metrics_cache=self.sample_metrics_cache,
        )
        # TRANSFER is index 4 in one-hot, and the 83rd element (index 82) is is_transfer
        self.assertEqual(obs[79], 1.0)
        self.assertEqual(obs[82], 1.0)

    def test_03_state_builder_lambda(self):
        """Test State Builder Lambda handler and local fallback decision."""
        event = {"body": json.dumps(self.sample_tx)}
        out = state_builder_handler(event)

        self.assertEqual(out["statusCode"], 200)
        self.assertEqual(len(out["observation"]), 83)
        self.assertIn("action", out)
        for stage in STAGES:
            self.assertIn(stage, out["action"])
            self.assertIn(out["action"][stage], (0, 1, 2))
        self.assertIn("explanation", out)
        self.assertIn("dominant_reward_component", out["explanation"])

    def test_04_microservice_stages_and_pipeline(self):
        """Test simulated microservice execution across all 5 stages."""
        for stage in STAGES:
            lat = simulate_stage_latency(stage, candidate=0)
            self.assertGreater(lat, 0.0)

        action = {"ingress": 0, "auth": 1, "routing": 0, "processor": 2, "db": 1}
        event = {
            "transaction_id": "test_tx_pipe",
            "transaction": self.sample_tx,
            "action": action,
            "sla_ms": 374.19,
            "timeout_ms": 748.38,
        }
        res = microservice_pipeline_handler(event)

        self.assertEqual(res["statusCode"], 200)
        self.assertIn("stage_latencies_ms", res)
        self.assertEqual(len(res["stage_latencies_ms"]), 5)

        computed_sum = round(sum(res["stage_latencies_ms"].values()), 2)
        self.assertAlmostEqual(res["e2e_ms"], computed_sum, places=1)
        self.assertEqual(res["success"], res["e2e_ms"] <= 748.38)
        self.assertEqual(res["sla_violation"], res["e2e_ms"] > 374.19)

    def test_05_reward_computation(self):
        """Test multi-objective reward calculation under SLA compliant and violating conditions."""
        # Case A: Low latency (compliant)
        reward_comp, exp_comp = compute_realized_reward(
            e2e_ms=100.0,
            sla_ms=374.19,
            timeout_ms=748.38,
            fraud_risk=0.02,
        )
        self.assertGreater(reward_comp, 0.0)
        self.assertEqual(exp_comp["terms"]["sla_violation_penalty"], 0.0)
        self.assertEqual(exp_comp["terms"]["success_bonus"], 1.0)

        # Case B: High latency (SLA violation & timeout failure)
        reward_viol, exp_viol = compute_realized_reward(
            e2e_ms=800.0,
            sla_ms=374.19,
            timeout_ms=748.38,
            fraud_risk=0.10,
        )
        self.assertLess(reward_viol, 0.0)
        self.assertEqual(exp_viol["terms"]["sla_violation_penalty"], -0.75)
        self.assertEqual(exp_viol["terms"]["success_bonus"], -1.0)

    @patch("src.aws.lambda_action_executor.log_transaction")
    def test_06_action_executor_lambda(self, mock_log):
        """Test Action Executor Lambda handles execution and logs audit records."""
        event = {
            "transaction_id": "tx_exec_001",
            "transaction": self.sample_tx,
            "action": {"ingress": 0, "auth": 1, "routing": 0, "processor": 0, "db": 1},
            "sla_ms": 374.19,
            "timeout_ms": 748.38,
        }
        res = action_executor_handler(event)
        self.assertEqual(res["statusCode"], 200)
        self.assertIn("reward", res)
        self.assertIn("explanation", res)
        mock_log.assert_called_once()

    def test_07_dynamodb_serialization_helpers(self):
        """Test float to Decimal conversion and round-tripping."""
        original = {"e2e_ms": 154.25, "scores": [0.1, 0.9], "name": "test"}
        dec = float_to_decimal(original)
        back = decimal_to_float(dec)
        self.assertEqual(back["name"], "test")
        self.assertAlmostEqual(back["e2e_ms"], 154.25)
        self.assertAlmostEqual(back["scores"][0], 0.1)

    def test_08_metrics_publisher_lambda(self):
        """Test Metrics Publisher Lambda formats proper HTTP response."""
        event = {
            "transaction_id": "tx_pub_001",
            "e2e_ms": 165.2,
            "success": True,
            "sla_violation": False,
            "reward": 0.85,
            "action": {"ingress": 0, "auth": 1, "routing": 0, "processor": 2, "db": 1},
            "stage_latencies_ms": {"ingress": 12.0, "auth": 25.0, "routing": 15.0, "processor": 65.0, "db": 48.2},
            "explanation": {"dominant_reward_component": "success_bonus"},
        }
        out = metrics_publisher_handler(event)
        self.assertEqual(out["statusCode"], 200)
        body = json.loads(out["body"])
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["transaction_id"], "tx_pub_001")

    def test_09_closed_loop_controller_simulation(self):
        """Test full closed-loop controller execution across a batch of transactions."""
        from src.aws.closed_loop_controller import ClosedLoopController

        controller = ClosedLoopController(sla_ms=374.19, timeout_ms=748.38)
        summary = controller.run_simulation(num_transactions=5, compare_static=True)

        self.assertEqual(summary["num_transactions"], 5)
        self.assertIn("autonomous_policy", summary)
        self.assertIn("static_baseline", summary)
        self.assertIn("mean_e2e_ms", summary["autonomous_policy"])
        self.assertIn("p95_e2e_ms", summary["autonomous_policy"])
        self.assertIn("success_rate", summary["autonomous_policy"])


if __name__ == "__main__":
    unittest.main()
