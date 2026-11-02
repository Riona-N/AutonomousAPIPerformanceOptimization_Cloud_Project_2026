"""Closed-Loop Controller and Evaluation Engine (Role 2).

Orchestrates the complete autonomous feedback loop:
1. Ingests payment requests (from stream, simulation, or batch).
2. Triggers State Builder -> RL Inference -> Action Executor -> Microservice Pipeline -> Metrics Publisher.
3. Observes how subsequent routing decisions adapt to changing infrastructure load.
4. Computes empirical statistics (P50, P95, P99 latency, SLA compliance, success rate, reward).
5. Compares against the static baseline to quantify optimization performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.aws.lambda_state_builder import lambda_handler as state_builder_handler
    from src.aws.lambda_action_executor import lambda_handler as action_executor_handler
    from src.aws.lambda_metrics_publisher import lambda_handler as metrics_publisher_handler
except ImportError:
    from lambda_state_builder import lambda_handler as state_builder_handler
    from lambda_action_executor import lambda_handler as action_executor_handler
    from lambda_metrics_publisher import lambda_handler as metrics_publisher_handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ClosedLoopController:
    """Manages the autonomous execution cycle across simulated cloud microservices."""

    def __init__(
        self,
        sla_ms: float = 374.19,
        timeout_ms: float = 748.38,
        data_dir: str | Path = "dataset/processed",
    ):
        self.sla_ms = float(sla_ms)
        self.timeout_ms = float(timeout_ms)
        self.data_dir = Path(data_dir)

        # Load transactions if available
        tx_path = self.data_dir / "linkage_tx_sample.parquet"
        if not tx_path.exists():
            tx_path = self.data_dir / "linkage_tx.parquet"

        if tx_path.exists():
            df = pd.read_parquet(tx_path)
            self.tx_pool = df.to_dict(orient="records")
        else:
            self.tx_pool = []

    def execute_single_transaction(self, tx_data: dict[str, Any]) -> dict[str, Any]:
        """Run one transaction through the complete 4-step closed-loop cycle."""
        # Step 1: State Builder (constructs 83-dim state + invokes policy)
        state_payload = {
            "transaction_id": str(tx_data.get("transaction_id", f"tx_{int(time.time() * 1000)}")),
            "type": tx_data.get("type", "PAYMENT"),
            "amount": float(tx_data.get("amount", 100.0)),
            "fraud_risk": float(tx_data.get("fraud_risk", 0.01)),
            "slot": int(tx_data.get("slot", 0)),
            "window_id": int(tx_data.get("window_id", 0)),
            "sla_ms": self.sla_ms,
            "timeout_ms": self.timeout_ms,
        }
        sb_out = state_builder_handler(state_payload)

        # Step 2: Action Executor & Microservice Pipeline
        exec_payload = {
            "transaction_id": sb_out["transaction_id"],
            "transaction": sb_out["transaction"],
            "action": sb_out["action"],
            "explanation": sb_out["explanation"],
            "sla_ms": self.sla_ms,
            "timeout_ms": self.timeout_ms,
        }
        ae_out = action_executor_handler(exec_payload)

        # Step 3: Metrics Publisher (CloudWatch emission & state cache refresh)
        pub_out = metrics_publisher_handler(ae_out)

        return {
            "transaction_id": ae_out["transaction_id"],
            "type": sb_out["transaction"]["type"],
            "action": ae_out["action"],
            "e2e_ms": ae_out["e2e_ms"],
            "stage_latencies_ms": ae_out["stage_latencies_ms"],
            "success": ae_out["success"],
            "sla_violation": ae_out["sla_violation"],
            "reward": ae_out["reward"],
            "explanation": ae_out["explanation"],
            "response": json.loads(pub_out["body"]),
        }

    def run_simulation(
        self,
        num_transactions: int = 100,
        compare_static: bool = True,
    ) -> dict[str, Any]:
        """Execute a batch of transactions and produce comparative evaluation metrics."""
        logger.info(f"Starting closed-loop simulation with {num_transactions} transactions...")

        results = []
        static_results = []

        tx_count = min(num_transactions, len(self.tx_pool)) if self.tx_pool else num_transactions

        for i in range(tx_count):
            if self.tx_pool:
                item = dict(self.tx_pool[i])
                item["transaction_id"] = str(item.get("txn_id", f"txn_{i}"))
            else:
                item = {
                    "transaction_id": f"synthetic_txn_{i}",
                    "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]),
                    "amount": float(np.random.exponential(500.0)),
                    "fraud_risk": float(np.random.beta(0.5, 20.0)),
                    "slot": i % 60,
                    "window_id": i % 30,
                }

            # Run autonomous closed loop
            res = self.execute_single_transaction(item)
            results.append(res)

            # Compare against static baseline (candidate 0 for all stages)
            if compare_static:
                static_action = {"ingress": 0, "auth": 0, "routing": 0, "processor": 0, "db": 0}
                static_exec_payload = {
                    "transaction_id": f"static_{res['transaction_id']}",
                    "transaction": item,
                    "action": static_action,
                    "sla_ms": self.sla_ms,
                    "timeout_ms": self.timeout_ms,
                }
                static_out = action_executor_handler(static_exec_payload)
                static_results.append(static_out)

        # Aggregate statistics
        latencies = [r["e2e_ms"] for r in results]
        sla_viols = [r["sla_violation"] for r in results]
        successes = [r["success"] for r in results]
        rewards = [r["reward"] for r in results]

        summary = {
            "num_transactions": tx_count,
            "autonomous_policy": {
                "mean_e2e_ms": float(np.mean(latencies)),
                "p50_e2e_ms": float(np.percentile(latencies, 50)),
                "p95_e2e_ms": float(np.percentile(latencies, 95)),
                "p99_e2e_ms": float(np.percentile(latencies, 99)),
                "sla_violation_rate": float(np.mean(sla_viols)),
                "success_rate": float(np.mean(successes)),
                "mean_reward": float(np.mean(rewards)),
            },
        }

        if compare_static and static_results:
            s_latencies = [r["e2e_ms"] for r in static_results]
            s_sla_viols = [r["sla_violation"] for r in static_results]
            s_successes = [r["success"] for r in static_results]
            s_rewards = [r["reward"] for r in static_results]

            summary["static_baseline"] = {
                "mean_e2e_ms": float(np.mean(s_latencies)),
                "p50_e2e_ms": float(np.percentile(s_latencies, 50)),
                "p95_e2e_ms": float(np.percentile(s_latencies, 95)),
                "p99_e2e_ms": float(np.percentile(s_latencies, 99)),
                "sla_violation_rate": float(np.mean(s_sla_viols)),
                "success_rate": float(np.mean(s_successes)),
                "mean_reward": float(np.mean(s_rewards)),
            }

            # Percentage improvements
            p95_improve = (summary["static_baseline"]["p95_e2e_ms"] - summary["autonomous_policy"]["p95_e2e_ms"]) / summary["static_baseline"]["p95_e2e_ms"] * 100.0
            sla_improve = (summary["static_baseline"]["sla_violation_rate"] - summary["autonomous_policy"]["sla_violation_rate"]) * 100.0

            summary["improvements"] = {
                "p95_latency_reduction_pct": round(p95_improve, 2),
                "sla_violation_reduction_pct_pts": round(sla_improve, 2),
            }

        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="Number of transactions to run")
    parser.add_argument("--out", type=Path, default=Path("results/closed_loop_evaluation.json"))
    args = parser.parse_args()

    controller = ClosedLoopController()
    summary = controller.run_simulation(num_transactions=args.count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== Closed Loop Evaluation Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
