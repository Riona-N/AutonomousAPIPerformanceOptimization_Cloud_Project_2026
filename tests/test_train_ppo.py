from __future__ import annotations

from pathlib import Path

from train.train_ppo import main


def test_train_ppo_smoke(tmp_path, monkeypatch):
    model_path = tmp_path / "ppo_agent"
    metrics_path = tmp_path / "ppo_metrics.json"
    tb_log = tmp_path / "tensorboard" / "ppo"
    checkpoint_dir = tmp_path / "checkpoints" / "ppo"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ppo.py",
            "--timesteps",
            "20",
            "--episodes",
            "1",
            "--seed",
            "7",
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
            "--tensorboard-log",
            str(tb_log),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-freq",
            "20",
        ],
    )

    main()

    assert model_path.with_suffix(".zip").exists()
    assert metrics_path.exists()
    assert tb_log.exists()
    assert checkpoint_dir.exists()
