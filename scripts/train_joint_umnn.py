from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import build_joint_umnn, load_joint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(R / "data/wind_data_all_zone.csv")
    x_flat = data.x_train.reshape(len(data.x_train), -1)
    y_flat = data.y_train.reshape(len(data.y_train), -1)
    x_mean, x_std = x_flat.mean(0), x_flat.std(0)
    y_mean, y_std = y_flat.mean(0), y_flat.std(0)
    x_std[x_std == 0] = 1; y_std[y_std == 0] = 1
    x_train = ((x_flat - x_mean) / x_std).astype(np.float32)
    y_train = ((y_flat - y_mean) / y_std).astype(np.float32)
    x_validation = ((data.x_validation.reshape(50, -1) - x_mean) / x_std).astype(np.float32)
    y_validation = ((data.y_validation.reshape(50, -1) - y_mean) / y_std).astype(np.float32)
    batch_size = int(0.1 * len(y_train))
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = build_joint_umnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    validation_x = torch.from_numpy(x_validation).to(device)
    validation_y = torch.from_numpy(y_validation).to(device)
    output = args.output or R / f"outputs/checkpoints/joint_umnn_z1-5_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    history = []; best = float("inf"); started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for condition, target in loader:
            condition, target = condition.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            likelihood, _ = model.compute_ll(target, condition)
            loss = -likelihood.mean(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            likelihood, _ = model.compute_ll(validation_y, validation_x)
            validation = float(-likelihood.mean())
        row = {"epoch": epoch, "train_nll": float(np.mean(losses)),
               "validation_nll": validation,
               "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        if np.isfinite(validation) and validation < best:
            best = validation; torch.save(model.state_dict(), output / "best_state.pt")
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / "best_state.pt", weights_only=True))
    config = {"target_size": 120, "condition_size": 480, "nb_steps": 1,
              "hidden": [300] * 4, "out_size": 20, "integrand_net": [40] * 3,
              "integration_steps": 50, "epochs": args.epochs,
              "batch_size": batch_size, "seed": args.seed}
    torch.save({"state_dict": model.state_dict(), "config": config,
                "best_validation_nll": best, "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std, "train_dates": data.train_dates,
                "validation_dates": data.validation_dates,
                "test_dates": data.test_dates}, output / "joint_umnn.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved={output / 'joint_umnn.pt'} device={device} best={best:.6f}")


if __name__ == "__main__":
    main()
