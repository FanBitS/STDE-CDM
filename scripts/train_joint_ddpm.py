from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import JointDDPM, load_joint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(R / "data/wind_data_all_zone.csv")
    model = JointDDPM().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x_train), torch.from_numpy(data.y_train)),
        batch_size=50, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_x = torch.from_numpy(data.x_validation).to(device)
    validation_y = torch.from_numpy(data.y_validation).to(device)
    output = args.output or R / f"outputs/checkpoints/joint_ddpm_z1-5_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf"); history = []
    fixed_generator = torch.Generator(device=device).manual_seed(12345)
    fixed_steps = torch.randint(0, 50, (50,), device=device, generator=fixed_generator)
    fixed_noise = torch.randn(validation_y.shape, device=device, generator=fixed_generator)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for nwp, target in loader:
            nwp, target = nwp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(nwp, target); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(target)
        model.eval()
        with torch.no_grad():
            validation = float(model.loss(validation_x, validation_y,
                                          fixed_steps, fixed_noise))
        row = {"epoch": epoch, "train_mse": total / len(data.y_train),
               "validation_mse": validation}
        history.append(row)
        if validation < best:
            best = validation
            torch.save(model.state_dict(), output / "best_state.pt")
        if epoch == 1 or epoch % 50 == 0:
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / "best_state.pt", weights_only=True))
    config = {"farms": 5, "nwp_features": 4, "channels": 64, "layers": 3,
              "steps": 50, "beta_start": 1e-4, "beta_end": 0.05,
              "epochs": args.epochs, "batch_size": 50, "seed": args.seed}
    torch.save({"state_dict": model.state_dict(), "config": config,
                "best_validation_mse": best, "train_dates": data.train_dates,
                "validation_dates": data.validation_dates,
                "test_dates": data.test_dates}, output / "joint_ddpm.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved={output / 'joint_ddpm.pt'} device={device} best={best:.6f}")


if __name__ == "__main__":
    main()
