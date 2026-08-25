from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import JointVAE, load_joint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 reproduces the source setting: 10%% of LS")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(R / "data/wind_data_all_zone.csv")
    model = JointVAE(
        latent_size=args.latent_size, hidden_size=args.hidden_size
    ).to(device)
    # Reproduce scale_data_multi: one StandardScaler for the flattened joint
    # condition and one for the flattened joint target, both fitted on LS only.
    x_train_flat = data.x_train.reshape(len(data.x_train), -1)
    y_train_flat = data.y_train.reshape(len(data.y_train), -1)
    x_mean, x_std = x_train_flat.mean(0), x_train_flat.std(0)
    y_mean, y_std = y_train_flat.mean(0), y_train_flat.std(0)
    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1
    def scale_x(values):
        return ((values.reshape(len(values), -1) - x_mean) / x_std).reshape(values.shape).astype(np.float32)
    def scale_y(values):
        return ((values.reshape(len(values), -1) - y_mean) / y_std).reshape(values.shape).astype(np.float32)
    x_train, y_train = scale_x(data.x_train), scale_y(data.y_train)
    x_validation, y_validation = scale_x(data.x_validation), scale_y(data.y_validation)
    batch_size = args.batch_size or int(0.1 * len(y_train))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_x = torch.from_numpy(x_validation).to(device)
    validation_y = torch.from_numpy(y_validation).to(device)
    output = args.output or R / f"outputs/checkpoints/joint_vae_z1-5_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    # Same wind-VAE optimizer settings as the source baseline.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=10 ** -3.4, weight_decay=10 ** -3.4
    )
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = []
        for condition, target in loader:
            condition, target = condition.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(condition, target)
            loss.backward()
            optimizer.step()
            training.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            # Fixed randomness makes checkpoint selection deterministic.
            torch.manual_seed(10000 + args.seed)
            validation, parts = model.loss(validation_x, validation_y)
        row = {
            "epoch": epoch,
            "train_elbo": float(np.mean(training)),
            "validation_elbo": float(validation),
            "validation_reconstruction": float(parts["reconstruction"]),
            "validation_kl": float(parts["kl"]),
        }
        history.append(row)
        if row["validation_elbo"] < best:
            best = row["validation_elbo"]
            torch.save(model.state_dict(), output / "best_state.pt")
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps(row), flush=True)

    model.load_state_dict(torch.load(output / "best_state.pt", weights_only=True))
    config = {
        "latent_size": args.latent_size,
        "hidden_size": args.hidden_size,
        "hidden_layers": 1,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "seed": args.seed,
    }
    torch.save({
        "state_dict": model.state_dict(), "config": config,
        "best_validation_elbo": best,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "train_dates": data.train_dates,
        "validation_dates": data.validation_dates,
        "test_dates": data.test_dates,
    }, output / "joint_vae.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved={output / 'joint_vae.pt'} device={device} best={best:.6f}")


if __name__ == "__main__":
    main()
