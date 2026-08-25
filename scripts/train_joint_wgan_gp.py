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
from stde_cdm import JointWGANGenerator, JointWGANCritic, load_joint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(R / "data/wind_data_all_zone.csv")

    x_flat = data.x_train.reshape(len(data.x_train), -1)
    y_flat = data.y_train.reshape(len(data.y_train), -1)
    x_mean, x_std = x_flat.mean(0), x_flat.std(0)
    y_mean, y_std = y_flat.mean(0), y_flat.std(0)
    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1
    x_train = ((x_flat - x_mean) / x_std).astype(np.float32)
    y_train = ((y_flat - y_mean) / y_std).astype(np.float32)
    x_validation = ((data.x_validation.reshape(50, -1) - x_mean) / x_std).astype(np.float32)
    y_validation = ((data.y_validation.reshape(50, -1) - y_mean) / y_std).astype(np.float32)
    batch_size = args.batch_size or int(0.1 * len(y_train))
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))

    generator = JointWGANGenerator().to(device)
    critic = JointWGANCritic().to(device)
    generator.initialize()
    critic.initialize()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4,
                                   betas=(0.0, 0.9), weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=2e-4,
                                   betas=(0.0, 0.9), weight_decay=1e-4)
    validation_x = torch.from_numpy(x_validation).to(device)
    validation_y = torch.from_numpy(y_validation).to(device)
    output = args.output or R / f"outputs/checkpoints/joint_wgan_gp_z1-5_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    critic_updates = 0

    for epoch in range(1, args.epochs + 1):
        generator.train(); critic.train()
        d_values, g_values, gp_values = [], [], []
        for condition, real in loader:
            condition, real = condition.to(device), real.to(device)
            noise = torch.randn(len(real), generator.latent_size, device=device)
            fake = generator(noise, condition)
            optimizer_d.zero_grad(set_to_none=True)
            d_loss, pieces = critic.loss(fake.detach(), real, condition)
            d_loss.backward(); optimizer_d.step()
            d_values.append(float(d_loss.detach()))
            gp_values.append(float(pieces["gradient_penalty"].detach()))
            critic_updates += 1
            if critic_updates % 5 == 0:
                optimizer_g.zero_grad(set_to_none=True)
                fake = generator(torch.randn(len(real), generator.latent_size, device=device), condition)
                g_loss = -critic(fake, condition).mean()
                g_loss.backward(); optimizer_g.step()
                g_values.append(float(g_loss.detach()))
        generator.eval(); critic.eval()
        # Fixed validation noise for a reproducible training trace only. Like
        # the source WGAN-GP, the final generator is retained.
        torch.manual_seed(10000 + args.seed)
        with torch.enable_grad():
            validation_fake = generator(torch.randn(50, generator.latent_size, device=device), validation_x)
            vd, vp = critic.loss(validation_fake, validation_y, validation_x)
            vg = -critic(validation_fake, validation_x).mean()
        row = {"epoch": epoch, "critic_ls": float(np.mean(d_values)),
               "generator_ls": float(np.mean(g_values)), "gp_ls": float(np.mean(gp_values)),
               "critic_vs": float(vd.detach()), "generator_vs": float(vg.detach()),
               "gp_vs": float(vp["gradient_penalty"].detach())}
        history.append(row)
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps(row), flush=True)

    config = {"latent_size": 64, "width": 256, "layers": 2,
              "lambda_gp": 10.0, "n_critic": 5, "epochs": args.epochs,
              "batch_size": batch_size, "seed": args.seed}
    torch.save({"generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(), "config": config,
                "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                "train_dates": data.train_dates, "validation_dates": data.validation_dates,
                "test_dates": data.test_dates}, output / "joint_wgan_gp.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved={output / 'joint_wgan_gp.pt'} device={device}")


if __name__ == "__main__":
    main()
