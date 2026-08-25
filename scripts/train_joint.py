from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stde_cdm import JointCLDM, load_joint  # noqa: E402


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the joint multisite CLDM distribution expert"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--embedding-epochs", type=int, default=500)
    parser.add_argument("--diffusion-epochs", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.embedding_epochs < 1 or args.diffusion_epochs < 1:
        raise ValueError("both training epoch counts must be at least one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)

    data = load_joint(ROOT / "data" / "wind_data_all_zone.csv")
    model = JointCLDM(channels=args.channels).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.x_train),
            torch.from_numpy(data.y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_x = torch.from_numpy(data.x_validation).to(device)
    validation_y = torch.from_numpy(data.y_validation).to(device)
    output = (
        args.output
        or ROOT / f"outputs/checkpoints/joint_cldm_z1-5_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[dict[str, float | int]]] = {
        "embedding": [],
        "diffusion": [],
    }

    optimizer = torch.optim.Adam(model.embeddings.parameters(), 1e-3)
    best_embedding = float("inf")
    for epoch in range(1, args.embedding_epochs + 1):
        model.train()
        training = []
        for nwp, target in loader:
            nwp, target = nwp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model.forecast(nwp), target)
            loss.backward()
            optimizer.step()
            training.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = float(
                torch.nn.functional.l1_loss(
                    model.forecast(validation_x), validation_y
                )
            )
        row = {
            "epoch": epoch,
            "train_mae": float(np.mean(training)),
            "validation_mae": validation,
        }
        history["embedding"].append(row)
        if validation < best_embedding:
            best_embedding = validation
            torch.save(model.embeddings.state_dict(), output / "best_embedding.pt")
        if epoch == 1 or epoch % 50 == 0:
            print("embedding", json.dumps(row), flush=True)

    model.embeddings.load_state_dict(
        torch.load(output / "best_embedding.pt", weights_only=True)
    )
    model.embeddings.requires_grad_(False)
    optimizer = torch.optim.Adam(model.denoiser.parameters(), 1e-3)
    best_diffusion = float("inf")
    fixed_generator = torch.Generator(device=device).manual_seed(12345)
    fixed_steps = torch.randint(
        0,
        model.steps,
        (len(validation_y),),
        generator=fixed_generator,
        device=device,
    )
    fixed_noise = torch.randn(
        validation_y.shape, generator=fixed_generator, device=device
    )
    for epoch in range(1, args.diffusion_epochs + 1):
        model.train()
        training = []
        for nwp, target in loader:
            nwp, target = nwp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(nwp, target)
            loss.backward()
            optimizer.step()
            training.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = float(
                model.loss(
                    validation_x,
                    validation_y,
                    fixed_steps,
                    fixed_noise,
                )
            )
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(training)),
            "validation_mse": validation,
        }
        history["diffusion"].append(row)
        if validation < best_diffusion:
            best_diffusion = validation
            torch.save(model.denoiser.state_dict(), output / "best_denoiser.pt")
        if epoch == 1 or epoch % 50 == 0:
            print("diffusion", json.dumps(row), flush=True)

    model.denoiser.load_state_dict(
        torch.load(output / "best_denoiser.pt", weights_only=True)
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "farms": 5,
            "channels": args.channels,
            "seed": args.seed,
            "best_validation_embedding_mae": best_embedding,
            "best_validation_noise_mse": best_diffusion,
            "train_dates": data.train_dates,
            "validation_dates": data.validation_dates,
            "test_dates": data.test_dates,
        },
        output / "joint_cldm.pt",
    )
    (output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(
        f"saved={output / 'joint_cldm.pt'} device={device} "
        f"embedding={best_embedding:.6f} diffusion={best_diffusion:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
