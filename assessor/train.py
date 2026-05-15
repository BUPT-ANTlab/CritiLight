import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from assessor.model import LightTrafficGAT, TrafficGNNDataset

MODULE_ROOT = Path(__file__).resolve().parent


def _resolve_local_path(env_name, default):
    raw = os.environ.get(env_name, default)
    path = Path(raw)
    if not path.is_absolute():
        path = MODULE_ROOT / path
    return path


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_valid = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            raw_loss = criterion(pred, batch.y)
            masked_loss = raw_loss * batch.mask
            total_loss += masked_loss.sum().item()
            total_valid += batch.mask.sum().item()

    return 0.0 if total_valid == 0 else total_loss / total_valid


def train_assessor():
    batch_size = int(os.environ.get("CRITILIGHT_ASSESSOR_BATCH_SIZE", "64"))
    epochs = int(os.environ.get("CRITILIGHT_ASSESSOR_EPOCHS", "10"))
    learning_rate = float(os.environ.get("CRITILIGHT_ASSESSOR_LR", "0.0001"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = _resolve_local_path("CRITILIGHT_ASSESSOR_DATASET", "datasets/traffic_data_jn.pkl")
    graph_path = _resolve_local_path("CRITILIGHT_ASSESSOR_GRAPH", "checkpoints/traffic_graph_jn1.pkl")
    model_output = _resolve_local_path("CRITILIGHT_ASSESSOR_MODEL_OUT", "checkpoints/traffic_gat_model_jn1.pth")
    loss_log = _resolve_local_path("CRITILIGHT_ASSESSOR_TRAIN_LOG", "../outputs/assessor/train_loss.txt")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    loss_log.parent.mkdir(parents=True, exist_ok=True)

    dataset = TrafficGNNDataset(str(dataset_path), str(graph_path))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = LightTrafficGAT(state_dim=60, action_dim=12, output_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="none")

    print(f"Model initialized on {device}. Start training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            raw_loss = criterion(pred, batch.y)
            masked_loss = raw_loss * batch.mask
            loss = masked_loss.sum() / (batch.mask.sum() + 1e-6)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        if (epoch + 1) % 5 == 0:
            val_loss = evaluate(model, test_loader, criterion, device)
            message = f"Epoch {epoch + 1:03d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}"
            print(message)
            with open(loss_log, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    torch.save(model.state_dict(), model_output)
    print(f"Model saved to {model_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the assessor model.")
    parser.add_argument("--dataset", help="Path to the assessor dataset pickle.")
    parser.add_argument("--graph", help="Path to the assessor graph pickle.")
    parser.add_argument("--model-out", help="Path to save the trained assessor model.")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override the batch size.")
    parser.add_argument("--lr", type=float, help="Override the learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset:
        os.environ["CRITILIGHT_ASSESSOR_DATASET"] = args.dataset
    if args.graph:
        os.environ["CRITILIGHT_ASSESSOR_GRAPH"] = args.graph
    if args.model_out:
        os.environ["CRITILIGHT_ASSESSOR_MODEL_OUT"] = args.model_out
    if args.epochs is not None:
        os.environ["CRITILIGHT_ASSESSOR_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ["CRITILIGHT_ASSESSOR_BATCH_SIZE"] = str(args.batch_size)
    if args.lr is not None:
        os.environ["CRITILIGHT_ASSESSOR_LR"] = str(args.lr)
    train_assessor()


if __name__ == "__main__":
    main()
