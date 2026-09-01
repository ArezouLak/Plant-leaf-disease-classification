from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    VGG16_BN_Weights,
    resnet18,
    vgg16_bn,
)

from custom_dataset import CustomDataset
from network_resnet18 import NetworkResNet18
from network_vgg16 import NetworkVGG16
from prepare_dataset import prepare_dataset


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(model_name, num_classes, pretrained):
    if model_name == "resnet18":
        if pretrained:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = NetworkResNet18(num_classes=num_classes)

    elif model_name == "vgg16_bn":
        if pretrained:
            model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
            model.classifier[6] = torch.nn.Linear(
                model.classifier[6].in_features, num_classes
            )
        else:
            model = NetworkVGG16(num_classes=num_classes)
    else:
        raise ValueError("model_name must be 'resnet18' or 'vgg16_bn'")

    return model


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_path, labels = prepare_dataset(data_dir)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        images_path,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths,
        rest_labels,
        test_size=0.5,
        random_state=42,
        stratify=rest_labels,
    )

    pd.DataFrame(
        {
            "image_path": test_paths,
            "label": test_labels,
            "class_name": label_encoder.inverse_transform(test_labels),
        }
    ).to_csv(output_dir / "test.csv", index=False)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_dataset = CustomDataset((train_paths, train_labels), transform)
    val_dataset = CustomDataset((val_paths, val_labels), transform)
    test_dataset = CustomDataset((test_paths, test_labels), transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_encoder.classes_)

    model = build_model(args.model, num_classes, args.pretrained).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device).long()

            logits = model(images)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels_batch).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device).long()

                logits = model(images)
                loss = criterion(logits, labels_batch)

                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == labels_batch).sum().item()

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_correct / len(train_dataset))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_correct / len(val_dataset))

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss={history['train_loss'][-1]:.4f} | "
            f"train acc={history['train_acc'][-1]:.4f} | "
            f"val loss={history['val_loss'][-1]:.4f} | "
            f"val acc={history['val_acc'][-1]:.4f}"
        )

    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device).long()
            logits = model(images)

            true_labels.extend(labels_batch.cpu().numpy())
            pred_labels.extend(logits.argmax(dim=1).cpu().numpy())

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_encoder.classes_,
        digits=4,
    )
    print(report)

    run_name = f"{args.model}_{'pretrained' if args.pretrained else 'scratch'}"
    (output_dir / f"{run_name}_classification_report.txt").write_text(report)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title(f"Training history: {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_training.png")
    plt.close()

    torch.save(model.state_dict(), output_dir / f"{run_name}_weights.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", choices=["resnet18", "vgg16_bn"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    main(parser.parse_args())
