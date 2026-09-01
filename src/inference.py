from pathlib import Path
import argparse
import random

import cv2 as cv
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    VGG16_BN_Weights,
    resnet18,
    vgg16_bn,
)

from network_resnet18 import NetworkResNet18
from network_vgg16 import NetworkVGG16


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Cashew anthracnose",
    "Cashew healthy",
    "Cassava green mite",
    "Maize streak virus",
]


def build_model(model_name, pretrained, num_classes=4):
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
    test_csv = Path(args.test_csv)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.pretrained)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    n = min(args.num_images, len(df))
    random_indices = random.sample(range(len(df)), n)

    with torch.no_grad():
        for i in random_indices:
            image_path = df.iloc[i]["image_path"]
            true_class = df.iloc[i]["class_name"]

            image_bgr = cv.imread(image_path)
            if image_bgr is None:
                print(f"Could not read: {image_path}")
                continue

            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0).to(device)

            logits = model(image_tensor)
            pred_id = logits.argmax(dim=1).item()
            pred_label = CLASS_NAMES[pred_id]
            confidence = torch.softmax(logits, dim=1)[0, pred_id].item()

            cv.putText(
                image_bgr,
                f"True: {true_class}",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv.putText(
                image_bgr,
                f"Pred: {pred_label} ({confidence:.2%})",
                (20, 70),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

            output_path = output_dir / f"prediction_{i}_{Path(image_path).name}"
            cv.imwrite(str(output_path), image_bgr)

            print(
                f"{i}: True={true_class}, "
                f"Predicted={pred_label}, Confidence={confidence:.2%}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="results/test.csv")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="results/inference")
    parser.add_argument("--model", choices=["resnet18", "vgg16_bn"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-images", type=int, default=10)
    main(parser.parse_args())
