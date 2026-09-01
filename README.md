# Plant Leaf Disease Classification

Deep-learning project for four-class plant leaf classification using **VGG16-BN** and **ResNet18**, trained both **from scratch** and with **ImageNet transfer learning**.

## Classes

- Cashew anthracnose
- Cashew healthy
- Cassava green mite
- Maize streak virus

## Experiments

| Architecture | Training strategy |
|---|---|
| VGG16-BN | From scratch |
| VGG16-BN | Pretrained / transfer learning |
| ResNet18 | From scratch |
| ResNet18 | Pretrained / transfer learning |

The pretrained models showed smoother learning curves and stronger classification performance than the corresponding scratch-trained models.

## Repository structure

```text
.
├── src/
│   ├── custom_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── network_resnet18.py
│   └── network_vgg16.py
├── data/
├── results/
│   ├── training_curves/
│   └── inference/
├── models/
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset layout

Keep the image dataset locally in a `dataset/` folder:

```text
dataset/
├── Cashew anthracnose/
├── Cashew healthy/
├── Cassava green mite/
└── Maize streak virus/
```

The dataset itself is excluded from GitHub through `.gitignore`.

## Training

Train ResNet18 from scratch:

```bash
python src/train.py --model resnet18
```

Train pretrained ResNet18:

```bash
python src/train.py --model resnet18 --pretrained
```

Train VGG16-BN from scratch:

```bash
python src/train.py --model vgg16_bn
```

Train pretrained VGG16-BN:

```bash
python src/train.py --model vgg16_bn --pretrained
```

## Evaluation

The training script generates a classification report with precision, recall, F1-score, and support for each class, along with training/validation loss and accuracy curves.

## Inference

Inference can be run on 10 randomly selected test images. Example:

```bash
python src/inference.py \
  --model resnet18 \
  --weights results/resnet18_scratch_weights.pth \
  --test-csv results/test.csv \
  --num-images 10
```

For a pretrained model, add:

```bash
--pretrained
```

## Results

Add your four training-history plots to `results/training_curves/` and your annotated prediction examples to `results/inference/`.

A final comparison table can be added here once the exact metrics from all four experiments are available.

## Key takeaway

Transfer learning provided more stable convergence and better classification performance in this four-class plant disease classification task.
