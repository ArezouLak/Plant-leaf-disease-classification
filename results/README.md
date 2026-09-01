## Results

The project results are organized into three categories:

### Training Curves

Training and validation loss/accuracy curves are available in:

`results/training_curves/`

These plots compare the convergence behavior of the scratch-trained and pretrained VGG16-BN and ResNet18 models.

### Classification Reports

Detailed classification reports are available in:

`results/classification_reports/`

The reports include class-wise:

- Precision
- Recall
- F1-score
- Support

The pretrained models achieved stronger overall classification performance compared with the corresponding models trained from scratch.

### Inference Examples

Inference was performed on 10 randomly selected test images for each model.

Annotated prediction examples are available in:

`results/inference/`

Each prediction image shows the true class, predicted class, and model confidence.
