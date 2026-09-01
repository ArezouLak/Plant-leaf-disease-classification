from pathlib import Path
from imutils import paths


def prepare_dataset(folder):
    """Return image paths and class labels from a class-folder dataset."""
    images_path = []
    labels = []

    image_list = list(paths.list_images(str(folder)))

    for image_path in image_list:
        label = Path(image_path).parent.name
        images_path.append(image_path)
        labels.append(label)

    return images_path, labels
