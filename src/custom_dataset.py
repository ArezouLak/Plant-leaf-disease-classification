from torch.utils.data import Dataset
import cv2 as cv


class CustomDataset(Dataset):
    def __init__(self, tensors, transforms=None):
        super().__init__()
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.tensors[0][index]
        label = self.tensors[1][index]

        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.tensors[0])
