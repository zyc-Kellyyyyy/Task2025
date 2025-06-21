import os
from torch.utils import data
from PIL import Image

#####Kelly
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            img_path, label = line.split()
            self.labels.append(int(label.strip()))
            self.imgs_path.append(img_path)

#####
    def __len__(self):
        return len(self.imgs_path)

#####
    def __getitem__(self, index):
        path = self.imgs_path[index]
        label = self.labels[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
