from glob import glob

from torch.utils.data import Dataset
from PIL import Image


class MSCOCO(Dataset):
    def __init__(self, root, transform, img_list=None):
        assert root[-1] == '/', "root to COCO dataset should end with \'/\', not {}.".format(
            root)

        if img_list:
            self.image_paths = []
            with open(img_list, 'r') as r:
                lines = r.read().splitlines()
                for line in lines:
                    self.image_paths.append(root + line)
        else:
            self.image_paths = sorted(glob(root + "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)
    

class Kodak(Dataset):
    def __init__(self, root, transform):

        assert root[-1] == '/', "root to Kodak dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)
