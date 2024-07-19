import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class ABAWDataset(Dataset):
    def __init__(self, cfg, split):
        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.FEAT_DIR)
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.target_size = (cfg.DATASET.RESCALE_TEM_LENGTH, cfg.MODEL.IN_FEAT_DIM)
        self.max_segment_num = cfg.DATASET.MAX_SEGMENT_NUM
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.base_dir = os.path.join(self.root, self.split)
        if self.split == self.train_split:
            self.datas = self._make_dataset_all(cfg.DATASET.TRAIN_TXT_PATH)
        else:
            self.datas = self._make_dataset_all(cfg.DATASET.VAL_TXT_PATH)

        self.class_label = cfg.DATASET.CLASS_IDX
        self.window_size = cfg.DATASET.WINDOW_SIZE

        patch_h = cfg.DATASET.PATCH_H
        patch_w = cfg.DATASET.PATCH_W
        self.train_transform = train_transforms(patch_h, patch_w)
        self.test_transform = test_transforms(patch_h, patch_w)

    def __len__(self):
            return len(self.datas)

    def _make_dataset_all(self, path):
        sample_list = []
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = list(set(lines[1:]))
            for line in lines:
                line_list = line.strip().split(',')
                assert len(line_list) == 16
                sample_list.append(line_list)
        return sample_list

    def __getitem__(self, index):
        item = self.datas[index]
        labels = item[1:]
        label_list = []
        for label in labels:
            label = float(label)
            label_list.append(label)
        image_path = os.path.join(self.root, item[0])
        image = Image.open(image_path).convert("RGB")
        if self.split == self.train_split:
            image = self.train_transform(image)
            return image, label_list
        else:
            image = self.test_transform(image)
            video_name = item[0]
            return image, label_list, video_name

def train_transforms(patch_h, patch_w):
    transform_list = [transforms.Resize((patch_h * 14, patch_w * 14)),
                      transforms.RandomCrop((patch_h * 14, patch_w * 14)),
                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                    ]
    return transforms.Compose(transform_list)

def test_transforms(patch_h, patch_w):
    transform_list = [transforms.Resize((patch_h * 14, patch_w * 14)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                    ]
    return transforms.Compose(transform_list)
