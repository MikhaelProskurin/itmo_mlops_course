import pandas as pd

import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class AirfieldDataset(Dataset):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            paths: list,
            transforms: torchvision.transforms.Compose = None
    ) -> None:
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.paths = paths

        if transforms is None:
            self.transforms = v2.Compose([
                v2.Resize((800, 800)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transforms = transforms

    """
        Custom dataset for custom object detection task,
        "aircraft identification on the airfield"
    """

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = Image.open(f'{self.paths[index]}')
        image = self.transforms(image)

        records = self.df[self.df['image_id'] == image_id]

        # converting bboxes into tensor
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        roi = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        roi = torch.as_tensor(roi, dtype=torch.float32)

        # if needed it labels might be different
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index]),
            'roi': roi
        }

        return image, targets, image_id
