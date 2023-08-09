from dataset import ObjectDetectionDataset
from torch.utils.data import DataLoader
from utils import collate_fn
from model import DETR
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    traindata = ObjectDetectionDataset(data_dir="Aquarium/train", annotation_file="Aquarium/train/_annotations.coco.json")
    testdata = ObjectDetectionDataset(data_dir="Aquarium/test", annotation_file="Aquarium/test/_annotations.coco.json")

    trainloader = DataLoader(traindata, batch_size=2, shuffle=False, collate_fn=collate_fn)
    testloader = DataLoader(testdata, batch_size=2, shuffle=False, collate_fn=collate_fn)

    image, targets = next(iter(trainloader))
    print(image, targets)
    model = DETR(num_classes=traindata.num_classes)
    trainer = Trainer(logger=WandbLogger(project="detr"), overfit_batches=1, num_sanity_val_steps=0)
    trainer.fit(model, trainloader, testloader)
