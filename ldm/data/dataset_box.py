import json
import cv2
import os
from basicsr.utils import img2tensor
import torch


class dataset_coco_box:
    def __init__(
        self,
        path_json,
        root_path_im,
        root_path_box,
        image_size,
        text_encoder,
        max_objs=30,
    ):
        super(dataset_coco_box, self).__init__()
        with open(path_json, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        data = data["annotations"]
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_box = root_path_box
        for file in data:
            name = "%012d.jpg" % file["image_id"]
            box_cords = [i["bbox"] for i in file["all_bbox"]]
            box_cords = [[i[0], i[1], i[0] + i[2], i[1] + i[3]] for i in box_cords]
            labels = [i["label"]["category"] for i in file["all_bbox"]]
            # box_cords_tensor = torch.tensor(box_cords)
            self.files.append(
                {
                    "name": name,
                    "sentence": file["caption"],
                    "boxes": box_cords,  # box_cords_tensor,
                    "labels": labels,
                }
            )
        self.text_encoder = text_encoder
        self.max_objs = max_objs

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file["name"]

        im = cv2.imread(os.path.join(self.root_path_im, name))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.0

        mask = cv2.imread(os.path.join(self.root_path_box, name))  # [:,:,0]
        mask = cv2.resize(mask, (512, 512))
        mask = (
            img2tensor(mask, bgr2rgb=True, float32=True) / 255.0
        )  # [0].unsqueeze(0)#/255.
        sentence = file["sentence"]
        boxes = file["boxes"]
        if len(boxes) < self.max_objs:
            boxes = boxes + (self.max_objs - len(boxes)) * [
                torch.tensor([0.0, 0.0, 0.0, 0.0])
            ]
        else:
            boxes = boxes[: self.max_objs]
        boxes = torch.tensor(boxes)
        if len(file["labels"]) < self.max_objs:
            file["labels"] = file["labels"] + (self.max_objs - len(file["labels"])) * [
                ""
            ]
        else:
            file["labels"] = file["labels"][: self.max_objs]
        labels = []
        for label in file["labels"]:
            encoded_label = self.text_encoder(label).squeeze(0)
            labels.append(encoded_label)

        labels = torch.stack(labels)

        return {
            "im": im,
            "mask": mask,
            "boxes": boxes,
            "labels": labels,
            "sentence": sentence,
        }

    def __len__(self):
        return len(self.files)
