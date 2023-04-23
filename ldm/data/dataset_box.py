import json
import cv2
import os
from basicsr.utils import img2tensor


class dataset_coco_box:
    def __init__(self, path_json, root_path_im, root_path_box, image_size):
        super(dataset_coco_box, self).__init__()
        with open(path_json, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        data = data["annotations"]
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_box = root_path_box
        for file in data:
            name = "%012d.png" % file["image_id"]
            self.files.append({"name": name, "sentence": file["caption"]})
        self.text_encoder = ...

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file["name"]
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace(".png", ".jpg")))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.0

        mask = cv2.imread(os.path.join(self.root_path_box, name))  # [:,:,0]
        mask = cv2.resize(mask, (512, 512))
        mask = (
            img2tensor(mask, bgr2rgb=True, float32=True) / 255.0
        )  # [0].unsqueeze(0)#/255.

        sentence = file["sentence"]
        return {
            "im": im,
            "mask": mask,
            "boxes": boxes,
            "labels": labels,
            "sentence": sentence,
        }

    def __len__(self):
        return len(self.files)
