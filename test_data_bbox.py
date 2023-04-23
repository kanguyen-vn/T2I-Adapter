from ldm.data.dataset_box import dataset_coco_box
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

text_encoder = FrozenCLIPEmbedder()

train_dataset = dataset_coco_box('captions-train-box.json', 
root_path_im='/home/deval/train2017',
root_path_box='/home/deval/train2017_color',
image_size=512,
text_encoder=text_encoder,
)

for item in train_dataset:
    print(item['boxes'].shape)
    print(item['labels'].shape)
    
    break