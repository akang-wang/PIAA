import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import scclip
import clip
import itaclip
import sclip
from tqdm import tqdm
from evaluate import compute_metrics
from torchvision import transforms


pascal_classes = [
   "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor"
]

coco_classes = [
                "person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse",
                "remote",  "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


nus_classes = [
               "airport", "animal", "beach", "bear", "birds",
               "boats", "book", "bridge", "buildings", "cars",
               "castle", "cat", "cityscape", "clouds", "computer",
               "coral", "cow", "dancing", "dog", "earthquake",
               "elk", "fire", "fish", "flags", "flowers",
               "food", "fox", "frost", "garden", "glacier",
               "grass", "harbor", "horses", "house", "lake",
               "leaf", "map", "military", "moon", "mountain",
               "nighttime", "ocean", "person", "plane", "plants",
               "police", "protest", "railroad", "rainbow", "reflection",
               "road", "rocks", "running", "sand", "sign",
               "sky", "snow", "soccer", "sports", "statue",
               "street", "sun", "sunset", "surf", "swimmers",
               "tattoo", "temple", "tiger", "tower", "town",
               "toy", "train", "tree", "valley", "vehicle",
               "water", "waterfall", "wedding", "whales", "window",
               "zebra"]

prompt_templates = ['a photo of a {}']




def compute_gda_adapter_probs_batch(gda_logits, temperature=500):
    with torch.no_grad():
        intermediate = gda_logits.clone()
        intermediate -= torch.max(intermediate, dim=-1, keepdim=True)[0]
        intermediate = torch.exp(intermediate / temperature)
        intermediate = intermediate / torch.sum(intermediate, dim=-1, keepdim=True)
        return intermediate



class Gaussian(nn.Module):
    def __init__(self, W, b):
        super().__init__()
        self.W = nn.Parameter(W, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)
    
    def forward(self, x, no_exp=False):
        logits = x @ self.W + self.b
        return logits

def load_text_features(model, device):
    text_features = []
    with torch.no_grad():
        for classname in args.classname:
            texts = [template.format(classname) for template in prompt_templates]       
            tokens = scclip.tokenize(texts).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=0)   
            features = features / features.norm()
            text_features.append(features)
    return torch.stack(text_features, dim=0).to(device)   



class COCODataset(Dataset):
    def __init__(self, img_root, split_file, preprocess):
        self.img_root = img_root
        file_list = tuple(open(split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        self.image_list = [x[0] + ".jpg" for x in file_list]
        self.all_label_list = [x[1:] for x in file_list]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_root, self.image_list[idx])
        pil_img = Image.open(image_path).convert("RGB")
        tensor_img = self.preprocess(pil_img)
        label_ids = [int(lid) for lid in self.all_label_list[idx]]
        return tensor_img, label_ids, image_path

def collate_fn(batch):
    tensors, labels, paths = zip(*batch)
    return torch.stack(tensors), list(labels), list(paths)

import random
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_piaa(args):
    set_random_seed(42)
    device = "cuda:6" if torch.cuda.is_available() else "cpu"


    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()

    if args.model == 'scclip':
        model_patch, preprocess = scclip.load("ViT-B/16", device=device, jit=False)
        model_patch.eval()
    elif args.model == 'itaclip':
        model_patch, preprocess = itaclip.load("ViT-B/16", device=device)
        model_patch.eval()
    elif args.model == 'sclip':
        model_patch, preprocess = sclip.load("ViT-B/16", device=device)
        model_patch.eval()
    else:
        raise ValueError(f"Unknown model: {args.model}")


    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))
    ])

    NUM_CLASSES = len(args.classname)
    adapter_checkpoint = torch.load(f'{args.dataname}_adapter.pth', map_location=device)

    W = adapter_checkpoint['W'].to(device)  
    b = adapter_checkpoint['b'].to(device)  
    
    adapter = Gaussian(W, b).to(device)
    adapter.eval()
    print("Loaded classifier. W shape:", adapter.W.shape, "b shape:", adapter.b.shape)

    dataset = COCODataset(args.img_root, args.split_file, preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=8,
                            collate_fn=collate_fn, pin_memory=True, 
                            prefetch_factor=4)

    all_preds = []
    all_gts = []

    text_features = load_text_features(model, device)

    with torch.no_grad():
        for batch_tensors, labels, paths in tqdm(dataloader, desc="Eval"):
            batch_tensors = batch_tensors.to(device, non_blocking=True) 

            # --------------------------
            # Global feature branch: standard CLIP [CLS] token
            # --------------------------
            cls_feats_batch = model.encode_image(batch_tensors)  
            cls_feats_batch = cls_feats_batch / cls_feats_batch.norm(dim=-1, keepdim=True)
            
            cls_logits_batch = cls_feats_batch @ text_features.T  
            cls_probs_batch = torch.softmax(cls_logits_batch * args.cls, dim=-1) 
            
            # --------------------------
            # Patch feature branch: specialized CLIP variant outputs
            # --------------------------
            if args.model == 'scclip':
                patch_feats_batch = model_patch.encode_image(batch_tensors, return_all=True)
            elif args.model == 'itaclip':
                patch_feats_batch = model_patch.encode_image(batch_tensors, return_all=True, attn_self=True, device=device)
                patch_feats_batch = patch_feats_batch[:, 1:, :]
            elif args.model == 'sclip':
                patch_feats_batch = model_patch.encode_image(batch_tensors, return_all=True, csa=True)
                patch_feats_batch = patch_feats_batch[:, 1:, :]

            patch_feats_batch = patch_feats_batch / patch_feats_batch.norm(dim=-1, keepdim=True)
            

            batch_size_actual, num_patches, feat_dim = patch_feats_batch.shape
            patch_feats_flat = patch_feats_batch.view(-1, feat_dim)  
            
            with torch.autocast("cuda:3", dtype=torch.float16):
                gda_logits_flat = adapter(patch_feats_flat, no_exp=True).float()  
            
            gda_logits_batch = gda_logits_flat.view(batch_size_actual, num_patches, -1)  
            gda_probs_batch = compute_gda_adapter_probs_batch(gda_logits_batch, temperature=500) 
            patch_logits_batch = torch.max(gda_probs_batch, dim=1)[0]       
            patch_probs_batch = torch.softmax(patch_logits_batch * args.patch, dim=-1) 
            combined_logits_batch = 0.1 * cls_probs_batch + 0.9 * patch_probs_batch 
            all_preds.append(combined_logits_batch.cpu())
            

            gt_batch = torch.zeros(batch_size_actual, NUM_CLASSES, dtype=torch.float32)
            for i, label_ids in enumerate(labels):
                for lid in label_ids:
                    gt_batch[i, lid] = 1
            all_gts.append(gt_batch)


    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(all_preds, all_gts)
    print("================================================")
    print(f"mAP {metrics['map']:.3f}")
    print("================================================")
    print(f"mAP: {metrics['ap']}")
    print("================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='voc07', choices=['voc07', 'coco', 'nus', 'voc12'])
    parser.add_argument('--cls', type=int, default=80)
    parser.add_argument('--patch', type=int, default=200)
    parser.add_argument('--model', type=str, default='sclip', choices=['scclip', 'itaclip', 'sclip'])
    args = parser.parse_args()

    if args.dataname == 'voc07':
        args.img_root = '/data/public/multi-label/data/pascal/VOCdevkit/VOC2007/JPEGImages'
        args.split_file = './imageset/voc07val_labels.txt'
        args.classname = pascal_classes
        args.cls = 80
        args.patch = 200
    elif args.dataname == 'voc12':
        args.img_root = '/data/public/multi-label/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        args.split_file = './imageset/voc12val_labels.txt'
        args.classname = pascal_classes
        args.cls = 80
        args.patch = 200
    elif args.dataname == 'coco':
        args.img_root = '/data/public/multi-label/data/coco/val2014'
        args.split_file = './imageset/cocoval_labels.txt'
        args.classname = coco_classes
        args.cls = 80
        args.patch = 400
    elif args.dataname == 'nus':
        args.img_root = '/data/public/multi-label/data/nuswide/Flickr'
        args.split_file = './imageset/nuswideval_labels.txt'
        args.classname = nus_classes
        args.cls = 10
        args.patch = 400


    test_piaa(args)
