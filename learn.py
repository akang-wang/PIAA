from torch.utils.data import Dataset, DataLoader
import os
import argparse
import torch
import torch.nn.functional as F
import scclip
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import scclip
import itaclip
import sclip



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


@torch.no_grad()
def param_estimation(image_features, banks, initial_mean):

    sorted_classes = sorted(banks.keys())
    device = image_features.device

    vecs = torch.cat([item[0].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0).float()
    labels = torch.tensor([class_idx for class_idx in sorted_classes for _ in banks[class_idx]], device=device)
    cache_pro = torch.cat([item[2].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0).float()
    
    mus = torch.cat([
        ((cache_pro[labels == i][:, i].unsqueeze(1) * vecs[labels == i]).sum(dim=0) / 
         cache_pro[labels == i][:, i].sum()).unsqueeze(0) 
        if i in banks.keys() else initial_mean[i].unsqueeze(0) 
        for i in range(initial_mean.shape[0])
    ]) 

    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in banks.keys()])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1], device=device))

    ps = torch.ones(initial_mean.shape[0], device=device) * 1. / initial_mean.shape[0]
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
    return W, b 

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
        self.preprocess = preprocess
        file_list = tuple(open(split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        self.image_list = [x[0] + ".jpg" for x in file_list]
        self.all_label_list = [x[1:] for x in file_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_root, self.image_list[idx])
        pil_img = Image.open(image_path).convert("RGB")
        tensor_img = self.preprocess(pil_img)
        return tensor_img

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def learn_piaa(args):
    set_random_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.model == 'scclip':
        model, preprocess = scclip.load("ViT-B/16", device=device, jit=False)
        model.eval()
    elif args.model == 'itaclip':
        model, preprocess = itaclip.load("ViT-B/16", device=device)
        model.eval()
    elif args.model == 'sclip':
        model, preprocess = sclip.load("ViT-B/16", device=device)
        model.eval()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    text_features = load_text_features(model, device).float()

    NUM_CLASSES = len(args.classname)

    dataset = COCODataset(args.img_root, args.split_file, preprocess)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=8, collate_fn=lambda x: torch.stack(x), 
                            pin_memory=True, prefetch_factor=4, persistent_workers=True)

    class_topk_features = {}
    class_topk_probs = {}
    class_topk_entropies = {}
    
    class_accumulated_features = {c: [] for c in range(NUM_CLASSES)}
    class_accumulated_entropies = {c: [] for c in range(NUM_CLASSES)}
    class_accumulated_probs = {c: [] for c in range(NUM_CLASSES)}
    
    UPDATE_INTERVAL = 20  
    start_time = time.time()

    with torch.inference_mode():
        for batch_idx, batch_tensors in enumerate(tqdm(dataloader, desc="Extracting patches")):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            

            if args.model == 'scclip':
                patch_feats_batch = model.encode_image(batch_tensors, return_all=True)
            elif args.model == 'itaclip':
                patch_feats_batch = model.encode_image(batch_tensors, return_all=True, attn_self=True, device=device)  
                patch_feats_batch = patch_feats_batch[:, 1:, :]  
            elif args.model == 'sclip':
                patch_feats_batch = model.encode_image(batch_tensors, return_all=True, csa=True)
                patch_feats_batch = patch_feats_batch[:, 1:, :]  

            patch_feats_batch = patch_feats_batch.float()

            patch_feats_batch = F.normalize(patch_feats_batch, dim=-1)
            
            B, N, D = patch_feats_batch.shape
            patch_feats_flat = patch_feats_batch.view(-1, D)
            
            logits_flat = torch.mm(patch_feats_flat, text_features.T)
            probs_flat = F.softmax(logits_flat * 60.0, dim=-1)
            entropies_flat = -(probs_flat * probs_flat.clamp_min(1e-10).log()).sum(dim=-1)
            pred_classes_flat = logits_flat.argmax(dim=-1)
            
            unique_classes = pred_classes_flat.unique()
            for c in unique_classes.tolist():
                mask = pred_classes_flat == c
                class_accumulated_features[c].append(patch_feats_flat[mask])
                class_accumulated_entropies[c].append(entropies_flat[mask])
                class_accumulated_probs[c].append(probs_flat[mask])

            if (batch_idx + 1) % UPDATE_INTERVAL == 0 or batch_idx == len(dataloader) - 1:
                classes_to_update = [c for c in range(NUM_CLASSES) if len(class_accumulated_features[c]) > 0]
                
                for c in classes_to_update:
                    curr_feats = torch.cat(class_accumulated_features[c], dim=0)
                    curr_entropies = torch.cat(class_accumulated_entropies[c], dim=0)
                    curr_probs = torch.cat(class_accumulated_probs[c], dim=0)

                    if c not in class_topk_features:
                        k = min(args.k, len(curr_entropies))
                        if k > 0:
                            _, topk_idx = torch.topk(curr_entropies, k, largest=False)
                            class_topk_features[c] = curr_feats[topk_idx]
                            class_topk_entropies[c] = curr_entropies[topk_idx]
                            class_topk_probs[c] = curr_probs[topk_idx]
                    else:
                        merged_feats = torch.cat([class_topk_features[c], curr_feats], dim=0)
                        merged_entropies = torch.cat([class_topk_entropies[c], curr_entropies], dim=0)
                        merged_probs = torch.cat([class_topk_probs[c], curr_probs], dim=0)
                        
                        k = min(args.k, len(merged_entropies))
                        _, topk_idx = torch.topk(merged_entropies, k, largest=False)
                        
                        class_topk_features[c] = merged_feats[topk_idx]
                        class_topk_entropies[c] = merged_entropies[topk_idx]
                        class_topk_probs[c] = merged_probs[topk_idx]
                    
                    class_accumulated_features[c] = []
                    class_accumulated_entropies[c] = []
                    class_accumulated_probs[c] = []
                
                torch.cuda.empty_cache()
    

    all_features = [class_topk_features[c] for c in sorted(class_topk_features.keys())]
    all_probs = [class_topk_probs[c] for c in sorted(class_topk_probs.keys())]
    
    combined_features = torch.cat(all_features, dim=0)
    combined_features = F.normalize(combined_features, dim=-1)
    all_probs_combined = torch.cat(all_probs, dim=0)
    
    del class_topk_features, class_topk_entropies, class_topk_probs
    del class_accumulated_features, class_accumulated_entropies, class_accumulated_probs
    torch.cuda.empty_cache()

    preds = all_probs_combined.argmax(dim=-1)
    banks = {}
    banks_indices = {}

    for pred_cls in preds.unique().tolist():
        mask = preds == pred_cls
        class_features_sel = combined_features[mask]
        class_probs_sel = all_probs_combined[mask]
        banks[pred_cls] = [(class_features_sel[i], i, class_probs_sel[i]) 
                           for i in range(len(class_features_sel))]
        banks_indices[pred_cls] = torch.where(mask)[0]
    
    initial_mean = text_features
    W, b = param_estimation(combined_features, banks, initial_mean)

    device = combined_features.device
    W_gda = W.to(device)  
    b_gda = b.to(device) 
    
    temperature = args.temperature
    
    logits_gda = torch.mm(combined_features, W_gda) + b_gda.unsqueeze(0)  
    probs_gda = F.softmax(logits_gda * temperature, dim=-1)  
    preds_gda = logits_gda.argmax(dim=-1) 
    
    class_good_patches = {}
    
    for pred_cls in preds_gda.unique().tolist():
        mask = preds_gda == pred_cls
        class_features = combined_features[mask]
        class_probs_gda = probs_gda[mask]
        class_probs_for_pred = class_probs_gda[:, pred_cls]
        mean_prob = class_probs_for_pred.mean().item()
        std_prob = class_probs_for_pred.std().item()
        prob_threshold = mean_prob +  std_prob 
        good_mask = class_probs_for_pred >= prob_threshold
        good_indices = torch.where(good_mask)[0]
        class_good_patches[pred_cls] = [
            (class_features[i], i.item(), class_probs_gda[i]) 
            for i in good_indices
        ]
    
    
    good_features_list = []
    good_probs_list = []
    
    for pred_cls in sorted(class_good_patches.keys()):
        if len(class_good_patches[pred_cls]) > 0:
            good_patches = class_good_patches[pred_cls]
            class_feats = torch.stack([patch[0] for patch in good_patches])
            class_probs = torch.stack([patch[2] for patch in good_patches])
            good_features_list.append(class_feats)
            good_probs_list.append(class_probs)
    
    combined_features = torch.cat(good_features_list, dim=0)
    combined_features = F.normalize(combined_features, dim=-1)
    all_probs_combined = torch.cat(good_probs_list, dim=0)
    
    banks = {}
    preds = all_probs_combined.argmax(dim=-1)
    
    for pred_cls in preds.unique().tolist():
        mask = preds == pred_cls
        class_features_sel = combined_features[mask]
        class_probs_sel = all_probs_combined[mask]
        banks[pred_cls] = [(class_features_sel[i], 0.0, class_probs_sel[i]) 
                           for i in range(len(class_features_sel))]

    
    initial_mean = text_features
    W, b = param_estimation(combined_features, banks, initial_mean)

    save_path = f'{args.dataname}_adapter.pth'
    torch.save({
        'W': W.cpu(),
        'b': b.cpu(),
    }, save_path)


    end_time = time.time()

    print("================================================")
    print(f"Learning complete!\nClassifier saved to: {save_path}\nTotal learning time: {end_time - start_time:.2f} seconds.")
    print("================================================")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='voc12')
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--temperature', type=int)
    parser.add_argument('--model', type=str, default='scclip')
    args = parser.parse_args()

    if args.dataname == 'voc07':
        args.img_root = '/data/public/multi-label/data/pascal/VOCdevkit/VOC2007/JPEGImages'
        args.split_file = './imageset/voc07train_labels.txt'
        args.classname = pascal_classes
        args.temperature = 100
    elif args.dataname == 'voc12':
        args.img_root = '/data/public/multi-label/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        args.split_file = './imageset/voc12train_labels.txt'
        args.classname = pascal_classes
        args.temperature = 100
    elif args.dataname == 'coco':
        args.img_root = '/data/public/multi-label/data/coco/train2014'
        args.split_file = './imageset/cocotrain_labels.txt'
        args.classname = coco_classes
        args.temperature = 0.01
    elif args.dataname == 'nus':
        args.img_root = '/data/public/multi-label/data/nuswide/Flickr'
        args.split_file = './imageset/nuswidetrain_labels.txt'
        args.classname = nus_classes
        args.temperature = 500


    learn_piaa(args)
