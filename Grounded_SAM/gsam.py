import os, sys
import cv2
sys.path.insert(0, sys.path[0]+"/../")
import torch
import numpy as np
from matplotlib import pyplot as plt
from segment_anything import (
    sam_model_registry,
    build_sam_hq,
    SamPredictor
)
import groundingdino.datasets.transforms as T
import torchvision.transforms as TS
from Grounded_SAM.grounded_sam_demo import load_model, get_grounding_output, save_mask_data, show_mask, show_box

def convert_SAM(sam_semantic_pred, object_category):
    '''
    sam_semantic_pred[0]: (N, 1, 480, 640)
    we need output of (480, 640, nc+1)
    '''
    try:
        masks, boxes_filt, pred_phrases = sam_semantic_pred
    except:
        return np.zeros((480, 640, len(object_category)))
    
    N, _, H, W = masks.shape
    masks = masks.cpu().numpy()
    all_cls_masks = np.zeros((H, W, len(object_category)))
    for i, cat in enumerate(object_category):
        for j in range(N):
            phrase = pred_phrases[j].split('(')[0] # e.g. bed(0.55)
            score = float(pred_phrases[j].split('(')[1].split(')')[0])
            if cat == phrase:
                all_cls_masks[:, :, i][masks[j][0] == 1] = score
                
    # choose the highest score for each point on (480, 640) and set it to 1
    argmax_indices = np.argmax(all_cls_masks, axis=2) # (480, 640)
    all_cls_masks_ = np.zeros((H, W, len(object_category)))
    for i in range(H):
        for j in range(W):
            if all_cls_masks[i, j, argmax_indices[i, j]] != 0:
                all_cls_masks_[i, j, argmax_indices[i, j]] = all_cls_masks[i, j, argmax_indices[i, j]]
            
    return all_cls_masks_

class GSAM():
    def __init__(self, cls: list, box_threshold=0.3, text_threshold=0.2, device='cuda', use_ram=False):
        self.config = 'Grounded_SAM/GroundingDINO_SwinT_OGC.py'
        
        # DINO
        self.dino_ckpt = 'Grounded_SAM/groundingdino_swint_ogc.pth'
        self.dino_model = load_model(self.config, self.dino_ckpt, device=device)
        
        # SAM
        self.transform_sam = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.sam_ckpt = 'Grounded_SAM/sam_vit_h_4b8939.pth'
        self.sam_model = SamPredictor(sam_model_registry["vit_h"](checkpoint=self.sam_ckpt).to(device))
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.set_text(cls)
        
    def set_text(self, cls):
        self.text_prompt = cls[0]
        for i in range(1, len(cls)):
            self.text_prompt += '.' + cls[i]
            
    def add_text(self, cls):
        prev_cls = self.text_prompt.split('.')
        full_cls = prev_cls[:-1] + cls + prev_cls[-1:] # "stairs" should be at the end
        self.set_text(full_cls)
    
    def predict(self, rgb):
        transformed_rgb, _ = self.transform_sam(rgb, None)
        boxes_filt, pred_phrases = get_grounding_output(
            self.dino_model, transformed_rgb, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device
        )
        size = rgb.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
            
        rgb = np.asarray(rgb)
        self.sam_model.set_image(rgb)
        transformed_boxes = self.sam_model.transform.apply_boxes_torch(boxes_filt, rgb.shape[:2]).to(self.device)
        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        return masks, boxes_filt, pred_phrases
    
    def __call__(self, rgb):
        return self.predict(rgb)
    
    def get_vis(self, rgb, predictions):
        masks, boxes_filt, pred_phrases = predictions
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig("current.png", bbox_inches="tight")
        plt.close()
        return cv2.resize(cv2.imread("current.png"), (640, 480), interpolation=cv2.INTER_NEAREST)
        
        
if __name__ == "__main__":
    model = GSAM(
        cls=['chair', 'window', 'table', 'bed', 'clock', 'door', 'wall'],
        text_threshold=0.3
    )
    from PIL import Image
    rgb = Image.open('Grounded_SAM/input/input_2.png').convert('RGB')
    print(rgb.size[0], rgb.size[1])
    masks, boxes_filt, pred_phrases = model.predict(rgb)
    print(masks.shape)
    print(boxes_filt)
    print(pred_phrases)
    # draw output image
    output_dir = "Grounded_SAM/output_sam"
    rgb.save(os.path.join(output_dir, "raw_image.jpg"))
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"), 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    save_mask_data("Grounded_SAM/output_sam/", masks, boxes_filt, pred_phrases)