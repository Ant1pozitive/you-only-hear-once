import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.ops as ops
from typing import List, Dict, Tuple

from .backbone import BiPathBackbone
from .heads import (
    AnchorFreeDetectionHead, 
    SegmentationHead, 
    decode_boxes, 
    task_aligned_assigner, 
    VarifocalLoss, 
    AudioIoULoss
)
from .memory_aug import AudioMemoryBank

class YOHO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        
        # Spectrogram transformation
        self.spec_transform = T.MelSpectrogram(
            sample_rate=cfg.spec.sample_rate,
            n_mels=cfg.spec.n_mels,
            hop_length=cfg.spec.hop_length,
            n_fft=cfg.spec.n_fft,
            f_max=cfg.spec.f_max,
            normalized=True
        )
        
        # Backbone
        self.backbone = BiPathBackbone(
            base_channels=cfg.model.base_channels,
            scales=cfg.model.scales
        )
        
        in_ch = cfg.model.base_channels * 4  # from FPN output
        
        # Detection head
        self.det_head = AnchorFreeDetectionHead(
            in_channels=in_ch,
            num_classes=self.num_classes,
            reg_max=cfg.model.reg_max
        )
        
        # Segmentation head
        self.seg_head = SegmentationHead(
            in_channels=in_ch,
            num_classes=self.num_classes,
            proto_channels=32
        ) if cfg.model.seg_enabled else None
        
        # Memory
        self.memory = AudioMemoryBank(
            num_slots=cfg.model.memory_slots,
            slot_dim=in_ch
        ) if cfg.model.use_memory else None
        
        # Strides of feature maps (P3=8, P4=16, P5=32)
        self.strides = [8, 16, 32]
        
        # Loss components
        self.vfl = VarifocalLoss()
        self.iou_loss = AudioIoULoss(time_weight=2.0, freq_weight=0.5)
        
        # Detection thresholds
        self.conf_thres = 0.25
        self.iou_thres = 0.45

    def forward(self, audio: torch.Tensor, targets: Dict = None) -> Dict:
        """
        Forward pass: computes predictions or loss
        """
        spec = self.spec_transform(audio).unsqueeze(1)  # [B, 1, n_mels, time]
        feats = self.backbone(spec)
        
        cls_scores, reg_preds = self.det_head(feats)
        
        if self.seg_head:
            proto, coeffs = self.seg_head(feats)
            seg_out = (proto, coeffs)
        else:
            seg_out = None
            
        preds = {"cls": cls_scores, "reg": reg_preds, "seg": seg_out}
        
        if self.training and targets is not None:
            loss_dict = self.compute_loss(preds, targets, spec.shape)
            return loss_dict
        
        return preds

    def compute_loss(self, preds: Dict, targets: Dict, spec_shape: Tuple[int, int, int, int]) -> Dict:
        """
        Compute full loss with batched task-aligned assignment
        targets: {'boxes': List of [N_gt, 4], 'cls': List of [N_gt], 'masks': Optional List}
        spec_shape: [B, 1, n_mels, time]
        """
        device = preds['cls'][0].device
        B = spec_shape[0]
        feat_shape = (spec_shape[2], spec_shape[3])  # (n_mels, time)
        
        all_pred_scores = []
        all_pred_boxes = []
        
        # 1. Decode predictions from all FPN levels and concatenate
        for i in range(len(preds['cls'])):
            cls = preds['cls'][i]  # [B, C, H, W]
            reg = preds['reg'][i]  # [B, 4*(reg_max+1), H, W]
            
            # [B, H*W, C]
            scores = cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes).sigmoid()
            # [B, H*W, 4]
            boxes = decode_boxes(reg, self.strides[i], feat_shape)
            
            all_pred_scores.append(scores)
            all_pred_boxes.append(boxes)
            
        pred_scores = torch.cat(all_pred_scores, dim=1)  # [B, Total_A, C]
        pred_boxes = torch.cat(all_pred_boxes, dim=1)    # [B, Total_A, 4]
        
        # 2. Pad Ground Truths for batched processing
        max_gt = max([len(b) for b in targets['boxes']]) if len(targets['boxes']) > 0 else 0
        max_gt = max(1, max_gt)  # Ensure at least size 1 to avoid tensor dimension errors
        
        gt_boxes = torch.zeros((B, max_gt, 4), device=device)
        gt_labels = torch.zeros((B, max_gt), dtype=torch.long, device=device)
        mask_gt = torch.zeros((B, max_gt), dtype=torch.bool, device=device)
        
        for b in range(B):
            n_gt = len(targets['boxes'][b])
            if n_gt > 0:
                gt_boxes[b, :n_gt] = targets['boxes'][b]
                gt_labels[b, :n_gt] = targets['cls'][b]
                mask_gt[b, :n_gt] = True
                
        # 3. Batched Task-Aligned Assignment
        assigned_ids, assigned_scores, pos_mask = task_aligned_assigner(
            pred_boxes, pred_scores, gt_boxes, gt_labels, mask_gt, self.iou_thres
        )
        
        # 4. Compute Classification Loss (Varifocal Loss)
        target_scores = torch.zeros_like(pred_scores)
        target_labels = torch.full_like(pred_scores, -1)
        
        for b in range(B):
            pm = pos_mask[b]
            if pm.sum() > 0:
                ids = assigned_ids[b][pm]
                classes = gt_labels[b, ids]
                target_labels[b, pm, classes] = 1
                target_scores[b, pm, classes] = assigned_scores[b, pm]
                
        loss_cls = self.vfl(pred_scores, target_scores, target_labels)
        
        # 5. Compute Box Regression Loss (Asymmetric Audio IoU)
        loss_box = torch.tensor(0.0, device=device)
        num_pos = pos_mask.sum()
        
        if num_pos > 0:
            pos_pred_boxes = pred_boxes[pos_mask]
            pos_gt_boxes = torch.zeros_like(pos_pred_boxes)
            
            idx = 0
            for b in range(B):
                pm = pos_mask[b]
                n_p = pm.sum()
                if n_p > 0:
                    pos_gt_boxes[idx:idx+n_p] = gt_boxes[b, assigned_ids[b, pm]]
                    idx += n_p
                    
            loss_box = self.iou_loss(pos_pred_boxes, pos_gt_boxes)
            
        # 6. Compute Segmentation Loss (Optional fallback)
        loss_seg = torch.tensor(0.0, device=device)
        if preds.get('seg') is not None and 'masks' in targets:
            proto, coeffs = preds['seg']
            # Highly simplified placeholder for segmentation integration
            pred_masks = (proto.mean(dim=1) * coeffs.mean(dim=1)).sigmoid()
            gt_masks = torch.stack(targets['masks']).to(device) if len(targets['masks']) > 0 else None
            if gt_masks is not None:
                # Resize pred_masks to match gt_masks if necessary
                pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=gt_masks.shape[-2:]).squeeze(1)
                loss_seg = F.binary_cross_entropy(pred_masks, gt_masks.float())
                
        total_loss = loss_cls + 5.0 * loss_box + 2.0 * loss_seg
        
        return {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "total_loss": total_loss,
            "num_pos": num_pos
        }

    @torch.inference_mode()
    def infer(self, audio: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45) -> Dict:
        """
        Standard single-pass inference with NMS
        """
        spec = self.spec_transform(audio).unsqueeze(1)
        feat_shape = (spec.shape[2], spec.shape[3])
        feats = self.backbone(spec)
        cls_scores, reg_preds = self.det_head(feats)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for i in range(len(cls_scores)):
            cls = cls_scores[i].sigmoid().permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            reg = reg_preds[i]
            
            scores, labels = cls.max(dim=1)
            # Reg requires 4D input for decode_boxes [bs, 4*(reg_max+1), h, w]
            boxes = decode_boxes(reg, self.strides[i], feat_shape)[0]  # Take first batch
            
            mask = scores > conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            if len(boxes) > 0:
                keep = ops.nms(boxes, scores, iou_thres)
                all_boxes.append(boxes[keep])
                all_scores.append(scores[keep])
                all_labels.append(labels[keep])
        
        if len(all_boxes) == 0:
            return {"boxes": torch.empty((0,4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
        
        # Merge multi-scale
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Final NMS across scales
        keep = ops.nms(boxes, scores, iou_thres)
        
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep]
        }

    @torch.inference_mode()
    def stream_infer(self, audio_stream: torch.Tensor, 
                     chunk_length_sec: float = 4.0,
                     overlap_sec: float = 1.0,
                     conf_thres: float = 0.25,
                     iou_thres: float = 0.45) -> Dict:
        """
        Streaming inference mode - process audio in overlapping chunks and merge predictions
        """
        sr = self.cfg.spec.sample_rate
        chunk_samples = int(chunk_length_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        step = chunk_samples - overlap_samples
        
        total_samples = audio_stream.shape[0]
        all_boxes = []
        all_scores = []
        all_labels = []
        
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio_stream[start:end]
            
            # Pad if shorter
            pad_len = chunk_samples - (end - start)
            if pad_len > 0:
                chunk = F.pad(chunk, (0, pad_len))
            
            chunk = chunk.unsqueeze(0)  # [1, chunk_samples]
            
            # Run inference
            pred = self.infer(chunk, conf_thres=conf_thres, iou_thres=iou_thres)
            
            if len(pred["boxes"]) > 0:
                # Adjust time to global (Offset the time axis, which are idx 0 and 2)
                offset = start / sr
                pred["boxes"][:, [0, 2]] += offset
                
                # Filter out predictions that fall into the padding area
                pad_time = pad_len / sr
                mask = pred["boxes"][:, 2] <= chunk_length_sec - pad_time
                for k in pred:
                    pred[k] = pred[k][mask]
                
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])
            
            start += step
        
        if len(all_boxes) == 0:
            return {"boxes": torch.empty((0,4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
        
        # Merge all chunks
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Cross-chunk NMS
        keep = ops.nms(boxes, scores, iou_thres)
        
        # Sort chronologically by the start time (x1)
        sort_idx = boxes[keep, 0].argsort()
        
        return {
            "boxes": boxes[keep][sort_idx],
            "scores": scores[keep][sort_idx],
            "labels": labels[keep][sort_idx]
        }
