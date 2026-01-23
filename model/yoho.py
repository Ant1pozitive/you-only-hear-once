import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.ops as ops  # for NMS
from typing import List, Dict, Tuple
from .backbone import BiPathBackbone
from .heads import AnchorFreeDetectionHead, SegmentationHead, decode_boxes, task_aligned_assigner, VarifocalLoss, CIoULoss
from .memory_aug import AudioMemoryBank

class YOHO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
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
            num_classes=cfg.model.num_classes,
            reg_max=cfg.model.reg_max
        )
        
        # Segmentation head
        self.seg_head = SegmentationHead(
            in_channels=in_ch,
            num_classes=cfg.model.num_classes,
            proto_channels=32
        ) if cfg.model.seg_enabled else None
        
        # Memory (optional)
        self.memory = AudioMemoryBank(
            num_slots=cfg.model.memory_slots,
            slot_dim=in_ch
        ) if cfg.model.use_memory else None
        
        # Strides of feature maps (P3=8, P4=16, P5=32)
        self.strides = [8, 16, 32]
        
        # Loss components
        self.vfl = VarifocalLoss()
        self.ciou = CIoULoss()
        
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
            loss_dict = self.compute_loss(preds, targets)
            return loss_dict
        
        return preds

    def compute_loss(self, preds: Dict, targets: Dict) -> Dict:
        """
        Compute full loss with task-aligned assignment
        targets: {'boxes': [B, N_gt, 4], 'cls': [B, N_gt], 'masks': [B, N_gt, H, W] optional}
        """
        cls_scores = preds['cls']
        reg_preds = preds['reg']
        seg_out = preds.get('seg', None)
        
        loss_cls = 0.0
        loss_box = 0.0
        loss_seg = 0.0
        num_pos = 0
        
        for i in range(len(cls_scores)):
            cls = cls_scores[i].permute(0, 2, 3, 1).reshape(1, -1, self.num_classes)  # [1, A, C]
            reg = reg_preds[i].permute(0, 2, 3, 1).reshape(1, -1, 4 * (self.cfg.model.reg_max + 1))  # [1, A, 4*(reg_max+1)]
            
            # Decode boxes
            pred_boxes = decode_boxes(reg[0], self.strides[i], self.cfg.spec.n_mels, reg.shape[-1])
            
            # Per-batch - assume B=1 for simplicity; extend for larger B
            gt_boxes = targets['boxes'][0]
            gt_labels = targets['cls'][0]
            
            if len(gt_boxes) == 0:
                continue
            
            # Assignment
            pred_scores = cls[0].sigmoid()
            assigned_ids, assigned_scores, pos_mask = task_aligned_assigner(
                pred_boxes, pred_scores, gt_boxes, gt_labels
            )
            
            num_pos += pos_mask.sum()
            
            # CLS loss
            pos_labels = gt_labels[assigned_ids[pos_mask]]
            loss_cls += self.vfl(
                pred_scores[pos_mask], assigned_scores, pos_labels
            )
            
            # Box loss
            pos_pred_boxes = pred_boxes[pos_mask]
            pos_gt_boxes = gt_boxes[assigned_ids[pos_mask]]
            loss_box += self.ciou(pos_pred_boxes, pos_gt_boxes)
        
        # Seg loss (simple BCE)
        if seg_out is not None and 'masks' in targets:
            proto, coeffs = seg_out
            gt_masks = targets['masks'][0]  # [N_gt, H, W]
            pred_masks = (proto.unsqueeze(0) * coeffs.unsqueeze(2).unsqueeze(3)).sum(1).sigmoid()  # approx
            loss_seg = F.binary_cross_entropy(pred_masks, gt_masks)
        
        total_loss = loss_cls + 5.0 * loss_box + 2.0 * loss_seg  # weighted
        
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
        feats = self.backbone(spec)
        cls_scores, reg_preds = self.det_head(feats)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for i in range(len(cls_scores)):
            cls = cls_scores[i].sigmoid().permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            reg = reg_preds[i].permute(0, 2, 3, 1).reshape(-1, 4 * (self.cfg.model.reg_max + 1))
            
            scores, labels = cls.max(dim=1)
            boxes = decode_boxes(reg.unsqueeze(0), self.strides[i], spec.shape[2:])[0]
            
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
        
        Args:
            audio_stream: long audio tensor [T]
            chunk_length_sec: length of each processing chunk
            overlap_sec: overlap between consecutive chunks
            
        Returns:
            Merged predictions dict
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
        chunk_id = 0
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
                # Adjust time to global
                offset = start / sr
                pred["boxes"][:, [0, 2]] += offset
                
                # Adjust for padding
                pad_time = pad_len / sr
                mask = pred["boxes"][:, 2] <= chunk_length_sec - pad_time
                for k in pred:
                    pred[k] = pred[k][mask]
                
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])
            
            start += step
            chunk_id += 1
        
        if len(all_boxes) == 0:
            return {"boxes": torch.empty((0,4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
        
        # Merge all chunks
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Cross-chunk NMS
        keep = ops.nms(boxes, scores, iou_thres)
        
        # Sort by time
        sort_idx = boxes[keep, 0].argsort()
        
        return {
            "boxes": boxes[keep][sort_idx],
            "scores": scores[keep][sort_idx],
            "labels": labels[keep][sort_idx]
        }