import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import List, Dict, Tuple
from .backbone import BiPathBackbone
from .heads import (
    AnchorFreeDetectionHead,
    decode_boxes,
    hungarian_matcher,
    VarifocalLoss,
    AudioIoULoss,
    box_iou
)

class YOHO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        self.max_dur_sec = cfg.data.max_dur_sec

        self.spec_transform = T.MelSpectrogram(
            sample_rate=cfg.data.spec.sample_rate,
            n_mels=cfg.data.spec.n_mels,
            hop_length=cfg.data.spec.hop_length,
            n_fft=cfg.data.spec.n_fft,
            f_max=cfg.data.spec.f_max,
            normalized=True
        )

        self.freq_masking = T.FrequencyMasking(freq_mask_param=cfg.data.aug.freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=cfg.data.aug.time_mask_param)

        self.backbone = BiPathBackbone(
            base_channels=cfg.model.base_channels,
            scales=cfg.model.scales,
            bifpn_layers=cfg.model.bifpn_layers
        )

        in_ch = cfg.model.base_channels * 4
        self.det_head = AnchorFreeDetectionHead(in_channels=in_ch, num_classes=self.num_classes, reg_max=cfg.model.reg_max)

        self.strides = [4, 8, 16]  # matches backbone downsampling
        self.vfl = VarifocalLoss()
        self.iou_loss = AudioIoULoss(time_weight=2.0, freq_weight=0.5)
        self.conf_thres = 0.25

    def forward(self, audio: torch.Tensor, targets: Dict = None) -> Dict:
        spec = self.spec_transform(audio)
        if self.training:
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)
        spec = spec.unsqueeze(1)
        feats = self.backbone(spec)
        cls_scores, reg_preds = self.det_head(feats)

        preds = {"cls": cls_scores, "reg": reg_preds}
        if self.training and targets is not None:
            return self.compute_loss(preds, targets, spec.shape)
        return preds

    def compute_loss(self, preds: Dict, targets: Dict, spec_shape: Tuple) -> Dict:
        device = preds['cls'][0].device
        B = spec_shape[0]
        feat_shape = (spec_shape[2], spec_shape[3])

        all_pred_scores = []
        all_pred_boxes = []
        for i in range(len(preds['cls'])):
            cls = preds['cls'][i].permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            reg = preds['reg'][i].flatten(2)  # flatten for DFL
            boxes = decode_boxes(reg, self.strides[i], feat_shape)
            all_pred_scores.append(cls)
            all_pred_boxes.append(boxes)

        pred_scores = torch.cat(all_pred_scores, dim=1)
        pred_boxes = torch.cat(all_pred_boxes, dim=1)

        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(B):
            p_scores_b = pred_scores[b]
            p_boxes_b = pred_boxes[b]
            g_boxes_b = targets['boxes'][b].to(device)
            g_labels_b = targets['labels'][b].to(device)

            target_scores = torch.zeros_like(p_scores_b)
            target_labels = torch.full((p_scores_b.shape[0],), -1, dtype=torch.long, device=device)

            if len(g_boxes_b) > 0:
                matched_p, matched_g = hungarian_matcher(p_boxes_b, p_scores_b, g_boxes_b, g_labels_b)
                if len(matched_p) > 0:
                    matched_classes = g_labels_b[matched_g]
                    target_labels[matched_p] = matched_classes
                    matched_ious = box_iou(p_boxes_b[matched_p], g_boxes_b[matched_g]).diag()
                    target_scores[matched_p, matched_classes] = matched_ious
                    loss_box += self.iou_loss(p_boxes_b[matched_p], g_boxes_b[matched_g])
                    num_pos += len(matched_p)

            loss_cls += self.vfl(p_scores_b, target_scores, target_labels)

        loss_cls /= B
        if num_pos > 0:
            loss_box /= num_pos
        total_loss = loss_cls + 5.0 * loss_box
        return {"loss_cls": loss_cls, "loss_box": loss_box, "total_loss": total_loss, "num_pos": num_pos}

    @torch.inference_mode()
    def infer(self, audio: torch.Tensor, conf_thres: float = 0.25) -> List[Dict]:
        """per-sample + normalized -> seconds"""
        spec = self.spec_transform(audio).unsqueeze(1)
        feat_shape = (spec.shape[2], spec.shape[3])
        feats = self.backbone(spec)
        cls_scores, reg_preds = self.det_head(feats)

        batch_size = audio.shape[0]
        all_sample_preds = []
        for b in range(batch_size):
            sample_boxes, sample_scores, sample_labels = [], [], []
            for i, stride in enumerate(self.strides):
                cls_b = cls_scores[i][b:b+1].sigmoid().permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                reg_b = reg_preds[i][b:b+1].flatten(2)
                scores_b, labels_b = cls_b.max(dim=1)
                boxes_b = decode_boxes(reg_b, stride, feat_shape)

                mask = scores_b > conf_thres
                sample_boxes.append(boxes_b[mask])
                sample_scores.append(scores_b[mask])
                sample_labels.append(labels_b[mask])

            boxes = torch.cat(sample_boxes, 0) if sample_boxes else torch.empty((0, 4), device=cls_scores[0].device)
            scores = torch.cat(sample_scores, 0) if sample_scores else torch.empty(0, device=cls_scores[0].device)
            labels = torch.cat(sample_labels, 0) if sample_labels else torch.empty(0, dtype=torch.long, device=cls_scores[0].device)

            # norm -> seconds
            boxes[:, [0, 2]] *= self.max_dur_sec
            all_sample_preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        return all_sample_preds
