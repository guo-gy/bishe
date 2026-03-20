import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import FPN, MultiModalBalance, Projector, TransformerDecoder


class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        # Optional modules for the improved variant
        self.use_balance = getattr(cfg, 'use_balance', False)
        self.balance_loss_weight = getattr(cfg, 'balance_loss_weight', 0.0)
        self.use_self_distill = getattr(cfg, 'use_self_distill', False)
        self.distill_weight = getattr(cfg, 'distill_weight', 0.0)
        self.distill_mask_ratio = getattr(cfg, 'distill_mask_ratio', 0.3)
        self.balance = None
        if self.use_balance:
            self.balance = MultiModalBalance(vis_dim=cfg.vis_dim,
                                             state_dim=cfg.word_dim,
                                             nhead=getattr(cfg, 'balance_num_head',
                                                           cfg.num_head),
                                             dropout=getattr(cfg,
                                                             'balance_dropout',
                                                             cfg.dropout))

    def _decode_logits(self, fq, word, pad_mask, state):
        b, c, h, w = fq.size()
        decoded = self.decoder(fq, word, pad_mask)
        if isinstance(decoded, list):
            decoded = decoded[-1]
        decoded = decoded.reshape(b, c, h, w)
        return self.proj(decoded, state)

    def _apply_spatial_mask(self, feat):
        if (not self.training) or self.distill_mask_ratio <= 0:
            return feat
        keep_prob = max(1.0 - self.distill_mask_ratio, 1e-6)
        mask = (torch.rand(feat.size(0), 1, feat.size(2), feat.size(3),
                           device=feat.device) < keep_prob).type_as(feat)
        return feat * mask / keep_prob

    def _apply_token_mask(self, word, pad_mask):
        if (not self.training) or self.distill_mask_ratio <= 0:
            return word
        keep_prob = max(1.0 - self.distill_mask_ratio, 1e-6)
        valid_tokens = (~pad_mask).unsqueeze(-1)
        mask = (torch.rand(word.size(0), word.size(1), 1, device=word.device) <
                keep_prob) & valid_tokens
        masked_word = word * mask.type_as(word) / keep_prob
        return masked_word.masked_fill(pad_mask.unsqueeze(-1), 0.)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        balance_loss = fq.new_tensor(0.0)
        if self.balance is not None:
            fq, word, state, balance_loss = self.balance(fq, word, state,
                                                         pad_mask)

        # b, 1, 104, 104
        pred = self._decode_logits(fq, word, pad_mask, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            if self.balance is not None and self.balance_loss_weight > 0:
                loss = loss + self.balance_loss_weight * balance_loss
            if self.use_self_distill and self.distill_weight > 0:
                masked_fq = self._apply_spatial_mask(fq)
                masked_word = self._apply_token_mask(word, pad_mask)
                aux_pred = self._decode_logits(masked_fq, masked_word,
                                               pad_mask, state)
                aux_loss = F.binary_cross_entropy_with_logits(aux_pred, mask)
                consistency = F.smooth_l1_loss(torch.sigmoid(aux_pred),
                                               torch.sigmoid(pred.detach()))
                loss = loss + self.distill_weight * (aux_loss + consistency)
            return pred.detach(), mask, loss
        else:
            return pred.detach()
