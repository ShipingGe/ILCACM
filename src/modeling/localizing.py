import torch
import torch.nn as nn
from modeling.captioning import T5Captioner
import torch.nn.functional as F
import math
from util.giou import giou_loss_1d


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


class MaskGenerator(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, layer_norm_eps, max_frames=100, num_layers=1, tau=1.,
                 gamma=0.1):
        super().__init__()

        self.tpe = nn.Parameter(torch.randn([16, d_model]))
        self.dropout = nn.Dropout(p=0.1)

        # self.pe = PositionalEncoding(d_model)
        self.pe = nn.Parameter(torch.randn([max_frames, d_model]))
        self.dropout2 = nn.Dropout(p=0.1)
        self.max_frames = max_frames

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   batch_first=True,
                                                   dim_feedforward=dim_feedforward,
                                                   layer_norm_eps=layer_norm_eps,
                                                   dropout=0.1,
                                                   activation='gelu',
                                                   norm_first=True)

        self.mask_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.center_fc = nn.Linear(d_model, 1, bias=False)
        self.width_fc = nn.Linear(d_model, 1, bias=False)

        self.tau = tau
        self.gamma = gamma

    def forward(self, video_features, num_sents, feature_mask, timestamps):
        video_features = video_features.half()

        max_sents = max(num_sents)
        idx = torch.arange(max_sents).unsqueeze(0)
        text_mask = idx < torch.tensor(num_sents).unsqueeze(1)
        text_mask = text_mask.to(video_features.device)

        queries = self.tpe[:max_sents].unsqueeze(0).expand(len(video_features), -1, -1) * text_mask.unsqueeze(-1)
        outputs = self.mask_decoder(self.dropout(queries),
                                    self.dropout2(video_features + self.pe),
                                    memory_key_padding_mask=~feature_mask.bool(),
                                    tgt_key_padding_mask=~text_mask.bool())

        outputs = self.norm(outputs)

        loss = 0.
        centers = []
        widths = []
        masks = []

        for output, mask, num_sent in zip(outputs, feature_mask, num_sents):
            output = output[:num_sent]
            center = torch.sigmoid(self.center_fc(output))
            width = torch.sigmoid(self.width_fc(output))

            center, ids = torch.sort(center, dim=0)
            width = width[ids].squeeze(-1)

            videolen = int(mask.sum().item())
            t = torch.linspace(0, 1, videolen).unsqueeze(0).expand(num_sent, -1).to(output.device)
            loc_mask = torch.exp(-(t - center) ** 2 / (2 * (width / self.tau) ** 2))

            # cauchy = 1 / math.pi * ((width / self.tau) / ((t - center) ** 2 + (width / self.tau) ** 2))
            # loc_mask = cauchy / torch.max(cauchy, dim=1, keepdim=True)[0]

            # hard 01 mask
            # left = torch.clamp(center - width / 2, 0, 1)
            # right = torch.clamp(center + width / 2, 0, 1)
            # loc_mask = (t > left).float() * (t < right).float()

            padding = torch.zeros([num_sent, self.max_frames - videolen], device=output.device)
            result = torch.cat([loc_mask, padding], dim=1)
            pred_mask = result.unsqueeze(-1)
            masks.append(pred_mask)

            loc_mask = F.normalize(loc_mask, p=2, dim=-1)
            diag = torch.eye(len(loc_mask), device=loc_mask.device)
            output_sim = loc_mask.matmul(loc_mask.t())
            norm_output = torch.masked_select(output_sim, ~diag.bool()).view(len(output_sim), -1)
            div_loss = torch.relu(norm_output - self.gamma).mean()
            loss += div_loss

            centers.append(center)
            widths.append(width)

        loss = loss / len(video_features)

        centers = torch.cat(centers, dim=0)
        widths = torch.cat(widths, dim=0)

        lefts = torch.clamp(centers - widths / 2, 0, 1)
        rights = torch.clamp(centers + widths / 2, 0, 1)

        masks = torch.cat(masks, dim=0)

        return lefts, rights, loss, masks
