import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, GPT2LMHeadModel, GPT2Tokenizer, \
    AutoTokenizer, AutoModelForCausalLM, GPT2Config, AutoConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
)
import re
import math


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


class VideoEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, layer_norm_eps, max_frames, num_layers=1):
        super().__init__()

        self.vpe = nn.Parameter(torch.randn([max_frames, d_model]))
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   layer_norm_eps=layer_norm_eps, batch_first=True, dropout=0.1,
                                                   norm_first=True, activation='gelu')

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.proj = nn.Sequential(nn.LayerNorm(d_model, eps=layer_norm_eps),
                                  nn.Linear(d_model, d_model, bias=False))

    def forward(self, video_features, feature_mask):
        video_features = video_features.half()

        video_features = self.dropout(video_features + self.vpe)

        video_features = self.encoder(video_features, src_key_padding_mask=~feature_mask.bool())
        video_features = self.proj(video_features)

        return video_features


class T5Captioner(nn.Module):
    def __init__(self, t5_path, t5_tokenizer, max_frames):
        super().__init__()

        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_path)

        self.t5_tokenizer = t5_tokenizer

        self.video_encoder = VideoEncoder(d_model=self.t5_model.config.d_model,
                                          nhead=self.t5_model.config.d_model // self.t5_model.config.d_kv,
                                          dim_feedforward=self.t5_model.config.d_ff,
                                          layer_norm_eps=self.t5_model.config.layer_norm_epsilon,
                                          max_frames=max_frames)

        for i in self.t5_model.encoder.parameters():
            i.requires_grad = False

    def forward(self, video_features, attention_mask, input_ids, feature_mask):
        video_features = self.video_encoder(video_features, feature_mask)

        encoded = BaseModelOutput(last_hidden_state=video_features)
        labels = input_ids.masked_fill(input_ids == self.t5_tokenizer.pad_token_id, -100)

        outputs = self.t5_model(
            encoder_outputs=encoded,
            attention_mask=torch.ones(*video_features.shape[:2]).to(video_features.device),
            decoder_attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )

        return outputs

    def generate(self, video_features, feature_mask):
        video_features = self.video_encoder(video_features, feature_mask)

        encoded = BaseModelOutput(last_hidden_state=video_features)

        outputs = self.t5_model.generate(encoder_outputs=encoded,
                                         attention_mask=torch.ones(*video_features.shape[:2]).to(video_features.device),
                                         do_sample=False,
                                         num_beams=4,
                                         # early_stopping=True,
                                         max_length=256,
                                         min_length=32,
                                         length_penalty=1.0,
                                         no_repeat_ngram_size=4)

        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text


class GPT2Captioner(nn.Module):
    def __init__(self, gpt2_path, gpt2_tokenizer, max_frames):
        super().__init__()

        # randomly initialized GPT-2:
        # gpt2config = AutoConfig.from_pretrained(gpt2_path)
        # gpt2_model = GPT2LMHeadModel(config=gpt2config)

        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path)
        gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

        self.gpt2_model = gpt2_model

        # better dim_feedforward=1024 if the video features are the CLIP features.
        self.video_encoder = VideoEncoder(d_model=self.gpt2_model.config.n_embd,
                                          nhead=self.gpt2_model.config.n_head,
                                          dim_feedforward=2048,
                                          layer_norm_eps=self.gpt2_model.config.layer_norm_epsilon,
                                          max_frames=max_frames,
                                          num_layers=6)

        self.tokenizer = gpt2_tokenizer
        self.max_frames = max_frames

    def forward(self, video_features, attention_mask, input_ids, feature_mask, num_sents, type_='full'):
        prefix_token = '<FULL>' if type_ == 'full' else '<MASK>'

        prompt = self.tokenizer([prefix_token + str(e) + ' events: ' for e in num_sents],
                                return_tensors='pt', padding=True)['input_ids'].to(input_ids.device)

        input_ids = torch.cat([prompt, input_ids], dim=1)
        video_features = self.video_encoder(video_features, feature_mask)
        text_embs = self.gpt2_model.transformer.wte(input_ids)

        cat_embs = torch.cat([video_features, text_embs], dim=1)
        labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        labels = torch.cat([torch.ones([len(video_features), self.max_frames + 1],
                                       device=video_features.device) * -100, labels[:, 1:]], dim=1).long()

        attention_mask = torch.cat([feature_mask,
                                    torch.ones([len(input_ids), prompt.shape[1]], device=input_ids.device),
                                    attention_mask], dim=1)

        outputs = self.gpt2_model(
            inputs_embeds=cat_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )

        return outputs

    def generate(self, video_features, feature_mask):

        prompt = self.tokenizer('<FULL>', return_tensors='pt')['input_ids'].expand(len(video_features), -1).to(
            video_features.device)

        video_features = self.video_encoder(video_features, feature_mask)

        prompt_embs = self.gpt2_model.transformer.wte(prompt).expand(len(video_features), -1, -1)
        prefix_embds = torch.cat([video_features, prompt_embs], dim=1)
        mask = torch.cat([feature_mask,
                          torch.ones([len(video_features), prompt.shape[1]], device=video_features.device)], dim=1)

        outputs = self.gpt2_model.generate(inputs_embeds=prefix_embds,
                                           attention_mask=mask,
                                           do_sample=False,
                                           num_beams=4,
                                           max_new_tokens=128,
                                           min_new_tokens=8,
                                           length_penalty=1.,
                                           no_repeat_ngram_size=4,
                                           pad_token_id=self.tokenizer.pad_token_id,
                                           eos_token_id=self.tokenizer.eos_token_id)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def generate_one_sentence(self, video_features, feature_mask):
        prompt = self.tokenizer('<MASK>' + '1 events: ', return_tensors='pt')['input_ids'].expand(
            len(video_features), -1).to(video_features.device)

        video_features = self.video_encoder(video_features, feature_mask)

        prompt_embs = self.gpt2_model.transformer.wte(prompt).expand(len(video_features), -1, -1)
        prefix_embds = torch.cat([video_features, prompt_embs], dim=1)
        mask = torch.cat([feature_mask,
                          torch.ones([len(video_features), prompt.shape[1]], device=video_features.device)], dim=1)

        outputs = self.gpt2_model.generate(inputs_embeds=prefix_embds,
                                           attention_mask=mask,
                                           do_sample=False,
                                           num_beams=4,
                                           max_new_tokens=32,
                                           min_new_tokens=4,
                                           length_penalty=1.,
                                           no_repeat_ngram_size=4,
                                           pad_token_id=self.tokenizer.pad_token_id,
                                           eos_token_id=self.tokenizer.eos_token_id)

        output_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
