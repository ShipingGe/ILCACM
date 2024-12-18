import torch
import torch.nn as nn

from modeling.captioning import GPT2Captioner
from modeling.localizing import MaskGenerator


class ILCACM(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()

        self.tokenizer = tokenizer
        self.captioner = GPT2Captioner(gpt2_path=args.base_model_path, gpt2_tokenizer=tokenizer,
                                       max_frames=args.max_frames)

        self.localizer = MaskGenerator(d_model=768,
                                       nhead=12,
                                       dim_feedforward=1024,
                                       layer_norm_eps=self.captioner.gpt2_model.config.layer_norm_epsilon,
                                       max_frames=args.max_frames,
                                       num_layers=1,
                                       tau=args.tau,
                                       gamma=args.gamma)

    def forward(self, video_features, feature_mask, num_sents=None, input_ids=None,
                localize=False, pos=True, is_full=False, timestamps=None):

        if is_full:
            outputs = self.captioner(video_features=video_features,
                                     feature_mask=feature_mask,
                                     attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                     input_ids=input_ids,
                                     num_sents=num_sents,
                                     type_='full')
            return outputs.loss

        lefts, rights, loc_loss, final_masks = self.localizer(video_features=video_features,
                                                              num_sents=num_sents,
                                                              feature_mask=feature_mask,
                                                              timestamps=timestamps)

        multi_video_features = []
        multi_feature_mask = []
        for i in range(len(num_sents)):
            num = num_sents[i]
            multi_video_features.append(video_features[i].unsqueeze(0).expand(num, -1, -1))
            multi_feature_mask.append(feature_mask[i].unsqueeze(0).expand(num, -1))
        multi_video_features = torch.cat(multi_video_features, dim=0)
        multi_feature_mask = torch.cat(multi_feature_mask, dim=0)

        if localize:
            pred_captions = self.captioner.generate_one_sentence(multi_video_features * final_masks, multi_feature_mask)

            return lefts, rights, pred_captions
        else:
            if pos:
                masked_vfeats = multi_video_features * final_masks
                outputs = self.captioner(video_features=masked_vfeats,
                                         feature_mask=multi_feature_mask,
                                         attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                         input_ids=input_ids,
                                         num_sents=[1] * len(multi_feature_mask),
                                         type_='pos')

            else:
                num_events = []
                start_ids = torch.cumsum(torch.tensor(num_sents), dim=0)[:-1]
                sp_input_tokens = torch.tensor_split(input_ids, start_ids.cpu())

                neg_masked_vfeats = multi_video_features * (1 - final_masks) * multi_feature_mask.unsqueeze(-1)
                dense_tokens = []
                for input_tokens in sp_input_tokens:
                    num = len(input_tokens)
                    cat_tokens = []
                    for tokens in input_tokens:
                        length = (
                                         tokens != self.tokenizer.pad_token_id).sum() - 1  # minus 1 for removing the last eos_token
                        tokens = tokens[:length]
                        cat_tokens.append(tokens)

                    for i in range(num):
                        tokens = cat_tokens[:]
                        tokens.pop(i)
                        num_events.append(len(tokens))
                        tokens = torch.cat(tokens, dim=0)
                        tokens = torch.cat([tokens, torch.LongTensor([self.tokenizer.eos_token_id]).to(tokens.device)],
                                           dim=0)
                        dense_tokens.append(tokens)

                bt_dense_tokens = []
                max_len = max(len(dense_tokens[i]) for i in range(len(dense_tokens)))
                for input_tokens in dense_tokens:
                    input_tokens = torch.cat([input_tokens,
                                              torch.ones(max_len - len(input_tokens),
                                                         device=input_tokens.device) * self.tokenizer.pad_token_id],
                                             dim=0)
                    bt_dense_tokens.append(input_tokens)
                bt_dense_tokens = torch.stack(bt_dense_tokens).long()
                outputs = self.captioner(video_features=neg_masked_vfeats,
                                         feature_mask=multi_feature_mask,
                                         attention_mask=(bt_dense_tokens != self.tokenizer.pad_token_id),
                                         input_ids=bt_dense_tokens,
                                         num_sents=num_events,
                                         type_='neg')

            return outputs.loss + loc_loss
