import numpy as np
import torch

import os
import json
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer


class ANDataset(Dataset):
    def __init__(self, caption_path, video_feats_dir, max_captions,
                 max_frames, tokenizer=None, max_tokens=1024, split='train'):
        super(ANDataset, self).__init__()

        with open(caption_path, 'r') as f:
            self.captions = json.load(f)
        self.video_ids = list(self.captions.keys())

        self.video_feats_dir = video_feats_dir

        assert len(os.listdir(self.video_feats_dir)) >= len(self.video_ids)

        self.max_frames = max_frames

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.split = split

        self.max_captions = max_captions

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, item):

        video_id = self.video_ids[item]

        video_feature = torch.from_numpy(np.load(os.path.join(self.video_feats_dir, video_id + '.npy')))
        visual_feats, feature_mask = self.frame_padding(video_feature, self.split)

        sentences = self.captions[video_id]['sentences']
        sentences = sentences[:self.max_captions]
        sentences = [sent.strip().capitalize() for sent in
                     sentences]  # Remove the space at the beginning of some sentences.
        sentences = [sent + '.' if sent[-1] != '.' else sent for sent in sentences]

        if len(sentences) == 0:
            sentences = ['None.']

        duration = self.captions[video_id]['duration']
        num_sent = len(sentences)

        # for fully-supervised
        if self.split == 'train':
            timestamps = self.captions[video_id]['timestamps'][:self.max_captions]
            timestamps = torch.tensor(timestamps) / duration
            center = (timestamps[:, 0] + timestamps[:, 1]) / 2
            width = timestamps[:, 1] - timestamps[:, 0]
            timestamps = torch.stack([center, width], dim=1)
        else:
            timestamps = torch.tensor([[0, 0]])

        sents = [sent + ' <|endoftext|>' for sent in sentences]
        input_tokens = self.tokenizer(sents, padding=True, max_length=self.max_tokens, truncation=True,
                                      return_tensors='pt', add_special_tokens=False)['input_ids']

        text_input_tokens = [self.tokenizer(sent + ' ', max_length=self.max_tokens, truncation=True, padding=False,
                                            add_special_tokens=False, return_tensors='pt')['input_ids'][0] for sent in
                             sentences]
        dense_input_tokens = torch.cat(text_input_tokens, dim=0)
        dense_input_tokens = dense_input_tokens[:-1]  # remove the last space.
        dense_input_tokens = dense_input_tokens[:self.max_tokens - 1]
        dense_input_tokens = torch.cat([dense_input_tokens, torch.LongTensor([self.tokenizer.eos_token_id])], dim=0)

        return video_id, duration, visual_feats, num_sent, input_tokens, feature_mask, dense_input_tokens, timestamps

    def frame_padding(self, feature, split):

        feature_mask = torch.ones([self.max_frames])
        if len(feature) > self.max_frames:
            ids = torch.linspace(0, len(feature) - 1, self.max_frames).long()
            new_feature = torch.index_select(feature, 0, ids)
        else:
            new_feature = torch.nn.functional.interpolate(feature.unsqueeze(0).permute(0, 2, 1), size=self.max_frames,
                                                          mode='linear', align_corners=True).squeeze(0).permute(1, 0)

        return new_feature, feature_mask


def collate_fn(batch, tokenizer):
    bs = len(batch)
    num_sents = [batch[i][3] for i in range(bs)]

    video_ids = [batch[i][0] for i in range(bs)]
    durations = [batch[i][1] for i in range(bs)]
    visual_feats = torch.stack([batch[i][2] for i in range(bs)])
    feature_mask = torch.stack([batch[i][5] for i in range(bs)])

    sent_lens = [batch[i][4].shape[1] for i in range(bs)]
    max_sent_len = np.max(np.array(sent_lens))
    batch_input_tokens = []

    bt_dense_tokens = []
    max_dense_len = max(len(batch[i][6]) for i in range(bs))

    for data in batch:
        input_tokens = data[4]
        input_tokens = torch.cat([input_tokens, torch.ones(input_tokens.shape[0],
                                                           max_sent_len - input_tokens.shape[
                                                               1]) * tokenizer.pad_token_id], dim=1)
        batch_input_tokens.append(input_tokens)

        dense_input_tokens = data[6]
        dense_input_tokens = torch.cat([dense_input_tokens,
                                        torch.ones(max_dense_len - len(dense_input_tokens)) * tokenizer.pad_token_id],
                                       dim=0)
        bt_dense_tokens.append(dense_input_tokens)

    batch_input_tokens = torch.cat(batch_input_tokens, dim=0).long()
    bt_dense_tokens = torch.stack(bt_dense_tokens).long()

    # the ground-truth timestamps are only used for fully-supervised settings.
    timestamps = [batch[i][7] for i in range(bs)]
    timestamps = torch.cat(timestamps, dim=0)

    return video_ids, durations, visual_feats, batch_input_tokens, num_sents, feature_mask, bt_dense_tokens, timestamps


def get_train_loader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.add_tokens(['<FULL>', '<MASK>'])

    if args.dataset_name == 'ActivityNet-1.3':
        train_set = ANDataset(caption_path=args.train_caption_path,
                              video_feats_dir=args.video_feats_dir,
                              tokenizer=tokenizer,
                              max_captions=args.max_captions,
                              max_frames=args.max_frames,
                              max_tokens=args.max_tokens)

        train_sampler = RandomSampler(train_set)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.train_batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=lambda batch: collate_fn(batch, tokenizer),
                                  sampler=train_sampler)

        return train_loader, tokenizer
    else:
        raise NotImplementedError


def get_val_loader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.add_tokens(['<FULL>', '<MASK>'])

    if args.dataset_name == 'ActivityNet-1.3':

        val_set = ANDataset(caption_path=args.val_caption_dense_pred,
                            video_feats_dir=args.video_feats_dir,
                            tokenizer=tokenizer,
                            max_captions=args.max_captions,
                            max_frames=args.max_frames,
                            max_tokens=args.max_tokens,
                            split='val')

        val_sampler = SequentialSampler(val_set)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=args.eval_batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda batch: collate_fn(batch, tokenizer),
                                sampler=val_sampler)

        return val_loader, tokenizer

    else:
        raise NotImplementedError


def get_val_video_loader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.add_tokens(['<FULL>', '<MASK>'])

    val_set = ANDataset(caption_path=args.val_caption_1,
                        video_feats_dir=args.video_feats_dir,
                        tokenizer=tokenizer,
                        max_captions=args.max_captions,
                        max_frames=args.max_frames,
                        max_tokens=args.max_tokens,
                        split='val')

    val_sampler = SequentialSampler(val_set)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.eval_batch_size,
                            num_workers=args.num_workers,
                            collate_fn=lambda batch: collate_fn(batch, tokenizer),
                            sampler=val_sampler)

    return val_loader
