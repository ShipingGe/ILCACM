# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import json
import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(file_dir, 'coco-caption'))  # Hack to allow the import of pycocoeval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.re.re import Re
# from pycocoevalcap.self_bleu.self_bleu import Self_Bleu
import numpy as np

Set = set
import re


def parse_sent(sent):
    res = re.sub('[^a-zA-Z]', ' ', sent)
    res = res.strip().lower().split()
    return res


def parse_para(para):
    para = para.replace('..', '.')
    para = para.replace('.', ' endofsent')
    return parse_sent(para)


class ANETcaptions(object):

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 verbose=False, all_scorer=False):
        # Check that the gt and submission files exist and load them
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.all_scorer = all_scorer
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose or self.all_scorer:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]

    def ensure_caption_key(self, data):
        if len(data) == 0:
            return data
        if not list(data.keys())[0].startswith('v_'):
            data = {'v_' + k: data[k] for k in data}
        return data

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission... {}".format(prediction_filename))
        submission = json.load(open(prediction_filename))
        # change to paragraph format
        para_submission = {}
        for id in submission.keys():
            # para_submission[id] = ''
            # for info in submission[id]:
            #     para_submission[id] += info['sentence'] + '. '
            para_submission[id] = ' '.join(submission[id]['sentences'])

        for para in para_submission.values():
            assert (type(para) == str or type(para) == unicode)
        # Ensure that every video is limited to the correct maximum number of proposals.
        return self.ensure_caption_key(para_submission)

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = Set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(self.ensure_caption_key(gt))
        if self.verbose:
            print("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids)))
        return gts

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
                return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        self.scores = self.evaluate_para()

    def evaluate_para(self):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos
        gt_vid_ids = self.get_gt_vid_ids()
        vid2idx = {k: i for i, k in enumerate(gt_vid_ids)}
        gts = {vid2idx[k]: [] for k in gt_vid_ids}
        for i, gt in enumerate(self.ground_truths):
            for k in gt_vid_ids:
                if k not in gt:
                    continue
                # gts[vid2idx[k]].append(' '.join(parse_sent(gt[k])))
                gts[vid2idx[k]].append(' '.join([sent.strip().lower() for sent in gt[k]['sentences']]))

        res = {vid2idx[k]: [' '.join(parse_sent(self.prediction[k]))] \
            if k in self.prediction and len(self.prediction[k]) > 0 else [''] for k in gt_vid_ids}
        para_res = {vid2idx[k]: [' '.join(parse_para(self.prediction[k]))] \
            if k in self.prediction and len(self.prediction[k]) > 0 else [''] for k in gt_vid_ids}

        # Each scorer will compute across all videos and take average score
        output = {}
        num = len(res)
        hard_samples = {}
        easy_samples = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...' % (scorer.method()))

            if method != 'Self_Bleu':
                score, scores = scorer.compute_score(gts, res)
            else:
                score, scores = scorer.compute_score(gts, para_res)
            scores = np.asarray(scores)

            if type(method) == list:
                for m in range(len(method)):
                    output[method[m]] = score[m]
                    if self.verbose:
                        print("%s: %0.3f" % (method[m], output[method[m]]))
                for m, i in enumerate(scores.argmin(1)):
                    if i not in hard_samples:
                        hard_samples[i] = []
                    hard_samples[i].append(method[m])
                for m, i in enumerate(scores.argmax(1)):
                    if i not in easy_samples:
                        easy_samples[i] = []
                    easy_samples[i].append(method[m])
            else:
                output[method] = score
                if self.verbose:
                    print("%s: %0.3f" % (method, output[method]))
                i = scores.argmin()
                if i not in hard_samples:
                    hard_samples[i] = []
                hard_samples[i].append(method)
                i = scores.argmax()
                if i not in easy_samples:
                    easy_samples[i] = []
                easy_samples[i].append(method)
        print('# scored video =', num)

        self.hard_samples = {gt_vid_ids[i]: v for i, v in hard_samples.items()}
        self.easy_samples = {gt_vid_ids[i]: v for i, v in easy_samples.items()}
        return output


def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             verbose=args.verbose,
                             all_scorer=args.all_scorer)
    evaluator.evaluate()
    output = {}
    # Output the results
    for metric, score in evaluator.scores.items():
        print('| %s: %2.4f' % (metric, 100 * score))
        output[metric] = score
    json.dump(output, open(args.output, 'w'))
    print(output)


def eval_para(prediction, referneces, verbose=False):
    args = type('args', (object,), {})()
    args.submission = prediction
    args.references = referneces
    args.all_scorer = True
    args.verbose = verbose

    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             verbose=args.verbose,
                             all_scorer=args.all_scorer)
    evaluator.evaluate()
    output = {}

    for metric, score in evaluator.scores.items():
        # print ('| %s: %2.4f'%(metric, 100*score))
        output['para_' + metric] = score
    return output
