import os
import torch
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
from functools import reduce

from args import get_args_parser
from util.metrics import MetricLogger
from dvc_eval import eval_dvc, eval_soda

from util.dataloader import get_train_loader, get_val_loader, get_val_video_loader
from modeling.model import ILCACM

from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import load_file

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train_one_epoch_dense(
        model: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        args,
        accelerator,
        tokenizer=None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    for i_batch, batch_data in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header, disabled=not accelerator.is_main_process)
    ):
        vids, durs, vfeats, batch_input_tokens, num_sents, fmask, dense_tokens, _ = batch_data

        loss = model(video_features=vfeats,
                     feature_mask=fmask,
                     input_ids=dense_tokens,
                     num_sents=num_sents,
                     is_full=True)

        losses = accelerator.gather(loss)
        loss_value = losses.mean().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate_dense(
        model: torch.nn.Module,
        args,
        accelerator,
        split="test",
        dataset_name="ActivityNet-1.3",
        tokenizer=None,
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    # saving preds
    val_loader = get_val_video_loader(args)
    val_loader = accelerator.prepare(val_loader)

    all_res = []

    for i_batch, batch_data in enumerate(
            metric_logger.log_every(val_loader, args.print_freq, header, disabled=not accelerator.is_main_process)
    ):
        vids, durs, vfeats, _, num_sents, fmask, _, _ = batch_data

        with torch.no_grad():
            results = model.module.captioner.generate(vfeats, fmask)

        res = []

        for i, vid in enumerate(vids):
            sents = results[i]
            sents = sents.split('events:')[1].strip() if 'events:' in sents else sents
            sentences = [sentence.strip() for sentence in sents.split('.') if sentence]
            sentences = [sent + '.' if sent[-1] != '.' else sent for sent in sentences]
            res.append({vid: {'sentences': sentences, 'duration': durs[i]}})

        outputs = accelerator.gather_for_metrics(res)
        all_res += outputs

    accelerator.wait_for_everyone()

    all_res = reduce(lambda a, b: a.update(b) or a, all_res, {})

    if args.save_dir and accelerator.is_main_process:
        if os.path.exists(args.val_caption_dense_pred):
            os.remove(args.val_caption_dense_pred)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        json.dump(all_res, open(args.val_caption_dense_pred, "w"))
        print('Full captioning stage finished.')


def train_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        args,
        accelerator,
        tokenizer=None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    for i_batch, batch_data in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header, disabled=not accelerator.is_main_process)
    ):
        vids, durs, vfeats, batch_input_tokens, num_sents, fmask, dense_tokens, timestamps = batch_data

        loss = model(video_features=vfeats,
                     num_sents=num_sents,
                     feature_mask=fmask,
                     input_ids=batch_input_tokens,
                     pos=True,
                     timestamps=timestamps)

        accelerator.backward(loss)
        optimizer.step()

        loss = model(video_features=vfeats,
                     num_sents=num_sents,
                     feature_mask=fmask,
                     input_ids=batch_input_tokens,
                     pos=False,
                     timestamps=timestamps)

        accelerator.backward(loss)
        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

        losses = accelerator.gather(loss)
        loss_value = losses.mean().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        args,
        accelerator,
        split="test",
        dataset_name="ActivityNet-1.3",
        tokenizer=None,
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    val_loader, _ = get_val_loader(args)
    val_loader = accelerator.prepare(val_loader)

    all_res = []
    start_time = time.time()
    for i_batch, batch_data in enumerate(
            metric_logger.log_every(val_loader, args.print_freq, header, disabled=not accelerator.is_main_process)
    ):
        vids, durs, vfeats, _, num_sents, fmask, _, _ = batch_data

        video_ids = []
        durations = []

        for i in range(len(num_sents)):
            num = num_sents[i]
            video_ids += [vids[i]] * num
            durations += [durs[i]] * num

        lefts, rights, pred_sents = model(video_features=vfeats,
                                          num_sents=num_sents,
                                          feature_mask=fmask,
                                          localize=True)

        res = {}
        for i, vid in enumerate(video_ids):

            text = pred_sents[i].strip()
            start = np.round(lefts[i].item() * durations[i], 2)
            end = np.round(rights[i].item() * durations[i], 2)
            if vid not in res:
                res[vid] = []
            res[vid].append({'sentence': text,
                             'timestamp': [start, end]})

        res = accelerator.gather_for_metrics([res])
        all_res += res

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(val_loader.dataset)
    metrics = {}
    if accelerator.is_main_process:
        if args.save_dir:
            pred_path = os.path.join(args.save_dir, dataset_name + f"_{split}_localization_preds.json", )
            json.dump({'results': results}, open(pred_path, "w", ))
        else:
            pred_path = {'results': results}

        if dataset_name == 'ActivityNet-1.3':
            references = args.val_caption_path

        else:
            raise NotImplementedError

        metrics.update(
            eval_dvc(pred_path, references, tious=[0.3, 0.5, 0.7, 0.9], max_proposals_per_video=1000,
                     verbose=False,
                     no_lang_eval=False))
        metrics.update(eval_soda(pred_path, references, verbose=False))
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        json.dump(
            metrics,
            open(
                os.path.join(args.save_dir, args.dataset_name + "summary.json"), "w"
            ),
        )
        print(total_time_str)


def main(args):
    # Fix seeds
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    accelerator = Accelerator()

    # Build model
    train_loader, tokenizer = get_train_loader(args)

    model = ILCACM(tokenizer, args)

    if accelerator.is_main_process:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of params:{} M".format(float(n_parameters / 1e6)))

    optimizer = torch.optim.AdamW(list(p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr,
                                  betas=(0.9, 0.99),
                                  eps=1e-08,
                                  weight_decay=0.01)

    num_training_steps = int(len(train_loader) * args.total_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.05 * num_training_steps),
                                                num_training_steps=num_training_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    if not args.eval:
        if accelerator.is_main_process:
            print("Start training")
        start_time = time.time()

        for epoch in range(args.cap_epochs):
            train_one_epoch_dense(
                model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                accelerator=accelerator,
                tokenizer=tokenizer)

        accelerator.wait_for_everyone()

        evaluate_dense(model=model,
                       dataset_name=args.dataset_name,
                       args=args,
                       split="test",
                       tokenizer=tokenizer,
                       accelerator=accelerator)

        accelerator.wait_for_everyone()

        for epoch in range(args.cap_epochs, args.total_epochs):
            if accelerator.is_main_process:
                print(f"Starting epoch {epoch}")
            train_one_epoch(
                model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                accelerator=accelerator,
                tokenizer=tokenizer)

        accelerator.wait_for_everyone()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if accelerator.is_main_process:
            print("Training time {}".format(total_time_str))

        if args.save_dir:
            checkpoint_path = os.path.join(args.save_dir, f"stage_2")
            accelerator.save_model(model, checkpoint_path)

    else:
        if accelerator.is_main_process:
            print(f"loading checkpoint...")
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_path = os.path.join(args.save_dir, f"stage_2", "model.safetensors")
        unwrapped_model.load_state_dict(load_file(checkpoint_path))

    accelerator.wait_for_everyone()

    evaluate(
        model=model,
        dataset_name=args.dataset_name,
        args=args,
        split="test",
        tokenizer=tokenizer,
        accelerator=accelerator
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
