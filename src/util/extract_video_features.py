import torch
import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import ffmpeg
from torch.utils.data.sampler import Sampler
from transformers import CLIPProcessor, CLIPModel


class RandomSequenceSampler(Sampler):
    def __init__(self, n_sample, seq_len):
        self.n_sample = n_sample
        self.seq_len = seq_len

    def _pad_ind(self, ind):
        zeros = np.zeros(self.seq_len - self.n_sample % self.seq_len)
        ind = np.concatenate((ind, zeros))
        return ind

    def __iter__(self):
        idx = np.arange(self.n_sample)
        if self.n_sample % self.seq_len != 0:
            idx = self._pad_ind(idx)
        idx = np.reshape(idx, (-1, self.seq_len))
        np.random.shuffle(idx)
        idx = np.reshape(idx, (-1))
        return iter(idx.astype(int))

    def __len__(self):
        return self.n_sample + (self.seq_len - self.n_sample % self.seq_len)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):
    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            video_dir,
            output_dir,
            framerate=1,
            size=112,
            centercrop=False,
    ):

        self.videos = os.listdir(video_dir)
        self.video_dir = video_dir

        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.output_dir = output_dir

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def __len__(self):
        return len(self.videos)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        num, denum = video_stream["avg_frame_rate"].split("/")
        frame_rate = int(num) / int(denum)
        return height, width, frame_rate

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.videos[idx])
        output_file = os.path.join(self.output_dir, self.videos[idx].split('.')[0])

        if not (os.path.isfile(output_file)) and os.path.isfile(video_path):
            print("Decoding video: {}".format(video_path))
            try:
                h, w, fr = self._get_video_dim(video_path)
            except:
                print("ffprobe failed at: {}".format(video_path))
                return {
                    "video": th.zeros(1),
                    "input": video_path,
                    "output": output_file,
                }
            if fr < 1:
                print("Corrupted Frame Rate: {}".format(video_path))
                return {
                    "video": th.zeros(1),
                    "input": video_path,
                    "output": output_file,
                }
            height, width = self._get_output_dim(h, w)

            try:
                cmd = (
                    ffmpeg.input(video_path)
                    .filter("fps", fps=self.framerate)
                    .filter("scale", width, height)
                )
                if self.centercrop:
                    x = int((width - self.size) / 2.0)
                    y = int((height - self.size) / 2.0)
                    cmd = cmd.crop(x, y, self.size, self.size)
                out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                    capture_stdout=True, quiet=True
                )
            except:
                print("ffmpeg error at: {}".format(video_path))
                return {
                    "video": th.zeros(1),
                    "input": video_path,
                    "output": output_file,
                }
            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = th.from_numpy(video.astype("float32"))
            video = video.permute(0, 3, 1, 2)
        else:
            video = th.zeros(1)

        return {"video": video, "input": video_path, "output": output_file}


parser = argparse.ArgumentParser(description="Easy video feature extractor")

parser.add_argument(
    "--video_dir",
    type=str,
)
parser.add_argument(
    "--output_dir",
    type=str,
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size for extraction"
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=1,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=3,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
parser.add_argument(
    "--feature_dim", type=int, default=768, help="output video feature dimension"
)
args = parser.parse_args()

dataset = VideoLoader(
    video_dir=args.video_dir,
    output_dir=args.output_dir,
    framerate=1,  # one feature per second max
    size=224,
    centercrop=True,
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)

CLIP_path = '?/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
processor = CLIPProcessor.from_pretrained(CLIP_path)
model = CLIPModel.from_pretrained(CLIP_path)
model.eval()
model.cuda()

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data["input"][0]
        output_file = data["output"][0]
        if len(data["video"].shape) > 3:
            print(
                "Computing features of video {}/{}: {}".format(
                    k + 1, n_dataset, input_file
                )
            )
            video = data["video"].squeeze()
            if len(video.shape) == 4:
                video = processor(images=video, return_tensors='pt')['pixel_values'].numpy()
                video = torch.from_numpy(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in tqdm(range(n_iter)):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model.get_image_features(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()

                np.save(output_file, features)
        else:
            print("Video {} already processed.".format(input_file))
