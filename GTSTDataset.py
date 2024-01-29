# %%
import os
from fileinput import filename
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

# build GTST dataset. using binary annotations


# %%
class GTST(Dataset):
    def __init__(
        self,
        root=".",
        video_list_fname="videos_with_annotations.txt",
        window_size=64,
        as_semantic=False,
        as_shots=False,
        ground_truth="only_preview",
    ):
        video_list = f"{root}/GTST/{video_list_fname}"
        with open(video_list) as f:
            videos = f.read().splitlines()

        # if video_list == '../GTST/videos.txt':
        videos = [el[15:-4] for el in videos]
        frames_anno_df = pd.read_pickle(
            f"{root}/GTST/GTST_frames_weak_annotations.pickle"
        )

        X = []
        y = []
        filenames = []
        metadata = []

        anno_folder = f"{root}/GTST/visual_shots" if as_shots else f"{root}/GTST/visual_clips"

        if as_semantic:
            feat_folder = (
                f"{root}/GTST/textual_shots"
                if as_shots
                else f"{root}/GTST/textual_clips"
            )
        else:
            feat_folder = anno_folder + "/rgbfeatures"


        for video in videos:
            # load features
            X.append(
                torch.tensor(np.load(f"{feat_folder}/{video}.npy"), dtype=torch.float32)
            )
            preview = torch.tensor(
                np.load(f"{anno_folder}/annotations/preview/{video}.npy"),
                dtype=torch.int32,
            )

            # load labels
            y.append(preview.int())

            # load filenames
            filenames.append(video)

            # load metadata
            metadata_per_ep = {}
            frame_level_gt = {}
            ep_name = video
            ep_nr = int(video[-4:])
            frame_level_gt["mask"] = torch.tensor(
                frames_anno_df[frames_anno_df.episode == ep_nr].body.values,
                dtype=torch.bool,
            )

            frame_level_gt["only_preview"] = torch.tensor(
                np.array(
                    frames_anno_df[frames_anno_df.episode == ep_nr]
                    .is_in_preview.fillna(99999)
                    .values,
                    dtype=np.float32,
                ),
                dtype=torch.int32,
            )[frame_level_gt["mask"]]

            metadata_per_ep["frame_level_gt"] = frame_level_gt

            if as_shots:
                metadata_per_ep["shot_level_mask"] = torch.tensor(
                    np.load(
                        f"{anno_folder}/annotations/meta_shot_level_mask/{ep_name}.npy"
                    ),
                    dtype=torch.int32,
                )
                metadata_per_ep["shot_to_frame_map"] = torch.tensor(
                    np.load(
                        f"{anno_folder}/annotations/meta_shot_to_frame_map/{ep_name}.npy"
                    ),
                    dtype=torch.int32,
                )
            else:
                metadata_per_ep["clip_level_mask"] = torch.tensor(
                    np.load(
                        f"{anno_folder}/annotations/meta_clip_level_mask/{ep_name}.npy"
                    ),
                    dtype=torch.int32,
                )
            metadata_per_ep["ground_truth"] = ground_truth
            metadata_per_ep["as_shots"] = as_shots
            metadata_per_ep["window_size"] = window_size
            metadata.append(metadata_per_ep)

        self.x = X
        self.y = y
        self.filename = filenames
        self.metadata = metadata

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        features = self.x[idx]
        labels = self.y[idx]
        filename = self.filename[idx]
        metadata = self.metadata[idx]
        # return features, labels, filename
        return {
            "features": features,
            "labels": labels,
            "filename": filename,
            "metadata": metadata,
        }

