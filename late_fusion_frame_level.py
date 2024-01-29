# %%
import json
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torchmetrics import Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

# %%

api = wandb.Api()

seeds = [10, 20, 30, 40, 50]

for model_name in ["trailerness_transformer"]:
    results_path = Path(f"./late_fusion_results/{model_name}")
    results_path.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        # Project is specified by <entity/project-name>
        # raise ValueError("change this to your WandB entity/project-name")
        runs = api.runs(
            "carlobretti/trailerness", ## change this to your WandB entity/project-name
            filters={
                "config.general.model_name": model_name,
                # "config.trainer.model_checkpoint_monitor": "val/frame_f1",
                # "config.trainer.max_epochs": 100,
                # "config.module.num_layers": 1,
                # "config.module.dropout": 0.5,
                # "config.module.model_dim": 64,
                # "config.module.input_dropout": 0.2,
                # "config.module.lr": 0.0005,
                "config.general.seed": seed,
            },
        )
        if len([run.id for run in runs]) != 4:
            raise ValueError("incorrect filter! only 4 runs per config")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        window_size = 64

        ## %%
        # threshold 0.5 here since here i'm using the already-sigmoided preds
        f1_module = F1Score(task="binary", threshold=0.5).to(device)
        recall_module = Recall(task="binary", threshold=0.5).to(device)
        precision_module = Precision(task="binary", threshold=0.5).to(device)
        accuracy_module = Accuracy(task="binary", threshold=0.5).to(device)

        ## %%
        runs_ls = []
        for run in tqdm(runs):
            configs = (
                json.dumps(run.config["general"], sort_keys=True)
                + json.dumps(run.config["module"], sort_keys=True)
                + json.dumps(run.config["trainer"], sort_keys=True)
                + run.config["datamodule"]["ground_truth"]
            )

            runs_ls.append(
                {
                    "config": configs,
                    "id": run.id,
                    "as_shots": run.config["datamodule"]["as_shots"],
                    "as_semantic": run.config["datamodule"]["as_semantic"],
                    "ground_truth": run.config["datamodule"]["ground_truth"],
                    "model": run.config["general"]["model_name"],
                    "module": run.config["module"],
                    "trainer": run.config["trainer"],
                    **run.summary,
                }
            )
        runs_df = pd.DataFrame(runs_ls)

        # #%%

        possible_combos = {
            "only visual, only clips": [[0, 0]],
            "only visual, only shots": [[0, 1]],
            "only semantic, only clips": [[1, 0]],
            "only semantic, only shots": [[1, 1]],
            "both semantic and visual, only shots": [[1, 1], [0, 1]],
            "both semantic and visual, only clips": [[1, 0], [0, 0]],
            "only semantic, both clips and shots": [[1, 1], [1, 0]],
            "only visual, both shots and clips": [[0, 1], [0, 0]],
            "weird mix, semantic shots and visual clips": [[1, 1], [0, 0]],
            "weird mix, visual shots, semantic clips": [[0, 1], [1, 0]],
            "weird mix, mostly semantic, mostly shots": [[1, 1], [0, 1], [1, 0]],
            "weird mix, mostly visual, mostly shots": [[1, 1], [0, 1], [0, 0]],
            "weird mix, mostly semantic, mostly clips": [[1, 1], [1, 0], [0, 0]],
            "weird mix, mostly visual, mostly clips": [[0, 1], [1, 0], [0, 0]],
            "everything together, clips, shots, visuals, semantic": [
                [0, 1],
                [1, 0],
                [0, 0],
                [1, 1],
            ],
        }
        # #%%
        # pick best 4 runs across the different scales and modalities based on the val frame level f1
        # for the different models

        pick_split = "val"
        best_run_per_stream = defaultdict(list)
        # best_runs_idx = []
        for model in runs_df["model"].unique():
            for k, v in possible_combos.items():
                if len(v) == 1:
                    id = runs_df[
                        (runs_df["model"] == model)
                        & (runs_df["as_shots"] == v[0][0])
                        & (runs_df["as_semantic"] == v[0][1])
                    ][f"{pick_split}/frame_f1"].idxmax()
                    best_run_per_stream[model].append(runs_ls[id])
                    # best_run_per_stream[k] = runs_df['id'].iloc[id]

        # best_runs = [runs_ls[i] for i in best_runs_idx]
        # #%%

        results = []
        frame_level_annotations = {}

        df = pd.read_pickle("./GTST/GTST_frames_weak_annotations.pickle")
        clip_anno_folder = f"./GTST/visual_clips/annotations"

        for combo_name, combo in tqdm(possible_combos.items()):
            # compute late fusion scores
            for model, group_runs in best_run_per_stream.items():
                # let's start from the first run in the group just to be able to iterate through the episodes
                first_run_path = Path(
                    glob(
                        f'./multirun/*/*/*/experiments/logs/*/*/wandb/latest-run/run-{group_runs[0]["id"]}.wandb'
                    )[0]
                )

                f1_accu = defaultdict(list)
                recall_accu = defaultdict(list)
                precision_accu = defaultdict(list)
                accuracy_accu = defaultdict(list)
                size_of_each_batch = defaultdict(list)

                runs_ids = []

                for ep in glob(f"{first_run_path.parents[2]}/*.pt"):
                    split = Path(ep).stem.split("_")[0]
                    if not split in ["val", "test"]:
                        continue
                    ep_name = "_".join(Path(ep).stem.split("_")[1:])
                    ep_nr = int(ep_name[-4:])

                    # load frame-level annotations and store them in a dict
                    if not ep_name in frame_level_annotations:
                        mask = torch.tensor(
                            df[df.episode == ep_nr].body.values,
                            dtype=torch.bool,
                            device=device,
                        )
                        only_preview = torch.tensor(
                            np.array(
                                df[df.episode == ep_nr]
                                .is_in_preview.fillna(99999)
                                .values,
                                dtype=np.float32,
                            ),
                            dtype=torch.int32,
                            device=device,
                        )
                        only_recap = torch.tensor(
                            np.array(
                                df[df.episode == ep_nr]
                                .is_in_recap.fillna(99999)
                                .values,
                                dtype=np.float32,
                            ),
                            dtype=torch.int32,
                            device=device,
                        )
                        preview_or_recap = torch.logical_or(
                            only_recap, only_preview
                        ).int()

                        # after obtaining unmasked annotations, mask them and store them
                        # also store mask to mask back predictions later
                        frame_level_annotations[ep_nr] = {
                            "mask": mask,
                            "only_preview": only_preview[mask],
                            "only_recap": only_recap[mask],
                            "preview_or_recap": preview_or_recap[mask],
                        }

                    # load annotations at a clip level
                    gts = [r["ground_truth"] for r in group_runs]
                    if len(set(gts)) > 1:
                        print(gts)
                        raise ValueError(
                            "runs have different ground truths - cannot fuse predictions"
                        )
                    if gts[0] == "preview_or_recap":
                        y = frame_level_annotations[ep_nr]["preview_or_recap"]
                    elif gts[0] == "only_preview":
                        y = frame_level_annotations[ep_nr]["only_preview"]

                    preds_list = []

                    for run in group_runs:
                        current_run_path = Path(
                            glob(
                                f'./multirun/*/*/*/experiments/logs/*/*/wandb/latest-run/run-{run["id"]}.wandb'
                            )[0]
                        )

                        # if run should not be included for this combo, continue
                        if [int(run["as_semantic"]), int(run["as_shots"])] not in combo:
                            continue

                        # keep track of runs ids actually used
                        if not run["id"] in runs_ids:
                            runs_ids.append(run["id"])

                        ep_run = glob(
                            f"{current_run_path.parents[2]}/{split}_{ep_name}.pt"
                        )[0]
                        preds = torch.load(ep_run).to(device)

                        # if model trained at shot level let's bring it back to frame level
                        if run["as_shots"]:
                            # load useful things to map back results from shot-level to frame level
                            clip_level_mask = torch.tensor(
                                np.load(
                                    f"{clip_anno_folder}/meta_clip_level_mask/{ep_name}.npy"
                                ),
                                device=device,
                            )
                            shot_anno_folder = (
                                "/".join(clip_anno_folder.split("/")[:-2])
                                + "/visual_shots/annotations"
                            )
                            shot_level_mask = torch.tensor(
                                np.load(
                                    f"{shot_anno_folder}/meta_shot_level_mask/{ep_name}.npy"
                                ),
                                dtype=torch.int32,
                                device=device,
                            )
                            shot_to_frame_map = np.load(
                                f"{shot_anno_folder}/meta_shot_to_frame_map/{ep_name}.npy"
                            )

                            # "inflate" predictions at a shot-level (masked) backed to the unmasked shot-level
                            shots_unmasked = torch.zeros_like(
                                shot_level_mask, dtype=torch.float, device=device
                            )
                            shots_unmasked[shot_level_mask == 1] = preds.squeeze()
                            shots_unmasked = shots_unmasked.cpu().numpy()

                            # copy preds at shot level to frame level
                            n_frames = frame_level_annotations[ep_nr]["mask"].size()[0]
                            frame_level_preds = np.zeros(n_frames)
                            for (shot_start, shot_end), pred in zip(
                                shot_to_frame_map, shots_unmasked
                            ):
                                frame_level_preds[shot_start : shot_end + 1] = pred

                            # mask back preds at a frame level
                            preds = torch.tensor(frame_level_preds, device=device)[
                                frame_level_annotations[ep_nr]["mask"]
                            ]
                        # if model trained at clip level let's bring it back to frame level
                        else:
                            clip_level_mask = torch.tensor(
                                np.load(
                                    f"{clip_anno_folder}/meta_clip_level_mask/{ep_name}.npy"
                                ),
                                device=device,
                            )

                            # "inflate" predictions at a clip-level (masked) backed to the unmasked clip-level
                            clip_unmasked = torch.zeros_like(
                                clip_level_mask, dtype=torch.float, device=device
                            )
                            clip_unmasked[clip_level_mask == 1] = preds.squeeze()
                            clip_unmasked = clip_unmasked.cpu().numpy()

                            n_frames = frame_level_annotations[ep_nr]["mask"].size()[0]
                            frame_level_preds = np.zeros(n_frames)
                            for i, pred in enumerate(clip_unmasked):
                                start_frame = i * window_size
                                end_frame = start_frame + window_size
                                # no need to add 1 here as window size is the length of the clip as a whole, not end point as in shots
                                frame_level_preds[start_frame:end_frame] = pred
                            # mask back preds at a frame level
                            preds = torch.tensor(frame_level_preds, device=device)[
                                frame_level_annotations[ep_nr]["mask"]
                            ]

                        preds_list.append(preds.squeeze())

                    preds_list = torch.stack(preds_list)

                    # average out preds from different modalities and scales
                    preds = preds_list.mean(dim=0)

                    preds, y = preds.squeeze(), y.squeeze()
                    size_of_each_batch[split].append(y.size()[0])
                    y = y.to(device)
                    f1_accu[split].append(f1_module(preds, y))
                    recall_accu[split].append(recall_module(preds, y))
                    precision_accu[split].append(precision_module(preds, y))
                    accuracy_accu[split].append(accuracy_module(preds, y))

                size_of_each_batch = {
                    split: torch.tensor(size_of_each_batch[split])
                    for split in size_of_each_batch.keys()
                }
                weighted_batches = {
                    split: size_of_each_batch[split]
                    / torch.sum(size_of_each_batch[split])
                    for split in size_of_each_batch.keys()
                }

                measures_results = {}

                for measure_name, measure in {
                    "f1": f1_accu,
                    "recall": recall_accu,
                    "precision": precision_accu,
                    "accuracy": accuracy_accu,
                }.items():
                    for split in measure.keys():
                        # redundant? yes but it's fine
                        measure_accu = torch.tensor(measure[split])
                        measure_balanced = torch.sum(
                            measure_accu * weighted_batches[split]
                        )
                        measures_results[
                            f"{split}/manual_frame_level_{measure_name}"
                        ] = measure_balanced

                results.append(
                    {
                        # "group_of_runs": group_config,
                        "model": model,
                        "runs_ids": runs_ids,
                        "ground_truth": {
                            r["id"]: r["ground_truth"] for r in group_runs
                        },
                        "module": {r["id"]: r["module"] for r in group_runs},
                        "trainer": {r["id"]: r["trainer"] for r in group_runs},
                        "fusing": combo_name,
                        **measures_results,
                    }
                )

            pd.DataFrame(results).to_pickle(
                f"{results_path}/late_fusion_results_best_on_{pick_split}_seed_{seed}.pkl"
            )
