# %%
from pathlib import Path

import pandas as pd

# %%
seeds = [10, 20, 30, 40, 50]
split = "val"

model_name = "trailerness_transformer"
dfs = {}
for seed in seeds:
    f_name = Path(
        f"./late_fusion_results/{model_name}/late_fusion_results_best_on_{split}_seed_{seed}.pkl"
    )
    dfs[seed] = pd.read_pickle(f_name)

df = pd.concat([df for df in dfs.values()])
# %%
fusions_of_interest_names = {
    "everything together, clips, shots, visuals, semantic": "Semantic clips, Semantic shots, Visual clips, Visual shots",
    "only semantic, both clips and shots": "Semantic clips, Semantic shots",
    "only visual, both shots and clips": "Visual clips, Visual shots",
    "both semantic and visual, only clips": "Semantic clips, Visual clips",
    "both semantic and visual, only shots": "Semantic shots, Visual shots",
    "only semantic, only shots": "Semantic shots",
    "only semantic, only clips": "Semantic clips",
    "only visual, only clips": "Visual clips",
    "only visual, only shots": "Visual shots",
    "weird mix, semantic shots and visual clips": "Semantic shots, Visual clips",
    "weird mix, visual shots, semantic clips": "Visual shots, Semantic clips",
    "weird mix, mostly semantic, mostly shots": "Semantic shots, Visual shots, Semantic clips",
    "weird mix, mostly visual, mostly shots": "Semantic shots, Visual shots, Visual clips",
    "weird mix, mostly semantic, mostly clips": "Semantic shots, Semantic clips, Visual clips",
    "weird mix, mostly visual, mostly clips": "Visual shots, Semantic clips, Visual clips",
}

df = df.replace(fusions_of_interest_names)

# %%
pd.set_option("display.max_rows", None)
for split in ["test", "val"]:
    df = df.rename(
        columns={
            f"{split}/manual_frame_level_f1": f"{split}/f1",
            f"{split}/manual_frame_level_precision": f"{split}/precision",
            f"{split}/manual_frame_level_recall": f"{split}/recall",
            f"{split}/manual_frame_level_accuracy": f"{split}/accuracy",
        }
    )


# %%

for split in ["test", "val"]:
    df.loc[
        :, [f"{split}/f1", f"{split}/precision", f"{split}/recall", f"{split}/accuracy"]
    ] = (
        df.loc[
            :,
            [
                f"{split}/f1",
                f"{split}/precision",
                f"{split}/recall",
                f"{split}/accuracy",
            ],
        ].astype(float)
        * 100
    )
pd.options.display.float_format = "{:,.1f}%".format

# %%

model = "trailerness_transformer"


df = df[df.model == model].loc[
    :,
    [
        "fusing",
        "val/f1",
        "val/precision",
        "val/recall",
        "val/accuracy",
        "test/f1",
        "test/precision",
        "test/recall",
        "test/accuracy",
    ],
]

# %%
df_mean = (
    df.groupby(by=["fusing"])
    .mean()
    .rename(
        columns={
            "val/f1": "val/f1_mean",
            "val/precision": "val/precision_mean",
            "val/recall": "val/recall_mean",
            "val/accuracy": "val/accuracy_mean",
            "test/f1": "test/f1_mean",
            "test/precision": "test/precision_mean",
            "test/recall": "test/recall_mean",
            "test/accuracy": "test/accuracy_mean",
        }
    )
)
df_std = (
    df.groupby(by=["fusing"])
    .std()
    .rename(
        columns={
            "val/f1": "val/f1_std",
            "val/precision": "val/precision_std",
            "val/recall": "val/recall_std",
            "val/accuracy": "val/accuracy_std",
            "test/f1": "test/f1_std",
            "test/precision": "test/precision_std",
            "test/recall": "test/recall_std",
            "test/accuracy": "test/accuracy_std",
        }
    )
)
df_combined = pd.merge(
    left=df_mean,
    right=df_std,
    on="fusing",
)

for col in [
    "val/f1",
    "val/precision",
    "val/recall",
    "val/accuracy",
    "test/f1",
    "test/precision",
    "test/recall",
    "test/accuracy",
]:
    df_combined[col] = [
        # f"{x[f'{col}_mean']:,.1f} {{\scriptsize $\pm$ {x[f'{col}_std']:,.1f}}} "
        f"{x[f'{col}_mean']:,.1f} +- {x[f'{col}_std']:,.1f} "
        for x in df_combined.to_dict("records")
    ]
print(f"{model=}")


# %%
binarized_bas = "Semantic clips, Semantic shots, Visual clips, Visual shots".split(", ")

for col in binarized_bas:
    df_combined[col] = df_combined.index.str.contains(col, regex=True)

# %%
fusing_order = [
    "Visual clips",
    "Semantic clips",
    "Visual shots",
    "Semantic shots",
    "Semantic clips, Visual clips",
    "Semantic shots, Visual shots",
    "Visual clips, Visual shots",
    "Semantic clips, Semantic shots",
    "Semantic shots, Visual clips",
    "Visual shots, Semantic clips",
    "Visual shots, Semantic clips, Visual clips",
    "Semantic shots, Semantic clips, Visual clips",
    "Semantic shots, Visual shots, Visual clips",
    "Semantic shots, Visual shots, Semantic clips",
    "Semantic clips, Semantic shots, Visual clips, Visual shots",
]

# %%


transformer_best_df = df_combined.sort_values(
    by="fusing", key=lambda column: column.map(lambda e: fusing_order.index(e))
).loc[
    :,
    [
        "Visual clips",
        "Semantic clips",
        "Visual shots",
        "Semantic shots",
        "test/f1",
        "test/precision",
        "test/recall",
    ],
]
print(f"{model_name} results table")
# %%
print(transformer_best_df)
# print(
#     transformer_best_df.to_latex(
#         column_format="p{1cm}p{1cm}p{1cm}p{1cm}p{0.4cm}p{0.4cm}p{0.4cm}",
#         float_format="{:,.1f}\%".format,
#         index=False,
#     )
#     .replace("False", "          ")
#     .replace("True", "\checkmark")
# )

# %%
