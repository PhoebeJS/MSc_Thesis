import sys
sys.path.append("..")
from functions import *


# Scatter plot of records for CER and WER

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

# Read the jsonl and pull out just what we need per record
records = []
with open("corpus_metrics/claude_fewshot_bl_specific_structured_run-2.jsonl") as f:
    for line in f:
        row = json.loads(line)
        records.append({
            "doc_id": row["document_id"],
            "page_id": row["page_id"],
            "cer": row["overall_CER"],
            "wer": row["overall_WER"],
            "page_type": row["page_type"]
        })

df = pd.DataFrame(records)
df["record_label"] = df["doc_id"].astype(str) + " / " + df["page_id"].astype(str)


# Scatter plot

colour_dict = {"text": "#26f3fa", "table": "#ed1d6a", "mixed": "#ffa11a"}
colored_points = df["page_type"].map(colour_dict)


sns.set_theme(style="whitegrid", context="paper")   # theme/style
fig, ax = plt.subplots(figsize=(8, 6))
for category in df["page_type"].unique():
    x = df[df["page_type"] == category]
    ax.scatter(x["cer"], x["wer"], c=colour_dict.get(category), label=category.capitalize(), s=40, edgecolor="black", alpha=0.7)
ax.legend(loc = "lower center", bbox_to_anchor = (0.5, -0.2), ncol = 2)
ax.set_title("CER vs WER per record")
#ax.set_xscale('log')
#ax.set_yscale('log')

# Label only the outliers directly on the plot (top N by CER, say),
# otherwise 300+ labels would make it unreadable.
sns.regplot(x="cer", y="wer", data=df, ax=ax, scatter=False, ci=95)
ax.set_xlabel("Character Error Rate (CER)")
ax.set_ylabel("Word Error Rate (WER)")

fig.tight_layout()
fig.savefig("cer_wer_scatter.png", dpi=200)
print("Saved plot to cer_wer_scatter.png")

# Lookup table to see records with worst error scores
# Sort by CER descending so the worst records are at the top 
print(df.sort_values("cer", ascending=False).head(20)[["record_label", "cer", "wer"]])




###############
###############

# Load in data
# Loading it in

# Excel sheet with recorded info fromo each run
file_path = "data/MSc_Thesis_Results.xlsx"  
xls = pd.ExcelFile(file_path)
print(xls.sheet_names)


# Reading each sheet into it's own df to be used separetly
df_bl_results = pd.read_excel(file_path, sheet_name="BL_data_results")
df_nhm_results = pd.read_excel(file_path, sheet_name="NHM_data_results")
df_ie_results = pd.read_excel(file_path, sheet_name="IE_results")


# Renaming columns to replace spaces with _ - also removing a headache
df_bl_results.columns = df_bl_results.columns.str.replace(" ", "_").str.lower()

df_nhm_results.columns = df_nhm_results.columns.str.replace(" ", "_").str.lower()

df_ie_results.columns = df_ie_results.columns.str.replace(" ", "_").str.lower()


### Bar plot pooled per model for all metrics for BL dataset
# Group by model and calculate mean and CI for each metric


# Given a set of conditions each with their own (mean, sd, n), this returns
# the correct combined mean and SD *as if* all the underlying records had
# been pooled together — without needing the raw records themselves.
#
# grand_mean = weighted average of the means, weighted by n
# combined_var = weighted average of (within-condition variance +
#                squared distance of that condition's mean from the grand mean)
# This second term is what a plain "average of the SDs" would miss.

def pool_mean_sd(means, sds, ns):
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    ns = np.asarray(ns, dtype=float)

    grand_mean = np.sum(ns * means) / np.sum(ns)
    combined_var = np.sum(ns * (sds**2 + (means - grand_mean)**2)) / np.sum(ns)
    combined_sd = np.sqrt(combined_var)

    return grand_mean, combined_sd


# Apply it per model, per metric
# `the data set  is the full summary table (one row per Model/Dataset/Prompt/Specificity
# combination). We group by Model, and within each group pool across every
# other condition using the function above.

df_bl_results = pd.read_excel(file_path, sheet_name="BL_data_results")
df_bl_results.columns = df_bl_results.columns.str.replace(" ", "_").str.lower()
df_bl_results = df_bl_results.rename(columns={"sd_cer.1": "sd_wer"})

metrics = {
    "CER":        ("overall_cer_mean", "sd_cer",  "n_cer"),
    "WER":        ("overall_wer_mean", "sd_wer",  "n_wer"),  
    "Levenshtein": ("overall_lev_mean", "sd_lev", "n_lev"),
    "TokenSort":   ("overall_token_sort_mean", "sd_token", "n_token_sort"),
}

# A note about pooling: the pooling formula itself doesn't care what a 'group' is. All it needs is a bunch of triplets (mean, sd, n) to combine. The meaning of 'group' comes entirely from how yo uslice your table before feeding it in


rows = []
# Take every row in my big table, & sort them into buckets - one bucket per unique model. So all of Qwen's rows (zerom one, few, specific, non) land in one bucket & all of claude's in another
# groupby doesn't just split the table, it hands back pairs. each pair is: "bucket_name", "bucket_contents". So here, model is the bucket name, & grouop is the smaller dataframe containing only rows where model equals that name
for model, group in df_bl_results.groupby("model"):
    for metric_name, (mean_col, sd_col, n_col) in metrics.items():
        # Nested loop to unpack dictionary of tuples
                # items() gives (key, value) pairs. Here key is metric name "CER" and value is tuple of 3 column names. We set 3 values instead of 1 (mean_col...) bc python will let you unpack a tuple into several variables in one go as long as the shape matches
                # Inside of each bucket, the pooling formula treats each row as one 'group i', so each get's their own mean, sd, & n 
                # pool_mean_sd folds all of those group i's into one number for Qwen/Claude overall
        grand_mean, combined_sd = pool_mean_sd(
            # group is the df built via groupby above
            # The square brackets mean give me this column on it's own. It then feeds those into the pool_mean_sd
            group[mean_col], group[sd_col], group[n_col]
        )
        total_n = group[n_col].sum()
        se = combined_sd / np.sqrt(total_n)
        ci_half_width = stats.norm.ppf(0.975) * se

        rows.append({"Model": model, "Metric": metric_name,
                      "mean": grand_mean, "sd": combined_sd,
                      "n": total_n, "ci": ci_half_width,
                      "ci_lower": grand_mean - ci_half_width,
                      "ci_upper": grand_mean + ci_half_width})

pooled = pd.DataFrame(rows)
print(pooled)

# Plot — one subplot per metric, bars coloured by model 
metric_names = pooled["Metric"].unique()
# plt.subplots returns two things bundled as a tuple: fig (overall fig) & axes (the individual subplot areas inside that fig)
# First two arguments of plt.subplots is rows & columns of subplot grid. 1 row & len(metric_names) makes 1 row x 4 columns so four side by side subplot boxes, one per metric
# figsize sets overall canvas size in inches - width is 4 times the number of metrics & height fixed at 4 in
fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
# Safety patch: if there's only 1 metric, axes would otherwise be a single unloopable object, so wrap it in a list so the rest of the code can always assume axes is something that can be looped over. 
if len(metric_names) == 1:
    axes = [axes]

## Aesthetics
# Font setting
sns.set_theme(style="whitegrid", context="paper")   # theme/style
hex = sns.color_palette(["#53B062", "#af4343"])


for ax, metric_name in zip(axes, metric_names):
    sub = pooled[pooled["Metric"] == metric_name]
    ax.bar(sub["Model"], sub["mean"], yerr=sub["ci"], capsize=5,
           color=hex[:len(sub)])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)


plt.rcParams["font.family"] = "Times New Roman"
fig.suptitle("Model comparison, pooled across prompt/specificity conditions")
fig.tight_layout()
fig.savefig("model_comparison_pooled.png", dpi=200)
print("Saved plot to model_comparison_pooled.png")





################
################
# Bar plot pooled per model for all metrics for NHM dataset
# Group by model and calculate mean and CI for each metric

df_nhm_results = pd.read_excel(file_path, sheet_name="NHM_data_results")
df_nhm_results.columns = df_nhm_results.columns.str.replace(" ", "_").str.lower()

metrics = {
    "Character Error Rate":        ("overall_cer_mean", "sd_cer",  "n_cer"),
    "Word Error Rate":        ("overall_wer_mean", "sd_wer",  "n_wer"),  
    "Levenshtein Distance": ("overall_lev_mean", "sd_lev", "n_lev"),
    "Token Sort Ratio":   ("overall_token_sort_mean", "sd_token", "n_token_sort"),
}

rows = []
for model, group in df_nhm_results.groupby("model"):
    for metric_name, (mean_col, sd_col, n_col) in metrics.items():
        # Nested loop to unpack dictionary of tuples
                # items() gives (key, value) pairs. Here key is metric name "CER" and value is tuple of 3 column names. We set 3 values instead of 1 (mean_col...) bc python will let you unpack a tuple into several variables in one go as long as the shape matches
                # Inside of each bucket, the pooling formula treats each row as one 'group i', so each get's their own mean, sd, & n 
                # pool_mean_sd folds all of those group i's into one number for Qwen/Claude overall
        grand_mean, combined_sd = pool_mean_sd(
            # group is the df built via groupby above
            # The square brackets mean give me this column on it's own. It then feeds those into the pool_mean_sd
            group[mean_col], group[sd_col], group[n_col]
        )
        total_n = group[n_col].sum()
        se = combined_sd / np.sqrt(total_n)
        ci_half_width = stats.norm.ppf(0.975) * se

        rows.append({"Model": model, "Metric": metric_name,
                      "mean": grand_mean, "sd": combined_sd,
                      "n": total_n, "ci": ci_half_width,
                      "ci_lower": grand_mean - ci_half_width,
                      "ci_upper": grand_mean + ci_half_width})

pooled = pd.DataFrame(rows)
print(pooled)

pooled.replace({"Model":{"Claude": "Claude Sonnet 5", "Qwen 32B": "Qwen2.5-VL-32B "}}, inplace=True)

# --- Step 3: plot — one subplot per metric, bars coloured by model ---
metric_names = pooled["Metric"].unique()
fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
if len(metric_names) == 1:
    axes = [axes]

## Aesthetics
# Font setting
sns.set_theme(style="whitegrid", context="paper")   # theme/style
hex = sns.color_palette(["#ffa11a", "#26f3fa"])


for ax, metric_name in zip(axes, metric_names):
    sub = pooled[pooled["Metric"] == metric_name]
    ax.bar(sub["Model"], sub["mean"], yerr=sub["ci"], capsize=5,
       color=hex[:len(sub)])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)


plt.rcParams["font.family"] = "Times New Roman"
fig.suptitle("Model comparison, pooled across prompt/specificity conditions")
fig.tight_layout()
fig.savefig("model_comparison_pooled.png", dpi=200)
print("Saved plot to model_comparison_pooled.png")



#############
#############

# Token Usage across models
# Made figures but am using printed tables for the thesis.


### BL dataset
df_bl_results = pd.read_excel(file_path, sheet_name="BL_data_results")
df_bl_results.columns = df_bl_results.columns.str.replace(" ", "_").str.lower()
df_bl_results = df_bl_results.rename(columns={"sd_cer.1": "sd_wer"})

metrics = {
    "Prompt Tokens":        ("prompt_tokens_mean", "sd_prompt_tokens",  "n_prompt_tokens"),
    "Completion Tokens":    ("completion_tokens_mean", "sd_completion_tokens",  "n_completion_tokens"),  
    "Total Tokens":         ("total_tokens_mean", "sd_total_tokens", "n_total_tokens")
}

rows = []
for (model, prompt), group in df_bl_results.groupby(["model", "image_one_or_few_shot?"]):
    for metric_name, (mean_col, sd_col, n_col) in metrics.items():
        grand_mean, combined_sd = pool_mean_sd(
            group[mean_col], group[sd_col], group[n_col]
        )
        rows.append({"Model": model, "Prompt": prompt, "Metric": metric_name,
                      "mean": grand_mean, "sd": combined_sd})

pooled = pd.DataFrame(rows)
print(pooled)

metric_names = pooled["Metric"].unique()
fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
if len(metric_names) == 1:
    axes = [axes]

## Aesthetics
# Font setting
sns.set_theme(style="whitegrid", context="paper")   # theme/style
hex = sns.color_palette(["#ffa11a", "#26f3fa", "#ed1d6a"])


for ax, metric_name in zip(axes, metric_names):
    sub = pooled[pooled["Metric"] == metric_name]
    ax.bar(sub["Prompt"], sub["mean"], yerr=sub["sd"], capsize=5,
           color=hex[:len(sub)])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)


plt.rcParams["font.family"] = "Times New Roman"
fig.suptitle("Token Usage Across Models")
fig.tight_layout()
fig.savefig("model_comparison_pooled.png", dpi=200)
print("Saved plot to model_comparison_pooled.png")



### NHM Dataset

df_nhm_results = pd.read_excel(file_path, sheet_name="NHM_data_results")
df_nhm_results.columns = df_nhm_results.columns.str.replace(" ", "_").str.lower()

metrics = {
    "Prompt Tokens":        ("prompt_tokens_mean", "sd_prompt_tokens",  "n_prompt_tokens"),
    "Completion Tokens":    ("completion_tokens_mean", "sd_completion_tokens",  "n_completion_tokens"),  
    "Total Tokens":         ("total_tokens_mean", "sd_total_tokens", "n_total_tokens")
}

rows = []
for (model, prompt), group in df_nhm_results.groupby(["model", "image_one_or_few_shot?"]):
    for metric_name, (mean_col, sd_col, n_col) in metrics.items():
        grand_mean, combined_sd = pool_mean_sd(
            group[mean_col], group[sd_col], group[n_col]
        )
        rows.append({"Model": model, "Prompt": prompt, "Metric": metric_name,
                      "mean": grand_mean, "sd": combined_sd})

pooled = pd.DataFrame(rows)
print(pooled)


metric_names = pooled["Metric"].unique()
fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
if len(metric_names) == 1:
    axes = [axes]

## Aesthetics
# Font setting
sns.set_theme(style="whitegrid", context="paper")   # theme/style
hex = sns.color_palette(["#ffa11a", "#26f3fa", "#ed1d6a"])


for ax, metric_name in zip(axes, metric_names):
    sub = pooled[pooled["Metric"] == metric_name]
    ax.bar(sub["Prompt"], sub["mean"], yerr=sub["sd"], capsize=5,
           color=hex[:len(sub)])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)


plt.rcParams["font.family"] = "Times New Roman"
fig.suptitle("Token Usage Across Models")
fig.tight_layout()
fig.savefig("model_comparison_pooled.png", dpi=200)
print("Saved plot to model_comparison_pooled.png")


## Effect of prompt specificity - specifically looking at Qwen 32B in the NHm dataset
nhm_qwen_results = df_nhm_results[df_nhm_results["model"] == "Qwen 32B"]

sd_cols = ["sd_cer", "sd_wer", "sd_lev", "sd_token"]

df_sd_long = nhm_qwen_results.melt(
    id_vars=["model", "image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"],
    value_vars=sd_cols,
    var_name="metric_sd", 
    value_name="sd"
)


df_nhm_results_long = nhm_qwen_results.melt(
    id_vars=["model", "image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], # Column to keep as-is (grouping variable)
    value_vars=["overall_cer_mean", "overall_wer_mean","overall_lev_mean", "overall_token_sort_mean"], # Columns to stack into one
    var_name="metric",  # What to call the new column holding the old column names
    value_name="score"  # What to call the new column holding the actual values
    )  

# Mapping each SD's metric sd label to existing metrics label
metric_map = {
    "sd_cer": "overall_cer_mean",
    "sd_wer": "overall_wer_mean",
    "sd_lev": "overall_lev_mean",
    "sd_token": "overall_token_sort_mean",
}
df_sd_long["metric"] = df_sd_long["metric_sd"].map(metric_map)

# Then merging the shared identifying columns to the metrics
id_vars = ["model", "image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"]

df_nhm_results_long = df_nhm_results_long.merge(
    df_sd_long[id_vars + ["metric", "sd"]],
    on=id_vars + ["metric"],
    how="left"
)

ci_lower_cols = ["cer_ci_lower", "wer_ci_lower", "lev_ci_lower", "tok_ci_lower"]
ci_upper_cols = ["cer_ci_upper", "wer_ci_upper", "lev_ci_upper", "tok_ci_upper"]

df_ci_lower_long = nhm_qwen_results.melt(
    id_vars=id_vars,
    value_vars=ci_lower_cols,
    var_name="metric_ci", 
    value_name="ci_lower"
)

df_ci_upper_long = nhm_qwen_results.melt(
    id_vars=id_vars,
    value_vars=ci_upper_cols,
    var_name="metric_ci", 
    value_name="ci_upper"
)

ci_map = {
    "cer_ci_lower": "overall_cer_mean", "cer_ci_upper": "overall_cer_mean",
    "wer_ci_lower": "overall_wer_mean", "wer_ci_upper": "overall_wer_mean",
    "lev_ci_lower": "overall_lev_mean", "lev_ci_upper": "overall_lev_mean",
    "tok_ci_lower": "overall_token_sort_mean", "tok_ci_upper": "overall_token_sort_mean",
}
df_ci_lower_long["metric"] = df_ci_lower_long["metric_ci"].map(ci_map)
df_ci_upper_long["metric"] = df_ci_upper_long["metric_ci"].map(ci_map)

# Merge CI bounds
df_nhm_results_long = df_nhm_results_long.merge(
    df_ci_lower_long[id_vars + ["metric", "ci_lower"]],
    on=id_vars + ["metric"], how="left"
).merge(
    df_ci_upper_long[id_vars + ["metric", "ci_upper"]],
    on=id_vars + ["metric"], how="left"
)

# COmpute half-width errorbar needs for plotting
df_nhm_results_long["ci"] = (df_nhm_results_long["ci_upper"] - df_nhm_results_long["ci_lower"]) / 2


# Bars = only rows where specificity actually applies (one-shot / few-shot)
bar_df = df_nhm_results_long[df_nhm_results_long["image_one_or_few_shot?"] != "no"]

# Reference points = zero-shot rows (specificity is NA here, so they sit outside the grid)
zero_df = df_nhm_results_long[df_nhm_results_long["image_one_or_few_shot?"] == "no"]

model_order = bar_df["model"].unique().tolist()  # fixes x-position order so bars and markers align

sns.set_theme(style="whitegrid", context="paper")   # theme/style
# Font setting
plt.rcParams["font.family"] = "Times New Roman"

# Creating a legend swatch to describe what the black dash means
zero_shot_handle = Line2D(
    [0], [0], marker="_", color="black", linestyle="None", markersize=15, markeredgewidth=2, label="Zero-shot"
)

w = 0.4
centres = (-2*w + w/2, -w + w/2, 0 + w/2, w + w/2)  # centres of the bars for each model

# Now we need to draw actual bars - for one subplot (one metric), need to loop over four combos (shot x specificity) and call ax.bar() once per combo, using centres as the x-position, colour keyed by shot, hatch keyed by specificity.

shot_color = {"one": "#ffa11a", "few": "#26f3fa"}  # Example color mapping for shot types
spec_hatch = {"yes": "///", "no": None}  # Example hatch mapping for specificity

combos = [("one", "no"), ("one", "yes"), ("few", "no"), ("few", "yes")]
offsets = (-0.6, -0.2, 0.2, 0.6)  # Adjusted offsets for better spacing
metrics = ["overall_cer_mean", "overall_wer_mean", "overall_lev_mean", "overall_token_sort_mean"]

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
x0 = 0 # single model for now

#############

# Zero shot overlay loop
# need to loop over (ax, metric) pairs (same as bar loop), look up that metric's zero-shot value for "Qwen 32B" from zero_df, and plot the dash.
for ax, metric in zip(axes, metrics):
    zero_shot = zero_df[
        (zero_df["model"] == "Qwen 32B") &
        (zero_df["metric"] == metric)
    ]
    y = zero_shot["score"].values[0]
    x_start = -0.1  # Start of the dash (left side)
    x_end = 0.1     # End of the dash (right side)
    ax.plot([x_start, x_end], [y, y], color="red")

#############

# Making a legend for shot colors & specificity hatches - using matplotlib.patches.Patch to do so bc I need to make fake sample patches that exist only to show up in the legend
from matplotlib.patches import Patch

# Patch essentially creates a rectangle with the specified facecolor and label, which can be used in the legend to represent different categories.
shot_handles = [
    Patch(facecolor=shot_color["one"], label = "One-shot"),
    Patch(facecolor=shot_color["few"], label = "Few-shot")
]

spec_handles = [
    Patch(facecolor="white", edgecolor="black", hatch=spec_hatch["no"], label = "Non-specific"),
    Patch(facecolor="white", edgecolor="black", hatch=spec_hatch["yes"], label = "Specific")
]

# Need a line equivalent to patch - can use matplotlib's line drawing class Line2D. Is the same type of object as ax.plot
zero_shot_handle = plt.Line2D([0], [0], color="red", label="Zero-shot")

# NB: handles are a plotting term - in matplotlib, handle means a reference to a drawable object, something you can hand a legen entry, colour or style off of

#############

plot_names = {
    "overall_cer_mean": "Character Error Rate",
    "overall_wer_mean": "Word Error Rate",
    "overall_lev_mean": "Levenshtein Distance",
    "overall_token_sort_mean": "Token Sort Ratio"
}

for ax, metric in zip(axes, metrics):
    for (shot, spec), offset in zip(combos, offsets): # Zip combines the two lists into pairs, so each iteration gives one combo and one offset
        # Filter bar_df to a single row matching: this model, shot, spec, this metric
        row = bar_df[
            (bar_df["model"] == "Qwen 32B") &
            (bar_df["image_one_or_few_shot?"] == shot) &
            (bar_df["if_using_ex,_is_ex_dataset_wholly_specific?"] == spec) &
            (bar_df["metric"] == metric)
        ]
        height = row["score"].values[0]
        ci = row["ci"].values[0]
        ax.bar(x0 + offset, height, width = w, color= shot_color[shot], hatch=spec_hatch[spec]) # Draw one bar at position x0, that tall, coloured by shot & hatched or not by spec
        ax.errorbar(x0 + offset, height, yerr=ci, fmt='none', ecolor='black', capsize=5) # Draw the error bar on top of the bar
    ax.set_title(plot_names.get(metric, metric))

fig.legend(handles=shot_handles, title="Shot type", loc="upper right", bbox_to_anchor=(1.0, 0.8))
fig.legend(handles=spec_handles, title="Specificity", loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.legend(handles=[zero_shot_handle], title="Zero-shot", loc="upper right", bbox_to_anchor=(1.0, 0.6))
plt.show()


#############
#############

# Getting mean & CI per model for records classified as 'table' or 'mixed' in BL dataset - filtered them out to report on just 'text' data since their GT is often not great being made by Transkribus

file_path = "data/MSc_results_table_mixed.xlsx"  
xls = pd.ExcelFile(file_path)
print(xls.sheet_names)


### Mixed record results
bl_mixed_results = pd.read_excel(file_path, sheet_name="Mixed")
bl_mixed_results.columns = bl_mixed_results.columns.str.replace(" ", "_").str.lower()
bl_mixed_results = bl_mixed_results.rename(columns={"sd_cer.1": "sd_wer"})

metrics = {
    "CER":        ("overall_cer_mean", "sd_cer",  "n_cer"),
    "WER":        ("overall_wer_mean", "sd_wer",  "n_wer"),  
    "Levenshtein": ("overall_lev_mean", "sd_lev", "n_lev"),
    "TokenSort":   ("overall_token_sort_mean", "sd_token", "n_token_sort"),
}

rows = []
for model, group in bl_mixed_results.groupby("model"):
    for metric_name, (mean_col, sd_col, n_col) in metrics.items():
        grand_mean, combined_sd = pool_mean_sd(
            group[mean_col], group[sd_col], group[n_col]
        )
        total_n = group[n_col].sum()

        se = combined_sd / np.sqrt(total_n)
        ci_mult = stats.norm.ppf(0.975)   # 1.96 — swap to stats.t.ppf(0.975, df=total_n-1) if n is small
        ci_half_width = ci_mult * se

        rows.append({
            "Model": model, "Metric": metric_name,
            "mean": grand_mean, "sd": combined_sd, "n": total_n,
            "ci_lower": grand_mean - ci_half_width,
            "ci_upper": grand_mean + ci_half_width,
        })

pooled = pd.DataFrame(rows)
print(pooled)

### Table record results

bl_table_results = pd.read_excel(file_path, sheet_name="Table")
bl_table_results.columns = bl_table_results.columns.str.replace(" ", "_").str.lower()
bl_table_results = bl_table_results.rename(columns={"sd_cer.1": "sd_wer"})

metrics = {
    "CER":        ("overall_cer_mean", "sd_cer",  "n_cer"),
    "WER":        ("overall_wer_mean", "sd_wer",  "n_wer"),  
    "Levenshtein": ("overall_lev_mean", "sd_lev", "n_lev"),
    "TokenSort":   ("overall_token_sort_mean", "sd_token", "n_token_sort"),
}


rows = []
for model, group in bl_table_results.groupby("model"):
    for metric_name, (mean_col, sd_col, n_col) in metrics.items():
        grand_mean, combined_sd = pool_mean_sd(
            group[mean_col], group[sd_col], group[n_col]
        )
        total_n = group[n_col].sum()

        se = combined_sd / np.sqrt(total_n)
        ci_mult = stats.norm.ppf(0.975)   # 1.96 — swap to stats.t.ppf(0.975, df=total_n-1) if n is small
        ci_half_width = ci_mult * se

        rows.append({
            "Model": model, "Metric": metric_name,
            "mean": grand_mean, "sd": combined_sd, "n": total_n,
            "ci_lower": grand_mean - ci_half_width,
            "ci_upper": grand_mean + ci_half_width,
        })

pooled = pd.DataFrame(rows)
print(pooled)
