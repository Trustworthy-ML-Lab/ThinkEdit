import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

###############################################################################
# 1) Utility: load 10-run data from results_samples10.json
###############################################################################
def load_ten_run_data(json_path):
    """
    Reads a JSON file:
       {
         "sample_results": [
           {
             "accuracy": 0.753..., 
             "avg_thinking_length": 5664.558..., 
             "think_lengths": [...],  // array of length N
             "correctness":   [...],  // matching 0/1 of length N
           },
           ...
           // total 10 items
         ]
       }
    Returns list of these 10 dicts. Returns [] if file missing or invalid.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("sample_results", [])

###############################################################################
# 2) Compute short-thinking stats (5%/10%/20%) + overall from 10 runs
###############################################################################
def get_basic_stats_for_10_runs(sample_results):
    """
    For each of the 10 runs in sample_results (each has:
      - "accuracy" (overall float),
      - "think_lengths" (list of floats),
      - "correctness"  (list of 0/1)
    ), compute:
      - overall average accuracy ± std (float in [0,1])
      - overall average reasoning length ± std
      - top 5% / 10% / 20% short-len accuracy ± std, length ± std
        (we will only store the *means* for the final two “short-len” tables).

    Returns a dict with keys:
      {
        "overall_accuracy": (mean_acc, std_acc),
        "overall_length":   (mean_len, std_len),
        "short_5":  (acc_mean, acc_std, len_mean, len_std),
        "short_10": (acc_mean, acc_std, len_mean, len_std),
        "short_20": (acc_mean, acc_std, len_mean, len_std),
      }
    """
    if not sample_results:
        return {}

    run_accuracies = []
    run_lengths = []  # We'll collect all think_lengths across runs
    short_len_acc_5  = []
    short_len_acc_10 = []
    short_len_acc_20 = []
    short_len_avg_5  = []
    short_len_avg_10 = []
    short_len_avg_20 = []

    for run in sample_results:
        # Overall accuracy for the run
        run_accuracies.append(run["accuracy"])

        think_lengths = np.array(run["think_lengths"], dtype=float)
        correctness   = np.array(run["correctness"],   dtype=float)

        # Accumulate for overall length
        run_lengths.extend(think_lengths.tolist())

        # Sort by ascending length
        sorted_idx = np.argsort(think_lengths)
        sorted_lens = think_lengths[sorted_idx]
        sorted_corr = correctness[sorted_idx]
        n = len(sorted_lens)

        def short_segment_stats(pct):
            """Return (acc, avg_len) for the shortest pct% in *this run*."""
            k = int(np.floor(pct * n))
            if k < 1:
                return None, None
            seg_lens = sorted_lens[:k]
            seg_corr = sorted_corr[:k]
            seg_acc  = seg_corr.mean()   # average correctness
            seg_avg_len = seg_lens.mean()
            return seg_acc, seg_avg_len

        acc5,  len5  = short_segment_stats(0.05)
        acc10, len10 = short_segment_stats(0.10)
        acc20, len20 = short_segment_stats(0.20)

        if acc5 is not None:
            short_len_acc_5.append(acc5)
            short_len_avg_5.append(len5)
        if acc10 is not None:
            short_len_acc_10.append(acc10)
            short_len_avg_10.append(len10)
        if acc20 is not None:
            short_len_acc_20.append(acc20)
            short_len_avg_20.append(len20)

    # mean ± std helper
    def mean_std(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return (None, None)
        if len(arr) == 1:
            return (arr[0], 0.0)
        return (arr.mean(), arr.std(ddof=1))

    # Overall accuracy ± std
    overall_acc_mean, overall_acc_std = mean_std(run_accuracies)

    # Overall length ± std
    overall_len_mean, overall_len_std = mean_std(run_lengths)

    def short_stats_wrapper(acc_list, len_list):
        """
        Return a tuple:
          (acc_mean, acc_std, len_mean, len_std).
        """
        if not acc_list:
            return (None, None, None, None)
        a_mean, a_std = mean_std(acc_list)
        l_mean, l_std = mean_std(len_list)
        return (a_mean, a_std, l_mean, l_std)

    short5  = short_stats_wrapper(short_len_acc_5,  short_len_avg_5)
    short10 = short_stats_wrapper(short_len_acc_10, short_len_avg_10)
    short20 = short_stats_wrapper(short_len_acc_20, short_len_avg_20)

    return {
        "overall_accuracy": (overall_acc_mean, overall_acc_std),
        "overall_length":   (overall_len_mean, overall_len_std),
        "short_5":  short5,   # (acc_mean, acc_std, len_mean, len_std)
        "short_10": short10,
        "short_20": short20
    }

###############################################################################
# 3) Combine 10 runs for threshold-based accuracy
###############################################################################
def combine_runs_for_threshold_plot(sample_results):
    """
    Concatenate 'think_lengths' and 'correctness' across all 10 runs 
    into one big array, so we can measure "accuracy below threshold" 
    for that entire combined set.
    """
    all_lens = []
    all_corr = []
    for run in sample_results:
        all_lens.extend(run["think_lengths"])
        all_corr.extend(run["correctness"])
    return np.array(all_lens, dtype=float), np.array(all_corr, dtype=float)

def accuracy_under_threshold(think_lengths, correctness, threshold):
    """
    Return fraction correct among examples with length < threshold.
    Return None if no examples below threshold.
    """
    mask = (think_lengths < threshold)
    if not np.any(mask):
        return None
    return correctness[mask].mean()

###############################################################################
# 4) Main Script
###############################################################################
def main():
    base_dir = "ThinkEdit_model_evaluation_results"  # adjust if necessary

    # Example sets of datasets and model pairs
    datasets = [
        "gsm8k",
        "mmlu_elementary_math",
        "MATH-level1",
        "MATH-level5",
        "MATH-500"
    ]

    # For threshold plots: The three "deepseek" models in the order you want
    deepseek_models_for_plot = [
        "deepseek-qwen-1.5b",
        "deepseek-llama3-8b",
        "deepseek-qwen-14b",
        "deepseek-qwen-32b"
    ]

    # For table comparison: each deepseek model paired with its intervened version
    model_pairs = [
        ("deepseek-qwen-32b",  "ThinkEdit-deepseek-qwen-32b"),
        ("deepseek-qwen-14b",  "ThinkEdit-deepseek-qwen-14b"),
        ("deepseek-llama3-8b", "ThinkEdit-deepseek-llama3-8b"),
        ("deepseek-qwen-1.5b", "ThinkEdit-deepseek-qwen-1.5b")
    ]

    # We'll store final stats in these dictionaries, keyed by (model, dataset):
    # 1) overall_results => (mean_acc[%], std_acc[%])
    # 2) overall_lengths => (mean_len, std_len)
    # 3) short_acc_results => (acc5%, acc10%, acc20%)
    # 4) short_len_results => (len5, len10, len20)
    overall_results = {}
    overall_lengths = {}
    short_acc_results = {}
    short_len_results = {}

    for ds in datasets:
        print("======================================")
        print(f"DATASET: {ds}")
        print("======================================")

        # Threshold plot figure
        plt.figure(figsize=(7,5))

        # We'll handle each model pair (original+intervened).
        for (original_model, intervened_model) in model_pairs:
            # Load original
            path_orig = os.path.join(base_dir, ds, original_model, "instruction_", "results_samples10.json")
            sr_orig = load_ten_run_data(path_orig)

            # Load intervened
            path_intv = os.path.join(base_dir, ds, intervened_model, "instruction_", "results_samples10.json")
            sr_intv = load_ten_run_data(path_intv)

            def process_ten_run(sr, model_name):
                if not sr:
                    return
                stats = get_basic_stats_for_10_runs(sr)
                (acc_mean, acc_std) = stats.get("overall_accuracy", (None, None))
                (len_mean, len_std) = stats.get("overall_length", (None, None))

                # If we have valid data, store them
                if acc_mean is not None:
                    overall_results[(model_name, ds)] = (acc_mean*100.0, acc_std*100.0)
                if len_mean is not None:
                    overall_lengths[(model_name, ds)] = (len_mean, len_std)

                # short_5, short_10, short_20 => (acc_mean, acc_std, len_mean, len_std)
                # We'll store just the means for the final tables
                def extract_short_means(short_tuple):
                    if not short_tuple or short_tuple[0] is None:
                        return None, None
                    # short_tuple[0] = accuracy, short_tuple[2] = length
                    return short_tuple[0]*100.0, short_tuple[2]

                s5  = extract_short_means(stats.get("short_5"))
                s10 = extract_short_means(stats.get("short_10"))
                s20 = extract_short_means(stats.get("short_20"))

                # Place them in short_acc_results and short_len_results
                for triple, idx in zip([s5, s10, s20], [0,1,2]):
                    if triple[0] is not None:
                        short_acc_results.setdefault((model_name, ds), [None, None, None])
                        short_len_results.setdefault((model_name, ds), [None, None, None])
                        short_acc_results[(model_name, ds)][idx] = triple[0]
                        short_len_results[(model_name, ds)][idx]  = triple[1]

            process_ten_run(sr_orig, original_model)
            process_ten_run(sr_intv, intervened_model)

        # Plot threshold lines for the three deepseek models
        for model_name in deepseek_models_for_plot:
            json_path = os.path.join(base_dir, ds, model_name, "instruction_", "results_samples10.json")
            sr = load_ten_run_data(json_path)
            if not sr:
                continue

            all_lens, all_corr = combine_runs_for_threshold_plot(sr)
            if len(all_lens) == 0:
                continue

            max_len = all_lens.max()
            thresholds = np.linspace(0, max_len, 30)
            x_vals, y_vals = [], []
            for th in thresholds:
                acc = accuracy_under_threshold(all_lens, all_corr, th)
                if acc is not None:
                    x_vals.append(th)
                    y_vals.append(acc)

            plt.plot(x_vals, y_vals, marker='o', label=model_name)

        plt.title(f"{ds.upper()}", fontsize=20)
        plt.xlabel("Reasoning Length Threshold", fontsize=14)
        plt.ylabel("Cumulative Accuracy", fontsize=14)
        plt.ylim([0, 1.0])
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(base_dir, ds, "accuracy_vs_threshold.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved for {ds}: {fig_path}\n")
        plt.close()

    ###########################################################################
    # 5) Build and print the four final tables:
    #    Table A: Overall Accuracy (mean ± std)
    #    Table B: Overall Reasoning Length (mean ± std)
    #    Table C: Accuracy (%) of top 5%/10%/20% shortest
    #    Table D: Average length (tokens) of top 5%/10%/20% shortest
    ###########################################################################

    dataset_order = [
        "gsm8k",
        "mmlu_elementary_math",
        "MATH-level1",
        "MATH-level5",
        "MATH-500"
    ]
    dataset_labels = ["GSM8K", "MMLU Elem. Math", "MATH-Level1", "MATH-Level5", "MATH-500"]

    all_model_pairs = model_pairs

    # -----------------------
    # Table A: Overall Accuracy
    # -----------------------
    print("\nTable A: Overall accuracy (mean ± std) over 10 runs\n")
    table_rows_A = []
    for (deep_model, intv_model) in all_model_pairs:
        row_orig = [deep_model, "Original"]
        row_attn = ["", "ThinkEdit"]

        for ds in dataset_order:
            val_orig = overall_results.get((deep_model, ds))
            if not val_orig:
                row_orig.append("N/A")
            else:
                row_orig.append(f"{val_orig[0]:.2f} ± {val_orig[1]:.2f}")

            val_attn = overall_results.get((intv_model, ds))
            if not val_attn:
                row_attn.append("N/A")
            else:
                row_attn.append(f"{val_attn[0]:.2f} ± {val_attn[1]:.2f}")

        table_rows_A.append(row_orig)
        table_rows_A.append(row_attn)

    headers_A = ["Model", "", *dataset_labels]
    print(tabulate(table_rows_A, headers=headers_A, tablefmt="fancy_grid"))
    print()

    # -----------------------
    # Table B: Overall Reasoning Length
    # -----------------------
    print("\nTable B: Overall reasoning length (mean ± std) over 10 runs\n")
    table_rows_B = []
    for (deep_model, intv_model) in all_model_pairs:
        row_orig = [deep_model, "Original"]
        row_attn = ["", "ThinkEdit"]

        for ds in dataset_order:
            val_orig_len = overall_lengths.get((deep_model, ds))
            if not val_orig_len:
                row_orig.append("N/A")
            else:
                row_orig.append(f"{val_orig_len[0]:.1f} ± {val_orig_len[1]:.1f}")

            val_attn_len = overall_lengths.get((intv_model, ds))
            if not val_attn_len:
                row_attn.append("N/A")
            else:
                row_attn.append(f"{val_attn_len[0]:.1f} ± {val_attn_len[1]:.1f}")

        table_rows_B.append(row_orig)
        table_rows_B.append(row_attn)

    headers_B = ["Model", "", *dataset_labels]
    print(tabulate(table_rows_B, headers=headers_B, tablefmt="fancy_grid"))
    print()

    # -----------------------
    # Table C: Accuracy of top 5%/10%/20%
    # -----------------------
    print("\nTable C: Accuracy (%) of the top 5% / 10% / 20% shortest\n")
    table_rows_C = []
    for (deep_model, intv_model) in all_model_pairs:
        row_orig = [deep_model, "Original"]
        row_attn = ["", "ThinkEdit"]

        for ds in dataset_order:
            triple = short_acc_results.get((deep_model, ds), None)
            if not triple or any(x is None for x in triple):
                row_orig.append("N/A")
            else:
                row_orig.append(f"{triple[0]:.2f} / {triple[1]:.2f} / {triple[2]:.2f}")

            triple2 = short_acc_results.get((intv_model, ds), None)
            if not triple2 or any(x is None for x in triple2):
                row_attn.append("N/A")
            else:
                row_attn.append(f"{triple2[0]:.2f} / {triple2[1]:.2f} / {triple2[2]:.2f}")

        table_rows_C.append(row_orig)
        table_rows_C.append(row_attn)

    headers_C = ["Model", "", *dataset_labels]
    print(tabulate(table_rows_C, headers=headers_C, tablefmt="fancy_grid"))
    print()

    # -----------------------
    # Table D: Average length of top 5%/10%/20%
    # -----------------------
    print("\nTable D: Average reasoning length (tokens) for the top 5% / 10% / 20% shortest\n")
    table_rows_D = []
    for (deep_model, intv_model) in all_model_pairs:
        row_orig = [deep_model, "Original"]
        row_attn = ["", "ThinkEdit"]

        for ds in dataset_order:
            triple_len = short_len_results.get((deep_model, ds), None)
            if not triple_len or any(x is None for x in triple_len):
                row_orig.append("N/A")
            else:
                row_orig.append(f"{triple_len[0]:.1f} / {triple_len[1]:.1f} / {triple_len[2]:.1f}")

            triple_len2 = short_len_results.get((intv_model, ds), None)
            if not triple_len2 or any(x is None for x in triple_len2):
                row_attn.append("N/A")
            else:
                row_attn.append(f"{triple_len2[0]:.1f} / {triple_len2[1]:.1f} / {triple_len2[2]:.1f}")

        table_rows_D.append(row_orig)
        table_rows_D.append(row_attn)

    headers_D = ["Model", "", *dataset_labels]
    print(tabulate(table_rows_D, headers=headers_D, tablefmt="fancy_grid"))
    print()


if __name__ == "__main__":
    main()
