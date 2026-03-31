import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets_base import FC_KIND, SC_KIND, SFC_KIND, FC_CONN_MASK, normalize_modal_type
from batch_runner import run_modality_batch


DATASET_CONFIGS = {
    # "HCD": {
    #     "atlas_names": ["aal116", "bna246", "schaefer200_s1"],
    #     "modalities": {
    #         "SC": list(SC_KIND.keys()),
    #     },
    # },
    "S1200": {
        "atlas_names": ["bna246", "schaefer200_s1"],
        "modalities": {
            "SC": list(SC_KIND.keys()),
            "FC": list(FC_KIND.keys()),
        },
    },
    "ABCD": {
        "atlas_names": ["bna246", "schaefer200_s1"],
        "modalities": {
            "SC": list(SC_KIND.keys()),
            "FC": list(FC_KIND.keys()),
        },
    },
}


def _expand_fc_conn_kinds(conn_kinds):
    """将 FC conn_kinds 展开为原始 + 正连接 + 负连接三种类型。

    datasets_base.FC_CONN_MASK 的键是带后缀的（如 pcc_rest_pos / pcc_rest_neg），
    基类名 pcc_rest 不在其中，因此需用「基类 + _pos/_neg 是否均在 FC_CONN_MASK」判断。
    """
    result = []
    for k in conn_kinds:
        result.append(k)
        pos_key = f"{k}_pos"
        neg_key = f"{k}_neg"
        if pos_key in FC_CONN_MASK and neg_key in FC_CONN_MASK:
            result.append(pos_key)
            result.append(neg_key)
    return result


def build_tasks():
    tasks = []
    for dataset_name, cfg in DATASET_CONFIGS.items():
        atlas_names = cfg["atlas_names"]
        for modality_name, conn_kinds in cfg["modalities"].items():
            if not conn_kinds:
                continue
            if normalize_modal_type(modality_name) == "FC":
                conn_kinds = _expand_fc_conn_kinds(conn_kinds)
            for atlas_name in atlas_names:
                tasks.append(
                    {
                        "dataset_name": dataset_name,
                        "modality_name": modality_name,
                        "atlas_names": [atlas_name],
                        "conn_kinds": conn_kinds,
                        "label_names": None,
                        "random_state": 42,
                    }
                )
    return tasks


def run_single_task(task):
    dataset_name = task["dataset_name"]
    modality_name = task["modality_name"]
    atlas_name = task["atlas_names"][0]
    conn_kinds = task["conn_kinds"]
    print(
        f"[dispatch] dataset={dataset_name} modality={modality_name} "
        f"atlas={atlas_name} types={conn_kinds}"
    )
    pred_df, summary_df = run_modality_batch(**task)
    return {
        "dataset_name": dataset_name,
        "modality_name": modality_name,
        "atlas_name": atlas_name,
        "n_pred_rows": int(len(pred_df)),
        "n_summary_rows": int(len(summary_df)),
    }


def main(max_workers=None):
    tasks = build_tasks()
    if not tasks:
        print("No tasks configured.")
        return []

    max_workers = max_workers or min(len(tasks), os.cpu_count() or 1)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_task, task): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[completed] dataset={result['dataset_name']} "
                    f"modality={result['modality_name']} atlas={result['atlas_name']} "
                    f"pred_rows={result['n_pred_rows']} summary_rows={result['n_summary_rows']}"
                )
            except Exception as exc:
                print(
                    f"[failed] dataset={task['dataset_name']} modality={task['modality_name']} "
                    f"atlas={task['atlas_names'][0]} error={exc}"
                )
    return results


if __name__ == "__main__":
    main()
