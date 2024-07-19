import pandas as pd
import numpy as np

def averaged_f1_au(pre_au, gt_au):
    f1_score = 0
    for i in range(12):
        per_pre_au = pre_au[:, i]
        per_gt_au = gt_au[:, i]
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for pre, gt in zip(per_pre_au, per_gt_au):
            if pre == 1 and gt == 1:
                tp = tp + 1
            if pre == 1 and gt == 0:
                fp = fp + 1
            if pre == 0 and gt == 1:
                fn = fn + 1
            if pre == 0 and gt == 0:
                tn = tn + 1

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        per_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_score = f1_score + per_f1_score
    avg_f1_score = f1_score / 12
    return avg_f1_score

def averaged_f1_expr(pre_expr, gt_expr):
    f1_score = 0
    for j in range(8):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for pre, gt in zip(pre_expr, gt_expr):
            if int(pre) == j and int(gt) == j:
                tp = tp + 1
            if int(pre) == j and int(gt) != j:
                fp = fp + 1
            if int(pre) != j and int(gt) == j:
                fn = fn + 1
            if int(pre) != j and int(gt) != j:
                tn = tn + 1
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        per_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_score = f1_score + per_f1_score
    avg_f1_score = f1_score / 8
    return avg_f1_score

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def main_metric(output, ann_path):
    ann_df = pd.read_csv(ann_path)
    pre_va = []
    pre_expr = []
    pre_au = []
    gt_va = []
    gt_expr = []
    gt_au = []
    pre_lines = []
    for i in output:
        bs = i.shape[0]
        for j in range(bs):
            pre_lines.append(i[j, :])

    for pl in pre_lines:
        new_ann_df = ann_df[ann_df.image.values[:] == pl[0]]
        assert len(new_ann_df) == 1
        # va, expr, au先记录下来，后面单独评估
        if float(new_ann_df['valence']) != -5 :
            pre_va.append([float(pl[j+1]) for j in range(2)])
            gt_va.append([float(new_ann_df['valence']), float(new_ann_df['arousal'])])

        if float(new_ann_df['expression']) != -1:
            pre_expr.append([float(pl[3])])
            gt_expr.append([float(new_ann_df['expression'])])

        if float(new_ann_df['au1']) != -1 :
            pre_au.append([float(pl[j+4]) for j in range(12)])
            gt_au.append([float(new_ann_df['au1']), float(new_ann_df['au2']), float(new_ann_df['au4']),
                           float(new_ann_df['au6']), float(new_ann_df['au7']), float(new_ann_df['au10']),
                           float(new_ann_df['au12']), float(new_ann_df['au15']), float(new_ann_df['au23']),
                           float(new_ann_df['au24']), float(new_ann_df['au25']), float(new_ann_df['au26'])])
    # AU
    gt_au = np.array(gt_au)
    pre_au = np.array(pre_au)
    f1_au_avg = averaged_f1_au(pre_au, gt_au)
    # EXPR
    gt_expr = np.array(gt_expr)
    pre_expr = np.array(pre_expr)
    f1_expr_avg = averaged_f1_expr(pre_expr, gt_expr)
    # VA
    gt_va = np.array(gt_va)
    pre_va = np.array(pre_va)
    va1 = CCC_score(gt_va[:, 0], pre_va[:, 0])
    va2 = CCC_score(gt_va[:, 1], pre_va[:, 1])
    avg_va = (va1 + va2) / 2

    # MTL
    results = f1_expr_avg + f1_au_avg + avg_va

    return results, f1_expr_avg, f1_au_avg , avg_va

