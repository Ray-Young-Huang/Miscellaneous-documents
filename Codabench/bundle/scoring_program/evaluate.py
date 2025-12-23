import os
import json
import h5py
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.ndimage import distance_transform_edt, binary_erosion


# ================== Dice + NSD 计算函数 ==================
def calculate_metric_percase(pred, gt, spacing=(1.0, 1.0), tolerance=2.0):
    """
    计算 Dice 系数和 NSD (Normalized Surface Distance)

    Args:
        pred: 预测的二值 mask (numpy array)
        gt:   真实标注的二值 mask (numpy array)
        spacing: 像素间距 (mm), 默认 (1.0, 1.0)
        tolerance: NSD 的容差阈值 (mm), 默认 2.0 mm

    Returns:
        dice: Dice 系数 (0-1)
        nsd:  Normalized Surface Distance (0-1)，越大越好
    """
    # 确保二值
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # 空集情况
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0

    # Dice
    intersection = np.logical_and(pred, gt).sum()
    dice = 2.0 * intersection / (pred.sum() + gt.sum())

    # 表面点
    pred_surface = pred.astype(bool) ^ binary_erosion(pred).astype(bool)
    gt_surface = gt.astype(bool) ^ binary_erosion(gt).astype(bool)

    if not pred_surface.any() or not gt_surface.any():
        return float(dice), float(dice)

    # 距离变换
    gt_dist_map = distance_transform_edt(~gt_surface, sampling=spacing)
    pred_dist_map = distance_transform_edt(~pred_surface, sampling=spacing)

    # 表面点到对方表面的距离
    pred_surf_dist = gt_dist_map[pred_surface]
    gt_surf_dist = pred_dist_map[gt_surface]

    # 在容差内的点
    pred_within = (pred_surf_dist <= tolerance).sum()
    gt_within = (gt_surf_dist <= tolerance).sum()

    total_points = len(pred_surf_dist) + len(gt_surf_dist)
    nsd = (pred_within + gt_within) / total_points

    return float(dice), float(nsd)


# ============= Codabench 路径约定 =============
# /home/.../input 被挂到 /app/input
# 其中：
#   /app/input/ref : reference_data (data/val 或 data/test)
#   /app/input/res : prediction_result (选手提交 zip 解压后)
BASE_INPUT_DIR = Path(os.getenv("INPUT_DIR", "/app/input"))
REF_DIR = BASE_INPUT_DIR / "ref"
PREDS_DIR = BASE_INPUT_DIR / "res"               # 预测 .h5 直接放在 /app/input/res 根目录

GT_LABELS = REF_DIR / "labels"    # GT 存在这里

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))


def main():
    print(f"[DEBUG] REF_DIR = {REF_DIR}")
    print(f"[DEBUG] PREDS_DIR = {PREDS_DIR}")
    print(f"[DEBUG] GT_LABELS = {GT_LABELS}")

    if not GT_LABELS.exists():
        raise FileNotFoundError(f"GT labels folder not found at {GT_LABELS}")

    # 从 GT labels 推断 case id 列表
    ids = sorted(p.stem for p in GT_LABELS.glob("*.h5"))
    if not ids:
        raise RuntimeError(f"No ground-truth .h5 files found in {GT_LABELS}")

    seg_dice_list = []
    seg_nsd_list = []
    cls_gts = []
    cls_preds = []

    for sid in ids:
        gt_path = GT_LABELS / f"{sid}.h5"
        pred_path = PREDS_DIR / f"{sid}.h5"

        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing prediction file for id '{sid}': {pred_path}"
            )

        # -------- 读取 GT --------
        with h5py.File(gt_path, "r") as f:
            gt_long = f["label_long"][:]        # GT 长轴
            gt_trans = f["label_trans"][:]      # GT 短轴
            cls_gt = int(f["cls_label"][()])    # GT 分类标签

        # -------- 读取预测 --------
        with h5py.File(pred_path, "r") as f:
            pred_long = f["pred_long"][:]       # 预测长轴
            pred_trans = f["pred_trans"][:]     # 预测短轴
            cls_pred = int(f["cls_pred"][()])   # 预测分类标签

        # -------- 尺寸检查 --------
        if pred_long.shape != gt_long.shape or pred_trans.shape != gt_trans.shape:
            raise ValueError(
                f"Shape mismatch for id '{sid}': "
                f"pred_long {pred_long.shape} vs gt_long {gt_long.shape}, "
                f"pred_trans {pred_trans.shape} vs gt_trans {gt_trans.shape}"
            )

        # -------- Segmentation 指标（Dice + NSD）--------
        dice_l, nsd_l = calculate_metric_percase(pred_long, gt_long)
        dice_t, nsd_t = calculate_metric_percase(pred_trans, gt_trans)

        seg_dice_list.append((dice_l + dice_t) / 2.0)
        seg_nsd_list.append((nsd_l + nsd_t) / 2.0)

        # -------- Classification 指标 --------
        cls_gts.append(cls_gt)
        cls_preds.append(cls_pred)

    # ====== 聚合 ======
    seg_dice = float(np.mean(seg_dice_list)) if seg_dice_list else 0.0
    seg_nsd = float(np.mean(seg_nsd_list)) if seg_nsd_list else 0.0
    seg_score = 0.5 * seg_dice + 0.5 * seg_nsd

    cls_f1 = float(f1_score(cls_gts, cls_preds, zero_division=0)) if cls_gts else 0.0

    overall_score = 0.5 * seg_score + 0.5 * cls_f1

    # ====== 写结果 ======
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "scores.txt", "w") as f:
        f.write(f"overall_score: {overall_score}\n")
        f.write(f"seg_score: {seg_score}\n")
        f.write(f"seg_dice: {seg_dice}\n")
        f.write(f"seg_nsd: {seg_nsd}\n")
        f.write(f"cls_f1: {cls_f1}\n")

    with open(OUTPUT_DIR / "detailed_results.json", "w") as f:
        json.dump(
            {
                "overall_score": overall_score,
                "seg_score": seg_score,
                "seg_dice": seg_dice,
                "seg_nsd": seg_nsd,
                "cls_f1": cls_f1,
                "num_cases": len(ids),
            },
            f,
            indent=2,
        )

    print(
        f"[INFO] overall_score={overall_score:.4f}, "
        f"seg_score={seg_score:.4f}, "
        f"seg_dice={seg_dice:.4f}, "
        f"seg_nsd={seg_nsd:.4f}, "
        f"cls_f1={cls_f1:.4f}"
    )


if __name__ == "__main__":
    main()
