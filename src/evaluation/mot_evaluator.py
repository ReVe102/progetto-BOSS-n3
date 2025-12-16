import os
import cv2
import numpy as np
import motmetrics as mm
from typing import List, Dict, Tuple, Optional, Any

try:
    # Local import when run inside repo with sys.path including 'src'
    from .gt_loader import init_gt_loader, get_gt_for_frame  # type: ignore
except Exception:
    # Fallback absolute import when used via sys.path hack
    from evaluation.gt_loader import init_gt_loader, get_gt_for_frame  # type: ignore

# Simple IoU calculator
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

class MotEvaluator:
    """
    Minimal, modular MOT metrics evaluator using motmetrics.
    - Call update(frame_id, gt, pred) per frame
    - Call compute() at the end to get summary

    GT/pred format:
    - list of dicts with keys: 'id' (int), 'bbox' (x1,y1,x2,y2)
    """
    def __init__(self, iou_threshold: float = 0.5, id_tag: str = "default"):
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.iou_threshold = iou_threshold
        self.id_tag = id_tag
        self._frame_counter = 0

    def _build_distance_matrix(self, gtb: List[np.ndarray], prb: List[np.ndarray]) -> np.ndarray:
        if len(gtb) == 0 or len(prb) == 0:
            return np.array([])
        D = np.zeros((len(gtb), len(prb)), dtype=float)
        for i, g in enumerate(gtb):
            for j, p in enumerate(prb):
                iou = iou_xyxy(g, p)
                D[i, j] = 1 - iou  # distance = 1 - IoU
        return D

    def update(self, frame_id: int, gt: List[Dict], pred: List[Dict]) -> None:
        self._frame_counter += 1
        gt_ids = [int(g['id']) for g in gt]
        gt_bboxes = [np.array(g['bbox'], dtype=float) for g in gt]
        pred_ids = [int(p['id']) for p in pred]
        pred_bboxes = [np.array(p['bbox'], dtype=float) for p in pred]

        if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
            # nothing to update, but keep accumulator consistent
            self.acc.update([], [], [])
            return

        distances = self._build_distance_matrix(gt_bboxes, pred_bboxes)
        # Apply IoU threshold by masking distances > (1 - thr)
        if distances.size > 0:
            thr_dist = 1 - self.iou_threshold
            # Mask non-matches with np.nan, which motmetrics treats as misses
            distances = np.where(distances <= thr_dist, distances, np.nan)

        self.acc.update(gt_ids, pred_ids, distances)

    def compute(self) -> Tuple[Any, Any, Any]:
        # Create metrics host with robust fallback across motmetrics versions
        try:
            mh = mm.metrics.create()
        except Exception:
            mh = mm.metrics.MetricsHost()
        summary = mh.compute(self.acc, metrics=[
            'num_frames',
            'mota', 'motp', 'idf1',
            'num_switches',
            'mostly_tracked', 'mostly_lost',
            'num_false_positives', 'num_misses'
        ], name=self.id_tag)
        return mh, self.acc, summary

    def print_summary(self) -> None:
        mh, acc, summary = self.compute()
        # Render summary with fallback to DataFrame string if renderer unavailable
        try:
            rendered = mm.io.render_summary(
                summary,
                formatters={
                    'mota': '{:.3f}'.format,
                    'motp': '{:.3f}'.format,
                    'idf1': '{:.3f}'.format
                },
                namemap={self.id_tag: 'Evaluation'}
            )
            print(rendered)
        except Exception:
            try:
                # summary is a pandas DataFrame
                print(summary.to_string())
            except Exception:
                print(summary)

    # Convenience runner: evaluate over a frames directory using GT CSV and a detector
    def evaluate_frames_dir(self,
                            frames_dir: str,
                            gt_csv: str,
                            detector,
                            pred_classes: Optional[List[int]] = None,
                            max_frames: Optional[int] = None) -> None:
        init_gt_loader(gt_csv)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = [f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in exts]
        files.sort()
        img_paths = [os.path.join(frames_dir, f) for f in files]

        for idx, img_path in enumerate(img_paths):
            if max_frames is not None and idx >= max_frames:
                break
            img = cv2.imread(img_path)
            if img is None:
                continue

            preds = detector.detect_and_track(img)
            if pred_classes is not None:
                preds = [p for p in preds if int(p.get('class_id', -1)) in pred_classes]

            gt = get_gt_for_frame(idx)
            self.update(idx, gt, preds)

        self.print_summary()
