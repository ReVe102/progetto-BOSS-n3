import os
import sys
import argparse
import cv2

# Ensure local 'src' package is importable when running from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(REPO_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from evaluation.gt_loader import init_gt_loader, get_gt_for_frame
from evaluation.mot_evaluator import MotEvaluator
from processing.detector import ObjectDetector


def list_images_sorted(frames_dir):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return [os.path.join(frames_dir, f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT metrics over a frame directory using gt.csv")
    parser.add_argument('--frames', default=os.path.join('assets', '0001'), help='Directory with ordered frames (default assets/0001)')
    parser.add_argument('--gt', default=os.path.join('assets', 'gt.csv'), help='Path to GT CSV (default assets/gt.csv)')
    parser.add_argument('--model', default='yolov8s.pt', help='YOLO model name/path (default yolov8s.pt)')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching (default 0.5)')
    parser.add_argument('--max-frames', type=int, default=None, help='Optional limit on frames processed')
    parser.add_argument('--pred-classes', nargs='*', type=int, default=[2, 5, 7],
                        help='Filter predictions to these COCO class IDs (car=2, bus=5, truck=7)')

    args = parser.parse_args()

    frames = list_images_sorted(args.frames)
    if not frames:
        raise RuntimeError(f"No image frames found in {args.frames}")

    # Init GT loader and evaluator
    init_gt_loader(args.gt)
    evaluator = MotEvaluator(iou_threshold=args.iou, id_tag=os.path.basename(args.frames.rstrip('/\\')) or 'run')

    # Init detector once
    detector = ObjectDetector(model_name=args.model)

    for idx, img_path in enumerate(frames):
        if args.max_frames is not None and idx >= args.max_frames:
            break
        img = cv2.imread(img_path)
        if img is None:
            continue

        preds = detector.detect_and_track(img)
        if args.pred_classes is not None:
            preds = [p for p in preds if int(p.get('class_id', -1)) in args.pred_classes]

        gt = get_gt_for_frame(idx)
        evaluator.update(idx, gt, preds)

    evaluator.print_summary()


if __name__ == '__main__':
    main()
