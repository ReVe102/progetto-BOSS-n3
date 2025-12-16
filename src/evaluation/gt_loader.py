import csv
from typing import List, Dict, Tuple, Optional

class GTLoader:
    """
    Ground-truth loader for MOT evaluation.
    Expects a CSV with header:
      frame,id,x1,y1,x2,y2
    - frame: int (frame index starting from 0)
    - id: int (stable object id across frames)
    - x1,y1,x2,y2: integers (pixel coords, xyxy)
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._by_frame = {}
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame = int(row['frame'])
                    oid = int(row['id'])
                    x1 = int(float(row['x1']))
                    y1 = int(float(row['y1']))
                    x2 = int(float(row['x2']))
                    y2 = int(float(row['y2']))
                except Exception:
                    # skip malformed rows
                    continue
                entry = {'id': oid, 'bbox': (x1, y1, x2, y2)}
                self._by_frame.setdefault(frame, []).append(entry)

    def get_gt_for_frame(self, frame_idx: int) -> List[Dict]:
        """Return list of {'id': int, 'bbox': (x1,y1,x2,y2)} for given frame."""
        return self._by_frame.get(frame_idx, [])

# Convenience function
_def_loader: Optional[GTLoader] = None

def init_gt_loader(csv_path: str) -> GTLoader:
    global _def_loader
    _def_loader = GTLoader(csv_path)
    return _def_loader

def get_gt_for_frame(frame_idx: int) -> List[Dict]:
    if _def_loader is None:
        return []
    return _def_loader.get_gt_for_frame(frame_idx)
