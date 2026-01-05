
from typing import List, Tuple, Dict
import numpy as np

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(1, (ax2-ax1)*(ay2-ay1))
    b_area = max(1, (bx2-bx1)*(by2-by1))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0

class IOUTracker:
    """
    Very lightweight IOU-based tracker:
      - Assigns track IDs by greedy IOU matching
      - Keeps tracks up to max_age frames without detection
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks: Dict[int, Tuple[Tuple[int,int,int,int], int]] = {}  # id -> (bbox, age)
        self.next_id = 1

    @property
    def empty(self) -> bool:
        return len(self.tracks) == 0

    def update(self, detections: List[List[int]]) -> List[int]:
        det_bboxes = [tuple(map(int, d[:4])) for d in detections]
        assigned = set()
        track_ids = [-1] * len(det_bboxes)

        # age all tracks
        for tid in list(self.tracks.keys()):
            bbox, age = self.tracks[tid]
            self.tracks[tid] = (bbox, age + 1)
            if self.tracks[tid][1] > self.max_age:
                del self.tracks[tid]

        # greedy match
        for i, db in enumerate(det_bboxes):
            best_iou = 0.0
            best_tid = None
            for tid, (tb, age) in self.tracks.items():
                if tid in assigned:
                    continue
                iou = _iou(tb, db)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_iou >= self.iou_threshold and best_tid is not None:
                self.tracks[best_tid] = (db, 0)  # reset age
                track_ids[i] = best_tid
                assigned.add(best_tid)
            else:
                # new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (db, 0)
                track_ids[i] = tid
                assigned.add(tid)

        return track_ids

    def predict(self) -> Dict[int, Tuple[int,int,int,int]]:
        # Return current boxes (simple static model).
        return {tid: bbox for tid, (bbox, age) in self.tracks.items() if age <= self.max_age}
