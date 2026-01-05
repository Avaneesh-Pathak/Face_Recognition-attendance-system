
import numpy as np
import cv2
from typing import Tuple, List

class LivenessGuard:
    """
    Lightweight heuristic liveness:
      - Sharpness (variance of Laplacian) > min_sharp
      - Color variance in HSV S-channel > min_sat_var (printed photos often low variance)
      - Temporal area jitter over frames (keeps short memory)
    All thresholds are conservative defaults and can be tuned in settings/UI.
    """
    def __init__(self, min_sharp: float = 60.0, min_sat_var: float = 150.0, min_area_jitter: float = 0.01):
        self.min_sharp = float(min_sharp)
        self.min_sat_var = float(min_sat_var)
        self.min_area_jitter = float(min_area_jitter)
        self._last_areas: List[float] = []  # keep last few areas

    def _sharpness(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _sat_variance(self, bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return float(np.var(hsv[:,:,1]))

    def _update_area(self, bbox: Tuple[int,int,int,int]) -> float:
        x1,y1,x2,y2 = bbox
        area = max(1.0, float((x2-x1)*(y2-y1)))
        self._last_areas.append(area)
        if len(self._last_areas) > 10:
            self._last_areas.pop(0)
        return area

    def is_live(self, frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> bool:
        x1,y1,x2,y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sharp = self._sharpness(gray)
        if sharp < self.min_sharp:
            return False

        svar = self._sat_variance(crop)
        if svar < self.min_sat_var:
            return False

        area = self._update_area((x1,y1,x2,y2))
        if len(self._last_areas) >= 3:
            jitter = np.std(self._last_areas) / (np.mean(self._last_areas) + 1e-6)
            if jitter < self.min_area_jitter:
                return False

        return True
