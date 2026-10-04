"""Demand classification on complete, zero-filled monthly panels."""

import numpy as np


def sb_class(q: np.ndarray) -> str | None:
    nonzero = int((q > 0).sum())
    if nonzero < 2:
        return None
    adi = len(q) / nonzero
    mean = q.mean()
    cv2 = (q.std(ddof=1) / mean) ** 2 if mean else float("nan")
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"
