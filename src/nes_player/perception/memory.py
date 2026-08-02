"""Object memory: cluster sprites by appearance, then learn what they do.

A cluster is an averaged 16×16 grey patch. For each one we accumulate how often
it was seen, how often the controlled object touched it, and what followed —
points gained, or a life lost. That turns into a verdict: rewarding, dangerous
or unknown, which is what colours the boxes on the dashboard.

The memory outlives episodes. Over a long stream the agent accumulates
knowledge about the enemies it keeps meeting instead of relearning them after
every death.
"""

from dataclasses import dataclass

import cv2
import numpy as np

PATCH = 16
MATCH_DIST = 28.0     # L2 over the normalised patch
CONTACT_PX = 22.0
EFFECT_WINDOW = 45    # frames after a contact in which points or a death count


@dataclass
class ObjectCluster:
    cluster_id: int
    proto: np.ndarray     # (PATCH, PATCH) float32, an EMA prototype
    seen: int = 0
    contacts: int = 0
    score_gain: float = 0.0
    deaths: int = 0

    @property
    def verdict(self) -> str:
        if self.contacts >= 1 and self.deaths > 0:
            return "danger"
        if self.contacts >= 1 and self.score_gain >= 50:
            return "reward"
        return "unknown"


@dataclass
class _PendingContact:
    cluster_id: int
    frame: int
    score_at: int


MAX_CLUSTERS = 256    # ponytail: past this we only match, never add. On a long
                      # stream a Python loop over thousands of clusters dropped
                      # the frame rate to two or three.


class ObjectMemory:
    def __init__(self) -> None:
        self.clusters: list[ObjectCluster] = []
        self._pending: list[_PendingContact] = []
        self._slot_cluster: dict[int, int] = {}
        self._protos: np.ndarray | None = None   # (K, PATCH, PATCH), for vectorised L2

    def begin_episode(self) -> None:
        """Forget what was still in flight; keep what was learned.

        Clusters are meant to outlive an episode — that is the whole point of a
        memory. Contacts awaiting an outcome are not: their frame numbers are
        episode-local, so after a reset `frame_index - p.frame` goes negative
        and stays inside the window forever, and the first death of the new
        episode gets blamed on an object from the old one.
        """
        self._pending.clear()
        self._slot_cluster.clear()

    def _patch(self, frame_rgb: np.ndarray, bbox) -> np.ndarray:
        x, y, w, h = bbox
        crop = frame_rgb[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            return np.zeros((PATCH, PATCH), np.float32)
        g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (PATCH, PATCH)).astype(np.float32)

    def _assign(self, patch: np.ndarray) -> ObjectCluster:
        best = None
        if self.clusters:
            # All prototypes at once: a Python loop here is O(K) per slot per
            # frame and gets slower the longer the session runs.
            d = np.sqrt(((self._protos - patch[None]) ** 2).mean(axis=(1, 2)))
            k = int(d.argmin())
            if d[k] < MATCH_DIST or len(self.clusters) >= MAX_CLUSTERS:
                best = self.clusters[k]
        if best is None:
            best = ObjectCluster(len(self.clusters), patch.copy())
            self.clusters.append(best)
            grown = patch[None].copy()
            self._protos = grown if self._protos is None else np.concatenate(
                [self._protos, grown])
        else:
            best.proto = 0.95 * best.proto + 0.05 * patch
            self._protos[best.cluster_id] = best.proto
        best.seen += 1
        return best

    def update(self, frame_rgb, slots, frame_index: int, score: int, died: bool) -> dict[int, str]:
        """Returns slot_id -> verdict, for rendering."""
        controlled = max(slots, key=lambda s: s.ctrl_prob, default=None)
        verdicts: dict[int, str] = {}
        for s in slots:
            if s.missed > 0:
                continue
            cluster = self._assign(self._patch(frame_rgb, s.bbox))
            self._slot_cluster[s.slot_id] = cluster.cluster_id
            if (controlled is not None and s is not controlled
                    and controlled.ctrl_prob > 0.7
                    and np.hypot(s.cx - controlled.cx, s.cy - controlled.cy) < CONTACT_PX):
                cluster.contacts += 1
                self._pending.append(_PendingContact(cluster.cluster_id, frame_index, score))
        # Resolve the consequences of earlier contacts. The window is checked
        # first, for both outcomes: it used to gate only the score, so a death
        # was credited to every contact still on the list no matter how old,
        # and — because expiry happened in the branch a death skipped — those
        # contacts were never dropped at all. An object touched once could
        # collect the blame for every death for the rest of the run.
        still = []
        for p in self._pending:
            if frame_index - p.frame >= EFFECT_WINDOW:
                continue           # too long ago to be a consequence of this
            c = self.clusters[p.cluster_id]
            if died:
                c.deaths += 1
                continue           # resolved: the contact got its answer
            if score > p.score_at:
                c.score_gain += score - p.score_at
                p.score_at = score
            still.append(p)
        self._pending = still
        for s in slots:
            cid = self._slot_cluster.get(s.slot_id)
            verdicts[s.slot_id] = self.clusters[cid].verdict if cid is not None else "unknown"
        if len(self._slot_cluster) > 4000:   # slot ids only ever increase
            alive = {s.slot_id for s in slots}
            self._slot_cluster = {k: v for k, v in self._slot_cluster.items() if k in alive}
        return verdicts

    def summary(self, top: int = 6) -> list[str]:
        rows = sorted(self.clusters, key=lambda c: -(c.contacts + c.seen / 1000))[:top]
        return [
            f"obj{c.cluster_id}: seen={c.seen} touch={c.contacts} "
            f"dScore={c.score_gain:.0f} deaths={c.deaths} [{c.verdict}]"
            for c in rows
        ]
