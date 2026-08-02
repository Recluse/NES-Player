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
# Distance below which two patches are the same object. Measured rather than
# guessed: over pairs of sightings that are certainly the same thing — one
# track, at most eight frames apart — against pairs that are certainly not,
# a threshold of 55 separates them with 0.85 balanced accuracy. It used to be
# 28, which is tight enough to call the same sprite a new object whenever it
# animates, and that is where 251 clusters in one run of Super Mario Bros.
# came from.
MATCH_DIST = 55.0
CONTACT_PX = 22.0
EFFECT_WINDOW = 45    # frames after a contact in which points count
# A death is reported far later than it happens. The score updates on the frame
# it changes, but the lives counter only drops once the dying animation has run
# and the level has restarted: measured on Super Mario Bros., the hero touches
# a Goomba at frame 291 and the counter moves at frame 504 — 213 frames later.
# With one window for both, every death arrived long after the contact that
# caused it had expired, so nothing on that game was ever labelled dangerous:
# 60 contacts, zero danger labels. The project already knew about this lag from
# the other side — `DEATH_SOUND_LOOKBACK = 140` in the play loop exists because
# the death jingle precedes the counter.
DEATH_WINDOW = 260



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

    def lethal_only(self, slot_id: int) -> bool:
        """Has this object killed us and never once paid?

        The difference between a genre where you fight and one where you jump,
        learned rather than configured. A Double Dragon thug kills us and also
        gives points when hit, so it is worth engaging. A Goomba kills us and
        walking into it has never produced a single point, so it is worth
        avoiding. Nothing here knows which game it is.
        """
        cid = self._slot_cluster.get(slot_id)
        if cid is None:
            return False
        c = self.clusters[cid]
        return c.deaths > 0 and c.score_gain <= 0

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
        """What an object looks like: a 16x16 grey crop, background included.

        Two richer descriptions were built and both measured worse. Masking the
        crop with the tracker's motion mask made no difference at all — a frame
        difference marks where the object was as well as where it is, so it
        outlines a smear. Separating the sprite from the scenery by colour, then
        describing it by silhouette and palette, scored 0.67 against this crop's
        0.85: with the background gone, everything on one screen has the same
        few colours and a coarse silhouette, so it accepts far too much.

        The measurement that mattered was of the threshold, not the descriptor.
        """
        x, y, w, h = bbox
        crop = frame_rgb[max(0, y): y + h, max(0, x): x + w]
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
        if died:
            # Blame the LAST thing touched, not everything still pending. With
            # a window wide enough to cover the counter's lag, several objects
            # are in flight at once, and crediting all of them is how a single
            # death used to make half the screen look lethal.
            recent = [p for p in self._pending
                      if 0 <= frame_index - p.frame < DEATH_WINDOW]
            if recent:
                culprit = max(recent, key=lambda p: p.frame)
                self.clusters[culprit.cluster_id].deaths += 1
            self._pending = []     # the episode's causal chain is answered
        else:
            still = []
            for p in self._pending:
                age = frame_index - p.frame
                if not 0 <= age < DEATH_WINDOW:
                    continue       # too long ago to be a consequence of anything
                # Points are credited on the short window — the score moves the
                # instant it moves — but the contact is KEPT for the long one,
                # because the death that may follow is reported much later. An
                # earlier version dropped it at 45 frames and so had nothing
                # left to blame when the death finally arrived.
                if age < EFFECT_WINDOW and score > p.score_at:
                    self.clusters[p.cluster_id].score_gain += score - p.score_at
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
