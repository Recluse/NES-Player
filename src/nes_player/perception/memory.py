"""Object memory: cluster sprites by appearance, then learn what they do.

A cluster is an averaged 16×16 grey patch. For each one we accumulate how often
it was seen, how often the controlled object touched it, and what followed —
points gained, or a life lost. That turns into a verdict: rewarding, dangerous
or unknown, which is what colours the boxes on the dashboard.

The memory outlives episodes. Over a long stream the agent accumulates
knowledge about the enemies it keeps meeting instead of relearning them after
every death.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nes_player.perception.motion import pick_hero

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
# it changes, but a death has to travel through the dying animation and the
# level restart first: measured on Super Mario Bros., the hero touches a Goomba
# at frame 291 and the lives counter moves at 504 — 213 frames later. The honest
# signal, the screen going black, is later still: 4 frames behind the counter on
# Mario and 129 on Double Dragon. So the window has to cover roughly 350 frames
# of lag, and 400 leaves a margin.
#
# With one window for both, every death arrived long after the contact that
# caused it had expired, so nothing on that game was ever labelled dangerous:
# 60 contacts, zero danger labels.
DEATH_WINDOW = 400
# What it takes to call something dangerous. One death after one contact used
# to be enough, and on a real archive that labels almost everything: the
# tallies from 200 explored segments include a cluster touched 307 times with
# 5 deaths (0.02) sitting beside one touched 29 times with 8 (0.28). The first
# is something you brush past constantly, the second is an enemy. With the old
# rule both were "danger" and the flag was on in 98% of frames, which tells a
# policy nothing. Two deaths minimum kills the single coincidences; the rate
# kills the ubiquitous.
DANGER_DEATHS = 2
DANGER_RATE = 0.15



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
        if self.deaths >= DANGER_DEATHS and self.deaths >= DANGER_RATE * self.contacts:
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

    def risk_of(self, slot_id: int) -> float:
        """How often touching this thing has ended in death, in [0, 1].

        The "danger" verdict turns out to be unreachable for the objects that
        matter. Built over 40k frames and 32 deaths, the three clusters with
        the most deaths against them all came back "reward": a Goomba is worth
        100 points when stomped and kills only when it is not, so it is touched
        far more often than it kills — 5 deaths in 133 contacts — and both
        `DANGER_RATE` and the reward test are satisfied by the same object.
        A planner does not need the label anyway. It needs a number to weigh a
        collision by, and the tally already holds one.
        """
        cid = self._slot_cluster.get(slot_id)
        if cid is None:
            return 0.0
        c = self.clusters[cid]
        return c.deaths / max(c.contacts, 1)

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
        controlled = pick_hero(slots)
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

    def save(self, path) -> None:
        """Keep what was learned about objects between sessions.

        The docstring at the top of this file has always said the memory
        outlives episodes, and inside one process it does — but nothing ever
        wrote it down, so every run started over knowing nothing and relearned
        the same Goombas from the same deaths. What is stored is the
        prototypes and their tallies; the pending contacts and slot mapping are
        deliberately dropped, being about a run rather than about objects.
        """
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            protos=(self._protos if self._protos is not None
                    else np.zeros((0, PATCH, PATCH), np.float32)),
            seen=np.array([c.seen for c in self.clusters], np.int64),
            contacts=np.array([c.contacts for c in self.clusters], np.int64),
            score_gain=np.array([c.score_gain for c in self.clusters], np.float32),
            deaths=np.array([c.deaths for c in self.clusters], np.int64),
        )

    @classmethod
    def load(cls, path) -> ObjectMemory:
        m = cls()
        d = np.load(path)
        m._protos = d["protos"].astype(np.float32)
        m.clusters = [
            ObjectCluster(i, m._protos[i], int(d["seen"][i]),
                          int(d["contacts"][i]), float(d["score_gain"][i]),
                          int(d["deaths"][i]))
            for i in range(len(m._protos))
        ]
        return m

    def verdict_of(self, patch_source, bbox) -> str:
        """What this thing on screen is, according to what has been learned.

        Matching only — nothing is added and no tally moves. A policy asking
        "is that dangerous" must not teach the memory that it saw something.
        """
        if not self.clusters:
            return "unknown"
        patch = self._patch(patch_source, bbox)
        d = np.sqrt(((self._protos - patch[None]) ** 2).mean(axis=(1, 2)))
        k = int(d.argmin())
        return self.clusters[k].verdict if d[k] < MATCH_DIST else "unknown"

    def summary(self, top: int = 6) -> list[str]:
        rows = sorted(self.clusters, key=lambda c: -(c.contacts + c.seen / 1000))[:top]
        return [
            f"obj{c.cluster_id}: seen={c.seen} touch={c.contacts} "
            f"dScore={c.score_gain:.0f} deaths={c.deaths} [{c.verdict}]"
            for c in rows
        ]
