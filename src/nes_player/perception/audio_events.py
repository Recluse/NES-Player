"""Detecting sound events without labels (spec §10.11).

Onsets come from log-mel spectral flux: a burst of energy above a running
statistic. Music hums along evenly and the threshold rejects it. The onset's
mel column is its signature, clustered by L2 distance into stable event ids, so
that "sound #3" reliably means the same thing — a coin, say — across an episode.
"""

from dataclasses import dataclass

import numpy as np
import torch

from nes_player.policy.bc import mel_transform


@dataclass
class AudioEvent:
    cluster_id: int
    frame_index: int
    strength: float
    is_new: bool


@dataclass
class _Cluster:
    proto: np.ndarray   # (32,) signature
    heard: int = 1
    deaths: int = 0     # played shortly before a death
    rewards: int = 0    # played as the score went up

    @property
    def verdict(self) -> str:
        if self.deaths >= 2 and self.deaths / self.heard > 0.25:
            return "danger"
        if self.rewards >= 2 and self.rewards / self.heard > 0.25:
            return "reward"
        return "unknown"


class AudioEventDetector:
    MATCH_DIST = 1.5    # strict: at 4.0 the death jingle merged with the music
    FLUX_K = 2.6        # threshold is median + K * MAD

    def __init__(self, sample_rate: int):
        self._mel = mel_transform(sample_rate)
        self._tail = np.zeros(512, np.float32)   # PCM tail, so windows stay continuous
        self._prev_col: np.ndarray | None = None
        self._flux_hist: list[float] = []
        self.clusters: list[_Cluster] = []
        self._cooldown = 0

    def push(self, pcm: np.ndarray, frame_index: int) -> list[AudioEvent]:
        wav = np.concatenate([self._tail, pcm.astype(np.float32) / 32768])
        self._tail = wav[-512:]
        with torch.no_grad():
            mel = torch.log(self._mel(torch.from_numpy(wav)) + 1e-5).numpy()
        events: list[AudioEvent] = []
        for col in mel.T:
            if self._prev_col is not None:
                flux = float(np.clip(col - self._prev_col, 0, None).sum())
                self._flux_hist.append(flux)
                # A short statistics window on purpose: when the music stops the
                # threshold has to drop quickly, or the jingles go undetected.
                self._flux_hist = self._flux_hist[-120:]
                med = float(np.median(self._flux_hist))
                mad = float(np.median(np.abs(np.asarray(self._flux_hist) - med))) + 1e-6
                self._cooldown -= 1
                if flux > med + self.FLUX_K * mad and len(self._flux_hist) > 60 \
                        and self._cooldown <= 0:
                    self._cooldown = 12   # about 60 ms between onsets
                    events.append(self._classify(col, frame_index, flux))
            self._prev_col = col
        return events

    def attribute(self, cluster_ids: list[int], kind: str) -> list[int]:
        """Attribute an outcome — 'death' or 'reward' — to recent sounds.
        Returns the cluster ids that acquired a verdict for the first time."""
        newly = []
        for cid in set(cluster_ids):
            c = self.clusters[cid]
            before = c.verdict
            if kind == "death":
                c.deaths += 1
            else:
                c.rewards += 1
            if before == "unknown" and c.verdict != "unknown":
                newly.append(cid)
        return newly

    MAX_CLUSTERS = 192

    def _classify(self, col: np.ndarray, frame_index: int, strength: float) -> AudioEvent:
        best = None
        if self.clusters:
            protos = np.stack([c.proto for c in self.clusters])
            d = np.sqrt(((protos - col[None]) ** 2).mean(axis=1))
            k = int(d.argmin())
            if d[k] < self.MATCH_DIST or len(self.clusters) >= self.MAX_CLUSTERS:
                best = k
        if best is None:
            self.clusters.append(_Cluster(col.copy()))
            return AudioEvent(len(self.clusters) - 1, frame_index, strength, True)
        c = self.clusters[best]
        c.proto = 0.9 * c.proto + 0.1 * col
        c.heard += 1
        return AudioEvent(best, frame_index, strength, False)
