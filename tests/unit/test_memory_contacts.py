"""A contact is only to blame for what happened soon after it, in this episode.

Both halves used to be wrong: a death was credited to every pending contact
regardless of age, and pending contacts survived a reset with episode-local
frame numbers, so the first death of a new episode was blamed on an object
from the old one.
"""

import numpy as np

from nes_player.perception.memory import DEATH_WINDOW, EFFECT_WINDOW, ObjectMemory
from nes_player.perception.motion import Slot


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (224, 240, 3), dtype=np.uint8)


def _slots(touching: bool) -> list[Slot]:
    hero = Slot(0, (100, 100, 16, 16), 108.0, 108.0)
    hero.ctrl_score = 20.0        # ctrl_prob ~ 1
    other = Slot(1, (110, 100, 16, 16), 118.0 if touching else 220.0, 108.0)
    return [hero, other]


def _contact(mem: ObjectMemory, frame: np.ndarray, at: int) -> int:
    """Make the hero touch the other object; return that object's cluster id."""
    mem.update(frame, _slots(True), at, score=0, died=False)
    return mem._slot_cluster[1]


def test_a_death_long_after_the_contact_is_not_blamed_on_it():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    before = mem.clusters[cid].deaths
    mem.update(f, _slots(False), DEATH_WINDOW + 10, score=0, died=True)
    assert mem.clusters[cid].deaths == before


def test_a_death_is_blamed_across_the_counter_lag():
    """The lives counter drops long after the hit that caused it.

    Measured on Super Mario Bros.: contact at frame 291, counter at 504. With
    the score window doing double duty the contact had already expired, so the
    game produced 60 contacts and not one danger label.
    """
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    before = mem.clusters[cid].deaths
    mem.update(f, _slots(False), 213, score=0, died=True)
    assert mem.clusters[cid].deaths == before + 1


def test_points_still_use_the_short_window():
    """A contact kept alive for the death window must not keep collecting
    points for four seconds afterwards."""
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    mem.update(f, _slots(False), EFFECT_WINDOW + 10, score=500, died=False)
    assert mem.clusters[cid].score_gain == 0


def test_one_contact_is_blamed_for_one_death_only():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    before = mem.clusters[cid].deaths
    for at in (5, 10, 15):
        mem.update(f, _slots(False), at, score=0, died=True)
    assert mem.clusters[cid].deaths == before + 1


def test_only_the_last_thing_touched_is_blamed():
    """A window wide enough for the counter's lag holds several contacts at
    once. Crediting all of them is how one death made half the screen lethal."""
    mem, f = ObjectMemory(), _frame()
    first = _contact(mem, f, 0)
    second_slots = _slots(True)
    second_slots[1].slot_id = 2
    mem.update(f, second_slots, 100, score=0, died=False)
    last = mem._slot_cluster[2]
    mem.update(f, _slots(False), 150, score=0, died=True)
    assert mem.clusters[last].deaths == 1
    if last != first:
        assert mem.clusters[first].deaths == 0


def test_pending_contacts_do_not_cross_an_episode_boundary():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 3599)
    before = mem.clusters[cid].deaths
    mem.begin_episode()
    mem.update(f, _slots(False), 1, score=0, died=True)
    assert mem.clusters[cid].deaths == before


def test_begin_episode_keeps_what_was_learned():
    mem, f = ObjectMemory(), _frame()
    _contact(mem, f, 0)
    n = len(mem.clusters)
    mem.begin_episode()
    assert len(mem.clusters) == n


def test_what_was_learned_survives_a_restart(tmp_path):
    """The file docstring has always claimed the memory outlives episodes, and
    inside one process it does — but nothing wrote it down, so every session
    relearned the same Goombas from the same deaths."""
    import numpy as np

    from nes_player.perception.memory import ObjectCluster, ObjectMemory

    m = ObjectMemory()
    m.clusters = [ObjectCluster(0, np.full((16, 16), 7.0, np.float32),
                                seen=90, contacts=12, score_gain=0.0, deaths=4)]
    m._protos = np.stack([c.proto for c in m.clusters])
    m.save(tmp_path / "mem.npz")

    back = ObjectMemory.load(tmp_path / "mem.npz")
    c = back.clusters[0]
    assert (c.seen, c.contacts, c.deaths) == (90, 12, 4)
    assert c.verdict == "danger"


def test_an_empty_memory_round_trips(tmp_path):
    from nes_player.perception.memory import ObjectMemory

    ObjectMemory().save(tmp_path / "mem.npz")
    assert ObjectMemory.load(tmp_path / "mem.npz").clusters == []


def test_asking_about_an_object_does_not_teach_the_memory():
    """A policy checking "is that dangerous" must not count as a sighting."""
    import numpy as np

    from nes_player.perception.memory import ObjectCluster, ObjectMemory

    m = ObjectMemory()
    m.clusters = [ObjectCluster(0, np.zeros((16, 16), np.float32),
                                seen=5, contacts=4, deaths=3)]
    m._protos = np.stack([c.proto for c in m.clusters])
    frame = np.zeros((32, 32, 3), np.uint8)
    assert m.verdict_of(frame, (0, 0, 8, 8)) == "danger"
    assert m.clusters[0].seen == 5 and len(m.clusters) == 1


def test_an_unseen_object_is_unknown_rather_than_the_nearest_thing():
    import numpy as np

    from nes_player.perception.memory import ObjectCluster, ObjectMemory

    m = ObjectMemory()
    m.clusters = [ObjectCluster(0, np.zeros((16, 16), np.float32),
                                seen=5, contacts=4, deaths=3)]
    m._protos = np.stack([c.proto for c in m.clusters])
    bright = np.full((32, 32, 3), 255, np.uint8)
    assert m.verdict_of(bright, (0, 0, 8, 8)) == "unknown"


def test_nothing_learned_yet_means_unknown():
    import numpy as np

    from nes_player.perception.memory import ObjectMemory

    assert ObjectMemory().verdict_of(np.zeros((8, 8, 3), np.uint8),
                                     (0, 0, 4, 4)) == "unknown"


def _cluster(contacts: int, deaths: int):
    import numpy as np

    from nes_player.perception.memory import ObjectCluster

    return ObjectCluster(0, np.zeros((16, 16), np.float32),
                         seen=contacts * 10, contacts=contacts, deaths=deaths)


def test_something_touched_constantly_and_rarely_fatal_is_not_an_enemy():
    """Measured on a real archive: 307 contacts, 5 deaths. Calling that danger
    put the flag on in 98% of frames, which tells a policy nothing."""
    assert _cluster(307, 5).verdict == "unknown"


def test_something_that_kills_a_third_of_the_time_is():
    assert _cluster(29, 8).verdict == "danger"


def test_one_death_after_one_contact_is_a_coincidence():
    assert _cluster(1, 1).verdict == "unknown"


def test_two_deaths_out_of_two_contacts_is_not():
    assert _cluster(2, 2).verdict == "danger"


def test_paying_out_still_reads_as_rewarding():
    c = _cluster(3, 0)
    c.score_gain = 80
    assert c.verdict == "reward"
