"""When the start moves back, and what counts as reaching the goal.

The emulator half needs a ROM. The rules do not, and the rules are where this
project's mistakes have lived: what gets trained on, and what counts as having
done the job.
"""

from nes_player.policy.robustify import (
    ATTEMPTS,
    LEVEL_SPAN,
    Curriculum,
    Rung,
    progress_of,
    reached_goal,
)


def dbg(x: int, level: int = 0) -> dict:
    return {"xscrollHi": x // 256, "xscrollLo": x % 256,
            "levelHi": level // 4, "levelLo": level % 4}


def rung(index: int, wins: int, attempts: int = ATTEMPTS) -> Rung:
    return Rung(index=index, attempts=attempts, wins=wins)


def test_getting_as_far_as_the_trajectory_is_a_win():
    goal = progress_of(dbg(3100))
    assert reached_goal(progress_of(dbg(3100)), goal)


def test_falling_just_short_still_counts():
    """The policy will not stop on the exact pixel the search stopped on."""
    goal = progress_of(dbg(3100))
    assert reached_goal(progress_of(dbg(3070)), goal, margin=48)
    assert not reached_goal(progress_of(dbg(3000)), goal, margin=48)


def test_finishing_a_level_outranks_any_distance_in_the_previous_one():
    """x restarts at zero in a new level, so a plain comparison would call a
    finished level a step backwards."""
    assert progress_of(dbg(20, level=2)) > progress_of(dbg(3900, level=1))


def test_levels_never_overlap_on_the_progress_axis():
    """A level longer than the stride would make the number ambiguous."""
    assert progress_of(dbg(LEVEL_SPAN - 1, level=0)) < progress_of(dbg(0, level=1))


def test_the_furthest_point_counts_not_the_last_one():
    """The level counter flips a moment before the camera resets, so the best
    frame is usually already in the past by the time it is checked."""
    goal = progress_of(dbg(3148, level=1))
    passed_through = progress_of(dbg(3148, level=1))
    now = progress_of(dbg(0, level=2))
    assert reached_goal(max(passed_through, now), goal)


def test_the_start_moves_back_when_the_rung_is_passed():
    c = Curriculum(total=1000, index=800, step=120)
    assert c.record(rung(800, wins=ATTEMPTS))
    assert c.index == 680


def test_the_start_holds_when_it_is_not():
    c = Curriculum(total=1000, index=800, step=120)
    assert not c.record(rung(800, wins=1))
    assert c.index == 800


def test_half_the_attempts_is_enough():
    assert rung(0, wins=ATTEMPTS // 2).passed
    assert not rung(0, wins=ATTEMPTS // 2 - 1).passed


def test_a_rung_is_not_judged_before_it_is_finished():
    """Two wins out of two is not a pass rate, it is two attempts."""
    assert not Rung(index=0, attempts=2, wins=2).passed


def test_repeated_failure_is_counted_so_the_loop_can_give_up():
    c = Curriculum(total=1000, index=800)
    for _ in range(3):
        c.record(rung(800, wins=0))
    assert c.stalled == 3


def test_a_win_clears_the_stall_count():
    c = Curriculum(total=1000, index=800)
    c.record(rung(800, wins=0))
    c.record(rung(800, wins=ATTEMPTS))
    assert c.stalled == 0


def test_the_start_never_goes_past_the_beginning():
    c = Curriculum(total=1000, index=50, step=120)
    c.record(rung(50, wins=ATTEMPTS))
    assert c.index == 0
    assert c.done


def test_progress_is_the_share_of_the_route_it_can_finish():
    assert Curriculum(total=1000, index=1000).progress == 0.0
    assert Curriculum(total=1000, index=250).progress == 0.75
    assert Curriculum(total=1000, index=0).progress == 1.0
