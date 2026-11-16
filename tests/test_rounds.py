"""Regression tests for round sequences around match restarts."""

import polars as pl
import pytest

from awpy.parsers.rounds import create_round_df


def _events(
    starts: list[int],
    freezes: list[int],
    ends: list[int],
    officials: list[int],
    winners: list[str | None],
    reasons: list[str | None],
) -> dict[str, pl.DataFrame]:
    return {
        "round_start": pl.DataFrame({"event": ["start"] * len(starts), "tick": starts}),
        "round_freeze_end": pl.DataFrame({"event": ["freeze_end"] * len(freezes), "tick": freezes}),
        "round_end": pl.DataFrame(
            {"event": ["end"] * len(ends), "tick": ends, "winner": winners, "reason": reasons},
            schema_overrides={"winner": pl.String, "reason": pl.String},
        ),
        "round_officially_ended": pl.DataFrame({"event": ["official_end"] * len(officials), "tick": officials}),
    }


@pytest.mark.parametrize("restart", [95, 86543])
def test_restart_resultless_end_does_not_discard_pistol_round(restart: int):
    """A result-less end emitted with a restart must not interrupt the new round."""
    events = _events(
        [0, restart, restart + 100],
        [restart + 10, restart + 110],
        [restart, restart + 80, restart + 180],
        [restart, restart + 100, restart + 200],
        [None, "T", "CT"],
        [None, "bomb_exploded", "t_killed"],
    )
    rounds = create_round_df(events)

    assert rounds["round_num"].to_list() == [1, 2]
    assert rounds["start"].to_list() == [restart, restart + 100]
    assert rounds["winner"].to_list() == ["t", "ct"]
    assert rounds["reason"].to_list() == ["bomb_exploded", "t_killed"]


def test_halftime_restart_preserves_previous_round_and_next_pistol():
    """Keep completed rounds on both sides of an abandoned restart segment."""
    events = _events(
        [1, 90, 120, 220],
        [10, 130, 230],
        [80, 120, 200, 300],
        [90, 120, 220, 310],
        ["CT", None, "T", "CT"],
        ["t_killed", None, "bomb_exploded", "t_killed"],
    )
    rounds = create_round_df(events)

    assert rounds["start"].to_list() == [1, 120, 220]
    assert rounds["end"].to_list() == [80, 200, 300]
    assert rounds["round_num"].to_list() == [1, 2, 3]


def test_resultless_end_without_a_restart_is_retained():
    """Unknown result metadata alone must not discard an otherwise complete round."""
    rounds = create_round_df(_events([1], [10], [80], [90], [None], [None]))

    assert rounds.height == 1
    assert rounds["end"].item() == 80
    assert rounds["winner"].item() is None


def test_scored_end_is_retained_even_if_a_start_artifact_shares_its_tick(monkeypatch: pytest.MonkeyPatch):
    """Only result-less restart artifacts may be removed."""
    events = _events([1, 80], [10], [80], [], ["CT"], ["t_killed"])
    events["round_officially_ended"] = pl.DataFrame(schema={"event": pl.String, "tick": pl.Int64})
    seen: list[int] = []

    def inspect_ends(rounds: pl.DataFrame, sequence: list[str]) -> list[int]:
        assert sequence == ["start", "freeze_end", "end", "official_end"]
        seen.extend(rounds.filter(pl.col("event") == "end")["tick"].to_list())
        return []

    monkeypatch.setattr("awpy.parsers.rounds._find_valid_round_indices", inspect_ends)
    with pytest.raises(ValueError, match="No valid round sequences"):
        create_round_df(events)
    assert seen == [80]


@pytest.mark.parametrize(
    ("freeze", "official"),
    [([10], [90]), ([], [90]), ([10], []), ([], [])],
)
def test_existing_incomplete_round_sequences_remain_supported(freeze: list[int], official: list[int]):
    """Missing freeze/official markers at EOF must retain the completed round."""
    events = _events([1], freeze, [80], official, ["CT"], ["t_killed"])
    if not freeze:
        events["round_freeze_end"] = pl.DataFrame(schema={"event": pl.String, "tick": pl.Int64})
    if not official:
        events["round_officially_ended"] = pl.DataFrame(schema={"event": pl.String, "tick": pl.Int64})
    rounds = create_round_df(events)

    assert rounds.height == 1
    assert rounds["end"].item() == 80
    assert rounds["official_end"].item() == (90 if official else 80)
