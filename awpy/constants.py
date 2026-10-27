"""Module for constants in Counter-Strike 2."""

# Sides
CT_SIDE = "ct"
T_SIDE = "t"

# Side values that are not a playing team. A connected coach or observer keeps an entity in
# the demo for the whole match, so they appear in tick data with one of these side values and
# no player state. They are not players and must not reach player_round_totals, which seeds the
# scoreboard. Compared case-insensitively because the value may be the raw team_name
# ("Spectator") or a converted one ("spectator", via converters.TEAM_MAP).
NON_PLAYING_SIDES = frozenset({"spectator", "unassigned"})

# Server
DEFAULT_SERVER_TICKRATE = 64 # CS2 locked to 64 tick

# Rounds
DEFAULT_FREEZE_TIME_IN_SECS = 20.0
DEFAULT_ROUND_TIME_IN_SECS = 115.0
DEFAULT_BOMB_TIME_IN_SECS = 40.0

# Grenades
DEFAULT_SMOKE_DURATION_IN_SECS = 20.0
DEFAULT_INFERNO_DURATION_IN_SECS = 7.03125
