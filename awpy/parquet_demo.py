"""Defines the ParquetDemo class."""

import json
import sys
import tempfile
import time
import zipfile
from functools import cached_property
from pathlib import Path

import polars as pl
from loguru import logger

import awpy.constants
import awpy.parsers.bomb
import awpy.parsers.events
import awpy.parsers.grenades
import awpy.parsers.rounds
import awpy.parsers.ticks
import awpy.parsers.utils

PROP_WARNING_LIMIT = 40
DEFAULT_PLAYER_PROPS = [
    "team_name",
    "team_clan_name",
    "X",
    "Y",
    "Z",
    "last_place_name",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "pitch",
    "yaw",
    "health",
    "armor_value",
    "inventory",
    "current_equip_value",
    "has_defuser",
    "has_helmet",
    "flash_duration",
    "accuracy_penalty",
    "zoom_lvl",
    "ping",
]

DEFAULT_WORLD_PROPS = [
    "game_time",
    "is_bomb_planted",
    "which_bomb_zone",
    "is_freeze_period",
    "is_warmup_period",
    "is_terrorist_timeout",
    "is_ct_timeout",
    "is_technical_timeout",
    "is_waiting_for_resume",
    "is_match_started",
    "game_phase",
]

DEFAULT_EVENT_LIST = [
    "round_freeze_end",  # Freeze time ends
    "round_officially_ended",  # Round officially declared over
    "round_start",  # Round start (not shown in list_game_events)
    "round_end",  # Round end (not shown in list_game_events)
    "player_spawn",  # Players spawn
    "player_given_c4",  # C4 is given to a player
    "bomb_pickup",  # Bomb is picked up
    "item_pickup",  # Items are picked up
    "weapon_fire",  # Weapon is fired
    "player_sound",  # Player makes a sound
    "player_hurt",  # Player takes damage
    "player_death",  # Player died
    "bomb_dropped",  # Bomb is dropped
    "bomb_planted",  # Bomb is planted
    "bomb_defused",  # Bomb is defused
    "bomb_exploded",  # Bomb explodes
    "flashbang_detonate",  # Flashbang detonates
    "hegrenade_detonate",  # HE grenade detonates
    "smokegrenade_detonate",  # Smoke grenade detonates
    "smokegrenade_expired",  # Smoke dissipates
    "inferno_startburn",  # Molotov/Incendiary starts burning
    "inferno_expire",  # Molotov/Incendiary burn ends
]


class ParquetDemo:
    """Class to parse and store a demo's data.

    This class encapsulates functionality for parsing various aspects of a demo file,
    including header information, event data, tick data, and grenade events. Instances
    of this class are typically created with a file path to the demo file.

    Example:
        demo = Demo(path=Path("path/to/demo.dem"), verbose=True)
    """

    def __init__(
        self,
        path: Path,
        map_name: str,
        patch_version: str,
        tickrate: int = awpy.constants.DEFAULT_SERVER_TICKRATE,
        inferno_duration: float = awpy.constants.DEFAULT_INFERNO_DURATION_IN_SECS,
        smoke_duration: float = awpy.constants.DEFAULT_SMOKE_DURATION_IN_SECS,
        *,
        verbose: bool = False,
    ) -> None:
        """Instantiate a ParquetDemo object.

        Initializes the ParquetDemo with the specified parquet demo file path.

        Sets up an empty DataFrame for tick data. Also allows setting custom grenade
        durations for infernos and smokes. If verbose mode is enabled, the logger is configured
        to output debug information.

        Args:
            path (Path): The filesystem path to the parquet file.
            verbose (bool, optional): If True, configures logging for verbose output.
                                      Defaults to False.
            tickrate (int, optional): The server tickrate.
                Defaults to awpy.constants.DEFAULT_SERVER_TICKRATE.
            inferno_duration (float, optional): Duration (in seconds) for infernos.
                Defaults to awpy.constants.DEFAULT_INFERNO_DURATION_IN_SECS.
            smoke_duration (float, optional): Duration (in seconds) for smokes.
                Defaults to awpy.constants.DEFAULT_SMOKE_DURATION_IN_SECS.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
        """
        print("Parsing Parquet Demo...")
        self.path = Path(path)
        if not self.path.exists():
            file_not_found_error_msg = f"File not found: {self.path}"
            raise FileNotFoundError(file_not_found_error_msg)

        self.header = self.parse_header(map_name, patch_version)
        self.ticks: pl.DataFrame = pl.DataFrame()
        self.tickrate = tickrate
        self.inferno_duration = inferno_duration
        self.smoke_duration = smoke_duration
        self.in_play_ticks = None

        if verbose:
            logger.remove()
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.remove()

    def __repr__(self) -> str:
        """Return a string representation of the Demo object.

        The representation includes the file path, map name extracted from the header,
        and the number of tick records.

        Returns:
            str: A string summarizing the Demo object.
        """
        if self.in_play_ticks is None:
            return f"Demo(path={self.path}, map={self.header.get('map_name')})"
        return f"Demo(path={self.path}, map={self.header.get('map_name')}, ticks={len(self.in_play_ticks)})"
    

    def parse_header(self, map_name, patch_version) -> dict:
        """Construct Dictionary of header information.
        Parquet files don't contain header info so we pass important bits saved from demo to parquet conversion.

        Returns:
            dict: A dictionary containing the parsed header information.
        """
        header_dict = {}
        header_dict["map_name"] = map_name
        header_dict["patch_version"] = patch_version
        self.header = header_dict
        return self.header

    def parse(
        self,
        events: list[str] | None = None,
        player_props: list[str] | None = None,
        other_props: list[str] | None = None,
    ) -> None:
        """Parse the parquet demo file for events, rounds, tick data, and grenade events.

        This comprehensive parsing method processes the demo file in several stages:
          1. It extracts event data for a specified set of events (or a default set if None is provided).
          2. It creates a rounds DataFrame from the event data.
          3. It extracts and filters valid in-play ticks.
          4. It parses player-specific tick data.
          5. It parses grenade events.
          6. It filters the data to include only valid ticks during match play.

        Args:
            events (list[str] | None, optional): List of event names to parse. If None,
                a default set of events defined by `default_events` is used.
            player_props (list[str] | None, optional): List of player-related properties to extract.
            other_props (list[str] | None, optional): List of additional properties to extract.

        Returns:
            None
        """
        start = time.perf_counter()
        logger.debug(f"Starting to parse {self.path}")

        # Get player props
        player_props = ["last_place_name", "X", "Y", "Z", "health", "team_name"] + (
            player_props if player_props is not None else []
        )
        player_props = list(set(player_props))

        # Parse events from demo
        events_to_parse = events if events is not None else self.default_events
        self.events = self.parse_events(
            events_to_parse,
            player_props=player_props,
            other_props=other_props,
        )

        # Get rounds and in-play ticks
        self.rounds = awpy.parsers.rounds.create_round_df(self.events)
        self.in_play_ticks = awpy.parsers.ticks.get_valid_ticks(
            self.parse_ticks_from_parquet(
                other_props=[
                    "game_time",
                    "is_bomb_planted",
                    "which_bomb_zone",
                    "is_freeze_period",
                    "is_warmup_period",
                    "is_terrorist_timeout",
                    "is_ct_timeout",
                    "is_technical_timeout",
                    "is_waiting_for_resume",
                    "is_match_started",
                    "game_phase",
                ]
            )
        )

        # Parse, filter, and apply round number to ticks
        self.ticks = self.parse_ticks_from_parquet(player_props=player_props)
        self.ticks = self.ticks.filter(pl.col("tick").is_in(self.in_play_ticks))
        self.ticks = awpy.parsers.rounds.apply_round_num(df=self.ticks, rounds_df=self.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )
        self.ticks = awpy.parsers.utils.fix_common_names(self.ticks)

        # Parse, filter, and apply round number to grenades
        self.grenades = self.parse_grenades()
        self.grenades = self.grenades.filter(pl.col("tick").is_in(self.in_play_ticks))
        self.grenades = awpy.parsers.rounds.apply_round_num(
            df=self.grenades, rounds_df=self.rounds, tick_col="tick"
        ).filter(pl.col("round_num").is_not_null())

        logger.success(f"Finished parsing {self.path}, took {time.perf_counter() - start:.2f} seconds")

    @cached_property
    def infernos(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of inferno events in the demo.

        Inferno events are derived from the "inferno_startburn" and "inferno_expire" events.
        The effective duration is computed as tickrate multiplied by inferno_duration (in seconds).

        Returns:
            pl.DataFrame: A DataFrame containing inferno event data.

        Raises:
            KeyError: If the necessary inferno events ("inferno_startburn" or "inferno_expire")
                        are not found in the parsed events.
        """
        starts = awpy.parsers.utils.get_event_from_parsed_events(self.events, "inferno_startburn")
        ends = awpy.parsers.utils.get_event_from_parsed_events(self.events, "inferno_expire")
        duration_in_ticks = self.tickrate * self.inferno_duration
        infernos = awpy.parsers.grenades.parse_timed_grenade_entity(starts, ends, duration_in_ticks)
        return awpy.parsers.rounds.apply_round_num(df=infernos, rounds_df=self.rounds, tick_col="start_tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def smokes(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of smoke grenade events in the demo.

        Smoke events are derived from the "smokegrenade_detonate" and "smokegrenade_expired" events.
        The effective duration is computed as tickrate multiplied by smoke_duration (in seconds).

        Returns:
            pl.DataFrame: A DataFrame containing smoke grenade event data.

        Raises:
            KeyError: If the necessary smoke grenade events ("smokegrenade_detonate" or "smokegrenade_expired")
                        are not found in the parsed events.
        """
        starts = awpy.parsers.utils.get_event_from_parsed_events(self.events, "smokegrenade_detonate")
        ends = awpy.parsers.utils.get_event_from_parsed_events(self.events, "smokegrenade_expired")
        duration_in_ticks = self.tickrate * self.smoke_duration
        smokes = awpy.parsers.grenades.parse_timed_grenade_entity(starts, ends, duration_in_ticks)
        return awpy.parsers.rounds.apply_round_num(df=smokes, rounds_df=self.rounds, tick_col="start_tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def kills(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of kill events parsed from the demo.

        This property extracts the 'player_death' events from the parsed events dictionary,
        and then processes these events using the awpy.parsers.events.parse_kills function to
        create a standardized DataFrame of kill data.

        Returns:
            pl.DataFrame: A DataFrame containing kill event data.

        Raises:
            KeyError: If 'player_death' events are not found in the parsed events, which likely indicates
                    that the demo has not been parsed yet. In this case, please run the .parse() method.
        """
        kills = awpy.parsers.utils.get_event_from_parsed_events(self.events, "player_death")
        kills = awpy.parsers.events.parse_kills(kills)
        return awpy.parsers.rounds.apply_round_num(df=kills, rounds_df=self.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def damages(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of damage events parsed from the demo.

        This property extracts the 'player_hurt' events from the parsed events dictionary,
        and then processes these events using the awpy.parsers.events.parse_damages function to
        create a standardized DataFrame of damage (player hurt) data.

        Returns:
            pl.DataFrame: A DataFrame containing damage event data.

        Raises:
            KeyError: If 'player_hurt' events are not found in the parsed events, which likely indicates
                    that the demo has not been parsed yet. In this case, please run the .parse() method.
        """
        damages = awpy.parsers.utils.get_event_from_parsed_events(self.events, "player_hurt")
        damages = awpy.parsers.events.parse_damages(damages)
        return awpy.parsers.rounds.apply_round_num(df=damages, rounds_df=self.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def footsteps(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of footstep events from the demo.

        Footstep events are derived from the 'player_sound' events parsed from the demo.
        These events are processed using the awpy.parsers.events.parse_footsteps function,
        which standardizes the data for further analysis. The resulting DataFrame typically
        contains details such as the position and timing of player sounds (often corresponding to footsteps).

        Returns:
            pl.DataFrame: A DataFrame containing footstep event data.

        Raises:
            KeyError: If 'player_sound' events are not found in the parsed events. This may
                    indicate that the demo has not been parsed yet. Please run the .parse() method.
        """
        footsteps = awpy.parsers.utils.get_event_from_parsed_events(self.events, "player_sound")
        footsteps = awpy.parsers.events.parse_footsteps(footsteps)
        return awpy.parsers.rounds.apply_round_num(df=footsteps, rounds_df=self.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def shots(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of shot events from the demo.

        Shot events are extracted from the 'weapon_fire' events parsed from the demo.
        These events are processed using the awpy.parsers.events.parse_shots function,
        which organizes the weapon fire events into a standardized format suitable for analysis.
        The resulting DataFrame typically includes details such as the weapon used and the time
        of the shot.

        Returns:
            pl.DataFrame: A DataFrame containing shot event data.

        Raises:
            KeyError: If 'weapon_fire' events are not found in the parsed events. This may
                    indicate that the demo has not been parsed yet. Please run the .parse() method.
        """
        shots = awpy.parsers.utils.get_event_from_parsed_events(self.events, "weapon_fire")
        shots = awpy.parsers.events.parse_shots(shots)
        return awpy.parsers.rounds.apply_round_num(df=shots, rounds_df=self.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def bomb(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of bomb events from the demo.

        This property leverages the awpy.parsers.bomb.parse_bomb function to parse and standardize
        bomb-related events extracted from the demo's events. These events include bomb drops, pickups,
        plantings, and explosions. The parsing process uses the demo's parsed events (self.events) and filters
        them based on the valid in-play tick values (self.in_play_ticks) to ensure that only bomb events occurring
        during active gameplay are included.

        The resulting DataFrame typically contains columns such as:
        - tick: The tick at which the bomb event occurred.
        - status: The bomb event status (e.g., "dropped", "carried", "planted", or "detonated").
        - Position coordinates (x, y, z) of the bomb, derived from tick data.
        - steamid: The steam ID of the player associated with the bomb event.
        - name: The name of the player associated with the bomb event.
        - bombsite: For planted events, the bombsite information; otherwise, null.

        Returns:
            pl.DataFrame: A DataFrame containing standardized bomb event data.

        Raises:
            KeyError: If the necessary bomb event data is missing from the parsed events.
        """
        bomb = awpy.parsers.bomb.parse_bomb(self.events, self.in_play_ticks)
        return awpy.parsers.rounds.apply_round_num(df=bomb, rounds_df=self.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )

    @cached_property
    def player_round_totals(self) -> pl.DataFrame:
        """Cached property that calculates and returns player round totals.

        This property computes the total number of rounds played by each player, both for each specific side
        (e.g., ct or t) and overall (regardless of side). It uses data from the ticks and rounds
        DataFrames available on the demo object.

        The process is as follows:
          1. Select unique combinations of player name, steamid, side, and round number from the ticks DataFrame.
          2. Join these unique records with the rounds DataFrame on the round number to filter and validate rounds.
          3. Group the joined data by name, steamid, and side to count rounds per player for each team side.
          4. Additionally, group by name and steamid to compute the total rounds played by each player,
             regardless of side, and label these records with side as "all".
          5. Concatenate the per-side and overall aggregates into a single DataFrame.

        Returns:
            pl.DataFrame: A DataFrame with the following columns:
                - name (str): The player's name.
                - steamid (str): The player's Steam ID.
                - side (str): The team side ("ct", "t", or "all" for total rounds).
                - n_rounds (int): The number of rounds played by the player on the given side.
        """
        logger.debug("Calculating player round totals...")
        player_sides_by_round = self.ticks.select(["name", "steamid", "side", "round_num"]).unique()

        # Merge with rounds DataFrame on "round".
        player_sides_by_round = player_sides_by_round.join(self.rounds, on="round_num", how="inner")

        # Aggregate rounds by player and team (i.e. side).
        player_side_rounds = player_sides_by_round.group_by(["name", "steamid", "side"]).agg(
            pl.count("round_num").alias("n_rounds")
        )

        # Aggregate total rounds by player (regardless of team) and label as "all".
        player_total_rounds = (
            player_sides_by_round.group_by(["name", "steamid"])
            .agg(pl.count("round_num").alias("n_rounds"))
            .with_columns(pl.lit("all").alias("side"))
            .select(["name", "steamid", "side", "n_rounds"])
        )

        # Concatenate the two results into one DataFrame.
        return pl.concat([player_side_rounds, player_total_rounds])

    @cached_property
    def server_cvars(self) -> pl.DataFrame:
        """Cached property that returns a Polars DataFrame of server configuration variable events.

        This property extracts server configuration variables by parsing the 'server_cvar'
        event from the demo using the parser's parse_event method. The returned data is
        converted from a pandas DataFrame to a Polars DataFrame, providing a standardized view
        of the server's configuration variables at the time of the demo.

        Returns:
            pl.DataFrame: A DataFrame containing server cvar event data.
        """
        server_cvar_dict = self.read_events_from_parquet(["server_cvar"])
        return server_cvar_dict["server_cvar"]

    @property
    def default_events(self) -> list[str]:
        """Get the default list of event names to parse from the demo file.

        This default list includes key events such as round events, player actions,
        bomb events, and grenade detonations that are typically of interest when analyzing a demo.

        Returns:
            list[str]: A list of default event names.
        """
        return [
            "round_freeze_end",  # Freeze time ends
            "round_officially_ended",  # Round officially declared over
            "player_spawn",  # Players spawn
            "player_given_c4",  # C4 is given to a player
            "bomb_pickup",  # Bomb is picked up
            "item_pickup",  # Items are picked up
            "weapon_fire",  # Weapon is fired
            "player_sound",  # Player makes a sound
            "player_hurt",  # Player takes damage
            "player_death",  # Player died
            "bomb_dropped",  # Bomb is dropped
            "bomb_planted",  # Bomb is planted
            "bomb_defused",  # Bomb is defused
            "bomb_exploded",  # Bomb explodes
            "flashbang_detonate",  # Flashbang detonates
            "hegrenade_detonate",  # HE grenade detonates
            "smokegrenade_detonate",  # Smoke grenade detonates
            "smokegrenade_expired",  # Smoke dissipates
            "inferno_startburn",  # Molotov/Incendiary starts burning
            "inferno_expire",  # Molotov/Incendiary burn ends
        ]

    @property
    def detected_events(self) -> list[str]:
        """Retrieve the list of events detected in the parquet file.

        This property queries the parquet file to obtain a list of all event
        types that have been identified within the demo.

        Returns:
            list[str]: A list of event names detected in the demo.
        """
        df = pl.read_parquet(self.path)
        return df["event_type"].unique().to_list()

    def read_events_from_parquet(
        self,
        events_to_parse: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Read event data directly from the Parquet file.

        This method extracts events from the parquet file by filtering rows where
        the 'event_type' column is not null. It then separates the events by type
        into individual DataFrames.

        Args:
            events_to_parse (list[str] | None, optional): List of event names to extract.
                If None, all events in the parquet file will be returned.

        Returns:
            dict[str, pl.DataFrame]: A dictionary mapping event names to their respective
            Polars DataFrames.
        """
        logger.debug(f"Reading events from parquet file: {self.path}")
        
        # Read the parquet file and filter for event rows (non-null event_type)
        df = pl.read_parquet(self.path)
        
        # Check if event_type column exists
        if "event_type" not in df.columns:
            logger.warning("No 'event_type' column found in parquet file")
            return {}
        
        # Filter to only event rows
        events_df = df.filter(pl.col("event_type").is_not_null())
        
        # Get default of all available event types if none specified
        if events_to_parse is None:
            events_to_parse = events_df["event_type"].unique().to_list()
        
        # Split events into separate DataFrames by type
        events_dict = {}
        for event in events_to_parse:
            event_df = events_df.filter(pl.col("event_type") == event)
            if not event_df.is_empty():
                # Drop the event_type column as it's redundant for individual event DataFrames
                event_df = event_df.drop("event_type")
                
                # Drop columns that only contain null values for this specific event
                non_null_cols = []
                for col in event_df.columns:
                    # Skip if all null
                    if event_df[col].is_null().all():
                        continue
                    # Skip user_steamid if it only contains "nan" strings
                    if col == "user_steamid" and event_df[col].dtype in [pl.Utf8, pl.String]:
                        if (event_df[col] == "nan").all():
                            continue
                    
                    non_null_cols.append(col)
                
                event_df = event_df.select(non_null_cols)
                
                events_dict[event] = event_df
        
        logger.debug(f"Extracted {len(events_dict)} event types from parquet file")
        return events_dict

    def parse_events(
        self,
        events_to_parse: list[str] | None = None,
        player_props: list[str] | None = None,
        other_props: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Parse event data from the Parquet file.

        This method extracts the specified events from the Parquet file
        into Polars DataFrames, and applies additional processing for
        round events by setting custom labels and filtering out invalid data.

        Args:
            events_to_parse (list[str] | None, optional): List of event names to parse.
                If None, no additional events beyond the parser's defaults will be processed.
            player_props (list[str] | None, optional): List of player-specific properties to extract.
            other_props (list[str] | None, optional): List of other properties to extract.

        Returns:
            dict[str, pl.DataFrame]: A dictionary mapping event names to their respective
            Polars DataFrames.
        """
        events: dict[str, pl.DataFrame] = self.read_events_from_parquet(events_to_parse)

        # Explicitly parse round start and round end events
        round_start_dict = self.read_events_from_parquet(["round_start"])
        if "round_start" in round_start_dict:
            events["round_start"] = round_start_dict["round_start"]
            
        round_end_dict = self.read_events_from_parquet(["round_end"])
        if "round_end" in round_end_dict:
            events["round_end"] = round_end_dict["round_end"]

        # Loop through and process each event
        for event_name, event in events.items():
            logger.debug(f"Parsing event: {event_name} with {len(event)} records")

            logger.debug(event_name)
            logger.debug(event.columns)
            logger.debug(event.head())

            events[event_name] = awpy.parsers.utils.fix_common_names(event)
            if event_name == "round_start":
                # Label the event as 'start'
                events[event_name] = events[event_name].with_columns(pl.lit("start").alias("event"))
            elif event_name == "round_end":
                # Label the event as 'end' and filter for rows with a valid winner.
                events[event_name] = (
                    events[event_name].with_columns(pl.lit("end").alias("event")).filter(pl.col("winner").is_not_null())
                )
            elif event_name == "round_officially_ended":
                # Label the event as 'official_end'
                events[event_name] = events[event_name].with_columns(pl.lit("official_end").alias("event"))
            elif event_name == "round_freeze_end":
                # Label the event as 'freeze_end'
                events[event_name] = events[event_name].with_columns(pl.lit("freeze_end").alias("event"))
        return events

    def parse_ticks_from_parquet(
        self,
        player_props: list[str] | None = None,
        other_props: list[str] | None = None,
    ) -> pl.DataFrame:
        """Parse tick data from the parquet file.

        This method extracts tick-related data from the parquet file.
        Ticks are identified as rows where the 'event_type' column is null. 
        It is possible to specify which player-related and other properties to include in the ticks DataFrame.

        Args:
            player_props (list[str] | None, optional): List of player-related properties to include.
            other_props (list[str] | None, optional): List of other properties to include.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the parsed tick data.
        """
        player_props = player_props if player_props is not None else []
        other_props = other_props if other_props is not None else []
        required_props = ["steamid", "name"]

        df = pl.read_parquet(self.path)
        tick_rows = df.filter(pl.col("event_type").is_null())
        ticks_df = tick_rows.select(required_props + player_props + other_props + ["tick"])

        return ticks_df
    

    def parse_grenades(self) -> pl.DataFrame:
        """Parse grenade event data from the demo file.

        This method extracts grenade-related events and converts
        the resulting data into a Polars DataFrame for further analysis.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the parsed grenade event data.
        """
        
        grenades_dict = self.read_events_from_parquet(["grenade_data"])
        grenade_df = grenades_dict["grenade_data"].rename(
            {
                "name": "thrower",
                "steamid": "thrower_steamid",
                "x": "X",
                "y": "Y",
                "z": "Z",
                "grenade_entity_id": "entity_id",
            }
        )
        return grenade_df.select(
            [
                "thrower_steamid",
                "thrower",
                "grenade_type",
                "tick",
                "X",
                "Y",
                "Z",
                "entity_id",
            ]
        )

    def compress(self, outpath: Path | None = None) -> None:
        """Compress the parsed contents of the demo file.

        Args:
            outpath (Path | None, optional): The output path for the compressed demo file.
                If None, the compressed file will be saved to the original file path.
        """
        self._raise_if_no_parser()
        start = time.perf_counter()
        logger.debug(f"Starting to compress parsed {self.path}")

        outpath = Path.cwd() if outpath is None else Path(outpath)
        zip_name = outpath / Path(self.path.stem + ".zip")

        with (
            tempfile.TemporaryDirectory() as tmpdirname,
            zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf,
        ):
            # Get the main dataframes
            for df_name, parsed_df in [
                ("kills", self.kills),
                ("damages", self.damages),
                ("footsteps", self.footsteps),
                ("shots", self.shots),
                ("grenades", self.grenades),
                ("infernos", self.infernos),
                ("smokes", self.smokes),
                ("bomb", self.bomb),
                ("ticks", self.ticks),
                ("rounds", self.rounds),
            ]:
                parsed_df_filename = Path(tmpdirname) / f"{df_name}.parquet"
                parsed_df.write_parquet(parsed_df_filename)
                zipf.write(parsed_df_filename, arcname=f"{df_name}.parquet")

            # Write header
            header_filename = Path(tmpdirname) / "header.json"
            with open(header_filename, "w", encoding="utf-8") as f:
                json.dump(self.header, f)
            zipf.write(header_filename, arcname="header.json")

            logger.success(f"Compressed demo data saved to {zip_name}, took {time.perf_counter() - start:.2f} seconds")
