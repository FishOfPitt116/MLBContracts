"""
Player Lookup Module

Provides local cache lookups and career span derivation from stats data,
eliminating the need for Chadwick Baseball Bureau registry lookups.

Key functions:
- get_player_by_fangraphs_id(fg_id) -> Optional[Player]
- get_players_by_name(first, last) -> List[Player]
- get_career_span_from_stats(player_id) -> Optional[Tuple[int, int]]
- filter_players_by_year(players, contract_year) -> List[Player]
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from .records import Player
from .save import read_players_from_file, read_batter_stats, read_pitcher_stats


def normalize_name(name: str) -> str:
    """
    Normalize a name for fuzzy matching.

    Handles:
    - Unicode accents (José → Jose)
    - Periods and spaces (J.D. → jd, J. D. → jd)
    - Case normalization
    - Common suffixes (Jr., Sr., III, etc.)

    Args:
        name: Raw name string

    Returns:
        Normalized lowercase name for comparison
    """
    # Convert to lowercase
    name = name.lower()

    # Normalize Unicode (decompose accents, then remove combining characters)
    # NFD decomposes é into e + combining accent, then we strip the accent
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Remove periods
    name = name.replace('.', '')

    # Handle initials: collapse consecutive single letters (J D -> jd)
    # Split into words, then combine adjacent single-letter words
    words = name.split()
    result = []
    initials_buffer = []

    for word in words:
        if len(word) == 1:
            initials_buffer.append(word)
        else:
            if initials_buffer:
                result.append(''.join(initials_buffer))
                initials_buffer = []
            result.append(word)

    if initials_buffer:
        result.append(''.join(initials_buffer))

    name = ' '.join(result)

    # Remove common suffixes
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name.strip()


def names_match(name1: str, name2: str) -> bool:
    """
    Check if two names match after normalization.

    Args:
        name1: First name to compare
        name2: Second name to compare

    Returns:
        True if names match after normalization
    """
    return normalize_name(name1) == normalize_name(name2)


# Module-level caches - initialized lazily
_PLAYER_CACHE: Optional[Dict[str, Player]] = None  # player_id -> Player
_FANGRAPHS_ID_TO_PLAYER: Optional[Dict[int, Player]] = None  # fg_id -> Player
_NAME_TO_PLAYERS: Optional[Dict[Tuple[str, str], List[Player]]] = None  # (first_lower, last_lower) -> Players
_CAREER_SPANS: Optional[Dict[str, Tuple[int, int]]] = None  # player_id -> (first_year, last_year)


def _ensure_player_cache_loaded():
    """Initialize player caches if not already loaded"""
    global _PLAYER_CACHE, _FANGRAPHS_ID_TO_PLAYER, _NAME_TO_PLAYERS

    if _PLAYER_CACHE is not None:
        return

    _PLAYER_CACHE = {}
    _FANGRAPHS_ID_TO_PLAYER = {}
    _NAME_TO_PLAYERS = {}

    for player in read_players_from_file():
        _PLAYER_CACHE[player.player_id] = player

        if player.fangraphs_id is not None and player.fangraphs_id != -1:
            _FANGRAPHS_ID_TO_PLAYER[player.fangraphs_id] = player

        # Use normalized names for fuzzy matching
        name_key = (normalize_name(player.first_name), normalize_name(player.last_name))
        if name_key not in _NAME_TO_PLAYERS:
            _NAME_TO_PLAYERS[name_key] = []
        _NAME_TO_PLAYERS[name_key].append(player)


def _ensure_career_spans_loaded():
    """Initialize career spans cache from stats data if not already loaded"""
    global _CAREER_SPANS

    if _CAREER_SPANS is not None:
        return

    _CAREER_SPANS = {}

    # Build career spans from batter and pitcher stats
    batter_stats = read_batter_stats()
    pitcher_stats = read_pitcher_stats()

    # Only use single-year stats (window_years=1) for career span calculation
    for stat in batter_stats:
        if stat.window_years != 1:
            continue
        player_id = stat.player_id
        year = stat.year

        if player_id not in _CAREER_SPANS:
            _CAREER_SPANS[player_id] = (year, year)
        else:
            first, last = _CAREER_SPANS[player_id]
            _CAREER_SPANS[player_id] = (min(first, year), max(last, year))

    for stat in pitcher_stats:
        if stat.window_years != 1:
            continue
        player_id = stat.player_id
        year = stat.year

        if player_id not in _CAREER_SPANS:
            _CAREER_SPANS[player_id] = (year, year)
        else:
            first, last = _CAREER_SPANS[player_id]
            _CAREER_SPANS[player_id] = (min(first, year), max(last, year))


def refresh_caches():
    """Force reload of all caches from disk"""
    global _PLAYER_CACHE, _FANGRAPHS_ID_TO_PLAYER, _NAME_TO_PLAYERS, _CAREER_SPANS
    _PLAYER_CACHE = None
    _FANGRAPHS_ID_TO_PLAYER = None
    _NAME_TO_PLAYERS = None
    _CAREER_SPANS = None
    _ensure_player_cache_loaded()
    _ensure_career_spans_loaded()


def add_player_to_cache(player: Player):
    """Add a new player to the in-memory caches"""
    _ensure_player_cache_loaded()

    _PLAYER_CACHE[player.player_id] = player

    if player.fangraphs_id is not None and player.fangraphs_id != -1:
        _FANGRAPHS_ID_TO_PLAYER[player.fangraphs_id] = player

    # Use normalized names for fuzzy matching
    name_key = (normalize_name(player.first_name), normalize_name(player.last_name))
    if name_key not in _NAME_TO_PLAYERS:
        _NAME_TO_PLAYERS[name_key] = []
    if player not in _NAME_TO_PLAYERS[name_key]:
        _NAME_TO_PLAYERS[name_key].append(player)


def get_player_by_id(player_id: str) -> Optional[Player]:
    """Get a player by their internal player_id"""
    _ensure_player_cache_loaded()
    return _PLAYER_CACHE.get(player_id)


def get_player_by_fangraphs_id(fg_id: int) -> Optional[Player]:
    """
    Get a player by their FanGraphs ID from local cache.

    Args:
        fg_id: FanGraphs player ID

    Returns:
        Player object if found in cache, None otherwise
    """
    _ensure_player_cache_loaded()
    return _FANGRAPHS_ID_TO_PLAYER.get(fg_id)


def get_players_by_name(first_name: str, last_name: str) -> List[Player]:
    """
    Get all players matching a name from local cache.

    Uses normalized name matching to handle:
    - Accented characters (José → Jose)
    - Periods/spacing variations (J.D. → JD)
    - Case differences

    Args:
        first_name: Player's first name
        last_name: Player's last name

    Returns:
        List of Player objects matching the name (may be empty)
    """
    _ensure_player_cache_loaded()
    name_key = (normalize_name(first_name), normalize_name(last_name))
    return _NAME_TO_PLAYERS.get(name_key, [])


def get_career_span_from_stats(player_id: str) -> Optional[Tuple[int, int]]:
    """
    Get a player's career span (first year, last year) derived from local stats data.

    This looks at the min/max years where the player has batting or pitching stats
    in the local dataset.

    Args:
        player_id: Internal player ID

    Returns:
        Tuple of (first_year, last_year) if player has stats, None otherwise
    """
    _ensure_career_spans_loaded()
    return _CAREER_SPANS.get(player_id)


def filter_players_by_year(players: List[Player], contract_year: int) -> List[Player]:
    """
    Filter a list of players to those active around a contract year.

    Uses career spans from stats to determine if a player was active.
    Allows 1 year buffer on either end (player could sign contract before debut
    or after last recorded season).

    Args:
        players: List of Player objects to filter
        contract_year: Year of the contract

    Returns:
        List of players whose career spans include the contract year (+/- 1 year)
    """
    _ensure_career_spans_loaded()

    filtered = []
    for player in players:
        career_span = _CAREER_SPANS.get(player.player_id)
        if career_span is None:
            # No stats data - can't determine if active, include them as candidate
            filtered.append(player)
            continue

        first_year, last_year = career_span
        # Allow 1 year buffer: player could sign contract year before debut or after last season
        if first_year - 1 <= contract_year <= last_year + 1:
            filtered.append(player)

    return filtered


def get_all_fangraphs_ids() -> Dict[int, str]:
    """
    Get a mapping of all FanGraphs IDs to internal player IDs.

    Returns:
        Dict mapping FanGraphs ID -> player_id
    """
    _ensure_player_cache_loaded()
    return {fg_id: player.player_id for fg_id, player in _FANGRAPHS_ID_TO_PLAYER.items()}


def get_all_players() -> List[Player]:
    """
    Get all players from the cache.

    Returns:
        List of all Player objects
    """
    _ensure_player_cache_loaded()
    return list(_PLAYER_CACHE.values())
