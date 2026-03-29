"""
FanGraphs Search Module

Provides direct FanGraphs stats search functionality, eliminating the need
for Chadwick Baseball Bureau registry lookups.

Key functions:
- search_fangraphs_by_name(first, last, year) -> List[Dict]
- get_active_players_for_year(year) -> Set[int]
- is_player_active_in_year(fg_id, year) -> bool
- scrape_baseball_reference_link(fangraphs_id) -> Optional[str]
"""

from bs4 import BeautifulSoup
from datetime import datetime
import requests
import time
from typing import Dict, List, Optional, Set, Tuple

from pybaseball import batting_stats, pitching_stats, cache

# Enable pybaseball caching to avoid repeated API calls
cache.enable()

from .player_lookup import normalize_name


# Module-level cache for active players by year
_ACTIVE_PLAYERS_CACHE: Dict[int, Set[int]] = {}  # year -> set of FanGraphs IDs

CURRENT_YEAR = datetime.now().year


def _ensure_year_cached(year: int):
    """Load active players for a year into cache if not already loaded"""
    if year in _ACTIVE_PLAYERS_CACHE:
        return

    _ACTIVE_PLAYERS_CACHE[year] = set()

    try:
        # Get all batters for the year (qual=0 means no minimum plate appearances)
        batting = batting_stats(year, qual=0)
        if not batting.empty and 'IDfg' in batting.columns:
            _ACTIVE_PLAYERS_CACHE[year].update(batting['IDfg'].dropna().astype(int).tolist())
    except Exception as e:
        print(f"Warning: Could not fetch batting stats for {year}: {e}")

    try:
        # Get all pitchers for the year
        pitching = pitching_stats(year, qual=0)
        if not pitching.empty and 'IDfg' in pitching.columns:
            _ACTIVE_PLAYERS_CACHE[year].update(pitching['IDfg'].dropna().astype(int).tolist())
    except Exception as e:
        print(f"Warning: Could not fetch pitching stats for {year}: {e}")


def get_active_players_for_year(year: int) -> Set[int]:
    """
    Get the set of FanGraphs IDs for all players active in a given year.

    This queries FanGraphs batting and pitching stats for the year.

    Args:
        year: The season year

    Returns:
        Set of FanGraphs player IDs who had stats in that year
    """
    _ensure_year_cached(year)
    return _ACTIVE_PLAYERS_CACHE.get(year, set())


def is_player_active_in_year(fg_id: int, year: int) -> bool:
    """
    Check if a player was active (had stats) in a given year.

    Args:
        fg_id: FanGraphs player ID
        year: The season year

    Returns:
        True if the player had batting or pitching stats in that year
    """
    active_players = get_active_players_for_year(year)
    return fg_id in active_players


def _parse_fangraphs_name(full_name: str) -> Tuple[str, str]:
    """Parse a FanGraphs full name into first and last name components."""
    if not full_name or not isinstance(full_name, str):
        return ("", "")
    parts = full_name.strip().split(" ", 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return (parts[0], "")


def search_fangraphs_by_name(first_name: str, last_name: str, year: int) -> List[Dict]:
    """
    Search FanGraphs stats for players matching a name in a given year.

    Uses normalized name matching to handle accents, periods, etc.

    Args:
        first_name: Player's first name
        last_name: Player's last name
        year: The season year to search

    Returns:
        List of dicts with keys: fg_id, name, year, stats_type ('batting' or 'pitching')
    """
    results = []
    seen_ids = set()

    first_normalized = normalize_name(first_name)
    last_normalized = normalize_name(last_name)

    try:
        batting = batting_stats(year, qual=0)
        if not batting.empty and 'Name' in batting.columns and 'IDfg' in batting.columns:
            for _, row in batting.iterrows():
                fg_first, fg_last = _parse_fangraphs_name(row['Name'])
                # Check if normalized names match
                if (normalize_name(fg_first) == first_normalized and
                    normalize_name(fg_last) == last_normalized):
                    fg_id = int(row['IDfg'])
                    if fg_id not in seen_ids:
                        seen_ids.add(fg_id)
                        results.append({
                            'fg_id': fg_id,
                            'name': row['Name'],
                            'year': year,
                            'stats_type': 'batting'
                        })
    except Exception as e:
        print(f"Warning: Could not search batting stats for {year}: {e}")

    try:
        pitching = pitching_stats(year, qual=0)
        if not pitching.empty and 'Name' in pitching.columns and 'IDfg' in pitching.columns:
            for _, row in pitching.iterrows():
                fg_first, fg_last = _parse_fangraphs_name(row['Name'])
                if (normalize_name(fg_first) == first_normalized and
                    normalize_name(fg_last) == last_normalized):
                    fg_id = int(row['IDfg'])
                    if fg_id not in seen_ids:
                        seen_ids.add(fg_id)
                        results.append({
                            'fg_id': fg_id,
                            'name': row['Name'],
                            'year': year,
                            'stats_type': 'pitching'
                        })
    except Exception as e:
        print(f"Warning: Could not search pitching stats for {year}: {e}")

    return results


def search_fangraphs_by_name_range(first_name: str, last_name: str, start_year: int, end_year: int) -> List[Dict]:
    """
    Search FanGraphs stats for players matching a name across a range of years.

    Args:
        first_name: Player's first name (case-insensitive)
        last_name: Player's last name (case-insensitive)
        start_year: First year to search
        end_year: Last year to search (inclusive)

    Returns:
        List of dicts with keys: fg_id, name, first_year, last_year
        Deduplicated by FanGraphs ID, with combined year range
    """
    results_by_id: Dict[int, Dict] = {}

    for year in range(start_year, end_year + 1):
        year_results = search_fangraphs_by_name(first_name, last_name, year)
        for result in year_results:
            fg_id = result['fg_id']
            if fg_id not in results_by_id:
                results_by_id[fg_id] = {
                    'fg_id': fg_id,
                    'name': result['name'],
                    'first_year': year,
                    'last_year': year
                }
            else:
                results_by_id[fg_id]['first_year'] = min(results_by_id[fg_id]['first_year'], year)
                results_by_id[fg_id]['last_year'] = max(results_by_id[fg_id]['last_year'], year)

    return list(results_by_id.values())


def scrape_baseball_reference_link(fangraphs_id: int) -> Optional[str]:
    """
    Scrape the Baseball Reference link from a player's FanGraphs page.

    Args:
        fangraphs_id: FanGraphs player ID

    Returns:
        Baseball Reference URL if found, None otherwise
    """
    fangraphs_url = f"https://www.fangraphs.com/players/x/{fangraphs_id}"

    try:
        time.sleep(1)  # Be polite to FanGraphs
        response = requests.get(fangraphs_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for the Baseball Reference link
        # FanGraphs typically has links to other sites in a specific section
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'baseball-reference.com/players/' in href:
                # Ensure it's a full URL
                if not href.startswith('http'):
                    href = 'https://www.baseball-reference.com' + href
                return href

    except Exception as e:
        print(f"Warning: Could not scrape FanGraphs page for ID {fangraphs_id}: {e}")

    return None


def get_career_span_from_fangraphs(fangraphs_id: int, min_year: int = 2000, max_year: int = None) -> Optional[Tuple[int, int]]:
    """
    Determine a player's career span by checking FanGraphs stats year by year.

    This is an expensive operation - use local stats data when possible.
    Uses binary search-like optimization to reduce API calls.

    Args:
        fangraphs_id: FanGraphs player ID
        min_year: Earliest year to check (default 2000)
        max_year: Latest year to check (default current year)

    Returns:
        Tuple of (first_year, last_year) if player found, None otherwise
    """
    if max_year is None:
        max_year = CURRENT_YEAR

    # First, find any year where the player was active
    # Start from recent years (more likely to find quickly)
    active_years = []

    # Check current year first, then work backwards
    for year in range(max_year, min_year - 1, -1):
        if is_player_active_in_year(fangraphs_id, year):
            active_years.append(year)
            # Once we find one, expand in both directions
            break

    if not active_years:
        return None

    # Found at least one year - now find the boundaries
    first_year = active_years[0]
    last_year = active_years[0]

    # Search backwards for first year
    for year in range(first_year - 1, min_year - 1, -1):
        if is_player_active_in_year(fangraphs_id, year):
            first_year = year
        else:
            # Gap found, stop searching
            break

    # Search forwards for last year (if not already at max)
    for year in range(last_year + 1, max_year + 1):
        if is_player_active_in_year(fangraphs_id, year):
            last_year = year
        else:
            # Gap found, stop searching
            break

    return (first_year, last_year)


def get_player_age_for_year(fangraphs_id: int, year: int) -> Optional[int]:
    """
    Get a player's age for a specific year from FanGraphs stats.

    Args:
        fangraphs_id: FanGraphs player ID
        year: The season year

    Returns:
        Player's age during that season, or None if not found
    """
    try:
        # Try batting stats first
        batting = batting_stats(year, qual=0)
        if not batting.empty and 'Age' in batting.columns and 'IDfg' in batting.columns:
            player_row = batting[batting['IDfg'] == fangraphs_id]
            if not player_row.empty:
                return int(player_row['Age'].iloc[0])

        # Try pitching stats
        pitching = pitching_stats(year, qual=0)
        if not pitching.empty and 'Age' in pitching.columns and 'IDfg' in pitching.columns:
            player_row = pitching[pitching['IDfg'] == fangraphs_id]
            if not player_row.empty:
                return int(player_row['Age'].iloc[0])

    except Exception as e:
        print(f"Warning: Could not get age for FG ID {fangraphs_id} in {year}: {e}")

    return None


def clear_cache():
    """Clear the active players cache"""
    global _ACTIVE_PLAYERS_CACHE
    _ACTIVE_PLAYERS_CACHE = {}
