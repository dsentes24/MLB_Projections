from __future__ import annotations
import os
import sys
import time
import json
import uuid
import math
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import date, datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# Logging Configuration
LOG_LEVEL = os.environ.get("PROJECTIONS_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("mlb_projections.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Projections")

# Constants & Park Factors
DEFAULT_TIMEOUT = 15
HTTP_RETRIES = 3
RETRY_BACKOFF = 1.5
FIP_CONSTANT = 3.1
WOBA_WEIGHTS = {
    'bb': 0.69,
    'hbp': 0.72,
    'single': 0.89,
    'double': 1.27,
    'triple': 1.62,
    'hr': 2.10,
}

PARK_FACTORS = {
    'Angel Stadium': {'runs': 1.00, 'hr': 1.02},
    'Busch Stadium': {'runs': 0.93, 'hr': 0.84},
    'Chase Field': {'runs': 0.95, 'hr': 0.87},
    'Citi Field': {'runs': 0.89, 'hr': 1.07},
    'Citizens Bank Park': {'runs': 1.06, 'hr': 1.22},
    'Comerica Park': {'runs': 1.03, 'hr': 0.97},
    'Coors Field': {'runs': 1.27, 'hr': 1.21},
    'Dodger Stadium': {'runs': 0.92, 'hr': 1.27},
    'Fenway Park': {'runs': 1.12, 'hr': 0.97},
    'Globe Life Field': {'runs': 0.97, 'hr': 0.96},
    'Great American Ball Park': {'runs': 1.07, 'hr': 1.28},
    'Guaranteed Rate Field': {'runs': 0.98, 'hr': 1.12},
    'Kauffman Stadium': {'runs': 1.03, 'hr': 0.84},
    'loanDepot park': {'runs': 0.90, 'hr': 0.72},
    'American Family Field': {'runs': 1.06, 'hr': 1.14},
    'Minute Maid Park': {'runs': 1.03, 'hr': 1.10},
    'Nationals Park': {'runs': 1.05, 'hr': 1.09},
    'Oracle Park': {'runs': 0.93, 'hr': 0.79},
    'Oriole Park at Camden Yards': {'runs': 0.94, 'hr': 0.91},
    'Petco Park': {'runs': 0.94, 'hr': 0.98},
    'PNC Park': {'runs': 0.97, 'hr': 0.79},
    'Progressive Field': {'runs': 1.08, 'hr': 0.98},
    'Rogers Centre': {'runs': 0.96, 'hr': 1.12},
    'T-Mobile Park': {'runs': 0.94, 'hr': 1.04},
    'Target Field': {'runs': 0.93, 'hr': 0.86},
    'Truist Park': {'runs': 1.09, 'hr': 0.93},
    'Wrigley Field': {'runs': 0.99, 'hr': 0.98},
    'Yankee Stadium': {'runs': 0.98, 'hr': 1.20},
    'Sutter Health Park': {'runs': 1.19, 'hr': 1.64},
    'George M. Steinbrenner Field': {'runs': 1.02, 'hr': 1.33},
}

# Utility Helpers
def clean_player_id(player_id: Any) -> Optional[int]:
    try:
        if player_id is None or pd.isna(player_id):
            return None
        return int(float(player_id))
    except (ValueError, TypeError) as e:
        logger.error("Failed to clean player ID %s: %s", player_id, e)
        return None

def get_park_factor(venue_name: str, stat: str = 'runs') -> float:
    try:
        if venue_name in PARK_FACTORS:
            return float(PARK_FACTORS[venue_name].get(stat, 1.0))
        logger.warning("No park factor for %s (stat=%s); using 1.00", venue_name, stat)
        return 1.0
    except Exception as e:
        logger.exception("get_park_factor error: %s", e)
        return 1.0

def advanced_weather_adjustment(weather: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not isinstance(weather, dict):
        return 1.0, 1.0
    try:
        temp = float(weather.get('temperature', 70))
        wind_speed = float(weather.get('windSpeed', 0))
        wind_dir = str(weather.get('windDirection', '')).lower()
        humidity = float(weather.get('humidity', 50))
        temp_factor = 1 + (temp - 70.0) * 0.0015
        wind_factor = 1.0
        if 'out' in wind_dir:
            wind_factor += wind_speed * 0.005
        elif 'in' in wind_dir:
            wind_factor -= wind_speed * 0.005
        elif 'cross' in wind_dir:
            wind_factor += wind_speed * 0.002
        hum_factor = 1 + (humidity - 50.0) * 0.0005
        runs_adj = max(min(temp_factor * wind_factor * hum_factor, 1.5), 0.5)
        hr_adj = max(min(temp_factor * (wind_factor ** 1.2) * hum_factor, 1.5), 0.5)
        return runs_adj, hr_adj
    except Exception:
        return 1.0, 1.0

_FETCH_CACHE = {}
def _cache_key(prefix: str, **kwargs: Any) -> str:
    return prefix + "::" + json.dumps(kwargs, sort_keys=True)

def http_get_json(url: str, *, timeout: int = DEFAULT_TIMEOUT, retries: int = HTTP_RETRIES) -> Optional[Dict[str, Any]]:
    key = _cache_key("GET", url=url)
    if key in _FETCH_CACHE:
        return _FETCH_CACHE[key]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            _FETCH_CACHE[key] = data
            return data
        except Exception as e:
            last_err = e
            wait = (RETRY_BACKOFF ** (attempt - 1))
            logger.warning("GET failed (attempt %d/%d): %s; retrying in %.2fs", attempt, retries, e, wait)
            time.sleep(wait)
    logger.error("GET failed after %d attempts: %s", retries, last_err)
    return None

# Normalization & Validation
def normalize_team_column(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    candidates = ["Team", "team", "teamName", "Tm", "TEAM", "name", "Name"]
    for c in candidates:
        if c in df.columns:
            if c != "Team":
                df = df.rename(columns={c: "Team"})
            return df
    return df

def coerce_float(series: pd.Series, default: float = 0.0) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)
    except Exception:
        return pd.Series([default] * len(series), index=series.index, dtype=float)

def parse_innings_pitched(ip_str: Any) -> float:
    try:
        ip_str = str(ip_str)
        if '.' in ip_str:
            whole, frac = ip_str.split('.')
            return float(whole) + float(frac) / 3
        return float(ip_str)
    except Exception:
        return 0.0

# Fetchers
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

def fetch_mlb_pitching_stats(season: str = "2025", recent_games: int = 0) -> pd.DataFrame:
    stat_type = "gameLog" if recent_games > 0 else "season"
    limit = 100000 if recent_games > 0 else 1000
    url = f"{MLB_API_BASE}/teams/stats?season={season}&stats={stat_type}&group=pitching&sportIds=1&limit={limit}"
    logger.info("Fetching pitching stats: season=%s, recent_games=%s", season, recent_games)
    data = http_get_json(url)
    pitching_stats = []
    try:
        if recent_games > 0:
            team_splits = defaultdict(list)
            for group in (data or {}).get('stats', []):
                for split in group.get('splits', []):
                    team = split.get('team', {})
                    team_name = team.get('name', 'Unknown')
                    split['date'] = split.get('date', '0000-00-00')
                    team_splits[team_name].append(split)
            for team_name, splits in team_splits.items():
                sorted_splits = sorted(splits, key=lambda s: s['date'], reverse=True)[:recent_games]
                if not sorted_splits:
                    continue
                agg = defaultdict(float)
                agg['gamesPlayed'] = len(sorted_splits)
                for rsplit in sorted_splits:
                    stat = rsplit.get('stat', {})
                    agg['wins'] += stat.get('wins', 0)
                    agg['runs'] += stat.get('runs', 0)
                    agg['homeRuns'] += stat.get('homeRuns', 0)
                    agg['strikeOuts'] += stat.get('strikeOuts', 0)
                    agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
                    agg['hits'] += stat.get('hits', 0)
                    agg['inningsPitched'] += parse_innings_pitched(stat.get('inningsPitched', '0'))
                    agg['hitByPitch'] += stat.get('hitByPitch', 0)
                ip = agg['inningsPitched']
                era = 9 * agg['runs'] / ip if ip > 0 else 0.0
                fip = (13 * agg['homeRuns'] + 3 * (agg['baseOnBalls'] + agg['hitByPitch']) - 2 * agg['strikeOuts']) / ip + FIP_CONSTANT if ip > 0 else 0.0
                whip = (agg['hits'] + agg['baseOnBalls']) / ip if ip > 0 else 0.0
                pitching_stats.append({
                    'Team': team_name,
                    'Games': agg['gamesPlayed'],
                    'Wins': agg['wins'],
                    'RunsAllowed': agg['runs'],
                    'HomeRunsAllowed': agg['homeRuns'],
                    'ERA': era,
                    'FIP': fip,
                    'WHIP': whip,
                    'Strikeouts': agg['strikeOuts'],
                    'Walks': agg['baseOnBalls'],
                    'InningsPitched': ip,
                    'Hits': agg['hits'],
                    'HitByPitch': agg['hitByPitch'],
                })
        else:
            for group in (data or {}).get('stats', []):
                for team in group.get('splits', []):
                    team_name = team.get('team', {}).get('name', 'Unknown')
                    stats_dict = team.get('stat', {})
                    ip = parse_innings_pitched(stats_dict.get('inningsPitched', '0'))
                    runs = stats_dict.get('runs', 0)
                    hr = stats_dict.get('homeRuns', 0)
                    so = stats_dict.get('strikeOuts', 0)
                    bb = stats_dict.get('baseOnBalls', 0)
                    hbp = stats_dict.get('hitByPitch', 0)
                    hits = stats_dict.get('hits', 0)
                    era = float(stats_dict.get('era', '0').replace('-', '0') or 0)
                    whip = float(stats_dict.get('whip', '0').replace('-', '0') or 0)
                    fip = float(stats_dict.get('fip', '0').replace('-', '0') or 0)
                    if fip == 0 and ip > 0:
                        fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
                    pitching_stats.append({
                        'Team': team_name,
                        'Games': stats_dict.get('gamesPlayed', 0),
                        'Wins': stats_dict.get('wins', 0),
                        'RunsAllowed': runs,
                        'HomeRunsAllowed': hr,
                        'ERA': era,
                        'FIP': fip,
                        'WHIP': whip,
                        'Strikeouts': so,
                        'Walks': bb,
                        'InningsPitched': ip,
                        'Hits': hits,
                        'HitByPitch': hbp,
                    })
    except Exception as e:
        logger.exception("Failed parsing pitching stats: %s", e)
    columns = ['Team', 'Games', 'Wins', 'RunsAllowed', 'HomeRunsAllowed', 'ERA', 'FIP', 'WHIP', 'Strikeouts', 'Walks', 'InningsPitched', 'Hits', 'HitByPitch']
    df = pd.DataFrame(pitching_stats if pitching_stats else [], columns=columns)
    if not df.empty:
        for col in df.columns:
            if col != 'Team':
                df[col] = coerce_float(df[col], 0.0)
    else:
        logger.warning("No pitching data fetched (season=%s, recent=%s)", season, recent_games)
    return df

def fetch_mlb_team_batting_stats(season: str = "2025", recent_games: int = 0) -> pd.DataFrame:
    stat_type = "gameLog" if recent_games > 0 else "season"
    limit = 100000 if recent_games > 0 else 1000
    url = f"{MLB_API_BASE}/teams/stats?season={season}&stats={stat_type}&group=hitting&sportIds=1&limit={limit}"
    logger.info("Fetching team batting stats: season=%s, recent_games=%s", season, recent_games)
    data = http_get_json(url)
    batting_stats = []
    try:
        if recent_games > 0:
            team_splits = defaultdict(list)
            for group in (data or {}).get('stats', []):
                for split in group.get('splits', []):
                    team = split.get('team', {})
                    team_name = team.get('name', 'Unknown')
                    split['date'] = split.get('date', '0000-00-00')
                    team_splits[team_name].append(split)
            for team_name, splits in team_splits.items():
                sorted_splits = sorted(splits, key=lambda s: s['date'], reverse=True)[:recent_games]
                if not sorted_splits:
                    continue
                agg = defaultdict(float)
                agg['gamesPlayed'] = len(sorted_splits)
                for rsplit in sorted_splits:
                    stat = rsplit.get('stat', {})
                    agg['homeRuns'] += stat.get('homeRuns', 0)
                    agg['atBats'] += stat.get('atBats', 0)
                    agg['plateAppearances'] += stat.get('plateAppearances', 0)
                    agg['hits'] += stat.get('hits', 0)
                    agg['doubles'] += stat.get('doubles', 0)
                    agg['triples'] += stat.get('triples', 0)
                    agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
                    agg['hitByPitch'] += stat.get('hitByPitch', 0)
                    agg['sacFlies'] += stat.get('sacFlies', 0)
                singles = agg['hits'] - agg['doubles'] - agg['triples'] - agg['homeRuns']
                total_bases = singles + 2 * agg['doubles'] + 3 * agg['triples'] + 4 * agg['homeRuns']
                slg = total_bases / max(agg['atBats'], 1)
                obp = (agg['hits'] + agg['baseOnBalls'] + agg['hitByPitch']) / max(agg['plateAppearances'], 1)
                ops = obp + slg
                woba_num = (
                    WOBA_WEIGHTS['bb'] * agg['baseOnBalls'] +
                    WOBA_WEIGHTS['hbp'] * agg['hitByPitch'] +
                    WOBA_WEIGHTS['single'] * singles +
                    WOBA_WEIGHTS['double'] * agg['doubles'] +
                    WOBA_WEIGHTS['triple'] * agg['triples'] +
                    WOBA_WEIGHTS['hr'] * agg['homeRuns']
                )
                woba_den = agg['atBats'] + agg['baseOnBalls'] + agg['sacFlies'] + agg['hitByPitch']
                woba = woba_num / max(woba_den, 1)
                batting_stats.append({
                    'Team': team_name,
                    'Games': agg['gamesPlayed'],
                    'HomeRuns': agg['homeRuns'],
                    'AtBats': agg['atBats'],
                    'PlateAppearances': agg['plateAppearances'],
                    'wOBA': woba,
                    'SLG': slg,
                    'OPS': ops,
                    'Hits': agg['hits'],
                    'Doubles': agg['doubles'],
                    'Triples': agg['triples'],
                    'BaseOnBalls': agg['baseOnBalls'],
                    'HitByPitch': agg['hitByPitch'],
                    'SacFlies': agg['sacFlies'],
                })
        else:
            for group in (data or {}).get('stats', []):
                for team in group.get('splits', []):
                    team_name = team.get('team', {}).get('name', 'Unknown')
                    stats_dict = team.get('stat', {})
                    hits = stats_dict.get('hits', 0)
                    doubles = stats_dict.get('doubles', 0)
                    triples = stats_dict.get('triples', 0)
                    hr = stats_dict.get('homeRuns', 0)
                    singles = hits - doubles - triples - hr
                    bb = stats_dict.get('baseOnBalls', 0)
                    hbp = stats_dict.get('hitByPitch', 0)
                    sf = stats_dict.get('sacFlies', 0)
                    ab = stats_dict.get('atBats', 0)
                    pa = stats_dict.get('plateAppearances', 0)
                    woba = float(stats_dict.get('woba', 0) or 0)
                    if woba == 0 and pa > 0:
                        woba_num = (
                            WOBA_WEIGHTS['bb'] * bb +
                            WOBA_WEIGHTS['hbp'] * hbp +
                            WOBA_WEIGHTS['single'] * singles +
                            WOBA_WEIGHTS['double'] * doubles +
                            WOBA_WEIGHTS['triple'] * triples +
                            WOBA_WEIGHTS['hr'] * hr
                        )
                        woba_den = ab + bb + sf + hbp
                        woba = woba_num / max(woba_den, 1)
                    batting_stats.append({
                        'Team': team_name,
                        'Games': stats_dict.get('gamesPlayed', 0),
                        'HomeRuns': hr,
                        'AtBats': ab,
                        'PlateAppearances': pa,
                        'wOBA': woba,
                        'SLG': float(stats_dict.get('slg', 0) or 0),
                        'OPS': float(stats_dict.get('ops', 0) or 0),
                        'Hits': hits,
                        'Doubles': doubles,
                        'Triples': triples,
                        'BaseOnBalls': bb,
                        'HitByPitch': hbp,
                        'SacFlies': sf,
                    })
    except Exception as e:
        logger.exception("Failed parsing team batting stats: %s", e)
    columns = ['Team', 'Games', 'HomeRuns', 'AtBats', 'PlateAppearances', 'wOBA', 'SLG', 'OPS', 'Hits', 'Doubles', 'Triples', 'BaseOnBalls', 'HitByPitch', 'SacFlies']
    df = pd.DataFrame(batting_stats if batting_stats else [], columns=columns)
    if not df.empty:
        for col in df.columns:
            if col != 'Team':
                df[col] = coerce_float(df[col], 0.0)
    else:
        logger.warning("No team batting data fetched (season=%s, recent=%s)", season, recent_games)
    return df

def fetch_mlb_player_batting_stats(season: str = "2025", recent_games: int = 0) -> pd.DataFrame:
    stat_type = "gameLog" if recent_games > 0 else "season"
    limit = 100000 if recent_games > 0 else 1000
    url = f"{MLB_API_BASE}/stats?stats={stat_type}&group=hitting&sportId=1&season={season}&limit={limit}"
    logger.info("Fetching player batting stats: season=%s, recent_games=%s", season, recent_games)
    data = http_get_json(url)
    player_stats = []
    try:
        if recent_games > 0:
            player_splits = defaultdict(list)
            for group in (data or {}).get('stats', []):
                for split in group.get('splits', []):
                    player = split.get('player', {})
                    player_id = clean_player_id(player.get('id', None))
                    if player_id is None:
                        continue
                    split['date'] = split.get('date', '0000-00-00')
                    player_splits[player_id].append(split)
            for player_id, splits in player_splits.items():
                sorted_splits = sorted(splits, key=lambda s: s['date'], reverse=True)[:recent_games]
                if not sorted_splits:
                    continue
                team_name = sorted_splits[0].get('team', {}).get('name', 'Unknown')
                player_name = sorted_splits[0].get('player', {}).get('fullName', 'Unknown')
                agg = defaultdict(float)
                agg['gamesPlayed'] = len(sorted_splits)
                for rsplit in sorted_splits:
                    stat = rsplit.get('stat', {})
                    agg['homeRuns'] += stat.get('homeRuns', 0)
                    agg['atBats'] += stat.get('atBats', 0)
                    agg['plateAppearances'] += stat.get('plateAppearances', 0)
                    agg['hits'] += stat.get('hits', 0)
                    agg['doubles'] += stat.get('doubles', 0)
                    agg['triples'] += stat.get('triples', 0)
                    agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
                    agg['hitByPitch'] += stat.get('hitByPitch', 0)
                    agg['sacFlies'] += stat.get('sacFlies', 0)
                singles = agg['hits'] - agg['doubles'] - agg['triples'] - agg['homeRuns']
                total_bases = singles + 2 * agg['doubles'] + 3 * agg['triples'] + 4 * agg['homeRuns']
                slg = total_bases / max(agg['atBats'], 1)
                obp = (agg['hits'] + agg['baseOnBalls'] + agg['hitByPitch']) / max(agg['plateAppearances'], 1)
                ops = obp + slg
                woba_num = (
                    WOBA_WEIGHTS['bb'] * agg['baseOnBalls'] +
                    WOBA_WEIGHTS['hbp'] * agg['hitByPitch'] +
                    WOBA_WEIGHTS['single'] * singles +
                    WOBA_WEIGHTS['double'] * agg['doubles'] +
                    WOBA_WEIGHTS['triple'] * agg['triples'] +
                    WOBA_WEIGHTS['hr'] * agg['homeRuns']
                )
                woba_den = agg['atBats'] + agg['baseOnBalls'] + agg['sacFlies'] + agg['hitByPitch']
                woba = woba_num / max(woba_den, 1)
                player_stats.append({
                    'PlayerID': player_id,
                    'PlayerName': player_name,
                    'Team': team_name,
                    'Games': agg['gamesPlayed'],
                    'HomeRuns': agg['homeRuns'],
                    'AtBats': agg['atBats'],
                    'PlateAppearances': agg['plateAppearances'],
                    'wOBA': woba,
                    'SLG': slg,
                    'OPS': ops,
                    'Hits': agg['hits'],
                    'Doubles': agg['doubles'],
                    'Triples': agg['triples'],
                    'BaseOnBalls': agg['baseOnBalls'],
                    'HitByPitch': agg['hitByPitch'],
                    'SacFlies': agg['sacFlies'],
                })
        else:
            for group in (data or {}).get('stats', []):
                for split in group.get('splits', []):
                    player = split.get('player', {})
                    player_id = clean_player_id(player.get('id', None))
                    if player_id is None:
                        continue
                    team_name = split.get('team', {}).get('name', 'Unknown')
                    stats_dict = split.get('stat', {})
                    hits = stats_dict.get('hits', 0)
                    doubles = stats_dict.get('doubles', 0)
                    triples = stats_dict.get('triples', 0)
                    hr = stats_dict.get('homeRuns', 0)
                    singles = hits - doubles - triples - hr
                    bb = stats_dict.get('baseOnBalls', 0)
                    hbp = stats_dict.get('hitByPitch', 0)
                    sf = stats_dict.get('sacFlies', 0)
                    ab = stats_dict.get('atBats', 0)
                    pa = stats_dict.get('plateAppearances', 0)
                    woba = float(stats_dict.get('woba', 0) or 0)
                    if woba == 0 and pa > 0:
                        woba_num = (
                            WOBA_WEIGHTS['bb'] * bb +
                            WOBA_WEIGHTS['hbp'] * hbp +
                            WOBA_WEIGHTS['single'] * singles +
                            WOBA_WEIGHTS['double'] * doubles +
                            WOBA_WEIGHTS['triple'] * triples +
                            WOBA_WEIGHTS['hr'] * hr
                        )
                        woba_den = ab + bb + sf + hbp
                        woba = woba_num / max(woba_den, 1)
                    player_stats.append({
                        'PlayerID': player_id,
                        'PlayerName': player.get('fullName', 'Unknown'),
                        'Team': team_name,
                        'Games': stats_dict.get('gamesPlayed', 0),
                        'HomeRuns': hr,
                        'AtBats': ab,
                        'PlateAppearances': pa,
                        'wOBA': woba,
                        'SLG': float(stats_dict.get('slg', 0) or 0),
                        'OPS': float(stats_dict.get('ops', 0) or 0),
                        'Hits': hits,
                        'Doubles': doubles,
                        'Triples': triples,
                        'BaseOnBalls': bb,
                        'HitByPitch': hbp,
                        'SacFlies': sf,
                    })
    except Exception as e:
        logger.exception("Failed parsing player batting stats: %s", e)
    columns = ['PlayerID', 'PlayerName', 'Team', 'Games', 'HomeRuns', 'AtBats', 'PlateAppearances', 'wOBA', 'SLG', 'OPS', 'Hits', 'Doubles', 'Triples', 'BaseOnBalls', 'HitByPitch', 'SacFlies']
    df = pd.DataFrame(player_stats if player_stats else [], columns=columns)
    if not df.empty:
        df['PlayerID'] = pd.to_numeric(df['PlayerID'], errors='coerce').astype('Int64')
        for col in df.columns:
            if col not in ['PlayerID', 'PlayerName', 'Team']:
                df[col] = coerce_float(df[col], 0.0)
    else:
        logger.warning("No player batting data fetched (season=%s, recent=%s)", season, recent_games)
    return df

def fetch_mlb_schedule(date_str: str = None) -> pd.DataFrame:
    if date_str is None:
        date_str = date.today().strftime('%Y-%m-%d')
    logger.info("Fetching schedule for %s", date_str)
    url = f"{MLB_API_BASE}/schedule?sportId=1&date={date_str}&hydrate=team,probablePitcher(note),linescore,game,stats,venue,weather,broadcasts"
    data = http_get_json(url)
    games = []
    try:
        for date_info in (data or {}).get('dates', []):
            for g in date_info.get('games', []):
                home_team = g.get('teams', {}).get('home', {}).get('team', {}).get('name', 'Unknown')
                away_team = g.get('teams', {}).get('away', {}).get('team', {}).get('name', 'Unknown')
                venue_name = g.get('venue', {}).get('name', 'Unknown')
                weather = g.get('weather', {}) or {}
                game_status = g.get('status', {}).get('detailedState', 'Scheduled')  # Use detailedState for more accuracy
                home_score = g.get('teams', {}).get('home', {}).get('score', 0) or 0
                away_score = g.get('teams', {}).get('away', {}).get('score', 0) or 0
                actual_winner = None
                if game_status in ['Final', 'Game Over', 'Completed Early']:
                    actual_winner = home_team if home_score > away_score else away_team if away_score > home_score else 'Tie'
                home_pitcher_id = clean_player_id(g.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('id', None))
                away_pitcher_id = clean_player_id(g.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('id', None))
                games.append({
                    'GameID': g.get('gamePk'),
                    'Date': date_str,
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'HomePitcher': g.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('fullName', 'TBD'),
                    'HomePitcherID': home_pitcher_id,
                    'AwayPitcher': g.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('fullName', 'TBD'),
                    'AwayPitcherID': away_pitcher_id,
                    'Venue': venue_name,
                    'Weather': weather,
                    'GameStatus': game_status,
                    'ActualWinner': actual_winner,
                    'HomeScore': home_score,
                    'AwayScore': away_score,
                })
    except Exception as e:
        logger.exception("Failed parsing schedule: %s", e)
    columns = ['GameID', 'Date', 'HomeTeam', 'AwayTeam', 'HomePitcher', 'HomePitcherID', 'AwayPitcher', 'AwayPitcherID', 'Venue', 'Weather', 'GameStatus', 'ActualWinner', 'HomeScore', 'AwayScore']
    df = pd.DataFrame(games if games else [], columns=columns)
    if not df.empty:
        df['GameID'] = pd.to_numeric(df['GameID'], errors='coerce').astype('Int64')
        df['HomePitcherID'] = pd.to_numeric(df['HomePitcherID'], errors='coerce').astype('Int64')
        df['AwayPitcherID'] = pd.to_numeric(df['AwayPitcherID'], errors='coerce').astype('Int64')
    else:
        logger.warning("Fetched 0 games for %s", date_str)
    return df

def fetch_game_player_stats(game_id: int, date_str: str) -> pd.DataFrame:
    logger.info("Fetching player stats for game %s on %s", game_id, date_str)
    url = f"{MLB_API_BASE}/game/{game_id}/boxscore"
    data = http_get_json(url)
    player_stats = []
    try:
        teams = (data or {}).get('teams', {})
        for team_type in ['home', 'away']:
            team_data = teams.get(team_type, {})
            team_name = team_data.get('team', {}).get('name', 'Unknown')
            players = team_data.get('players', {}) or {}
            for player_id_str, pdata in players.items():
                player_id = clean_player_id(player_id_str.replace('ID', '') if player_id_str.startswith('ID') else player_id_str)
                if player_id is None:
                    continue
                batting = pdata.get('stats', {}).get('batting', {}) or {}
                ab = batting.get('atBats', 0)
                if ab > 0:
                    player_stats.append({
                        'PlayerID': player_id,
                        'PlayerName': pdata.get('person', {}).get('fullName', 'Unknown'),
                        'Team': team_name,
                        'HomeRuns': batting.get('homeRuns', 0),
                        'Hits': batting.get('hits', 0),
                        'GameID': game_id,
                    })
    except Exception as e:
        logger.exception("Failed parsing game player stats for game %s: %s", game_id, e)
    columns = ['PlayerID', 'PlayerName', 'Team', 'HomeRuns', 'Hits', 'GameID']
    df = pd.DataFrame(player_stats if player_stats else [], columns=columns)
    if not df.empty:
        df['PlayerID'] = pd.to_numeric(df['PlayerID'], errors='coerce').astype('Int64')
        df['GameID'] = pd.to_numeric(df['GameID'], errors='coerce').astype('Int64')
    if df.empty:
        logger.warning("No player stats fetched for game %s", game_id)
    return df

def fetch_game_boxscore(game_id: int) -> Optional[Dict[str, Any]]:
    logger.info("Fetching linescore for game %s", game_id)
    url = f"{MLB_API_BASE}/schedule?gamePk={game_id}&hydrate=linescore"
    data = http_get_json(url)
    if not data or not data.get('dates'):
        logger.error("No linescore data for game %s", game_id)
        return None
    try:
        game = data['dates'][0]['games'][0]
        game_status = game.get('status', {}).get('detailedState', 'Scheduled')
        home_score = game.get('teams', {}).get('home', {}).get('score', 0) or 0
        away_score = game.get('teams', {}).get('away', {}).get('score', 0) or 0
        actual_winner = None
        if game_status in ['Final', 'Game Over', 'Completed Early']:
            if home_score > away_score:
                actual_winner = 'Home'
            elif away_score > home_score:
                actual_winner = 'Away'
            else:
                actual_winner = 'Tie'
        return {
            'GameStatus': game_status,
            'HomeScore': home_score,
            'AwayScore': away_score,
            'ActualWinner': actual_winner,
        }
    except Exception as e:
        logger.exception("Failed parsing linescore for game %s: %s", game_id, e)
        return None

def fetch_pitcher_stats(pitcher_id: Optional[int], season: str = "2025", recent_games: int = 0) -> Optional[Dict[str, Any]]:
    pitcher_id = clean_player_id(pitcher_id)
    if pitcher_id is None:
        return None
    stat_type = 'gameLog' if recent_games > 0 else 'season'
    limit = 100 if recent_games > 0 else 1
    url = f"{MLB_API_BASE}/people/{pitcher_id}/stats?stats={stat_type}&group=pitching&season={season}&limit={limit}"
    logger.info("Fetching pitcher stats for ID %s (season=%s, recent_games=%s)", pitcher_id, season, recent_games)
    data = http_get_json(url)
    try:
        splits = []
        for group in (data or {}).get('stats', []):
            splits.extend(group.get('splits', []))
        if recent_games > 0:
            splits = sorted(splits, key=lambda s: s.get('date', '0000-00-00'), reverse=True)[:recent_games]
            if not splits:
                return None
            agg = defaultdict(float)
            agg['gamesPlayed'] = len(splits)
            for split in splits:
                stat = split.get('stat', {})
                agg['inningsPitched'] += parse_innings_pitched(stat.get('inningsPitched', '0'))
                agg['homeRuns'] += stat.get('homeRuns', 0)
                agg['strikeOuts'] += stat.get('strikeOuts', 0)
                agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
                agg['hitByPitch'] += stat.get('hitByPitch', 0)
            ip = agg['inningsPitched']
            hr = agg['homeRuns']
            so = agg['strikeOuts']
            bb = agg['baseOnBalls']
            hbp = agg['hitByPitch']
            fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT if ip > 0 else 0.0
            return {
                'PitcherID': pitcher_id,
                'Games': agg['gamesPlayed'],
                'InningsPitched': ip,
                'HomeRunsAllowed': hr,
                'Strikeouts': so,
                'Walks': bb,
                'FIP': fip,
            }
        else:
            if not splits:
                return None
            stats_dict = splits[0].get('stat', {})
            ip = parse_innings_pitched(stats_dict.get('inningsPitched', '0'))
            hr = stats_dict.get('homeRuns', 0)
            so = stats_dict.get('strikeOuts', 0)
            bb = stats_dict.get('baseOnBalls', 0)
            hbp = stats_dict.get('hitByPitch', 0)
            fip = float(stats_dict.get('fip', 0) or 0)
            if fip == 0 and ip > 0:
                fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
            return {
                'PitcherID': pitcher_id,
                'Games': stats_dict.get('gamesPlayed', 0),
                'InningsPitched': ip,
                'HomeRunsAllowed': hr,
                'Strikeouts': so,
                'Walks': bb,
                'FIP': fip,
            }
    except Exception as e:
        logger.exception("Failed parsing pitcher stats for ID %s: %s", pitcher_id, e)
    return None

def fetch_all_pitcher_stats(pitcher_ids: Sequence[int], season: str = "2025", recent_games: int = 0) -> Dict[int, Optional[Dict[str, Any]]]:
    pitcher_ids = list(set([pid for pid in pitcher_ids if pid is not None]))
    if not pitcher_ids:
        return {}
    logger.info("Fetching stats for %d pitchers in parallel", len(pitcher_ids))
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_pitcher_stats, pid, season, recent_games): pid for pid in pitcher_ids}
        results = {}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                results[pid] = future.result()
            except Exception as e:
                logger.error("Failed to fetch stats for pitcher %s: %s", pid, e)
                results[pid] = None
    return results

# Modeling Utilities
def blend_value(base: float, recent: Optional[float], weight_recent: float = 0.3) -> float:
    if recent is not None and np.isfinite(recent):
        return (1 - weight_recent) * base + weight_recent * recent
    return base

def safe_ols_two_row(
    home_woba: float, away_woba: float,
    home_era: float, away_era: float,
    home_fip: float, away_fip: float,
    home_whip: float, away_whip: float,
    home_pitcher_fip: Optional[float], away_pitcher_fip: Optional[float],
    home_runs_allowed: float, away_runs_allowed: float,
    park_runs: float, runs_adj: float
) -> Tuple[float, float]:
    try:
        home_woba = max(min(home_woba, 0.450), 0.200)
        away_woba = max(min(away_woba, 0.450), 0.200)
        home_era = max(min(home_era, 7.0), 2.0)
        away_era = max(min(away_era, 7.0), 2.0)
        home_fip = max(min(home_fip, 7.0), 2.0)
        away_fip = max(min(away_fip, 7.0), 2.0)
        home_whip = max(min(home_whip, 2.0), 0.8)
        away_whip = max(min(away_whip, 2.0), 0.8)
        home_pitcher_fip = max(min(home_pitcher_fip, 7.0), 2.0) if home_pitcher_fip is not None else home_fip
        away_pitcher_fip = max(min(away_pitcher_fip, 7.0), 2.0) if away_pitcher_fip is not None else away_fip
        home_runs_allowed = max(min(home_runs_allowed, 1.0), 0.0)
        away_runs_allowed = max(min(away_runs_allowed, 1.0), 0.0)
        park_runs = max(min(park_runs, 1.5), 0.5)
        runs_adj = max(min(runs_adj, 1.5), 0.5)

        logger.debug(
            "OLS Inputs: home_woba=%.3f, away_woba=%.3f, home_era=%.3f, away_era=%.3f, "
            "home_fip=%.3f, away_fip=%.3f, home_whip=%.3f, away_whip=%.3f, "
            "home_pitcher_fip=%.3f, away_pitcher_fip=%.3f, home_runs_allowed=%.3f, away_runs_allowed=%.3f, "
            "park_runs=%.3f, runs_adj=%.3f",
            home_woba, away_woba, home_era, away_era, home_fip, away_fip, home_whip, away_whip,
            home_pitcher_fip, away_pitcher_fip, home_runs_allowed, away_runs_allowed, park_runs, runs_adj
        )

        X = np.array([
            [home_woba, away_era, away_fip, away_whip, away_pitcher_fip, away_runs_allowed, park_runs * runs_adj],
            [away_woba, home_era, home_fip, home_whip, home_pitcher_fip, home_runs_allowed, park_runs * runs_adj],
        ], dtype=float)
        y = np.array([4.5, 4.5], dtype=float)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        home_pred = model.predict([1.0, home_woba, away_era, away_fip, away_whip, away_pitcher_fip, away_runs_allowed, park_runs * runs_adj])[0]
        away_pred = model.predict([1.0, away_woba, home_era, home_fip, home_whip, home_pitcher_fip, home_runs_allowed, park_runs * runs_adj])[0]

        logger.debug("OLS Predictions: home_pred=%.3f, away_pred=%.3f", home_pred, away_pred)

        if not np.isfinite(home_pred) or not np.isfinite(away_pred):
            raise ValueError("Non-finite predictions")
        return max(home_pred, 0.5), max(away_pred, 0.5)
    except Exception as e:
        logger.debug("OLS failed: %s", e)
        woba_diff = home_woba - away_woba
        era_diff = away_era - home_era
        fip_diff = (away_pitcher_fip or away_fip) - (home_pitcher_fip or home_fip)
        whip_diff = away_whip - home_whip
        runs_allowed_diff = away_runs_allowed - home_runs_allowed
        park_adjust = park_runs * runs_adj
        home_pred = 4.5 + (10.0 * woba_diff + 0.4 * era_diff + 0.2 * fip_diff + 0.3 * whip_diff + 0.5 * runs_allowed_diff) * park_adjust
        away_pred = 4.5 - (10.0 * woba_diff + 0.4 * era_diff + 0.2 * fip_diff + 0.3 * whip_diff + 0.5 * runs_allowed_diff) * park_adjust

        logger.debug(
            "Fallback Predictions: home_pred=%.3f, away_pred=%.3f, woba_diff=%.3f, era_diff=%.3f, "
            "fip_diff=%.3f, whip_diff=%.3f, runs_allowed_diff=%.3f, park_adjust=%.3f",
            home_pred, away_pred, woba_diff, era_diff, fip_diff, whip_diff, runs_allowed_diff, park_adjust
        )

        return max(home_pred, 0.5), max(away_pred, 0.5)

# Projections — Game Outcomes
def project_game_outcomes(
    schedule_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
    team_batting_df: pd.DataFrame,
    recent_pitching_df: pd.DataFrame,
    recent_team_batting_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Generating advanced game projections")
    projections = []
    pitcher_stats_dict = fetch_all_pitcher_stats(pd.concat([schedule_df['HomePitcherID'], schedule_df['AwayPitcherID']]).unique())

    for _, game in schedule_df.iterrows():
        try:
            home_team = game['HomeTeam']
            away_team = game['AwayTeam']
            venue = game['Venue']
            weather = game['Weather']
            game_id = game['GameID']
            home_pitcher_id = game['HomePitcherID']
            away_pitcher_id = game['AwayPitcherID']

            # Check linescore for updated game status and scores
            boxscore = fetch_game_boxscore(game_id)
            game_status = game['GameStatus']
            home_score = game.get('HomeScore')
            away_score = game.get('AwayScore')
            actual_winner = game.get('ActualWinner')
            if boxscore:
                home_score = boxscore['HomeScore']
                away_score = boxscore['AwayScore']
                actual_winner = boxscore['ActualWinner']
                if actual_winner == 'Home':
                    actual_winner = home_team
                elif actual_winner == 'Away':
                    actual_winner = away_team

            home_pitching = pitching_df[pitching_df['Team'] == home_team].iloc[0] if not pitching_df[pitching_df['Team'] == home_team].empty else None
            away_pitching = pitching_df[pitching_df['Team'] == away_team].iloc[0] if not pitching_df[pitching_df['Team'] == away_team].empty else None
            home_batting = team_batting_df[team_batting_df['Team'] == home_team].iloc[0] if not team_batting_df[team_batting_df['Team'] == home_team].empty else None
            away_batting = team_batting_df[team_batting_df['Team'] == away_team].iloc[0] if not team_batting_df[team_batting_df['Team'] == away_team].empty else None

            if any(x is None for x in [home_pitching, away_pitching, home_batting, away_batting]):
                logger.warning("Skipping game %s: missing data for %s vs %s", game_id, home_team, away_team)
                continue

            home_era = blend_value(home_pitching['ERA'], recent_pitching_df[recent_pitching_df['Team'] == home_team]['ERA'].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == home_team].empty else None)
            away_era = blend_value(away_pitching['ERA'], recent_pitching_df[recent_pitching_df['Team'] == away_team]['ERA'].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == away_team].empty else None)
            home_fip = blend_value(home_pitching['FIP'], recent_pitching_df[recent_pitching_df['Team'] == home_team]['FIP'].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == home_team].empty else None)
            away_fip = blend_value(away_pitching['FIP'], recent_pitching_df[recent_pitching_df['Team'] == away_team]['FIP'].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == away_team].empty else None)
            home_whip = blend_value(home_pitching['WHIP'], recent_pitching_df[recent_pitching_df['Team'] == home_team]['WHIP'].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == home_team].empty else None)
            away_whip = blend_value(away_pitching['WHIP'], recent_pitching_df[recent_pitching_df['Team'] == away_team]['WHIP'].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == away_team].empty else None)
            home_woba = blend_value(home_batting['wOBA'], recent_team_batting_df[recent_team_batting_df['Team'] == home_team]['wOBA'].iloc[0] if not recent_team_batting_df[recent_team_batting_df['Team'] == home_team].empty else None)
            away_woba = blend_value(away_batting['wOBA'], recent_team_batting_df[recent_team_batting_df['Team'] == away_team]['wOBA'].iloc[0] if not recent_team_batting_df[recent_team_batting_df['Team'] == away_team].empty else None)
            home_runs_allowed = home_pitching['RunsAllowed'] / max(home_pitching['InningsPitched'], 1)
            away_runs_allowed = away_pitching['RunsAllowed'] / max(away_pitching['InningsPitched'], 1)

            home_pitcher_stats = pitcher_stats_dict.get(home_pitcher_id)
            away_pitcher_stats = pitcher_stats_dict.get(away_pitcher_id)
            home_pitcher_fip = home_pitcher_stats['FIP'] if home_pitcher_stats and home_pitcher_stats['InningsPitched'] > 0 else None
            away_pitcher_fip = away_pitcher_stats['FIP'] if away_pitcher_stats and away_pitcher_stats['InningsPitched'] > 0 else None

            park_runs = get_park_factor(venue, 'runs')
            runs_adj, _ = advanced_weather_adjustment(weather)

            home_pred, away_pred = safe_ols_two_row(
                home_woba, away_woba, home_era, away_era, home_fip, away_fip, home_whip, away_whip,
                home_pitcher_fip, away_pitcher_fip, home_runs_allowed, away_runs_allowed, park_runs, runs_adj
            )
            home_expected_runs = home_pred * park_runs * runs_adj
            away_expected_runs = away_pred * park_runs * runs_adj

            exp = 1.83
            home_win_prob = (home_expected_runs ** exp) / (home_expected_runs ** exp + away_expected_runs ** exp)
            home_win_prob = min(max(home_win_prob + 0.035 + np.random.uniform(-0.02, 0.02), 0.1), 0.9)
            away_win_prob = 1 - home_win_prob
            prob_diff = abs(home_win_prob - away_win_prob)

            projected_winner = home_team if home_win_prob >= 0.5 else away_team
            projection_status = None
            if actual_winner is not None and actual_winner != 'Tie':
                projection_status = 'Correct' if projected_winner == actual_winner else 'Incorrect'
            elif game_status == 'In Progress':
                if home_score > away_score:
                    projection_status = 'Leading' if projected_winner == home_team else 'Trailing'
                elif away_score > home_score:
                    projection_status = 'Leading' if projected_winner == away_team else 'Trailing'
                else:
                    projection_status = 'Tied'

            projections.append({
                'GameID': game_id,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'HomeWinProb': round(home_win_prob * 100, 1),
                'AwayWinProb': round(away_win_prob * 100, 1),
                'ProbDiff': prob_diff,
                'ProjectedWinner': projected_winner,
                'HomeExpectedRuns': round(home_expected_runs, 1),
                'AwayExpectedRuns': round(away_expected_runs, 1),
                'TotalExpectedRuns': round(home_expected_runs + away_expected_runs, 1),
                'GameStatus': game_status,
                'ProjectionStatus': projection_status,
                'HomeScore': home_score,
                'AwayScore': away_score,
            })
        except Exception as e:
            logger.exception("Error projecting game %s: %s", game_id, e)
    columns = ['GameID', 'HomeTeam', 'AwayTeam', 'HomeWinProb', 'AwayWinProb', 'ProbDiff', 'ProjectedWinner', 'HomeExpectedRuns', 'AwayExpectedRuns', 'TotalExpectedRuns', 'GameStatus', 'ProjectionStatus', 'HomeScore', 'AwayScore']
    df = pd.DataFrame(projections if projections else [], columns=columns)
    df['GameID'] = pd.to_numeric(df['GameID'], errors='coerce').astype('Int64')
    # Sort by probability difference (largest to smallest)
    df = df.sort_values(by='ProbDiff', ascending=False)
    return df

# Projections — Home Run Hitters
def project_home_run_hitters(
    schedule_df: pd.DataFrame,
    player_batting_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
    recent_player_batting_df: pd.DataFrame,
    recent_pitching_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Generating advanced HR projections")
    projections = []
    pitcher_stats_dict = fetch_all_pitcher_stats(pd.concat([schedule_df['HomePitcherID'], schedule_df['AwayPitcherID']]).unique())

    for _, game in schedule_df.iterrows():
        try:
            home_team = game['HomeTeam']
            away_team = game['AwayTeam']
            home_pitcher_id = game['HomePitcherID']
            away_pitcher_id = game['AwayPitcherID']
            game_id = game['GameID']
            venue = game['Venue']
            weather = game['Weather']
            date_str = game['Date']

            # Check linescore for updated game status
            boxscore = fetch_game_boxscore(game_id)
            game_status = game['GameStatus']
            if boxscore:
                game_status = boxscore['GameStatus']

            actual_stats = pd.DataFrame()
            if game_status != 'Scheduled':
                actual_stats = fetch_game_player_stats(game_id, date_str)

            home_pitcher_stats = pitcher_stats_dict.get(home_pitcher_id)
            away_pitcher_stats = pitcher_stats_dict.get(away_pitcher_id)

            park_hr = get_park_factor(venue, 'hr')
            _, hr_adj = advanced_weather_adjustment(weather)

            for team, opp_pitcher_stats, opp_pitcher_id, opp_team in [
                (home_team, away_pitcher_stats, away_pitcher_id, away_team),
                (away_team, home_pitcher_stats, home_pitcher_id, home_team),
            ]:
                team_batting = player_batting_df[player_batting_df['Team'] == team] if 'Team' in player_batting_df.columns else pd.DataFrame()
                if team_batting.empty:
                    continue

                team_recent_b = recent_player_batting_df[recent_player_batting_df['Team'] == team] if 'Team' in recent_player_batting_df.columns else pd.DataFrame()

                opp_pitching = pitching_df[pitching_df['Team'] == opp_team].iloc[0] if not pitching_df[pitching_df['Team'] == opp_team].empty else {'HomeRunsAllowed': 0, 'Games': 1}
                opp_recent_p = recent_pitching_df[recent_pitching_df['Team'] == opp_team].iloc[0] if not recent_pitching_df[recent_pitching_df['Team'] == opp_team].empty else None

                hr_allowed_pi_base = opp_pitching['HomeRunsAllowed'] / max(opp_pitching['Games'], 1) / 9.0
                hr_allowed_pi = blend_value(hr_allowed_pi_base, opp_recent_p['HomeRunsAllowed'] / max(opp_recent_p['Games'], 1) / 9.0, 0.4) if opp_recent_p is not None else hr_allowed_pi_base

                if opp_pitcher_stats and opp_pitcher_stats['InningsPitched'] > 0:
                    hr_allowed_pi = blend_value(hr_allowed_pi, opp_pitcher_stats['HomeRunsAllowed'] / opp_pitcher_stats['InningsPitched'], 0.6)

                tb = team_batting.copy()
                tb['HRRate'] = tb['HomeRuns'] / tb['PlateAppearances'].clip(lower=1)

                if not team_recent_b.empty:
                    recent_pa = team_recent_b['PlateAppearances'].sum()
                    recent_weight = min(recent_pa / 200, 0.5)
                    recent_hr_rate = team_recent_b['HomeRuns'].sum() / team_recent_b['PlateAppearances'].clip(lower=1).sum()
                    tb['HRRate'] = (1 - recent_weight) * tb['HRRate'] + recent_weight * recent_hr_rate

                tb['ExpectedHR'] = tb['HRRate'] * 4.2 * (1 + (hr_allowed_pi - 0.1)) * park_hr * hr_adj
                lam = tb['ExpectedHR'].clip(lower=1e-6)
                tb['HRProb'] = 1 - stats.poisson.pmf(0, lam)

                if tb.empty or 'HRProb' not in tb.columns:
                    continue
                tb = tb[tb['PlateAppearances'] >= 50]
                if tb.empty:
                    continue
                top_idx = tb["HRProb"].idxmax()
                if pd.isna(top_idx):
                    continue
                top_player = tb.loc[top_idx]

                projection_status = None
                if not actual_stats.empty:
                    player_actual = actual_stats[actual_stats['PlayerID'] == top_player['PlayerID']]
                    if game_status in ['Final', 'Game Over', 'Completed Early']:
                        projection_status = 'Correct' if not player_actual.empty and player_actual['HomeRuns'].iloc[0] > 0 else 'Incorrect'
                    elif game_status == 'In Progress':
                        projection_status = 'Hit So Far' if not player_actual.empty and player_actual['HomeRuns'].iloc[0] > 0 else 'Not Yet'

                projections.append({
                    'GameID': game_id,
                    'PlayerName': top_player['PlayerName'],
                    'Team': team,
                    'HRProb': round(top_player['HRProb'] * 100, 1),
                    'ProjectionStatus': projection_status,
                })
        except Exception as e:
            logger.exception("HR projection error for game %s: %s", game_id, e)
    columns = ['GameID', 'PlayerName', 'Team', 'HRProb', 'ProjectionStatus']
    df = pd.DataFrame(projections if projections else [], columns=columns)
    df['GameID'] = pd.to_numeric(df['GameID'], errors='coerce').astype('Int64')
    return df

# Backtesting
def backtest_game_projections(projections_df: pd.DataFrame) -> Dict[str, float]:
    valid = projections_df[projections_df['ProjectionStatus'].isin(['Correct', 'Incorrect'])]
    if valid.empty:
        return {"brier_score": float('nan'), "accuracy": 0.0}
    actual_home_win = (valid['ProjectionStatus'] == 'Correct').astype(int)
    predicted_prob = valid['HomeWinProb'] / 100.0
    brier_score = np.mean((predicted_prob - actual_home_win) ** 2)
    accuracy = np.mean(valid['ProjectionStatus'] == 'Correct')
    return {"brier_score": brier_score, "accuracy": accuracy}

def backtest_hr_projections(hr_projections_df: pd.DataFrame) -> Dict[str, float]:
    valid = hr_projections_df[hr_projections_df['ProjectionStatus'].isin(['Correct', 'Incorrect'])]
    if valid.empty:
        return {"brier_score": float('nan'), "hit_rate": 0.0}
    actual = (valid['ProjectionStatus'] == 'Correct').astype(int)
    predicted_prob = valid['HRProb'] / 100.0
    brier_score = np.mean((predicted_prob - actual) ** 2)
    hit_rate = np.mean(actual)
    return {"brier_score": brier_score, "hit_rate": hit_rate}

# Aggregate functions (add these)
def aggregate_team_pitching_stats(gamelog_data: Dict[str, Any], as_of: Optional[str] = None, recent_games: int = 0) -> pd.DataFrame:
    team_splits = defaultdict(list)
    for group in (gamelog_data or {}).get('stats', []):
        for split in group.get('splits', []):
            team_name = split.get('team', {}).get('name', 'Unknown')
            split_date = split.get('date', '0000-00-00')
            if as_of and split_date >= as_of:
                continue
            team_splits[team_name].append(split)
    pitching_stats = []
    for team_name, splits in team_splits.items():
        sorted_splits = sorted(splits, key=lambda s: s.get('date', '0000-00-00'), reverse=True)
        if recent_games > 0:
            sorted_splits = sorted_splits[:recent_games]
        if not sorted_splits:
            continue
        agg = defaultdict(float)
        agg['gamesPlayed'] = len(sorted_splits)
        for rsplit in sorted_splits:
            stat = rsplit.get('stat', {})
            agg['wins'] += stat.get('wins', 0)
            agg['runs'] += stat.get('runs', 0)
            agg['homeRuns'] += stat.get('homeRuns', 0)
            agg['strikeOuts'] += stat.get('strikeOuts', 0)
            agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
            agg['hits'] += stat.get('hits', 0)
            agg['inningsPitched'] += parse_innings_pitched(stat.get('inningsPitched', '0'))
            agg['hitByPitch'] += stat.get('hitByPitch', 0)
        ip = agg['inningsPitched']
        era = 9 * agg['runs'] / ip if ip > 0 else 0.0
        fip = (13 * agg['homeRuns'] + 3 * (agg['baseOnBalls'] + agg['hitByPitch']) - 2 * agg['strikeOuts']) / ip + FIP_CONSTANT if ip > 0 else 0.0
        whip = (agg['hits'] + agg['baseOnBalls']) / ip if ip > 0 else 0.0
        pitching_stats.append({
            'Team': team_name,
            'Games': agg['gamesPlayed'],
            'Wins': agg['wins'],
            'RunsAllowed': agg['runs'],
            'HomeRunsAllowed': agg['homeRuns'],
            'ERA': era,
            'FIP': fip,
            'WHIP': whip,
            'Strikeouts': agg['strikeOuts'],
            'Walks': agg['baseOnBalls'],
            'InningsPitched': ip,
            'Hits': agg['hits'],
            'HitByPitch': agg['hitByPitch'],
        })
    df = pd.DataFrame(pitching_stats)
    for col in df.columns:
        if col != 'Team':
            df[col] = coerce_float(df[col])
    return df

def aggregate_team_batting_stats(gamelog_data: Dict[str, Any], as_of: Optional[str] = None, recent_games: int = 0) -> pd.DataFrame:
    team_splits = defaultdict(list)
    for group in (gamelog_data or {}).get('stats', []):
        for split in group.get('splits', []):
            team_name = split.get('team', {}).get('name', 'Unknown')
            split_date = split.get('date', '0000-00-00')
            if as_of and split_date >= as_of:
                continue
            team_splits[team_name].append(split)
    batting_stats = []
    for team_name, splits in team_splits.items():
        sorted_splits = sorted(splits, key=lambda s: s.get('date', '0000-00-00'), reverse=True)
        if recent_games > 0:
            sorted_splits = sorted_splits[:recent_games]
        if not sorted_splits:
            continue
        agg = defaultdict(float)
        agg['gamesPlayed'] = len(sorted_splits)
        for rsplit in sorted_splits:
            stat = rsplit.get('stat', {})
            agg['homeRuns'] += stat.get('homeRuns', 0)
            agg['atBats'] += stat.get('atBats', 0)
            agg['plateAppearances'] += stat.get('plateAppearances', 0)
            agg['hits'] += stat.get('hits', 0)
            agg['doubles'] += stat.get('doubles', 0)
            agg['triples'] += stat.get('triples', 0)
            agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
            agg['hitByPitch'] += stat.get('hitByPitch', 0)
            agg['sacFlies'] += stat.get('sacFlies', 0)
        singles = agg['hits'] - agg['doubles'] - agg['triples'] - agg['homeRuns']
        total_bases = singles + 2 * agg['doubles'] + 3 * agg['triples'] + 4 * agg['homeRuns']
        slg = total_bases / max(agg['atBats'], 1)
        obp = (agg['hits'] + agg['baseOnBalls'] + agg['hitByPitch']) / max(agg['plateAppearances'], 1)
        ops = obp + slg
        woba_num = (
            WOBA_WEIGHTS['bb'] * agg['baseOnBalls'] +
            WOBA_WEIGHTS['hbp'] * agg['hitByPitch'] +
            WOBA_WEIGHTS['single'] * singles +
            WOBA_WEIGHTS['double'] * agg['doubles'] +
            WOBA_WEIGHTS['triple'] * agg['triples'] +
            WOBA_WEIGHTS['hr'] * agg['homeRuns']
        )
        woba_den = agg['atBats'] + agg['baseOnBalls'] + agg['sacFlies'] + agg['hitByPitch']
        woba = woba_num / max(woba_den, 1)
        batting_stats.append({
            'Team': team_name,
            'Games': agg['gamesPlayed'],
            'HomeRuns': agg['homeRuns'],
            'AtBats': agg['atBats'],
            'PlateAppearances': agg['plateAppearances'],
            'wOBA': woba,
            'SLG': slg,
            'OPS': ops,
            'Hits': agg['hits'],
            'Doubles': agg['doubles'],
            'Triples': agg['triples'],
            'BaseOnBalls': agg['baseOnBalls'],
            'HitByPitch': agg['hitByPitch'],
            'SacFlies': agg['sacFlies'],
        })
    df = pd.DataFrame(batting_stats)
    for col in df.columns:
        if col != 'Team':
            df[col] = coerce_float(df[col])
    return df

def aggregate_pitcher_stats(player_id: int, gamelog_data: Dict[str, Any], as_of: Optional[str] = None, recent_games: int = 0) -> Optional[Dict[str, Any]]:
    splits = []
    for group in (gamelog_data or {}).get('stats', []):
        for split in group.get('splits', []):
            pid = clean_player_id(split.get('player', {}).get('id'))
            if pid != player_id:
                continue
            split_date = split.get('date', '0000-00-00')
            if as_of and split_date >= as_of:
                continue
            splits.append(split)
    if not splits:
        return None
    sorted_splits = sorted(splits, key=lambda s: s.get('date', '0000-00-00'), reverse=True)
    if recent_games > 0:
        sorted_splits = sorted_splits[:recent_games]
    if not sorted_splits:
        return None
    agg = defaultdict(float)
    agg['gamesPlayed'] = len(sorted_splits)
    for split in sorted_splits:
        stat = split.get('stat', {})
        agg['inningsPitched'] += parse_innings_pitched(stat.get('inningsPitched', '0'))
        agg['homeRuns'] += stat.get('homeRuns', 0)
        agg['strikeOuts'] += stat.get('strikeOuts', 0)
        agg['baseOnBalls'] += stat.get('baseOnBalls', 0)
        agg['hitByPitch'] += stat.get('hitByPitch', 0)
    ip = agg['inningsPitched']
    hr = agg['homeRuns']
    so = agg['strikeOuts']
    bb = agg['baseOnBalls']
    hbp = agg['hitByPitch']
    fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT if ip > 0 else 0.0
    return {
        'PitcherID': player_id,
        'Games': agg['gamesPlayed'],
        'InningsPitched': ip,
        'HomeRunsAllowed': hr,
        'Strikeouts': so,
        'Walks': bb,
        'FIP': fip,
    }

def fetch_all_gamelogs(season: str = "2025") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    team_pitching_url = f"{MLB_API_BASE}/teams/stats?season={season}&stats=gameLog&group=pitching&sportIds=1&limit=100000"
    team_batting_url = f"{MLB_API_BASE}/teams/stats?season={season}&stats=gameLog&group=hitting&sportIds=1&limit=100000"
    player_pitching_url = f"{MLB_API_BASE}/stats?stats=gameLog&group=pitching&sportId=1&season={season}&limit=100000"
    team_pitching_gamelog = http_get_json(team_pitching_url)
    team_batting_gamelog = http_get_json(team_batting_url)
    player_pitching_gamelog = http_get_json(player_pitching_url)
    return team_pitching_gamelog, team_batting_gamelog, player_pitching_gamelog

# The backtest_season function (modified to return results)
def backtest_season(season: str = "2025", recent_games: int = 30) -> Dict:
    start_date = date(2025, 3, 18)  # Adjust if season start is different
    end_date = date.today() - timedelta(days=1)
    team_pitching_gamelog, team_batting_gamelog, player_pitching_gamelog = fetch_all_gamelogs(season)
    all_projections = []
    dates = pd.date_range(start=start_date, end=end_date)
    for d in dates:
        date_str = d.strftime('%Y-%m-%d')
        schedule_df = fetch_mlb_schedule(date_str)
        if schedule_df.empty:
            continue
        pitching_df = aggregate_team_pitching_stats(team_pitching_gamelog, as_of=date_str, recent_games=0)
        recent_pitching_df = aggregate_team_pitching_stats(team_pitching_gamelog, as_of=date_str, recent_games=recent_games)
        team_batting_df = aggregate_team_batting_stats(team_batting_gamelog, as_of=date_str, recent_games=0)
        recent_team_batting_df = aggregate_team_batting_stats(team_batting_gamelog, as_of=date_str, recent_games=recent_games)
        pitcher_ids = pd.concat([schedule_df['HomePitcherID'], schedule_df['AwayPitcherID']]).dropna().unique()
        pitcher_stats_dict = {}
        for pid in pitcher_ids:
            stats = aggregate_pitcher_stats(int(pid), player_pitching_gamelog, as_of=date_str, recent_games=0)
            if stats:
                pitcher_stats_dict[pid] = stats
        projections_df = project_game_outcomes(
            schedule_df, pitching_df, team_batting_df, recent_pitching_df, recent_team_batting_df
        )
        valid_df = projections_df[projections_df['GameStatus'].isin(['Final', 'Game Over', 'Completed Early'])]
        all_projections.append(valid_df)
    full_df = pd.concat(all_projections, ignore_index=True)
    full_df['FavoriteProb'] = full_df[['HomeWinProb', 'AwayWinProb']].max(axis=1)
    full_df['Bin'] = (full_df['FavoriteProb'] // 5 * 5).astype(int)
    full_df['FavoriteWon'] = (full_df['ProjectionStatus'] == 'Correct').astype(int)
    
    high_conf = full_df[full_df['FavoriteProb'] >= 75]
    results = {}
    if not high_conf.empty:
        record = high_conf['FavoriteWon'].sum()
        total = len(high_conf)
        win_rate = record / total
        results['high_conf'] = {'record': f"{int(record)}-{int(total - record)}", 'win_rate': f"{win_rate:.2%}"}
    else:
        results['high_conf'] = {'record': 'No games', 'win_rate': ''}
    
    grouped = full_df.groupby('Bin').agg(
        Games=('FavoriteWon', 'count'),
        Wins=('FavoriteWon', 'sum')
    ).reset_index()
    grouped['Losses'] = grouped['Games'] - grouped['Wins']
    grouped['WinRate'] = grouped['Wins'] / grouped['Games']
    grouped['BinRange'] = grouped['Bin'].astype(str) + '-' + (grouped['Bin'] + 5).astype(str) + '%'
    results['grouped'] = grouped[['BinRange', 'Wins', 'Losses', 'WinRate']].to_dict('records')
    
    return results

def prepare_data(season: str = "2025", recent_games: int = 30, date_str: Optional[str] = None) -> Tuple[pd.DataFrame, ...]:
    if date_str is None:
        date_str = date.today().strftime('%Y-%m-%d')
    logger.info("Preparing data for %s with recent_games=%d", date_str, recent_games)
    with ThreadPoolExecutor(max_workers=7) as executor:
        f_pitch = executor.submit(fetch_mlb_pitching_stats, season)
        f_rec_pitch = executor.submit(fetch_mlb_pitching_stats, season, recent_games)
        f_team_bat = executor.submit(fetch_mlb_team_batting_stats, season)
        f_rec_team_bat = executor.submit(fetch_mlb_team_batting_stats, season, recent_games)
        f_player_bat = executor.submit(fetch_mlb_player_batting_stats, season)
        f_rec_player_bat = executor.submit(fetch_mlb_player_batting_stats, season, recent_games)
        f_sched = executor.submit(fetch_mlb_schedule, date_str)
        pitching_df = f_pitch.result()
        recent_pitching_df = f_rec_pitch.result()
        team_batting_df = f_team_bat.result()
        recent_team_batting_df = f_rec_team_bat.result()
        player_batting_df = f_player_bat.result()
        recent_player_batting_df = f_rec_player_bat.result()
        schedule_df = f_sched.result()
    for df in [pitching_df, recent_pitching_df, team_batting_df, recent_team_batting_df, player_batting_df, recent_player_batting_df]:
        normalize_team_column(df)
    return schedule_df, pitching_df, team_batting_df, player_batting_df, recent_pitching_df, recent_team_batting_df, recent_player_batting_df

# Streamlit App
st.title("MLB Projections Dashboard")

# Initialize Session State for Data
if 'data' not in st.session_state:
    st.session_state.data = {}

# Auto-Refresh
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")  # Refresh every 5 minutes

# User Inputs
st.subheader("Select Parameters")
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Select Date",
        value=date.today(),
        min_value=date(2025, 1, 1),
        max_value=date(2025, 12, 31),
        format="YYYY-MM-DD",
    )
with col2:
    recent_games = st.slider(
        "Recent Games for Stats Blending",
        min_value=5,
        max_value=30,
        value=30,
        step=5,
    )

# Prepare Data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(date_str, recent_games):
    logger.info("Preparing data for %s with recent_games=%d", date_str, recent_games)
    schedule_df, pitching_df, team_batting_df, player_batting_df, recent_pitching_df, recent_team_batting_df, recent_player_batting_df = prepare_data(
        date_str=date_str, recent_games=recent_games
    )
    projections_df = project_game_outcomes(schedule_df, pitching_df, team_batting_df, recent_pitching_df, recent_team_batting_df)
    hr_projections_df = project_home_run_hitters(schedule_df, player_batting_df, pitching_df, recent_player_batting_df, recent_pitching_df)
    return {
        'projections': projections_df,
        'hr_projections': hr_projections_df,
        'schedule': schedule_df,
    }

# Load Data
date_str = selected_date.strftime('%Y-%m-%d')
data = load_data(date_str, recent_games)
st.session_state.data = data

# Tabs
tab1, tab2, tab3 = st.tabs(["Game Outcomes", "HR Projections", "Backtest"])

with tab1:
    st.subheader("Game Outcomes")
    projections_df = data['projections']
    if projections_df.empty:
        st.write("No game projections available.")
    else:
        display_cols = [
            'HomeTeam', 'AwayTeam', 'HomeWinProb', 'AwayWinProb', 'ProjectedWinner',
            'HomeExpectedRuns', 'AwayExpectedRuns', 'TotalExpectedRuns', 'GameStatus',
            'ProjectionStatus', 'HomeScore', 'AwayScore'
        ]
        # Conditional Formatting
        def style_df(df):
            def highlight_status(row):
                if row['ProjectionStatus'] == 'Correct':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                elif row['ProjectionStatus'] == 'Incorrect':
                    return ['background-color: #f8d7da; color: #721c24'] * len(row)
                elif row['ProjectionStatus'] == 'Leading':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                elif row['ProjectionStatus'] == 'Trailing':
                    return ['background-color: #f8d7da; color: #721c24'] * len(row)
                elif row['ProjectionStatus'] == 'Tied':
                    return ['background-color: #fff3cd; color: #856404'] * len(row)
                return [''] * len(row)
            return df.style.apply(highlight_status, axis=1).format(
                {col: '{:.1f}' for col in ['HomeWinProb', 'AwayWinProb', 'HomeExpectedRuns', 'AwayExpectedRuns', 'TotalExpectedRuns']}
            )
        st.dataframe(style_df(projections_df[display_cols]), use_container_width=True)

with tab2:
    st.subheader("HR Projections")
    hr_projections_df = data['hr_projections']
    if hr_projections_df.empty:
        st.write("No HR projections available.")
    else:
        display_cols = ['PlayerName', 'Team', 'HRProb', 'ProjectionStatus']
        hr_projections_df = hr_projections_df.sort_values(by='HRProb', ascending=False)
        def style_hr_df(df):
            def highlight_status(row):
                if row['ProjectionStatus'] == 'Correct':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                elif row['ProjectionStatus'] == 'Incorrect':
                    return ['background-color: #f8d7da; color: #721c24'] * len(row)
                elif row['ProjectionStatus'] == 'Hit So Far':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                elif row['ProjectionStatus'] == 'Not Yet':
                    return ['background-color: #fff3cd; color: #856404'] * len(row)
                return [''] * len(row)
            return df.style.apply(highlight_status, axis=1).format({'HRProb': '{:.1f}'})
        st.dataframe(style_hr_df(hr_projections_df[display_cols]), use_container_width=True)

with tab3:
    st.subheader("Backtest Results")
    if selected_date < date.today():
        game_back = backtest_game_projections(data['projections'])
        hr_back = backtest_hr_projections(data['hr_projections'])
        st.write(f"**Daily Game Backtest**: Brier Score = {game_back['brier_score']:.4f}, Accuracy = {game_back['accuracy']:.3f}")
        st.write(f"**Daily HR Backtest**: Brier Score = {hr_back['brier_score']:.4f}, Hit Rate = {hr_back['hit_rate']:.3f}")
    else:
        st.write("Daily backtest not available for current or future dates.")
    
    # Season-Long Backtest
    st.subheader("Season-Long Backtest for Win Probabilities")
    season_backtest_results = backtest_season()
    high_conf = season_backtest_results.get('high_conf', {})
    st.write(f"Record for projections over 75%: {high_conf.get('record', 'N/A')} ({high_conf.get('win_rate', '')})")
    grouped_data = season_backtest_results.get('grouped', [])
    if grouped_data:
        grouped_df = pd.DataFrame(grouped_data)
        grouped_df['WinRate'] = grouped_df['WinRate'].apply(lambda x: f'{x:.2%}')
        st.dataframe(grouped_df[['BinRange', 'Wins', 'Losses', 'WinRate']], use_container_width=True)
    else:
        st.write("No season-long data available.")

# Custom CSS
st.markdown("""
<style>
body {
    font-family: Arial, sans-serif;
}
.stTabs [role="tab"] {
    font-size: 16px;
}
.stDataFrame {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
