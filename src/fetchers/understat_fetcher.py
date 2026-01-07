"""Fetch xG data from Understat for EPL teams."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from understatapi import UnderstatClient
from scipy.stats import poisson

logger = logging.getLogger(__name__)


@dataclass
class TeamXGData:
    """xG statistics for a team."""
    team_name: str
    matches_played: int

    # Rolling averages (last N matches)
    xg_for: float       # Expected goals scored
    xga_against: float  # Expected goals conceded
    xpts: float         # Expected points

    # Home/Away splits
    home_xg: float
    home_xga: float
    away_xg: float
    away_xga: float

    # Recent form (last 5)
    form_xg: list[float] = field(default_factory=list)
    form_xga: list[float] = field(default_factory=list)

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UnderstatFetcher:
    """Fetch xG data from Understat."""

    LEAGUE = "epl"
    FORM_WINDOW = 5      # Last 5 matches for form
    ROLLING_WINDOW = 10  # Last 10 matches for averages

    def __init__(self, season: int = 2025):
        """Initialize with season year (e.g., 2025 for 2024/25 season)."""
        self.season = season
        self.client = UnderstatClient()

    async def fetch_all_teams(self) -> dict[str, TeamXGData]:
        """Fetch xG data for all EPL teams."""
        async with self.client as client:
            # Get team season data
            teams = await client.get_teams(self.LEAGUE, self.season)

            results = {}
            for team in teams:
                team_data = await self._process_team(client, team)
                if team_data:
                    results[team_data.team_name] = team_data

            logger.info(f"Fetched xG data for {len(results)} teams")
            return results

    async def _process_team(self, client, team_summary) -> Optional[TeamXGData]:
        """Process single team's xG data."""
        try:
            team_name = team_summary["title"]

            # Get detailed match-by-match data
            matches = await client.get_team_results(team_name, self.season)

            if not matches:
                return None

            # Calculate rolling averages
            recent_matches = matches[-self.ROLLING_WINDOW:]
            form_matches = matches[-self.FORM_WINDOW:]

            # Separate home/away
            home_matches = [m for m in recent_matches if m["h"]["title"] == team_name]
            away_matches = [m for m in recent_matches if m["a"]["title"] == team_name]

            # Calculate aggregates
            xg_for_list = []
            xga_against_list = []
            xpts_list = []

            for m in recent_matches:
                is_home = m["h"]["title"] == team_name
                if is_home:
                    xg_for_list.append(float(m["xG"]["h"]))
                    xga_against_list.append(float(m["xG"]["a"]))
                else:
                    xg_for_list.append(float(m["xG"]["a"]))
                    xga_against_list.append(float(m["xG"]["h"]))

                # Calculate xPTS from xG
                xpts_list.append(self._xg_to_xpts(
                    float(m["xG"]["h"]) if is_home else float(m["xG"]["a"]),
                    float(m["xG"]["a"]) if is_home else float(m["xG"]["h"]),
                    is_home,
                ))

            # Home/Away specific
            home_xg = sum(float(m["xG"]["h"]) for m in home_matches) / max(len(home_matches), 1)
            home_xga = sum(float(m["xG"]["a"]) for m in home_matches) / max(len(home_matches), 1)
            away_xg = sum(float(m["xG"]["a"]) for m in away_matches) / max(len(away_matches), 1)
            away_xga = sum(float(m["xG"]["h"]) for m in away_matches) / max(len(away_matches), 1)

            # Form data (last 5)
            form_xg = []
            form_xga = []
            for m in form_matches:
                is_home = m["h"]["title"] == team_name
                if is_home:
                    form_xg.append(float(m["xG"]["h"]))
                    form_xga.append(float(m["xG"]["a"]))
                else:
                    form_xg.append(float(m["xG"]["a"]))
                    form_xga.append(float(m["xG"]["h"]))

            return TeamXGData(
                team_name=team_name,
                matches_played=len(matches),
                xg_for=sum(xg_for_list) / len(xg_for_list) if xg_for_list else 0,
                xga_against=sum(xga_against_list) / len(xga_against_list) if xga_against_list else 0,
                xpts=sum(xpts_list) / len(xpts_list) if xpts_list else 0,
                home_xg=home_xg,
                home_xga=home_xga,
                away_xg=away_xg,
                away_xga=away_xga,
                form_xg=form_xg,
                form_xga=form_xga,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error processing team {team_summary.get('title', 'unknown')}: {e}")
            return None

    def _xg_to_xpts(self, xg_for: float, xg_against: float, is_home: bool) -> float:
        """Estimate expected points from xG using Poisson."""
        # Home advantage adjustment
        if is_home:
            xg_for *= 1.1
            xg_against *= 0.9

        win_prob = 0.0
        draw_prob = 0.0

        for h in range(10):
            for a in range(10):
                prob = poisson.pmf(h, xg_for) * poisson.pmf(a, xg_against)
                if h > a:
                    win_prob += prob
                elif h == a:
                    draw_prob += prob

        return win_prob * 3 + draw_prob * 1
