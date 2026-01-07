"""Fetch ELO ratings from clubelo.com."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class TeamELO:
    """ELO rating for a team."""
    team_name: str
    elo_rating: float
    rank: int
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ClubELOFetcher:
    """Fetch ELO ratings from clubelo.com."""

    BASE_URL = "http://clubelo.com"

    # Team name mapping (clubelo name -> standard name)
    TEAM_MAPPING = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Spurs": "Tottenham",
        "Wolves": "Wolverhampton Wanderers",
        "Newcastle": "Newcastle United",
        "Brighton": "Brighton and Hove Albion",
        "West Ham": "West Ham United",
        "Nott'm Forest": "Nottingham Forest",
        "Leicester": "Leicester City",
        "Ipswich": "Ipswich Town",
        "Southampton": "Southampton",
        "Brentford": "Brentford",
        "Everton": "Everton",
        "Crystal Palace": "Crystal Palace",
        "Fulham": "Fulham",
        "Bournemouth": "AFC Bournemouth",
        "Aston Villa": "Aston Villa",
        "Liverpool": "Liverpool",
        "Chelsea": "Chelsea",
        "Arsenal": "Arsenal",
    }

    # Reverse mapping for lookup
    REVERSE_MAPPING = {v: k for k, v in TEAM_MAPPING.items()}

    async def fetch_epl_ratings(self) -> dict[str, TeamELO]:
        """Fetch current ELO ratings for all EPL teams."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.BASE_URL}/ENG")
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", class_="ranking")

            if not table:
                logger.error("Could not find ELO ranking table")
                return {}

            results = {}
            rows = table.find_all("tr")[1:]  # Skip header

            for rank, row in enumerate(rows, 1):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    team_name = cols[0].text.strip()
                    try:
                        elo_rating = float(cols[1].text.strip())
                    except ValueError:
                        continue

                    # Normalize team name
                    normalized_name = self.TEAM_MAPPING.get(team_name, team_name)

                    results[normalized_name] = TeamELO(
                        team_name=normalized_name,
                        elo_rating=elo_rating,
                        rank=rank,
                        fetched_at=datetime.now(timezone.utc),
                    )

            logger.info(f"Fetched ELO ratings for {len(results)} teams")
            return results

    def calculate_win_probability(self, home_elo: float, away_elo: float) -> dict[str, float]:
        """Calculate win probabilities from ELO difference.

        Using the standard ELO formula with home advantage adjustment.

        Args:
            home_elo: Home team's ELO rating
            away_elo: Away team's ELO rating

        Returns:
            Dict with 'home', 'draw', 'away' probabilities
        """
        # Home advantage in ELO points (typically ~65-100 points)
        home_advantage = 65

        # Adjusted ELO
        adjusted_home_elo = home_elo + home_advantage
        elo_diff = adjusted_home_elo - away_elo

        # Expected score (probability of home win in binary outcome)
        expected_home = 1 / (1 + 10 ** (-elo_diff / 400))

        # Convert to 1X2 probabilities (simplified model)
        # Draw probability based on ELO closeness
        draw_factor = 0.28 - abs(elo_diff) / 2000  # ~28% base, reduces with larger diff
        draw_factor = max(0.15, min(0.32, draw_factor))  # Clamp 15-32%

        home_win = expected_home * (1 - draw_factor)
        away_win = (1 - expected_home) * (1 - draw_factor)

        return {
            "home": home_win,
            "draw": draw_factor,
            "away": away_win,
        }

    def get_team_elo(
        self,
        team_name: str,
        ratings: dict[str, TeamELO]
    ) -> Optional[TeamELO]:
        """Get ELO for a team, handling name variations."""
        # Direct lookup
        if team_name in ratings:
            return ratings[team_name]

        # Try reverse mapping
        mapped_name = self.REVERSE_MAPPING.get(team_name)
        if mapped_name and mapped_name in ratings:
            return ratings[mapped_name]

        # Fuzzy match (partial name)
        team_lower = team_name.lower()
        for name, elo in ratings.items():
            if team_lower in name.lower() or name.lower() in team_lower:
                return elo

        return None
