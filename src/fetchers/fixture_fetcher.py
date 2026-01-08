"""Fetch upcoming EPL fixtures from football-data.org."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.storage.database import db
from src.storage.models import Match, OddsSnapshot, TeamStats

logger = logging.getLogger(__name__)


# Team name mapping (football-data.org to our standard names)
TEAM_NAME_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion FC": "Brighton",
    "Ipswich Town FC": "Ipswich",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Southampton FC": "Southampton",
}


@dataclass
class Fixture:
    """Upcoming fixture data."""
    external_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    matchday: int
    status: str  # SCHEDULED, TIMED, IN_PLAY, FINISHED


class FixtureFetcher:
    """Fetch EPL fixtures from football-data.org API."""

    BASE_URL = "https://api.football-data.org/v4"
    EPL_CODE = "PL"  # Premier League competition code

    def __init__(self, api_key: Optional[str] = None):
        """Initialize fetcher.

        Args:
            api_key: Optional football-data.org API key. If not provided, loads from settings.
        """
        if api_key:
            self.api_key = api_key
        else:
            # Load from settings/environment
            from src.config.settings import settings
            self.api_key = settings.football_data.api_key

        self.client = httpx.Client(timeout=30)

    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name to match our database format."""
        return TEAM_NAME_MAP.get(name, name.replace(" FC", "").strip())

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Auth-Token"] = self.api_key
        return headers

    def fetch_upcoming_fixtures(self, days_ahead: int = 14) -> list[Fixture]:
        """Fetch upcoming EPL fixtures.

        Args:
            days_ahead: Number of days ahead to fetch (max 14 for free tier)

        Returns:
            List of upcoming fixtures
        """
        try:
            url = f"{self.BASE_URL}/competitions/{self.EPL_CODE}/matches"
            params = {"status": "SCHEDULED,TIMED"}

            response = self.client.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()

            data = response.json()
            fixtures = []

            for match in data.get("matches", []):
                kickoff = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))

                fixture = Fixture(
                    external_id=f"fd-{match['id']}",
                    home_team=self._normalize_team_name(match["homeTeam"]["name"]),
                    away_team=self._normalize_team_name(match["awayTeam"]["name"]),
                    kickoff=kickoff,
                    matchday=match.get("matchday", 0),
                    status=match["status"],
                )
                fixtures.append(fixture)

            logger.info(f"Fetched {len(fixtures)} upcoming fixtures")
            return fixtures

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded. Try again later or add API key.")
            elif e.response.status_code == 403:
                logger.warning("API key required. Using sample fixtures.")
                return self._generate_sample_fixtures()
            else:
                logger.error(f"HTTP error fetching fixtures: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            return []

    def _generate_sample_fixtures(self) -> list[Fixture]:
        """Generate sample upcoming fixtures for testing."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)

        # Sample EPL matchday fixtures (typical gameweek)
        # Team names must match database (from historical_data_seeder)
        sample_matches = [
            ("Arsenal", "Manchester City", 1),
            ("Liverpool", "Chelsea", 1),
            ("Manchester United", "Tottenham", 2),
            ("Aston Villa", "Newcastle United", 2),
            ("Brighton & Hove Albion", "West Ham United", 3),
            ("Everton", "Fulham", 3),
            ("Bournemouth", "Crystal Palace", 4),
            ("Brentford", "Wolverhampton", 4),
            ("Leicester City", "Southampton", 5),
            ("Nottingham Forest", "Ipswich", 5),
        ]

        fixtures = []
        for i, (home, away, day_offset) in enumerate(sample_matches):
            kickoff = now + timedelta(days=day_offset, hours=15)  # 3PM kick-offs

            fixture = Fixture(
                external_id=f"sample-{i+1}",
                home_team=home,
                away_team=away,
                kickoff=kickoff,
                matchday=20,  # Mid-season
                status="SCHEDULED",
            )
            fixtures.append(fixture)

        logger.info(f"Generated {len(fixtures)} sample fixtures")
        return fixtures

    def seed_fixtures(self, days_ahead: int = 14) -> dict:
        """Fetch and seed upcoming fixtures to database.

        Returns:
            Dict with seeding results
        """
        fixtures = self.fetch_upcoming_fixtures(days_ahead)

        imported = 0
        skipped = 0
        with_odds = 0

        with db.session() as session:
            for fixture in fixtures:
                # Check if match already exists
                existing = session.query(Match).filter_by(
                    external_id=fixture.external_id
                ).first()

                if existing:
                    skipped += 1
                    continue

                # Create match
                match = Match(
                    external_id=fixture.external_id,
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    kickoff=fixture.kickoff,
                    is_completed=False,
                )
                session.add(match)
                session.flush()

                # Create estimated odds from team stats
                odds_created = self._create_estimated_odds(session, match)
                if odds_created:
                    with_odds += 1

                imported += 1

        return {
            "total": len(fixtures),
            "imported": imported,
            "skipped": skipped,
            "with_odds": with_odds,
        }

    def _create_estimated_odds(self, session, match: Match) -> bool:
        """Create estimated odds based on team stats.

        Uses ELO ratings and xG to estimate fair odds.
        """
        # Get team stats
        home_stats = (
            session.query(TeamStats)
            .filter(TeamStats.team_name == match.home_team)
            .order_by(TeamStats.snapshot_date.desc())
            .first()
        )
        away_stats = (
            session.query(TeamStats)
            .filter(TeamStats.team_name == match.away_team)
            .order_by(TeamStats.snapshot_date.desc())
            .first()
        )

        if not home_stats or not away_stats:
            logger.debug(f"No stats for {match.home_team} vs {match.away_team}")
            return False

        # Estimate probabilities from ELO
        # Clamp elo_diff to prevent overflow
        raw_elo_diff = home_stats.elo_rating - away_stats.elo_rating
        elo_diff = max(-1000, min(raw_elo_diff, 1000))
        home_elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # Apply home advantage (~4-5% boost)
        home_prob = min(0.85, home_elo_prob + 0.04)
        away_prob = max(0.05, (1 - home_prob) * 0.55)
        draw_prob = max(0.10, 1 - home_prob - away_prob)

        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        # Convert to decimal odds (with ~5% margin)
        margin = 1.05
        home_odds = margin / home_prob
        draw_odds = margin / draw_prob
        away_odds = margin / away_prob

        # Estimate O/U 2.5 from xG
        total_xg = home_stats.xg_for + away_stats.xg_for
        over_prob = min(0.75, max(0.25, 0.35 + (total_xg - 2.5) * 0.12))
        under_prob = 1 - over_prob

        over_odds = margin / over_prob
        under_odds = margin / under_prob

        # Estimate BTTS from xG
        # Both teams likely to score if both have good xG
        btts_factor = min(home_stats.xg_for, away_stats.xg_for)
        btts_yes_prob = min(0.70, max(0.30, 0.45 + btts_factor * 0.08))
        btts_no_prob = 1 - btts_yes_prob

        btts_yes_odds = margin / btts_yes_prob
        btts_no_odds = margin / btts_no_prob

        # Create odds snapshot
        now = datetime.now(timezone.utc)
        hours_to_kick = (match.kickoff - now).total_seconds() / 3600

        snapshot = OddsSnapshot(
            match_id=match.id,
            snapshot_time=now,
            hours_to_kickoff=hours_to_kick,
            bf_home_odds=round(home_odds, 2),
            bf_draw_odds=round(draw_odds, 2),
            bf_away_odds=round(away_odds, 2),
            bf_over_2_5_odds=round(over_odds, 2),
            bf_under_2_5_odds=round(under_odds, 2),
            bf_btts_yes_odds=round(btts_yes_odds, 2),
            bf_btts_no_odds=round(btts_no_odds, 2),
        )
        session.add(snapshot)

        return True

    def close(self):
        """Close HTTP client."""
        self.client.close()


# Singleton instance
fixture_fetcher = FixtureFetcher()
