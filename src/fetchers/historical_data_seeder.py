"""Historical data seeder from football-data.co.uk."""
import logging
from datetime import datetime
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from src.storage.database import db
from src.storage.models import Match, OddsSnapshot, TeamStats

logger = logging.getLogger(__name__)


# Team name normalization: CSV name -> Standard name
TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
    "Sheffield United": "Sheffield Utd",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Norwich": "Norwich City",
    "Burnley": "Burnley FC",
}


class HistoricalDataSeeder:
    """Seed database with football-data.co.uk CSVs."""

    SEASONS = ["2223", "2324", "2425"]
    BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

    # Required CSV columns
    REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ODDS_COLUMNS = ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]

    def seed_all_seasons(self) -> dict:
        """Download and import all configured seasons.

        Returns:
            {"total": N, "imported": M, "skipped": K, "errors": L}
        """
        stats = {"total": 0, "imported": 0, "skipped": 0, "errors": 0}

        for season in self.SEASONS:
            logger.info(f"Seeding season {season}...")
            try:
                count = self.seed_season(season)
                stats["imported"] += count["imported"]
                stats["skipped"] += count["skipped"]
                stats["total"] += count["total"]
            except Exception as e:
                logger.error(f"Failed to seed season {season}: {e}")
                stats["errors"] += 1

        logger.info(f"Seeding complete: {stats}")
        return stats

    def seed_season(self, season: str) -> dict:
        """Seed single season, return counts."""
        df = self._download_csv(season)
        if df is None or df.empty:
            return {"total": 0, "imported": 0, "skipped": 0}

        imported = 0
        skipped = 0

        for _, row in df.iterrows():
            if self._import_match(row, season):
                imported += 1
            else:
                skipped += 1

        logger.info(f"Season {season}: {imported} imported, {skipped} skipped")
        return {"total": len(df), "imported": imported, "skipped": skipped}

    def _download_csv(self, season: str) -> Optional[pd.DataFrame]:
        """Fetch CSV from football-data.co.uk."""
        url = self.BASE_URL.format(season=season)
        logger.info(f"Downloading {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            df = pd.read_csv(StringIO(response.text))

            # Validate required columns
            missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return None

            # Filter to completed matches only (have goals)
            df = df.dropna(subset=["FTHG", "FTAG"])

            logger.info(f"Downloaded {len(df)} completed matches for season {season}")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download CSV: {e}")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV: {e}")
            return None

    def _normalize_team_name(self, name: str) -> str:
        """Map CSV team name to standard name."""
        return TEAM_NAME_MAP.get(name, name)

    def _import_match(self, row: pd.Series, season: str) -> bool:
        """Create Match + OddsSnapshot from CSV row.

        Returns:
            True if imported, False if skipped (duplicate or error)
        """
        try:
            # Parse date (dd/mm/yyyy or dd/mm/yy)
            date_str = str(row["Date"])
            try:
                kickoff = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                kickoff = datetime.strptime(date_str, "%d/%m/%y")

            # Normalize team names
            home_team = self._normalize_team_name(str(row["HomeTeam"]))
            away_team = self._normalize_team_name(str(row["AwayTeam"]))

            # Create external_id from date and teams
            external_id = f"fd_{season}_{kickoff.strftime('%Y%m%d')}_{home_team}_{away_team}"
            external_id = external_id.replace(" ", "_")

            with db.session() as session:
                # Check for existing match
                existing = session.query(Match).filter_by(external_id=external_id).first()
                if existing:
                    return False

                # Create match
                match = Match(
                    external_id=external_id,
                    home_team=home_team,
                    away_team=away_team,
                    kickoff=kickoff,
                    home_goals=int(row["FTHG"]),
                    away_goals=int(row["FTAG"]),
                    is_completed=True,
                )
                session.add(match)
                session.flush()

                # Create odds snapshot with Bet365 odds
                snapshot = OddsSnapshot(
                    match_id=match.id,
                    snapshot_time=kickoff,
                    hours_to_kickoff=0.0,
                    bf_home_odds=self._safe_float(row.get("B365H")),
                    bf_draw_odds=self._safe_float(row.get("B365D")),
                    bf_away_odds=self._safe_float(row.get("B365A")),
                    bf_over_2_5_odds=self._safe_float(row.get("B365>2.5")),
                    bf_under_2_5_odds=self._safe_float(row.get("B365<2.5")),
                )
                session.add(snapshot)

            return True

        except Exception as e:
            logger.warning(f"Failed to import match: {e}")
            return False

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _odds_to_prob(self, odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 1.0:
            return 0.0
        return 1.0 / odds

    def seed_team_stats(self) -> dict:
        """Create TeamStats from historical match data.

        Calculates rolling averages for each team and creates
        TeamStats snapshots that can be used for ML training.

        Returns:
            {"teams": N, "stats_created": M}
        """
        stats_created = 0
        teams_processed = set()

        with db.session() as session:
            # Get all matches ordered by date
            matches = (
                session.query(Match)
                .filter(Match.is_completed == True)
                .order_by(Match.kickoff)
                .all()
            )

            if not matches:
                logger.warning("No matches found for team stats")
                return {"teams": 0, "stats_created": 0}

            # Build team history
            team_matches = {}  # team_name -> list of match data
            for match in matches:
                # Home team data
                if match.home_team not in team_matches:
                    team_matches[match.home_team] = []
                team_matches[match.home_team].append({
                    "date": match.kickoff,
                    "goals_for": match.home_goals,
                    "goals_against": match.away_goals,
                    "is_home": True,
                })

                # Away team data
                if match.away_team not in team_matches:
                    team_matches[match.away_team] = []
                team_matches[match.away_team].append({
                    "date": match.kickoff,
                    "goals_for": match.away_goals,
                    "goals_against": match.home_goals,
                    "is_home": False,
                })

            # Create TeamStats for each team at key points
            for team_name, history in team_matches.items():
                if len(history) < 5:
                    continue

                teams_processed.add(team_name)

                # Create stats at every 5 matches
                for i in range(5, len(history) + 1, 5):
                    recent = history[max(0, i - 10):i]  # Last 10 matches
                    home_matches = [m for m in recent if m["is_home"]]
                    away_matches = [m for m in recent if not m["is_home"]]

                    # Calculate averages
                    avg_goals_for = sum(m["goals_for"] for m in recent) / len(recent)
                    avg_goals_against = sum(m["goals_against"] for m in recent) / len(recent)

                    home_goals_for = sum(m["goals_for"] for m in home_matches) / len(home_matches) if home_matches else avg_goals_for
                    home_goals_against = sum(m["goals_against"] for m in home_matches) / len(home_matches) if home_matches else avg_goals_against
                    away_goals_for = sum(m["goals_for"] for m in away_matches) / len(away_matches) if away_matches else avg_goals_for
                    away_goals_against = sum(m["goals_against"] for m in away_matches) / len(away_matches) if away_matches else avg_goals_against

                    # Estimate xG from goals (simplified)
                    xg_for = avg_goals_for * 1.05  # Slight inflation for xG
                    xga_against = avg_goals_against * 1.05

                    # Calculate ELO-like rating (simplified)
                    goal_diff = avg_goals_for - avg_goals_against
                    elo_rating = 1500 + (goal_diff * 100)

                    # Check for existing stats
                    snapshot_date = recent[-1]["date"]
                    existing = session.query(TeamStats).filter(
                        TeamStats.team_name == team_name,
                        TeamStats.snapshot_date == snapshot_date,
                    ).first()

                    if existing:
                        continue

                    team_stat = TeamStats(
                        team_name=team_name,
                        snapshot_date=snapshot_date,
                        xg_for=xg_for,
                        xga_against=xga_against,
                        xpts=avg_goals_for * 2.5,  # Rough xPts estimate
                        home_xg=home_goals_for * 1.05,
                        away_xg=away_goals_for * 1.05,
                        home_xga=home_goals_against * 1.05,
                        away_xga=away_goals_against * 1.05,
                        elo_rating=elo_rating,
                        elo_rank=1,  # Will update later
                        shots_per_game=avg_goals_for * 10,  # Rough estimate
                        shots_against_per_game=avg_goals_against * 10,
                        shots_on_target_per_game=avg_goals_for * 4,
                        goal_volatility=1.0,
                        xg_volatility=0.5,
                        shot_conversion_rate=0.1,
                        xg_overperformance=0.0,
                    )
                    session.add(team_stat)
                    stats_created += 1

        logger.info(f"Created {stats_created} team stats for {len(teams_processed)} teams")
        return {"teams": len(teams_processed), "stats_created": stats_created}


# Singleton instance
historical_seeder = HistoricalDataSeeder()
