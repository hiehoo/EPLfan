"""APScheduler service for periodic tasks."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SchedulerService:
    """Manages scheduled tasks for data fetching and alerting."""

    def __init__(self):
        """Initialize scheduler service."""
        self._scheduler = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    def _get_scheduler(self):
        """Lazy initialization of scheduler."""
        if self._scheduler is None:
            try:
                from apscheduler.schedulers.asyncio import AsyncIOScheduler

                self._scheduler = AsyncIOScheduler()
            except ImportError:
                logger.error("apscheduler not installed")
                raise
        return self._scheduler

    def start(self):
        """Start the scheduler with all jobs."""
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger

        scheduler = self._get_scheduler()

        # Primary scan - every N minutes
        scheduler.add_job(
            self.scan_and_alert,
            IntervalTrigger(minutes=settings.scan_interval_minutes),
            id="primary_scan",
            name="Primary Signal Scan",
            replace_existing=True,
        )

        # Daily data refresh - 6 AM UTC
        scheduler.add_job(
            self.daily_refresh,
            CronTrigger(hour=6, minute=0),
            id="daily_refresh",
            name="Daily Data Refresh",
            replace_existing=True,
        )

        # Matchday intensive scan check - every 5 min
        scheduler.add_job(
            self.check_matchday_scans,
            IntervalTrigger(minutes=5),
            id="matchday_check",
            name="Matchday Scan Check",
            replace_existing=True,
        )

        scheduler.start()
        self._is_running = True
        logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))

    def stop(self):
        """Stop the scheduler."""
        if self._scheduler and self._is_running:
            self._scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Scheduler stopped")

    async def scan_and_alert(self):
        """Main scan job - fetch data, analyze, and alert."""
        from src.fetchers import DataOrchestrator
        from src.models import match_analyzer
        from src.alerts import TelegramAlertService
        from src.alerts.templates import SignalAlert
        from src.storage.database import db
        from src.storage.models import Match, Prediction

        logger.info("Starting scheduled scan")

        telegram = TelegramAlertService()

        try:
            # 1. Fetch latest data
            orchestrator = DataOrchestrator()
            await orchestrator.fetch_all()

            # 2. Get upcoming matches
            with db.session() as session:
                now = datetime.now(timezone.utc)
                upcoming = (
                    session.query(Match)
                    .filter(
                        Match.kickoff >= now,
                        Match.kickoff <= now + timedelta(days=7),
                        Match.is_completed == False,
                    )
                    .all()
                )

                match_data = [
                    {
                        "id": m.id,
                        "external_id": m.external_id,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "kickoff": m.kickoff,
                    }
                    for m in upcoming
                ]

            # 3. Analyze each match and collect signals
            all_signals = []
            for match in match_data:
                signals = await self._analyze_and_store(match)
                all_signals.extend(signals)

            # 4. Send alerts
            if all_signals:
                await telegram.send_batch_alerts(all_signals)
                logger.info(f"Sent {len(all_signals)} signals")
            else:
                logger.info("No signals above threshold")

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            await telegram.send_error_alert("ScanError", str(e))

    async def _analyze_and_store(self, match: dict) -> list:
        """Analyze a single match and store prediction.

        Args:
            match: Match data dict with id, home_team, away_team, kickoff

        Returns:
            List of SignalAlert objects for actionable signals
        """
        from src.models import match_analyzer
        from src.alerts.templates import SignalAlert
        from src.storage.database import db
        from src.storage.models import OddsSnapshot, TeamStats, Prediction

        signals = []

        with db.session() as session:
            # Get latest odds
            odds = (
                session.query(OddsSnapshot)
                .filter(OddsSnapshot.match_id == match["id"])
                .order_by(OddsSnapshot.snapshot_time.desc())
                .first()
            )

            if not odds:
                return []

            # Get team stats
            home_stats = (
                session.query(TeamStats)
                .filter(TeamStats.team_name == match["home_team"])
                .order_by(TeamStats.snapshot_date.desc())
                .first()
            )
            away_stats = (
                session.query(TeamStats)
                .filter(TeamStats.team_name == match["away_team"])
                .order_by(TeamStats.snapshot_date.desc())
                .first()
            )

            if not home_stats or not away_stats:
                return []

            # Run analysis
            try:
                match_probs, weighted_probs, edge_analysis = match_analyzer.analyze(
                    match_id=str(match["id"]),
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    kickoff=match["kickoff"],
                    home_xg=home_stats.home_xg,
                    away_xg=away_stats.away_xg,
                    home_xga=home_stats.home_xga,
                    away_xga=away_stats.away_xga,
                    home_elo=home_stats.elo_rating,
                    away_elo=away_stats.elo_rating,
                    bf_home_odds=odds.bf_home_odds or 2.5,
                    bf_draw_odds=odds.bf_draw_odds or 3.5,
                    bf_away_odds=odds.bf_away_odds or 3.0,
                    bf_over_2_5_odds=odds.bf_over_2_5_odds,
                    bf_under_2_5_odds=odds.bf_under_2_5_odds,
                    bf_btts_yes_odds=odds.bf_btts_yes_odds,
                    bf_btts_no_odds=odds.bf_btts_no_odds,
                    pm_home_price=odds.pm_home_price,
                    pm_draw_price=odds.pm_draw_price,
                    pm_away_price=odds.pm_away_price,
                    pm_over_2_5_price=odds.pm_over_2_5_price,
                    pm_under_2_5_price=odds.pm_under_2_5_price,
                    pm_btts_yes_price=odds.pm_btts_yes_price,
                    pm_btts_no_price=odds.pm_btts_no_price,
                )

                # Store prediction
                prediction = Prediction(
                    match_id=match["id"],
                    preset_used=weighted_probs.preset_name,
                    p_home_model=weighted_probs.fair_1x2.get("home", 0),
                    p_draw_model=weighted_probs.fair_1x2.get("draw", 0),
                    p_away_model=weighted_probs.fair_1x2.get("away", 0),
                    p_home_market=match_probs.polymarket_1x2.get("home", 0),
                    p_draw_market=match_probs.polymarket_1x2.get("draw", 0),
                    p_away_market=match_probs.polymarket_1x2.get("away", 0),
                    edge_1x2=edge_analysis.signal_1x2.edge if edge_analysis.signal_1x2 else 0,
                    signal_1x2=edge_analysis.signal_1x2.outcome if edge_analysis.signal_1x2 and edge_analysis.signal_1x2.is_actionable else "none",
                    p_over_2_5_model=weighted_probs.fair_ou.get("2.5", {}).get("over", 0),
                    p_over_2_5_market=match_probs.polymarket_ou.get("2.5", {}).get("over", 0),
                    edge_ou=edge_analysis.signal_ou.edge if edge_analysis.signal_ou else 0,
                    signal_ou=edge_analysis.signal_ou.outcome if edge_analysis.signal_ou and edge_analysis.signal_ou.is_actionable else "none",
                    p_btts_yes_model=weighted_probs.fair_btts.get("yes", 0),
                    p_btts_yes_market=match_probs.polymarket_btts.get("yes", 0),
                    edge_btts=edge_analysis.signal_btts.edge if edge_analysis.signal_btts else 0,
                    signal_btts=edge_analysis.signal_btts.outcome if edge_analysis.signal_btts and edge_analysis.signal_btts.is_actionable else "none",
                )
                session.add(prediction)

                # Build signal alerts for actionable signals
                kickoff_str = match["kickoff"].strftime("%Y-%m-%d %H:%M UTC")

                if edge_analysis.signal_1x2 and edge_analysis.signal_1x2.is_actionable:
                    sig = edge_analysis.signal_1x2
                    signals.append(
                        SignalAlert(
                            match_id=match["id"],
                            home_team=match["home_team"],
                            away_team=match["away_team"],
                            kickoff=kickoff_str,
                            market="1X2",
                            selection=sig.outcome.upper(),
                            fair_prob=sig.fair_prob,
                            market_prob=sig.market_prob,
                            edge=abs(sig.edge),
                            weight_profile=weighted_probs.preset_name,
                        )
                    )

                if edge_analysis.signal_ou and edge_analysis.signal_ou.is_actionable:
                    sig = edge_analysis.signal_ou
                    signals.append(
                        SignalAlert(
                            match_id=match["id"],
                            home_team=match["home_team"],
                            away_team=match["away_team"],
                            kickoff=kickoff_str,
                            market="O/U 2.5",
                            selection=sig.outcome.upper(),
                            fair_prob=sig.fair_prob,
                            market_prob=sig.market_prob,
                            edge=abs(sig.edge),
                            weight_profile=weighted_probs.preset_name,
                        )
                    )

                if edge_analysis.signal_btts and edge_analysis.signal_btts.is_actionable:
                    sig = edge_analysis.signal_btts
                    signals.append(
                        SignalAlert(
                            match_id=match["id"],
                            home_team=match["home_team"],
                            away_team=match["away_team"],
                            kickoff=kickoff_str,
                            market="BTTS",
                            selection=sig.outcome.upper(),
                            fair_prob=sig.fair_prob,
                            market_prob=sig.market_prob,
                            edge=abs(sig.edge),
                            weight_profile=weighted_probs.preset_name,
                        )
                    )

            except Exception as e:
                logger.error(f"Analysis failed for match {match['id']}: {e}")

        return signals

    async def daily_refresh(self):
        """Daily refresh of team statistics and ELO ratings."""
        from src.fetchers import DataOrchestrator

        logger.info("Starting daily refresh")

        try:
            orchestrator = DataOrchestrator()
            await orchestrator.fetch_all()
            logger.info("Daily refresh complete")
        except Exception as e:
            logger.error(f"Daily refresh failed: {e}")

    async def check_matchday_scans(self):
        """Check if we need intensive scanning for imminent matches."""
        from src.storage.database import db
        from src.storage.models import Match

        with db.session() as session:
            now = datetime.now(timezone.utc)
            imminent = (
                session.query(Match)
                .filter(
                    Match.kickoff >= now,
                    Match.kickoff <= now + timedelta(hours=4),
                    Match.is_completed == False,
                )
                .count()
            )

        if imminent > 0:
            logger.info(f"Matchday mode: {imminent} matches within 4h")
            await self.scan_and_alert()


# Singleton instance
_scheduler: Optional[SchedulerService] = None


def get_scheduler() -> SchedulerService:
    """Get scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerService()
    return _scheduler
