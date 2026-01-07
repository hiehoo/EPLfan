"""CLI commands for EPL Bet Indicator."""
import asyncio
import sys

import click


@click.group()
def cli():
    """EPL Bet Indicator CLI."""
    pass


@cli.command()
def run():
    """Run the indicator service (scheduler + alerts)."""
    from src.main import main

    click.echo("Starting EPL Bet Indicator service...")
    main()


@cli.command()
def dashboard():
    """Run Streamlit dashboard."""
    import subprocess

    click.echo("Starting Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/dashboard.py"])


@cli.command()
def fetch():
    """Run a manual data fetch."""
    from src.config import setup_logging
    from src.fetchers import DataOrchestrator

    setup_logging()

    async def _fetch():
        orchestrator = DataOrchestrator()
        try:
            result = await orchestrator.fetch_all()
            click.echo(f"Fetch complete:")
            click.echo(f"  - Betfair matches: {len(result.get('betfair', []))}")
            click.echo(f"  - xG teams: {len(result.get('xg', {}))}")
            click.echo(f"  - ELO teams: {len(result.get('elo', {}))}")
            click.echo(f"  - Polymarket markets: {len(result.get('polymarket', []))}")
        finally:
            await orchestrator.close()

    asyncio.run(_fetch())


@cli.command()
@click.option("--match-id", type=int, help="Match ID to analyze")
def analyze(match_id: int):
    """Analyze a specific match."""
    from src.config import setup_logging
    from src.models import match_analyzer
    from src.storage.database import db
    from src.storage.models import Match, OddsSnapshot, TeamStats
    import json

    setup_logging()

    if not match_id:
        click.echo("Error: --match-id is required")
        return

    with db.session() as session:
        match = session.query(Match).filter_by(id=match_id).first()
        if not match:
            click.echo(f"Match {match_id} not found")
            return

        odds = (
            session.query(OddsSnapshot)
            .filter_by(match_id=match_id)
            .order_by(OddsSnapshot.snapshot_time.desc())
            .first()
        )

        home_stats = (
            session.query(TeamStats)
            .filter_by(team_name=match.home_team)
            .order_by(TeamStats.snapshot_date.desc())
            .first()
        )

        away_stats = (
            session.query(TeamStats)
            .filter_by(team_name=match.away_team)
            .order_by(TeamStats.snapshot_date.desc())
            .first()
        )

        if not odds or not home_stats or not away_stats:
            click.echo("Missing data for analysis")
            return

        try:
            match_probs, weighted_probs, edge_analysis = match_analyzer.analyze(
                match_id=str(match.id),
                home_team=match.home_team,
                away_team=match.away_team,
                kickoff=match.kickoff,
                home_xg=home_stats.home_xg,
                away_xg=away_stats.away_xg,
                home_xga=home_stats.home_xga,
                away_xga=away_stats.away_xga,
                home_elo=home_stats.elo_rating,
                away_elo=away_stats.elo_rating,
                bf_home_odds=odds.bf_home_odds or 2.5,
                bf_draw_odds=odds.bf_draw_odds or 3.5,
                bf_away_odds=odds.bf_away_odds or 3.0,
            )

            result = {
                "match": f"{match.home_team} vs {match.away_team}",
                "kickoff": str(match.kickoff),
                "fair_1x2": weighted_probs.fair_1x2,
                "fair_ou": weighted_probs.fair_ou,
                "fair_btts": weighted_probs.fair_btts,
                "best_signal": (
                    {
                        "market": edge_analysis.best_signal.market_type,
                        "selection": edge_analysis.best_signal.outcome,
                        "edge": f"{edge_analysis.best_signal.edge:.1%}",
                    }
                    if edge_analysis.best_signal
                    else None
                ),
            }
            click.echo(json.dumps(result, indent=2, default=str))
        except Exception as e:
            click.echo(f"Analysis failed: {e}")


@cli.command()
def test_telegram():
    """Send a test Telegram message."""
    from src.config import setup_logging
    from src.alerts import TelegramAlertService

    setup_logging()

    async def _test():
        service = TelegramAlertService()
        if not service.is_configured:
            click.echo("Error: Telegram not configured")
            click.echo("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            return

        success = await service.send_alert("🧪 Test message from EPL Bet Indicator")
        if success:
            click.echo("Test message sent successfully!")
        else:
            click.echo("Failed to send test message")

    asyncio.run(_test())


@cli.command()
def scan():
    """Run a single scan cycle."""
    from src.config import setup_logging
    from src.services import get_scheduler

    setup_logging()

    async def _scan():
        scheduler = get_scheduler()
        await scheduler.scan_and_alert()
        click.echo("Scan complete")

    asyncio.run(_scan())


@cli.command()
def init_db():
    """Initialize database tables."""
    from src.config import setup_logging
    from src.storage.database import db

    setup_logging()
    db.create_tables()
    click.echo(f"Database initialized at {db.db_path}")


@cli.command()
@click.option("--force", is_flag=True, help="Force full retrain")
def train_models(force: bool):
    """Train ML classification models."""
    from src.config import setup_logging
    from src.models.training_pipeline import training_pipeline

    setup_logging()

    click.echo("Starting ML model training...")

    if force:
        results = training_pipeline.run_full_training()
    else:
        results = training_pipeline.run_incremental_update()

    # Display results
    click.echo("\nTraining Results:")
    click.echo("-" * 40)

    for stage, data in results.get("stages", {}).items():
        click.echo(f"\n{stage}:")
        if isinstance(data, dict):
            for key, val in data.items():
                click.echo(f"  {key}: {val}")
        else:
            click.echo(f"  {data}")

    click.echo("\nTraining complete!")


@cli.command()
def status():
    """Show system status."""
    from src.storage.database import db
    from src.storage.models import Match, OddsSnapshot, Prediction, TeamStats
    from src.config.settings import settings

    with db.session() as session:
        match_count = session.query(Match).count()
        snapshot_count = session.query(OddsSnapshot).count()
        prediction_count = session.query(Prediction).count()
        team_count = session.query(TeamStats).count()

    click.echo("EPL Bet Indicator Status")
    click.echo("=" * 40)
    click.echo(f"Database: {settings.database_path}")
    click.echo(f"Matches tracked: {match_count}")
    click.echo(f"Odds snapshots: {snapshot_count}")
    click.echo(f"Predictions: {prediction_count}")
    click.echo(f"Team stats entries: {team_count}")
    click.echo("")
    click.echo("Configuration:")
    click.echo(f"  Scan interval: {settings.scan_interval_minutes} min")
    click.echo(f"  1X2 threshold: {settings.edge_threshold_1x2:.0%}")
    click.echo(f"  O/U threshold: {settings.edge_threshold_ou:.0%}")
    click.echo(f"  BTTS threshold: {settings.edge_threshold_btts:.0%}")
    click.echo(f"  Telegram configured: {settings.telegram.is_configured()}")
    click.echo(f"  Betfair configured: {settings.betfair.is_configured()}")


if __name__ == "__main__":
    cli()
