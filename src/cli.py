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


def _validate_odds(odds_value: float, default: float) -> float:
    """Validate odds are in reasonable range."""
    if odds_value and 1.01 <= odds_value <= 1000:
        return odds_value
    return default


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

        # Extract data to dicts to avoid DetachedInstanceError
        match_data = {
            "id": match.id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "kickoff": match.kickoff,
        }

        odds_data = {
            "bf_home_odds": _validate_odds(odds.bf_home_odds, 2.5),
            "bf_draw_odds": _validate_odds(odds.bf_draw_odds, 3.5),
            "bf_away_odds": _validate_odds(odds.bf_away_odds, 3.0),
        }

        home_stats_data = {
            "home_xg": home_stats.home_xg,
            "home_xga": home_stats.home_xga,
            "elo_rating": home_stats.elo_rating,
        }

        away_stats_data = {
            "away_xg": away_stats.away_xg,
            "away_xga": away_stats.away_xga,
            "elo_rating": away_stats.elo_rating,
        }

    # Use extracted dicts outside session
    try:
        match_probs, weighted_probs, edge_analysis = match_analyzer.analyze(
            match_id=str(match_data["id"]),
            home_team=match_data["home_team"],
            away_team=match_data["away_team"],
            kickoff=match_data["kickoff"],
            home_xg=home_stats_data["home_xg"],
            away_xg=away_stats_data["away_xg"],
            home_xga=home_stats_data["home_xga"],
            away_xga=away_stats_data["away_xga"],
            home_elo=home_stats_data["elo_rating"],
            away_elo=away_stats_data["elo_rating"],
            bf_home_odds=odds_data["bf_home_odds"],
            bf_draw_odds=odds_data["bf_draw_odds"],
            bf_away_odds=odds_data["bf_away_odds"],
        )

        result = {
            "match": f"{match_data['home_team']} vs {match_data['away_team']}",
            "kickoff": str(match_data["kickoff"]),
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
@click.option("--seasons", default=3, help="Number of seasons to seed (1-3)")
def seed_historical(seasons: int):
    """Seed historical EPL data from football-data.co.uk."""
    from src.config import setup_logging
    from src.fetchers.historical_data_seeder import HistoricalDataSeeder

    setup_logging()

    click.echo(f"Seeding {seasons} season(s) of historical EPL data...")
    click.echo("Source: football-data.co.uk")

    seeder = HistoricalDataSeeder()

    # Limit seasons
    seeder.SEASONS = seeder.SEASONS[-seasons:]

    results = seeder.seed_all_seasons()

    click.echo("\nSeeding Results:")
    click.echo("-" * 40)
    click.echo(f"Total matches found: {results['total']}")
    click.echo(f"Imported: {results['imported']}")
    click.echo(f"Skipped (duplicates): {results['skipped']}")
    click.echo(f"Errors: {results['errors']}")


@cli.command()
def seed_team_stats():
    """Generate TeamStats from historical match data."""
    from src.config import setup_logging
    from src.fetchers.historical_data_seeder import HistoricalDataSeeder

    setup_logging()

    click.echo("Generating TeamStats from historical matches...")

    seeder = HistoricalDataSeeder()
    results = seeder.seed_team_stats()

    click.echo("\nTeam Stats Results:")
    click.echo("-" * 40)
    click.echo(f"Teams processed: {results['teams']}")
    click.echo(f"Stats entries created: {results['stats_created']}")


@cli.command()
@click.option("--days", default=180, help="Lookback period in days")
@click.option("--market", default="all", help="Market: 1x2, ou, btts, or all")
@click.option("--min-edge", default=0.0, help="Minimum edge filter")
@click.option("--output", default="table", help="Output format: table or json")
def backtest(days: int, market: str, min_edge: float, output: str):
    """Run backtest on historical predictions."""
    from src.config import setup_logging
    from src.models.backtester import Backtester, BacktestParams

    setup_logging()

    click.echo(f"Running backtest: {days} days, market={market}, min_edge={min_edge}")

    backtester = Backtester()
    params = BacktestParams(days=days, market=market, min_edge=min_edge)
    report = backtester.run(params)

    if output == "json":
        click.echo(backtester.to_json(report))
    else:
        _print_backtest_table(report)


def _print_backtest_table(report):
    """Print backtest report as formatted table."""
    click.echo("\n" + "=" * 50)
    click.echo("BACKTEST RESULTS")
    click.echo("=" * 50)

    click.echo(f"\nPeriod: {report.period_start.date()} to {report.period_end.date()}")
    click.echo(f"Predictions: {report.n_predictions}")
    click.echo(f"Correct: {report.n_correct}")

    click.echo("\n--- Performance ---")
    click.echo(f"Win Rate: {report.win_rate:.1%}")
    click.echo(f"ROI: {report.roi_pct:+.2f}%")
    click.echo(f"Profit Factor: {report.profit_factor:.2f}")

    click.echo("\n--- Statistical ---")
    click.echo(f"Brier Score: {report.brier_score:.4f}")
    click.echo(f"Calibration Error: {report.calibration_error:.4f}")

    click.echo("\n--- Risk ---")
    click.echo(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    click.echo(f"Max Drawdown: {report.max_drawdown:.2f}")
    click.echo(f"Win Streak: {report.max_win_streak}")
    click.echo(f"Lose Streak: {report.max_lose_streak}")

    if report.by_market:
        click.echo("\n--- By Market ---")
        for market, stats in report.by_market.items():
            click.echo(f"\n{market.upper()}:")
            click.echo(f"  Predictions: {stats.n_predictions}")
            click.echo(f"  Win Rate: {stats.win_rate:.1%}")
            click.echo(f"  ROI: {stats.roi_pct:+.2f}%")
            click.echo(f"  Avg Edge: {stats.avg_edge:.2%}")


@cli.command()
@click.option("--target", default="all", help="Target: ou, btts, or all")
@click.option("--min-samples", default=100, help="Minimum training samples")
@click.option("--force", is_flag=True, help="Save model even if worse")
def retrain(target: str, min_samples: int, force: bool):
    """Retrain ML models with holdout validation."""
    from src.config import setup_logging
    from src.models.training_pipeline import training_pipeline

    setup_logging()

    click.echo(f"Retraining models: target={target}, min_samples={min_samples}")

    results = training_pipeline.retrain_with_validation(
        target=target,
        min_samples=min_samples,
        force_save=force,
    )

    _print_retrain_results(results)


def _print_retrain_results(results: dict):
    """Print retrain results as formatted output."""
    click.echo("\n" + "=" * 50)
    click.echo("RETRAIN RESULTS")
    click.echo("=" * 50)

    click.echo(f"\nStatus: {results['status']}")

    if results["status"] == "error":
        click.echo(f"Reason: {results.get('reason', 'unknown')}")
        if "n_examples" in results:
            click.echo(f"Examples found: {results['n_examples']}")
        return

    click.echo(f"Train samples: {results.get('n_train', 'N/A')}")
    click.echo(f"Test samples: {results.get('n_test', 'N/A')}")

    if "targets" in results:
        for target, data in results["targets"].items():
            click.echo(f"\n{target.upper()} Model:")
            click.echo(f"  Status: {data['status']}")
            if data.get("old_accuracy"):
                click.echo(f"  Old Accuracy: {data['old_accuracy']:.1%}")
            click.echo(f"  New Accuracy: {data['new_accuracy']:.1%}")
            click.echo(f"  Improvement: {data['improvement']:+.1%}")


@cli.command()
@click.option("--days", default=14, help="Days ahead to fetch (default 14)")
@click.option("--api-key", envvar="FOOTBALL_DATA_API_KEY", help="football-data.org API key")
def seed_fixtures(days: int, api_key: str):
    """Fetch and seed upcoming EPL fixtures."""
    from src.config import setup_logging
    from src.fetchers.fixture_fetcher import FixtureFetcher

    setup_logging()

    click.echo(f"Fetching upcoming EPL fixtures ({days} days ahead)...")

    fetcher = FixtureFetcher(api_key=api_key)
    try:
        results = fetcher.seed_fixtures(days_ahead=days)

        click.echo("\nFixture Seeding Results:")
        click.echo("-" * 40)
        click.echo(f"Total fixtures found: {results['total']}")
        click.echo(f"Imported: {results['imported']}")
        click.echo(f"Skipped (existing): {results['skipped']}")
        click.echo(f"With estimated odds: {results['with_odds']}")
    finally:
        fetcher.close()


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
