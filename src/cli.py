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


@cli.command("backfill-predictions")
@click.option("--batch-size", default=100, help="Batch size for processing")
@click.option("--dry-run", is_flag=True, help="Show what would be done without saving")
def backfill_predictions(batch_size: int, dry_run: bool):
    """Generate predictions for completed matches missing predictions."""
    from src.config import setup_logging
    from src.storage.database import db
    from src.storage.models import Match, OddsSnapshot, Prediction, TeamStats
    from src.models import MatchAnalyzer
    from datetime import datetime

    setup_logging()

    click.echo("Backfilling predictions for completed matches...")

    # Create analyzer instance
    analyzer = MatchAnalyzer(use_ml=True)

    stats = {"processed": 0, "skipped_no_odds": 0, "skipped_no_stats": 0, "errors": 0}

    with db.session() as session:
        # Get completed matches without predictions
        subq = session.query(Prediction.match_id).distinct()
        matches = (
            session.query(Match)
            .filter(
                Match.is_completed == True,
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
                ~Match.id.in_(subq)
            )
            .order_by(Match.kickoff)
            .all()
        )

        total = len(matches)
        click.echo(f"Found {total} matches needing predictions")

        if dry_run:
            click.echo("[DRY RUN] Would process these matches:")
            for m in matches[:10]:
                click.echo(f"  - {m.home_team} vs {m.away_team} ({m.kickoff.date()})")
            if total > 10:
                click.echo(f"  ... and {total - 10} more")
            return

        # Process in batches
        for i in range(0, total, batch_size):
            batch = matches[i:i + batch_size]
            click.echo(f"\nProcessing batch {i // batch_size + 1} ({i + 1}-{min(i + batch_size, total)} of {total})")

            for match in batch:
                try:
                    result = _process_match_for_backfill(session, match, analyzer)
                    if result == "success":
                        stats["processed"] += 1
                    elif result == "no_odds":
                        stats["skipped_no_odds"] += 1
                    elif result == "no_stats":
                        stats["skipped_no_stats"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    click.echo(f"  Error processing {match.home_team} vs {match.away_team}: {e}")

            # Commit batch
            session.commit()
            click.echo(f"  Committed. Processed: {stats['processed']}")

    # Summary
    click.echo("\n" + "=" * 50)
    click.echo("BACKFILL COMPLETE")
    click.echo("=" * 50)
    click.echo(f"Processed: {stats['processed']}")
    click.echo(f"Skipped (no odds): {stats['skipped_no_odds']}")
    click.echo(f"Skipped (no stats): {stats['skipped_no_stats']}")
    click.echo(f"Errors: {stats['errors']}")


def _process_match_for_backfill(session, match, analyzer) -> str:
    """Process a single match for backfill. Returns status string."""
    from src.storage.models import OddsSnapshot, TeamStats, Prediction

    # Get odds snapshot
    odds = (
        session.query(OddsSnapshot)
        .filter_by(match_id=match.id)
        .order_by(OddsSnapshot.snapshot_time.desc())
        .first()
    )
    if not odds or not odds.bf_home_odds:
        return "no_odds"

    # Get team stats closest to match date
    home_stats = (
        session.query(TeamStats)
        .filter(
            TeamStats.team_name == match.home_team,
            TeamStats.snapshot_date <= match.kickoff
        )
        .order_by(TeamStats.snapshot_date.desc())
        .first()
    )
    away_stats = (
        session.query(TeamStats)
        .filter(
            TeamStats.team_name == match.away_team,
            TeamStats.snapshot_date <= match.kickoff
        )
        .order_by(TeamStats.snapshot_date.desc())
        .first()
    )

    if not home_stats or not away_stats:
        return "no_stats"

    # Run analyzer with ML features
    is_elite = getattr(home_stats, 'is_elite', False) or getattr(away_stats, 'is_elite', False)

    match_probs, weighted_probs, edge_analysis = analyzer.analyze(
        match_id=str(match.id),
        home_team=match.home_team,
        away_team=match.away_team,
        kickoff=match.kickoff,
        home_xg=home_stats.home_xg or 1.2,
        away_xg=away_stats.away_xg or 1.2,
        home_xga=home_stats.home_xga or 1.2,
        away_xga=away_stats.away_xga or 1.2,
        home_elo=home_stats.elo_rating or 1500,
        away_elo=away_stats.elo_rating or 1500,
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
        # ML features
        home_shots_pg=getattr(home_stats, 'shots_per_game', None),
        away_shots_pg=getattr(away_stats, 'shots_per_game', None),
        home_volatility=getattr(home_stats, 'goal_volatility', None),
        away_volatility=getattr(away_stats, 'goal_volatility', None),
        home_conversion=getattr(home_stats, 'shot_conversion_rate', None),
        away_conversion=getattr(away_stats, 'shot_conversion_rate', None),
        is_elite_match=is_elite,
    )

    # Extract signal outcomes
    signal_1x2 = edge_analysis.signal_1x2.outcome if edge_analysis.signal_1x2 else "none"
    signal_ou = edge_analysis.signal_ou.outcome if edge_analysis.signal_ou else "none"
    signal_btts = edge_analysis.signal_btts.outcome if edge_analysis.signal_btts else "none"

    # Calculate actual outcomes
    h, a = match.home_goals, match.away_goals
    actual_1x2 = "home" if h > a else ("away" if a > h else "draw")
    actual_ou = "over" if (h + a) > 2.5 else "under"
    actual_btts = "yes" if (h > 0 and a > 0) else "no"

    # Create prediction using correct attribute names
    prediction = Prediction(
        match_id=match.id,
        preset_used=weighted_probs.preset_name or "balanced",
        # 1X2 - use fair_1x2 dict from WeightedProbabilities
        p_home_model=weighted_probs.fair_1x2.get("home", 0.33),
        p_draw_model=weighted_probs.fair_1x2.get("draw", 0.33),
        p_away_model=weighted_probs.fair_1x2.get("away", 0.33),
        p_home_market=match_probs.betfair_1x2.get("home", 0.33),
        p_draw_market=match_probs.betfair_1x2.get("draw", 0.33),
        p_away_market=match_probs.betfair_1x2.get("away", 0.33),
        edge_1x2=edge_analysis.signal_1x2.edge if edge_analysis.signal_1x2 else 0,
        signal_1x2=signal_1x2,
        # O/U - use fair_ou dict
        p_over_2_5_model=weighted_probs.fair_ou.get("2.5", {}).get("over", 0.5),
        p_over_2_5_market=match_probs.betfair_ou.get("2.5", {}).get("over", 0.5),
        edge_ou=edge_analysis.signal_ou.edge if edge_analysis.signal_ou else 0,
        signal_ou=signal_ou,
        # BTTS - use fair_btts dict
        p_btts_yes_model=weighted_probs.fair_btts.get("yes", 0.5),
        p_btts_yes_market=match_probs.betfair_btts.get("yes", 0.5),
        edge_btts=edge_analysis.signal_btts.edge if edge_analysis.signal_btts else 0,
        signal_btts=signal_btts,
        # Outcome correctness
        is_1x2_correct=(signal_1x2 == actual_1x2) if signal_1x2 != "none" else None,
        is_ou_correct=(signal_ou == actual_ou) if signal_ou != "none" else None,
        is_btts_correct=(signal_btts == actual_btts) if signal_btts != "none" else None,
    )

    session.add(prediction)
    return "success"


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
