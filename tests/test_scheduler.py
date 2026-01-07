"""Tests for scheduler service."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.services.scheduler import SchedulerService, get_scheduler


class TestSchedulerService:
    """Tests for SchedulerService."""

    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        scheduler = SchedulerService()
        assert scheduler._scheduler is None
        assert scheduler.is_running is False

    def test_scheduler_singleton(self):
        """Test get_scheduler returns singleton."""
        # Reset singleton
        import src.services.scheduler as scheduler_module
        scheduler_module._scheduler = None

        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_scheduler_start(self):
        """Test scheduler start adds jobs."""
        with patch("apscheduler.schedulers.asyncio.AsyncIOScheduler") as mock_scheduler_class:
            mock_scheduler = MagicMock()
            mock_scheduler.get_jobs.return_value = ["job1", "job2", "job3"]
            mock_scheduler_class.return_value = mock_scheduler

            scheduler = SchedulerService()
            scheduler.start()

            assert scheduler.is_running is True
            mock_scheduler.start.assert_called_once()

            # Check jobs were added
            assert mock_scheduler.add_job.call_count == 3

    def test_scheduler_stop(self):
        """Test scheduler stop."""
        with patch("apscheduler.schedulers.asyncio.AsyncIOScheduler") as mock_scheduler_class:
            mock_scheduler = MagicMock()
            mock_scheduler.get_jobs.return_value = []
            mock_scheduler_class.return_value = mock_scheduler

            scheduler = SchedulerService()
            scheduler.start()
            scheduler.stop()

            assert scheduler.is_running is False
            mock_scheduler.shutdown.assert_called_once_with(wait=False)

    def test_scheduler_stop_when_not_running(self):
        """Test scheduler stop when not running is safe."""
        scheduler = SchedulerService()
        scheduler.stop()  # Should not raise

        assert scheduler.is_running is False


class TestSchedulerScanAndAlert:
    """Tests for scan_and_alert job."""

    @pytest.mark.asyncio
    async def test_scan_and_alert_handles_errors(self):
        """Test scan handles errors gracefully."""
        scheduler = SchedulerService()

        with patch("src.fetchers.DataOrchestrator") as mock_orch:
            mock_instance = AsyncMock()
            mock_instance.fetch_all = AsyncMock(side_effect=Exception("Network error"))
            mock_orch.return_value = mock_instance

            with patch("src.alerts.TelegramAlertService") as mock_tg:
                mock_tg_instance = AsyncMock()
                mock_tg.return_value = mock_tg_instance

                await scheduler.scan_and_alert()

                # Should send error alert
                mock_tg_instance.send_error_alert.assert_called_once()


class TestSchedulerDailyRefresh:
    """Tests for daily_refresh job."""

    @pytest.mark.asyncio
    async def test_daily_refresh_success(self):
        """Test daily refresh fetches data."""
        scheduler = SchedulerService()

        with patch("src.fetchers.DataOrchestrator") as mock_orch:
            mock_instance = AsyncMock()
            mock_instance.fetch_all = AsyncMock(return_value={})
            mock_orch.return_value = mock_instance

            await scheduler.daily_refresh()

            mock_instance.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_daily_refresh_handles_errors(self):
        """Test daily refresh handles errors."""
        scheduler = SchedulerService()

        with patch("src.fetchers.DataOrchestrator") as mock_orch:
            mock_instance = AsyncMock()
            mock_instance.fetch_all = AsyncMock(side_effect=Exception("API error"))
            mock_orch.return_value = mock_instance

            # Should not raise
            await scheduler.daily_refresh()


class TestSchedulerMatchdayScans:
    """Tests for matchday intensive scanning."""

    @pytest.mark.asyncio
    async def test_check_matchday_scans_no_imminent(self):
        """Test matchday check with no imminent matches."""
        scheduler = SchedulerService()

        with patch("src.storage.database.db") as mock_db:
            # Create a proper context manager mock
            mock_session_cm = MagicMock()
            mock_session = MagicMock()
            mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cm.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.count.return_value = 0
            mock_db.session.return_value = mock_session_cm

            with patch.object(scheduler, "scan_and_alert", new_callable=AsyncMock) as mock_scan:
                await scheduler.check_matchday_scans()

                # Should not trigger scan
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_matchday_scans_with_imminent(self):
        """Test matchday check triggers scan when matches imminent."""
        scheduler = SchedulerService()

        with patch("src.storage.database.db") as mock_db:
            # Create a proper context manager mock
            mock_session_cm = MagicMock()
            mock_session = MagicMock()
            mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cm.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.count.return_value = 2
            mock_db.session.return_value = mock_session_cm

            with patch.object(scheduler, "scan_and_alert", new_callable=AsyncMock) as mock_scan:
                await scheduler.check_matchday_scans()

                # Should trigger scan
                mock_scan.assert_called_once()


class TestSchedulerAnalyzeAndStore:
    """Tests for _analyze_and_store helper."""

    @pytest.mark.asyncio
    async def test_analyze_and_store_no_odds(self):
        """Test analysis returns empty when no odds available."""
        scheduler = SchedulerService()

        match = {
            "id": 1,
            "external_id": "ext1",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        }

        with patch("src.storage.database.db") as mock_db:
            # Create proper context manager mock
            mock_session_cm = MagicMock()
            mock_session = MagicMock()
            mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cm.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
            mock_db.session.return_value = mock_session_cm

            signals = await scheduler._analyze_and_store(match)

            assert signals == []

    @pytest.mark.asyncio
    async def test_analyze_and_store_no_team_stats(self):
        """Test analysis returns empty when no team stats available."""
        scheduler = SchedulerService()

        match = {
            "id": 1,
            "external_id": "ext1",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
        }

        with patch("src.storage.database.db") as mock_db:
            # Create proper context manager mock
            mock_session_cm = MagicMock()
            mock_session = MagicMock()
            mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cm.__exit__ = MagicMock(return_value=False)

            # Return odds but no team stats
            mock_odds = MagicMock()
            mock_odds.bf_home_odds = 2.5

            call_count = [0]

            def query_side_effect(model):
                query = MagicMock()
                call_count[0] += 1
                # First call is for OddsSnapshot
                if call_count[0] == 1:
                    query.filter.return_value.order_by.return_value.first.return_value = mock_odds
                else:
                    # Team stats calls
                    query.filter.return_value.order_by.return_value.first.return_value = None
                return query

            mock_session.query.side_effect = query_side_effect
            mock_db.session.return_value = mock_session_cm

            signals = await scheduler._analyze_and_store(match)

            assert signals == []
