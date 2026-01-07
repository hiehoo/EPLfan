# Code Standards & Guidelines

**Version:** 2.0
**Status:** Phase 5 - Full Stack Implementation
**Updated:** January 7, 2026

## Python Standards (3.11+)

### Type Safety

**Requirement:** 100% type hints on public APIs, mypy strict mode

```python
# Good
from typing import Optional, Dict
from pathlib import Path

def get_weight_profile(market_type: str, hours_to_kickoff: float) -> Dict[str, float]:
    """Get adaptive weights for market and time."""
    pass

# Bad - missing return type
def get_weight_profile(market_type: str, hours_to_kickoff: float):
    pass

# Bad - missing parameter type
def get_weight_profile(market_type, hours_to_kickoff: float) -> Dict[str, float]:
    pass
```

**Rules:**
- All function parameters must have type hints
- All function returns must have type hints (except implicit `None`)
- Use `Optional[T]` for nullable, never `T | None` in public APIs (for Python 3.11 compat)
- Generic collections from `typing` module: `Dict`, `List`, `Tuple`
- Import from `typing` not `collections.abc` (for clarity)

### Code Style

**Formatting:** Ruff with line length 100

```bash
# Check
ruff check src/

# Fix
ruff check --fix src/
```

**Naming Conventions:**
- `Classes`: PascalCase (e.g., `BetfairSettings`)
- `Functions/Variables`: snake_case (e.g., `get_weight_profile`)
- `Constants`: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT_SECONDS`)
- `Private`: Leading underscore (e.g., `_internal_helper()`)

**Docstrings:** Google style

```python
def calculate_edge(our_prob: float, market_prob: float) -> float:
    """Calculate probabilistic edge percentage.

    Args:
        our_prob: Our estimated probability [0, 1]
        market_prob: Market-implied probability [0, 1]

    Returns:
        Edge as percentage (e.g., 0.05 = 5%)

    Raises:
        ValueError: If probabilities outside [0, 1]
    """
    if not (0 <= our_prob <= 1) or not (0 <= market_prob <= 1):
        raise ValueError("Probabilities must be in [0, 1]")
    return (our_prob - market_prob) / market_prob
```

### Testing Standards

**Coverage:** >95% on critical paths, 100% on models/calculations

**Structure:**
```python
# tests/test_module.py
import pytest
from src.module import my_function

class TestMyFunction:
    """Test my_function behavior."""

    def test_happy_path(self):
        """Test normal operation."""
        result = my_function(valid_input)
        assert result == expected_value

    def test_edge_case(self):
        """Test boundary conditions."""
        result = my_function(edge_case_input)
        assert result == expected_edge_value

    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

**Naming:**
- `test_<function>_<scenario>` (e.g., `test_calculate_edge_normal_case`)
- Describe expected behavior in docstring
- One assertion focus per test (ideally)

### Configuration & Secrets

**No Secrets in Code:**
```python
# Bad - hardcoded
API_KEY = "sk_live_abc123"

# Good
from src.config import settings
API_KEY = settings.betfair.app_key  # From .env
```

**Environment Variables:**
- Use Pydantic `BaseSettings` with `SettingsConfigDict`
- Prefix env vars by domain (e.g., `BETFAIR_USERNAME`)
- Provide `.env.example` with template (no real values)

**Configuration Files (YAML):**
```yaml
# src/config/weights/1x2_weights.yaml
market_trust:
  description: "Strategy for near kick-off"
  betfair: 0.55
  xg: 0.20
  # Weights must sum to 1.0
```

Rules:
- YAML for non-secret config (weights, time thresholds)
- Include descriptions for profiles
- Validate weights sum to 1.0 at load time

### Module Organization

```
src/
├── config/           # Settings, weights, logging
├── fetchers/         # API clients (Betfair, Understat, etc.)
├── models/           # Core calculations (Poisson, edges)
├── storage/          # Database layer (SQLAlchemy models)
└── ui/               # Streamlit dashboard, Telegram bot
```

**Module __init__.py:**
```python
"""Module description."""
from .submodule import function_a, ClassB

__all__ = ["function_a", "ClassB"]
```

### Error Handling

**Pattern:**
```python
# Log then raise
import logging

logger = logging.getLogger(__name__)

try:
    data = fetch_betfair_odds(match_id)
except requests.Timeout:
    logger.error(f"Betfair timeout for match {match_id}", exc_info=True)
    raise  # Re-raise, let caller decide handling

# Graceful degradation
try:
    xg_data = fetch_understat_xg()
except Exception as e:
    logger.warning(f"xG fetch failed, using cached: {e}")
    xg_data = cache.get("xg_latest")  # Fallback
```

**Rules:**
- Log at appropriate level (debug/info/warning/error)
- Use `exc_info=True` for exceptions
- Custom exceptions OK for domain logic
- Let errors bubble up, don't silently swallow

### Async Patterns (Future)

When async is needed in Phase 2+:
```python
# Use asyncio, not threading
import asyncio
from typing import Coroutine

async def fetch_multiple_odds(match_ids: List[int]) -> Dict[int, Odds]:
    """Fetch odds for multiple matches concurrently."""
    tasks = [fetch_odds_async(mid) for mid in match_ids]
    return dict(zip(match_ids, await asyncio.gather(*tasks)))

# In CLI/scheduler:
if __name__ == "__main__":
    result = asyncio.run(fetch_multiple_odds([1, 2, 3]))
```

## Pydantic Standards (v2+)

**Validation Pattern:**
```python
from pydantic import BaseModel, Field, field_validator

class Match(BaseModel):
    """EPL match data."""

    match_id: int = Field(..., gt=0)  # Must be positive
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    kickoff_time: datetime

    @field_validator("home_team", "away_team")
    @classmethod
    def validate_team_name(cls, v: str) -> str:
        """Ensure team names are not identical."""
        return v.strip().upper()

    @field_validator("kickoff_time")
    @classmethod
    def validate_kickoff_future(cls, v: datetime) -> datetime:
        """Kickoff must be in future."""
        if v <= datetime.now(tz=timezone.utc):
            raise ValueError("Kickoff time must be in future")
        return v
```

**Settings Pattern (BaseSettings):**
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class MySettings(BaseSettings):
    """Load from .env with env prefix."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MY_APP_",
        extra="ignore"  # Ignore unknown env vars
    )

    api_key: str
    timeout_seconds: int = 30  # Default
```

## Logging Standards

**Setup:**
```python
import logging

# In src/config/__init__.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/app.log"),
    ]
)
```

**Per-Module:**
```python
# In any module
import logging

logger = logging.getLogger(__name__)  # Use module name

logger.debug("Detailed info for devs")
logger.info("User-facing status updates")
logger.warning("Unexpected but recoverable")
logger.error("Something failed, needs investigation")
logger.critical("System down, immediate action needed")
```

**Rules:**
- Use module `__name__` for logger (enables granular filtering)
- Include context in messages: `logger.error(f"Failed to fetch {match_id}: {e}")`
- Use `exc_info=True` for exception logs: `logger.error("Error", exc_info=True)`
- Never log secrets (passwords, API keys)

## Git Commit Standards

**Message Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature (data fetcher, model, etc.)
- `fix`: Bug fix
- `refactor`: Code reorganization without feature change
- `test`: Test additions/modifications
- `docs`: Documentation
- `chore`: Config, dependencies, build scripts

**Example:**
```
feat(fetchers): add Understat xG data source

- Implement UnderstatScraper class with caching
- Parse season xG/xGA metrics
- Add 90%+ test coverage
- Validates data freshness (max 24h staleness)

Closes #12
```

**Scope:** Area of change (config, fetchers, models, tests, docs)
**Subject:** Imperative mood, lowercase, no period, <50 chars
**Body:** What/why, not how (code shows how), wrap at 72 chars
**Footer:** Issue references, breaking changes

## Linting & Type Checking

**Pre-Commit (Phase 2+):**
```bash
#!/bin/bash
# scripts/lint.sh
ruff check src/ tests/
mypy src/ --strict
pytest tests/ --cov=src --cov-fail-under=95
```

**CI/CD (Phase 3+):**
```yaml
# .github/workflows/test.yml
- run: ruff check src/ tests/
- run: mypy src/ --strict
- run: pytest tests/ --cov=src --cov-fail-under=95
```

## Documentation Standards

**Code Comments:**
- Explain *why*, not *what* (code is self-documenting)
- Use for complex algorithms, not simple statements

```python
# Good - explains reasoning
# We weight Betfair heavily late in week because market efficiency
# increases as match approaches and more sophisticated traders enter
weights["betfair"] = 0.55

# Bad - obvious from code
# Set betfair weight to 0.55
weights["betfair"] = 0.55
```

**Docstrings:**
- Google style for all public functions/classes
- Include Args, Returns, Raises, Examples (if complex)

```python
def get_poisson_distribution(
    lambda_param: float,
    max_goals: int = 10
) -> List[float]:
    """Get Poisson probability distribution for goals.

    Modified Poisson accounts for increased probability of
    exact 1-goal outcomes in football matches.

    Args:
        lambda_param: Expected value (mean) of distribution
        max_goals: Truncate distribution at this value

    Returns:
        List of probabilities for 0 to max_goals inclusive

    Raises:
        ValueError: If lambda_param < 0

    Example:
        >>> probs = get_poisson_distribution(lambda_param=1.5)
        >>> sum(probs)
        0.9999...  # Close to 1.0 due to truncation
    """
    pass
```

## CLI Commands Pattern

When adding CLI commands in `src/cli.py`:

```python
import click
from typing import Optional

@click.group()
def cli():
    """EPL Bet Indicator CLI."""
    pass

@cli.command()
@click.option('--interval', type=int, default=15, help='Scan interval in minutes')
def run(interval: int) -> None:
    """Start scheduler daemon (continuous scanning)."""
    from src.services.scheduler import run_scheduler
    run_scheduler(interval_minutes=interval)

@cli.command()
@click.argument('match_id', type=int)
def analyze(match_id: int) -> None:
    """Detailed probability analysis for match."""
    from src.models import EdgeDetector
    result = EdgeDetector.analyze(match_id)
    click.echo(result)
```

**Rules:**
- Use Click for CLI library
- Command names: snake_case (run, test_telegram, init_db)
- Type hints on all parameters
- Help text for all options
- Error handling with click.echo() or sys.exit(1)
- Async: wrap with asyncio.run() if needed

**Implemented Commands:**
1. `run` - Start scheduler daemon
2. `dashboard` - Launch Streamlit UI
3. `fetch` - Fetch all API data
4. `analyze <match_id>` - Detailed analysis
5. `scan` - Single scan cycle
6. `status` - System status
7. `init_db` - Initialize database
8. `test_telegram` - Verify Telegram

## Streamlit UI Pattern

When building Streamlit pages:

```python
# src/ui/pages/my_page.py
import streamlit as st
from src.storage import db

def render():
    """Render page content."""
    st.header("Page Title")

    # Fetch data
    data = db.query_alerts()

    # Display
    st.dataframe(data)

    # Refresh
    if st.button("Refresh"):
        st.rerun()

if __name__ == "__main__":
    render()
```

**Rules:**
- Each page: separate file in `src/ui/pages/`
- Cached functions with `@st.cache_data`
- Auto-refresh on data update
- Responsive layout (use columns)
- Color-coded indicators (green, yellow, red)

## Scheduler Pattern (Phase 5)

When implementing background jobs:

```python
# src/services/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
import signal

scheduler = BackgroundScheduler()

def run_scheduler(interval_minutes: int = 15):
    """Run background scheduler."""

    def signal_handler(signum, frame):
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    scheduler.add_job(
        scan_job,
        'interval',
        minutes=interval_minutes,
        id='main_scan'
    )
    scheduler.start()

async def scan_job():
    """Fetch -> Calculate -> Alert workflow."""
    try:
        data = await fetch_all()
        edges = calculate_edges(data)
        await send_alerts(edges)
    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
```

**Rules:**
- Use APScheduler for job scheduling
- Graceful shutdown: SIGTERM handling
- Async support with asyncio.run()
- Job timeout: 60s with error recovery
- Log all job execution & errors

## Alert Template Pattern

When creating alert messages:

```python
# src/alerts/templates.py
def alert_template(match: Match, edge: Edge) -> str:
    """Format alert message."""
    return (
        f"[MATCH] {match.home_team} vs {match.away_team}\n"
        f"Market: {edge.market_type.upper()}\n"
        f"Edge: {edge.percent:.1%}\n"
        f"Our: {edge.our_prob:.0%} | Market: {edge.market_prob:.0%}"
    )
```

**Rules:**
- Format probabilities as percentages (0-100%)
- Include match, market, edge, probability
- Keep under 160 chars (Telegram optimal)
- Support all 3 market types (1X2, OU, BTTS)

## Acceptance Criteria Template

When implementing features, verify:

- [ ] Code passes `ruff check` with no warnings
- [ ] Code passes `mypy` in strict mode
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage >95% on critical paths: `pytest --cov`
- [ ] Docstrings added for all public functions
- [ ] No hardcoded secrets or credentials
- [ ] Error handling with appropriate logging
- [ ] Type hints on all public APIs
- [ ] Commit message follows standards
- [ ] CLI commands: help text + async support
- [ ] Streamlit pages: responsive + cached
- [ ] Alerts: formatted + retry logic

---

**Standards Enforced:** Phase 1-5 (all code follows strict type safety & testing)
