# EPL Bet Indicator v2 - Phase 5 Complete

Multi-market sports betting indicator for EPL prediction markets (Polymarket). Full-stack application with Poisson probability model, time-adaptive weights, Streamlit dashboard (4 pages), Telegram alerts, APScheduler automation, and 137 passing tests.

## Status

**Phase 5 Complete:** 4,400 LOC across 6 modules | 2,350 LOC tests | <1 month delivery

## Key Features

- **3 Markets:** 1X2 (match result), Over/Under (1.5/2.5/3.5), BTTS
- **4 Data Sources:** Betfair Exchange, Understat (xG), Club ELO, Polymarket
- **Unified Poisson Model:** Calculate once, derive all markets
- **Time-Adaptive Weights:** Auto-select strategy (analytics_first, balanced, market_trust)
- **Edge Detection:** Identify opportunities >5-7% across markets
- **Streamlit Dashboard:** 4 pages (live signals, match analysis, historical, settings)
- **Telegram Alerts:** Real-time formatted notifications with retry logic
- **APScheduler:** 15-min configurable scanning + graceful shutdown
- **CLI Commands:** 8 utilities (run, dashboard, fetch, analyze, scan, status, init_db, test_telegram)
- **SQLite Persistence:** Audit trail + backtesting data

## Setup

### Prerequisites

1. **Python 3.11** (tested with 3.11.x, 3.12+ may work but not tested)
2. Betfair account with API access
3. Telegram bot token

### Installation

```bash
# Clone repository
git clone <repo-url>
cd epl-bet-indicator

# Create virtual environment (use Python 3.11 specifically)
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Betfair SSL Certificates

1. Go to [Betfair Developer Portal](https://developer.betfair.com/)
2. Create self-signed certificates
3. Place in `./certs/` directory:
   - `client-2048.crt`
   - `client-2048.key`

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|----------|-------------|
| `BETFAIR_USERNAME` | Betfair account username |
| `BETFAIR_PASSWORD` | Betfair account password |
| `BETFAIR_APP_KEY` | Betfair API application key |
| `BETFAIR_CERTS_PATH` | Path to SSL certificates directory |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Chat ID to send alerts to |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Quick Start

```bash
# Initialize database
python -m src.cli init_db

# Test Telegram connectivity
python -m src.cli test_telegram

# Launch dashboard
python -m src.cli dashboard

# Run scheduler (continuous scanning)
python -m src.cli run --interval 15

# Single scan cycle
python -m src.cli scan

# Fetch data once
python -m src.cli fetch

# Analyze specific match
python -m src.cli analyze 12345

# Check system status
python -m src.cli status
```

## Project Structure

```
src/                          # 4,400 LOC - Core application
├── alerts/ (362 LOC)         # Telegram notifications + templates
├── config/ (173 LOC)         # Pydantic settings + weight profiles
├── fetchers/ (971 LOC)       # Betfair, Understat, Club ELO, Polymarket
├── models/ (876 LOC)         # Poisson, probabilities, weights, edges
├── services/ (344 LOC)       # APScheduler daemon
├── storage/ (180 LOC)        # SQLite ORM + queries
├── ui/ (1,155 LOC)           # Streamlit dashboard (4 pages + components)
├── cli.py (222 LOC)          # 8 CLI commands
└── main.py (123 LOC)         # Entry point

tests/                        # 2,350 LOC - 137 passing tests
docs/                         # Updated documentation (Phase 5)
```

See `docs/` for detailed architecture, standards, and codebase summary.

## Weight Profiles

The indicator uses time-based weight profiles that automatically adjust:

| Profile | Hours to Kickoff | Strategy |
|---------|------------------|----------|
| Analytics First | >96 hours | Trust xG/ELO before market adjusts |
| Balanced | 24-96 hours | Equal trust in all sources |
| Market Trust | <24 hours | Trust efficient market pricing |

## Edge Thresholds (Configurable)

| Market | Min Edge | Env Var |
|--------|----------|---------|
| 1X2 | 5% | EDGE_THRESHOLD_1X2 |
| Over/Under | 7% | EDGE_THRESHOLD_OU |
| BTTS | 7% | EDGE_THRESHOLD_BTTS |

## Testing

**Coverage:** 137 tests, 2,350 LOC
- Unit: Config, calculations, edge detection
- Integration: End-to-end workflows
- Database: Persistence & queries
- API: Fetcher error handling

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=src  # Coverage report
```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Fetch | <2s | <500ms (cached) |
| Calculate | <1s (5 matches) | ~200ms |
| Alert latency | <5 min | <1 min |
| Memory | <1GB | ~150MB |
| Tests pass | 100% | 137/137 ✓ |

## Documentation

- `docs/project-overview-pdr.md` - Product requirements & PDR
- `docs/codebase-summary.md` - Module descriptions (4,400 LOC verified)
- `docs/code-standards.md` - Coding patterns & acceptance criteria
- `docs/system-architecture.md` - Data flows, deployment, integration

## License

MIT
