# EPL Bet Indicator v2

Multi-market sports betting indicator for EPL prediction markets (Polymarket).

## Features

- **3 Markets:** 1X2, Over/Under (1.5, 2.5, 3.5), BTTS
- **4 Data Sources:** Betfair Exchange, Understat (xG), Club ELO, Polymarket
- **Unified Poisson Model:** Calculate once, derive all markets
- **Market-Specific Weights:** Different profiles for each market type
- **Time-Based Auto-Adjustment:** Weights shift closer to kick-off
- **Streamlit Dashboard:** Visual analytics
- **Telegram Alerts:** Real-time notifications

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

### Running

```bash
# Run dashboard
streamlit run src/ui/dashboard.py

# Run Telegram bot
python -m src.ui.telegram_bot
```

## Architecture

```
src/
├── fetchers/      # Data acquisition (Betfair, Understat, ClubELO, Polymarket)
├── models/        # Core calculations (Poisson, weights, edge detection)
├── storage/       # SQLite persistence
├── ui/            # Streamlit dashboard + Telegram bot
└── config/        # Settings and weight configurations
```

See `plans/260106-1749-epl-bet-indicator/plan.md` for detailed documentation.

## Weight Profiles

The indicator uses time-based weight profiles that automatically adjust:

| Profile | Hours to Kickoff | Strategy |
|---------|------------------|----------|
| Analytics First | >96 hours | Trust xG/ELO before market adjusts |
| Balanced | 24-96 hours | Equal trust in all sources |
| Market Trust | <24 hours | Trust efficient market pricing |

## Edge Thresholds

| Market | Min Edge |
|--------|----------|
| 1X2 | 5% |
| Over/Under | 7% |
| BTTS | 7% |

## License

MIT
