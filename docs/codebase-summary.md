# EPL Bet Indicator - Codebase Summary

**Status:** ML Enhancement Complete - Hybrid Edge Detection
**Date:** January 7, 2026
**Version:** 3.0
**Total LOC:** 5,400+ (src) + 3,200+ (tests) = 8,600+

## Overview

Comprehensive sports betting indicator for EPL with multi-market edge detection (1X2, Over/Under, BTTS), **hybrid Poisson + ML ensemble**, true edge calculation with base rates, time-adaptive weights, Streamlit dashboard (4 pages + ML Insights), Telegram alerts, A/B testing framework, and APScheduler automation. **218 passing tests** across all components.

## Project Structure

```
epl-bet-indicator/
├── src/                          # 5,400+ LOC - Core application
│   ├── alerts/                   # 362 LOC - Telegram notifications
│   │   ├── __init__.py
│   │   ├── telegram_service.py   # Service with retry logic
│   │   └── templates.py          # Alert message formatting
│   ├── config/                   # 173 LOC - Configuration & settings
│   │   ├── __init__.py           # Logging setup
│   │   ├── settings.py           # Pydantic models (validated env vars)
│   │   └── weights/
│   │       ├── 1x2_weights.yaml  # Match result profiles
│   │       ├── ou_weights.yaml   # Over/Under profiles
│   │       └── btts_weights.yaml # Both teams to score profiles
│   ├── fetchers/                 # 971 LOC - API data fetchers
│   │   ├── __init__.py
│   │   ├── betfair_fetcher.py    # Exchange odds with SSL auth
│   │   ├── clubelo_fetcher.py    # Team strength ratings
│   │   ├── polymarket_fetcher.py # Prediction market prices
│   │   └── understat_fetcher.py  # xG/xGA + shots metrics scraper
│   ├── models/                   # 1,900+ LOC - Probability, ML & edge calculations
│   │   ├── __init__.py
│   │   ├── edge_detector.py      # Edge detection + HybridEdgeDetector
│   │   ├── market_probabilities.py # Derive 1X2/OU/BTTS from Poisson
│   │   ├── poisson_matrix.py     # Modified Poisson distribution
│   │   ├── weight_engine.py      # Time-based weight application
│   │   ├── base_rate_tracker.py  # Historical outcome tracking (NEW)
│   │   ├── feature_builder.py    # ML feature engineering (NEW)
│   │   ├── ml_classifier.py      # LGB/XGB/RF ensemble (NEW)
│   │   ├── training_data.py      # Training data collection (NEW)
│   │   ├── training_pipeline.py  # End-to-end training (NEW)
│   │   ├── ab_testing.py         # A/B testing framework (NEW)
│   │   └── performance_logger.py # Prediction metrics (NEW)
│   ├── services/                 # 344 LOC - Background services
│   │   ├── __init__.py
│   │   └── scheduler.py          # APScheduler daemon with graceful shutdown
│   ├── storage/                  # 180 LOC - Database persistence
│   │   ├── __init__.py
│   │   ├── database.py           # SQLite connection & queries
│   │   └── models.py             # SQLAlchemy ORM models
│   ├── ui/                       # 1,400+ LOC - Streamlit dashboard
│   │   ├── __init__.py
│   │   ├── dashboard.py          # Main app (multipage router)
│   │   ├── components/           # Reusable UI elements
│   │   │   ├── __init__.py
│   │   │   ├── edge_badge.py     # Color-coded edge indicator
│   │   │   └── probability_bar.py # Visual probability display
│   │   └── pages/                # 4 dashboard pages
│   │       ├── __init__.py
│   │       ├── live_signals.py   # Alerts + model source indicator
│   │       ├── match_analysis.py # Probability + ML Insights tab
│   │       ├── historical.py     # Edge tracking (7-day, 30-day)
│   │       └── settings_page.py  # Configuration UI
│   ├── __init__.py
│   ├── cli.py                    # 250+ LOC - CLI commands (9 commands)
│   └── main.py                   # 123 LOC - Entry point
├── tests/                        # 3,200+ LOC - Test suite (218 tests passing)
├── .env.example                  # Environment template
├── pyproject.toml                # Dependencies & metadata
├── README.md                     # User documentation
└── data/
    └── epl_indicator.db          # SQLite persistence
```

## Core Modules

### Configuration Layer (`src/config/`)

**settings.py** - Pydantic settings with validation (170 LOC)
- `BetfairSettings`: username, password, app_key, certs_path
- `TelegramSettings`: bot_token, chat_id
- `PolymarketSettings`: private_key (optional)
- `AppSettings`: log_level, database_path, scan_interval, edge thresholds
- `load_weight_config(market_type)`: Load YAML profiles
- `get_weight_profile(market_type, hours_to_kickoff)`: Adaptive selection

**Weight Profiles** (3 time-based strategies per market)
- Early (>96h): Analytics-first (trust xG 35%, ELO 25%, Betfair 30%)
- Mid (24-96h): Balanced approach (equal trust)
- Late (<24h): Market-trust (Betfair 55%, xG 20%)

### Data Fetchers Layer (`src/fetchers/`)

**BetfairFetcher** (300 LOC)
- SSL certificate authentication
- Parse implied odds from EPL market
- Retry logic: 3 attempts with exponential backoff
- Cache: 5-min in-memory, 24h disk

**UnderstatFetcher** (250 LOC)
- Scrape xG/xGA metrics for teams
- Season data parsing
- Cache fallback if current season unavailable

**ClubELOFetcher** (200 LOC)
- Fetch team strength ratings
- Weekly update frequency
- Public API integration

**PolymarketFetcher** (220 LOC)
- py-clob-client integration
- Query orderbook (bid/ask/depth)
- List active EPL markets

### Processing Layer (`src/models/`)

**PoissonMatrix** (180 LOC)
- Calculate probability distribution: 0-10 goals
- Modified Poisson accounts for 1-goal bias in football
- Input: xG/xGA, home advantage (1.4 multiplier)

**MarketProbabilities** (250 LOC)
- Derive from Poisson matrix:
  - P(Home Win), P(Draw), P(Away Win) → 1X2
  - P(Over 1.5/2.5/3.5) → Under/Over
  - P(BTTS Yes) → Both Teams Score
- Validation: all probabilities in [0, 1]

**WeightEngine** (280 LOC)
- Apply time-based weights to signals
- Formula: Sum(signal * weight) across 4 sources
- Support 3 profile types (analytics_first, balanced, market_trust)

**EdgeDetector** (280 LOC) + **HybridEdgeDetector** (170 LOC)
- Calculate edge: (Our Prob - Market Prob) / Market Prob
- True edge: accounts for base rates
- Filter by threshold (1X2: 5%, OU: 7%, BTTS: 7%)
- HybridEdgeDetector: blends Poisson + ML with confidence weighting

**MLClassifier** (285 LOC) - NEW
- Ensemble classifier: LightGBM + XGBoost + RandomForest
- Probability calibration with CalibratedClassifierCV
- Graceful degradation (works with RF only if LGB/XGB unavailable)
- Confidence levels: high (>0.7), medium (0.5-0.7), low (<0.5)

**FeatureBuilder** (180 LOC) - NEW
- Build 14-feature vector from match data
- Features: xG, xGA, shots, shots on target, ELO, form
- Rolling window support (5-match, 10-match)

**TrainingDataCollector** (189 LOC) - NEW
- Mine historical matches from database
- Generate training labels (1x2, over_2_5, btts)
- CSV export/import for offline analysis

**TrainingPipeline** (116 LOC) - NEW
- 5-stage workflow: collect → prep → train OU → train BTTS → importance
- Requires 100+ completed matches minimum
- Incremental update support (10+ new matches)

**ABTestTracker** (245 LOC) - NEW
- Log predictions from Poisson, ML, and Hybrid
- Track actual outcomes and calculate win rates
- Compute agreement rate between models
- JSON persistence per day

**PerformanceLogger** (112 LOC) - NEW
- Real-time prediction logging (JSONL format)
- Daily/weekly aggregation stats
- Grouped by market type

### Storage Layer (`src/storage/`)

**Database Models** (180 LOC)
```sql
matches:  date, home_team, away_team, goals, season
alerts:   match_id, market_type, our_prob, market_prob, edge, timestamp
edges:    match_id, market_type, hours_to_kickoff, edge, timestamp
```

**Queries:**
- Store alert when edge >threshold
- Retrieve for dashboard (live signals, historical)
- Backtest queries (win rate, ROI by market)

### Dashboard (`src/ui/`)

**Main Dashboard** (dashboard.py, 400 LOC)
- Multipage Streamlit router
- Real-time data refresh
- Responsive mobile layout

**Pages:**
1. **Live Signals** - Active alerts with edge breakdown, market data
2. **Match Analysis** - Detailed probability distribution for selected match
3. **Historical** - 7-day, 30-day edge tracking charts, win rate stats
4. **Settings** - Configure edge thresholds, scan interval, API keys

**Components:**
- `EdgeBadge`: Color-coded indicator (green >7%, yellow 5-7%, red <5%)
- `ProbabilityBar`: Visual probability scale 0-100%

### Alerts System (`src/alerts/`)

**TelegramService** (200 LOC)
- Format alert messages with match details
- Send via python-telegram-bot
- Retry logic: 3 attempts with backoff
- Error handling & logging

**Templates** (150 LOC)
- `alert_template`: `[MATCH] Team A vs B | Market: 1X2 | Edge: 8.3%`
- Format probabilities, odds, timestamps
- Support for all 3 market types

### Scheduler (`src/services/`)

**APScheduler Daemon** (340 LOC)
- Run scan every 15 min (configurable)
- Fetch → Calculate → Filter → Alert workflow
- Graceful shutdown: SIGTERM handling
- Job timeout: 60s with error recovery

### CLI (`src/cli.py`)

**9 Commands:**
1. `run` - Start scheduler daemon (continuous scanning)
2. `dashboard` - Launch Streamlit UI
3. `fetch` - Fetch all API data (one-off)
4. `analyze <match_id>` - Detailed probability analysis
5. `scan` - Execute single scan cycle
6. `status` - Check system status, next scan time
7. `init_db` - Initialize SQLite database
8. `test_telegram` - Verify Telegram connectivity
9. `train-models` - Train ML classifiers (NEW)

## Key Features Implemented

### Multi-Market Edge Detection
- **1X2**: Home Win, Draw, Away Win
- **Over/Under**: 1.5, 2.5, 3.5 goals
- **BTTS**: Both Teams to Score (Yes/No)

### Time-Adaptive Weights
- Auto-select profile based on hours to kickoff
- 3 strategies: analytics_first, balanced, market_trust
- Adjustable thresholds per market

### ML-Enhanced Predictions (NEW)
- **Hybrid Edge Detection**: Blends Poisson + ML ensemble
- **True Edge**: Accounts for historical base rates
- **Confidence Weighting**: High (60% ML), Medium (40% ML), Low (20% ML)
- **Ensemble Classifier**: LightGBM + XGBoost + RandomForest
- **14-Feature Vector**: xG, shots, ELO, form, H2H
- **A/B Testing**: Track Poisson vs ML vs Hybrid accuracy
- **Performance Logging**: Daily/weekly prediction metrics

### Data Integration
- 4 sources: Betfair odds, Understat xG, Club ELO, Polymarket
- Retry logic with exponential backoff
- Cache: in-memory (5m) + disk (24h)

### Automation
- APScheduler: 15-min intervals (configurable)
- Telegram alerts: real-time with retry
- SQLite persistence: audit trail & backtesting

## Testing

**Coverage:** 3,200+ LOC across 218 passing tests
**Test Types:**
- Unit tests: Config, calculations, edge detection, ML classifiers
- Integration tests: End-to-end workflows, hybrid detection
- Database tests: Persistence & queries
- API tests: Fetcher error handling
- ML tests: Training, prediction, calibration

**Key Test Suites:**
- `test_config.py` - Settings validation
- `test_models.py` - Poisson, probabilities, edges
- `test_fetchers.py` - API integration with mocks
- `test_alerts.py` - Telegram formatting & retry
- `test_scheduler.py` - Job execution & shutdown
- `test_ui.py` - Dashboard components
- `test_ml_classifier.py` - ML ensemble training & prediction (NEW)
- `test_integration.py` - Hybrid detection, A/B testing (NEW)
- `test_features.py` - Feature builder validation (NEW)

## Dependencies

**Core:**
- pandas, numpy, scipy - Data manipulation & statistics
- pydantic>=2.0 - Settings validation
- python-dotenv, pyyaml - Configuration
- scikit-learn>=1.3.0 - ML foundation (NEW)

**ML (Optional):**
- lightgbm>=4.0.0 - Gradient boosting (optional)
- xgboost>=2.0.0 - Extreme gradient boosting (optional)

**API Clients:**
- betfairlightweight - Betfair Exchange
- understatapi - xG metrics
- py-clob-client - Polymarket CLOB
- httpx, beautifulsoup4 - HTTP & web scraping

**UI & Alerts:**
- streamlit>=1.28 - Dashboard
- plotly>=5.18 - Charts
- python-telegram-bot>=20.0 - Alerts

**Backend:**
- sqlalchemy>=2.0 - ORM
- apscheduler>=3.10 - Job scheduling

**Dev:**
- pytest, pytest-cov - Testing
- ruff, mypy - Linting/type checking

## Configuration

### Environment Variables
```
# Betfair
BETFAIR_USERNAME
BETFAIR_PASSWORD
BETFAIR_APP_KEY
BETFAIR_CERTS_PATH

# Telegram
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID

# App
LOG_LEVEL (default: INFO)
DATABASE_PATH (default: ./data/epl_indicator.db)
SCAN_INTERVAL_MINUTES (default: 15)
EDGE_THRESHOLD_1X2 (default: 0.05)
EDGE_THRESHOLD_OU (default: 0.07)
EDGE_THRESHOLD_BTTS (default: 0.07)
```

### Weight Profiles (YAML)

Three files in `src/config/weights/`:
- `1x2_weights.yaml` - Match result profiles
- `ou_weights.yaml` - Over/Under profiles
- `btts_weights.yaml` - Both teams to score profiles

Each with 3 time-based strategies (analytics_first, balanced, market_trust).

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Fetch latency | <2s | <500ms (cached) |
| Calculation | <1s (5 matches) | ~200ms |
| Alert latency | <5 min | <1 min |
| Memory | <1GB | ~150MB |
| Test coverage | >95% critical paths | 100% on models |

## Phase Completion Status

| Phase | Deliverable | Status | LOC |
|-------|-------------|--------|-----|
| 1 | Config system | ✓ | 173 |
| 2 | Data fetchers | ✓ | 971 |
| 3 | Models & processing | ✓ | 876 |
| 4 | Dashboard | ✓ | 1,155 |
| 5 | Alerts & scheduler | ✓ | 706 |
| 6 | Testing & validation | ✓ | 2,350 tests |
| ML-1 | Base Rate & True Edge | ✓ | 330 |
| ML-2 | Features & Windows | ✓ | 360 |
| ML-3 | ML Classification | ✓ | 590 |
| ML-4 | Integration & Testing | ✓ | 820 |

**Total Implementation:** 5,400+ LOC (src) + 3,200+ LOC (tests)

## Key Design Decisions

1. **Pydantic v2** - Type-safe settings with full validation
2. **YAML weight configs** - Easy adjustment without redeployment
3. **SQLite** - Local persistence for dev, PostgreSQL option for prod
4. **Streamlit** - Rapid dashboard development
5. **APScheduler** - Reliable background job execution
6. **py-clob-client** - Direct Polymarket integration
7. **Retry logic** - Resilient API calls with exponential backoff
8. **Hybrid ML** - Ensemble classifier (LGB/XGB/RF) with graceful degradation
9. **True Edge** - Base rate accounting prevents false confidence
10. **A/B Testing** - Measure ML improvement vs Poisson baseline

## What's Next

**ML Enhancement Complete:**
- ✓ Base rate tracking and true edge calculation
- ✓ Feature engineering (14-feature vector)
- ✓ ML ensemble classifier
- ✓ Hybrid edge detection
- ✓ A/B testing framework
- ✓ Performance logging

**Future Enhancements:**
- Async I/O for parallel API fetches
- Redis caching layer
- PostgreSQL for production
- Kubernetes deployment
- Multi-user support
- Automated model retraining (quarterly)
- Feature drift monitoring

---

**Document Version:** 3.0 (Jan 7, 2026)
**Sync Status:** In sync with codebase (5,400+ LOC verified)
