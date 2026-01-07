# System Architecture

**Version:** 3.0 (ML Enhancement)
**Status:** Fully implemented & tested - ML Enhancement complete
**Updated:** January 7, 2026

## High-Level Overview

EPL Bet Indicator is a data pipeline that:
1. **Ingests** data from 4 external sources hourly
2. **Calculates** probabilistic edges using modified Poisson model
3. **Filters** opportunities by minimum edge threshold
4. **Alerts** via Telegram when edge detected
5. **Persists** all decisions for backtesting & analysis

```
External APIs ──> Fetchers ──> Processing ──> Storage ──> UI/Alerts
(Betfair,etc)      (Parse)     (Calculate)    (SQLite)   (Dashboard,
                                                          Telegram)
```

## Component Architecture

### 1. Configuration Layer (`src/config/`)

**Purpose:** Centralized settings, secrets management, logging

**Components:**
- `settings.py` - Pydantic models for env validation
- `weights/*.yaml` - Market-specific weight profiles
- `__init__.py` - Logging auto-setup

**Key Classes:**
```
AppSettings (parent)
├── BetfairSettings (username, password, app_key, certs_path)
├── TelegramSettings (bot_token, chat_id)
├── PolymarketSettings (private_key)
└── Edge thresholds (1x2: 5%, ou: 7%, btts: 7%)
```

**Data Flow:**
```
.env file ──> Pydantic Validators ──> AppSettings singleton
              ├─ Log level validation
              ├─ Path existence checks
              └─ Edge threshold bounds
```

**Logging Setup:**
- Stdout handler (console output)
- File handler (data/app.log, append mode)
- Format: `timestamp | LEVEL | logger_name | message`
- Auto-called on import (no manual setup needed)

### 2. Data Fetchers Layer (`src/fetchers/`) - COMPLETE (971 LOC)

**Purpose:** Abstract API clients, handle errors, cache responses

**Implemented Components:**
```
BetfairFetcher (300 LOC)
├── authenticate (SSL cert + username/password)
├── get_market_odds(match_id) → {1: 1.5, X: 3.2, 2: 4.1}
├── list_epl_markets() → List[Match]
├── Retry logic (3 attempts, exponential backoff: 1s/2s/4s)
└── Cache: 5-min in-memory, 24h disk

UnderstatFetcher (250 LOC)
├── scrape_season_data() → Dict[team, {xG, xGA, shots}]
├── cache_to_disk with TTL validation
└── Fallback to previous season if current unavailable

ClubELOFetcher (200 LOC)
├── get_elo_ratings() → Dict[team, float]
├── weekly update frequency
└── Cached ratings with 7-day TTL

PolymarketFetcher (220 LOC, py-clob-client)
├── get_orderbook(market_id) → {bid: price, ask: price, depth}
├── list_epl_markets_on_polymarket() → List[MarketInfo]
└── Market resolution tracking
```

**Error Handling:**
```
Network Error ──> Retry with exponential backoff (max 3 attempts)
                  ├─ 1s, 2s, 4s delays
                  └─ Use cached data if age < 30 min

API Rate Limit ──> Wait 60s, then retry

Data Parse Error ──> Log error, skip market, continue
```

**Caching Strategy:**
- In-memory: Recent fetches (TTL 5 min)
- Disk: Full responses (TTL 24 hours)
- Purpose: Reduce API calls, enable offline dev

### 3. Processing Layer (`src/models/`) - COMPLETE (1,900+ LOC)

**Purpose:** Calculate probabilities, edges, and ML-enhanced predictions

**Core Model: Modified Poisson Distribution (180 LOC)**

```
Input:
  - Home team xG (expected goals for)
  - Home team xGA (expected goals against)
  - Away team xG
  - Away team xGA
  - Home field advantage factor (~1.4)

Process:
  1. Adjust xG by home/away + team strength
  2. Calculate Poisson lambda for each team
  3. Generate probability tables (0-10 goals)
  4. Aggregate for 1X2 / Over-Under / BTTS

Output:
  - P(Home Win): 0-1
  - P(Draw): 0-1
  - P(Away Win): 0-1
  - P(Over 1.5), P(Over 2.5), P(Over 3.5)
  - P(BTTS Yes/No)
  - All with validation: sum to 1.0, bounds [0, 1]
```

**Components (1,900+ LOC total):**
- `PoissonMatrix` (180 LOC) - Distribution generation
- `MarketProbabilities` (250 LOC) - Derive 1X2/OU/BTTS
- `WeightEngine` (280 LOC) - Adaptive weight application
- `EdgeDetector` (280 LOC) - Edge calculation & filtering
- `BaseRateTracker` (150 LOC) - Historical outcome tracking
- `FeatureBuilder` (180 LOC) - ML feature engineering (14 features)
- `MLClassifier` (285 LOC) - Ensemble classifier (LGB/XGB/RF)
- `TrainingDataCollector` (189 LOC) - Training data mining
- `TrainingPipeline` (116 LOC) - End-to-end training orchestration
- `ABTestTracker` (245 LOC) - Model comparison framework
- `PerformanceLogger` (112 LOC) - Real-time metrics

**Weight Application:**

```python
def weighted_probability(
    betfair_prob: float,  # From odds
    xg_prob: float,       # From model
    elo_prob: float,
    form_prob: float,
    market_type: str,
    hours_to_kickoff: float
) -> float:
    weights = get_weight_profile(market_type, hours_to_kickoff)

    return (
        betfair_prob * weights["betfair"] +
        xg_prob * weights["xg"] +
        elo_prob * weights["elo"] +
        form_prob * weights["form"]
    )
```

**Edge Calculation:**

```python
def calculate_edge(our_prob: float, market_prob: float) -> float:
    """Returns decimal edge (0.05 = 5% edge)"""
    if market_prob == 0:
        return 0
    return (our_prob - market_prob) / market_prob
```

**True Edge Calculation (ML Enhancement):**

```python
def calculate_true_edge(model_prob: float, market_prob: float, base_rate: float) -> float:
    """Account for base rate when calculating edge.
    True edge = min(naive_edge, skill_component)
    Prevents false confidence from high base rate outcomes.
    """
    naive_edge = model_prob - market_prob
    skill_component = model_prob - max(base_rate, 1 - base_rate)
    return min(naive_edge, skill_component) if naive_edge > 0 else naive_edge
```

**ML Classification Ensemble:**

```
MLClassifier Components:
├── LightGBM (optional, requires lightgbm package)
├── XGBoost (optional, requires xgboost package)
└── RandomForest (always available via scikit-learn)

Features (14-dimensional vector):
├── Home xG, xGA, shots, shots on target
├── Away xG, xGA, shots, shots on target
├── Home ELO, Away ELO, ELO diff
├── Form indicators (recent results)
└── Historical head-to-head (if available)

Confidence Weighting:
├── High confidence (>0.7): ML 60%, Poisson 40%
├── Medium confidence (0.5-0.7): ML 40%, Poisson 60%
└── Low confidence (<0.5): ML 20%, Poisson 80%
```

**Hybrid Edge Detection:**

```
HybridEdgeDetector (extends EdgeDetector):
├── Blends Poisson + ML predictions
├── Confidence-based weighting
├── Agreement score tracking (0-1)
├── Graceful degradation if ML unavailable
└── Returns HybridEdgeSignal with full provenance
```

**Validation:**
- All probabilities in [0, 1]
- Weights sum to exactly 1.0
- Edge thresholds by market type
- True edge accounts for base rates

### 4. Storage Layer (`src/storage/`) - Phase 3

**Purpose:** Persist data for analysis, audit trail, backtesting

**Database Schema:**
```sql
CREATE TABLE matches (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    home_goals INT,
    away_goals INT,
    season INT NOT NULL,
    UNIQUE(date, home_team, away_team)
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    match_id INT NOT NULL REFERENCES matches(id),
    market_type VARCHAR(10) NOT NULL,  -- '1x2', 'ou', 'btts'
    market_outcome VARCHAR(20) NOT NULL,  -- 'home', 'over_2.5', 'yes'
    our_probability FLOAT NOT NULL,
    market_probability FLOAT NOT NULL,
    edge_percent FLOAT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    executed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE edge_history (
    id INTEGER PRIMARY KEY,
    match_id INT NOT NULL,
    market_type VARCHAR(10) NOT NULL,
    hours_to_kickoff FLOAT NOT NULL,
    edge_percent FLOAT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
```

**Access Patterns:**
```python
# Store alert
db.alerts.insert(
    match_id=12345,
    market_type="1x2",
    our_prob=0.52,
    market_prob=0.48,
    edge=0.083
)

# Query for backtest
edges = db.alerts.query(
    market_type="1x2",
    executed=True,
    min_edge=0.05
).all()
```

**Persistence Strategy:**
- SQLite for Phase 1-2 (dev/testing)
- PostgreSQL option for production (Phase 4+)
- Backup daily to S3 (future)

### 5. UI Layer (`src/ui/`) - COMPLETE (1,400+ LOC)

**Streamlit Dashboard (1,400+ LOC total):**
```
Main Router: multipage Streamlit app

Pages (4):
├── Live Signals - Active alerts with edge breakdown
│   ├── Real-time active alerts table
│   ├── Edge badge (green >7%, yellow 5-7%, red <5%)
│   ├── Model source indicator (🔀 Hybrid / 🤖 ML / 📊 Poisson)
│   ├── Market data: bid/ask/implied odds
│   └── 15-min auto-refresh
│
├── Match Analysis - Detailed probability breakdown (5 tabs)
│   ├── 1X2 tab: Match result probabilities
│   ├── Over/Under tab: Goal line analysis
│   ├── BTTS tab: Both teams to score
│   ├── Poisson Matrix tab: Scoreline probabilities
│   └── ML Insights tab (NEW):
│       ├── Poisson vs ML vs Hybrid comparison
│       ├── Model confidence levels
│       ├── Agreement score indicator
│       └── Feature importance (if available)
│
├── Historical - Edge tracking & performance
│   ├── 7-day edge history (line chart by market)
│   ├── 30-day statistics (win rate, avg edge)
│   ├── Market comparison: 1X2 vs OU vs BTTS
│   └── ROI projection (if backtested)
│
└── Settings - Configuration UI
    ├── Edge thresholds (1X2, OU, BTTS)
    ├── Scan interval (minutes)
    ├── API key toggle (for safety)
    └── Database stats (alerts count, last update)

Components:
├── EdgeBadge - Color-coded indicator with tooltip
├── ProbabilityBar - Visual scale 0-100%
└── ModelSourceIndicator - ML/Poisson/Hybrid badge
```

**Alerts System (362 LOC, Phase 5):**
```
TelegramService (200 LOC):
├── Format alert message with match details
├── Send via python-telegram-bot
├── Retry logic: 3 attempts with backoff
├── Error logging & handling
└── Rate limit awareness

Templates (150 LOC):
├── alert_template: "[MATCH] Team A vs B | Market: 1X2 | Edge: 8.3%"
├── probability formatting (0-100%)
├── Timestamp formatting
└── Support all 3 market types
```

## Data Flow Diagrams

### Scan Cycle (IMPLEMENTED)

```
┌─────────────────────────────────────────────────────────────┐
│  APScheduler (Every 15 minutes, configurable)              │
│  Handler: src/services/scheduler.py (340 LOC)             │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  Fetch Phase (Sequential)           │
    │  ├─ Betfair odds (cached 5m)        │
    │  ├─ Understat xG/xGA (cached 24h)  │
    │  ├─ Club ELO ratings (cached 7d)    │
    │  └─ Polymarket orderbook (cached 1h) │
    │  Timeout: 60s, fallback to cache    │
    │  Retry: 3 attempts, exp backoff     │
    └────────┬────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────┐
    │  Processing Phase (FAST)            │
    │  ├─ Build match data (5+ active)   │
    │  ├─ Calculate Poisson probs         │
    │  ├─ Apply time-based weights        │
    │  ├─ Calculate edges (all markets)   │
    │  └─ Validate: [0,1], sum=1.0       │
    │  Duration: ~200ms, RAM: ~150MB      │
    └────────┬────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────┐
    │  Filtering & Alerting               │
    │  ├─ Filter: edge > threshold?       │
    │  ├─ Check: duplicate in 6h?         │
    │  ├─ Store to SQLite (audit trail)   │
    │  └─ Send Telegram if new            │
    │  Duration: <1 min, retry 3x         │
    └─────────────────────────────────────┘
```

**Graceful Shutdown (IMPLEMENTED):**
- SIGTERM handling: APScheduler stops accepting new jobs
- Wait max 30s for current job completion
- Flush pending alerts
- Close DB connection
- Exit code 0 (success)

### Alert Generation Flow

```
New Edge Detected (8.3% for 1X2 Home Win)
    ↓
┌─ Compare to threshold (5% min) ✓ Pass
├─ Check duplicate in last 6 hours ✓ New
├─ Store to alerts table
└─ Send via Telegram
    ├─ Format message
    ├─ Retry 3x if failed
    └─ Update UI dashboard
```

## Deployment Architecture

### Local Development
```
├── .env (local secrets)
├── data/epl_indicator.db (SQLite)
├── logs/app.log
└── certs/ (Betfair SSL)
```

### Production (Future, Phase 4+)
```
┌────────────────────────────────────┐
│  Linux VM (Oracle/GCP/AWS)         │
├────────────────────────────────────┤
│  Docker Container                  │
│  ├─ Python 3.11 runtime           │
│  ├─ EPL Indicator app             │
│  └─ APScheduler daemon            │
│                                    │
│  PostgreSQL (Cloud SQL)            │
│  Redis (Cache layer)               │
│  Telegram Bot (@epl_bot)          │
│  Streamlit (Public dashboard)      │
└────────────────────────────────────┘
```

**CI/CD Pipeline (Phase 3+):**
```
Git Push ──> GitHub Actions
            ├─ Lint (ruff)
            ├─ Type check (mypy)
            ├─ Test (pytest 95%+)
            └─ If all pass: Deploy to staging
                          ├─ Run backtest
                          └─ If profitable: Deploy to prod
```

## Integration Points

### External APIs

| Service | Phase | Auth | Latency | Fallback |
|---------|-------|------|---------|----------|
| Betfair | 2 | SSL cert | 500ms | Cache 24h |
| Understat | 2 | Web scrape | 2s | Cache 24h |
| Club ELO | 2 | Public API | 200ms | Cache 7d |
| Polymarket | 2 | py-clob-client | 300ms | Cache 1h |
| Telegram | 5 | Bot token | 500ms | Queue + retry |

### Database

**Development:** SQLite file-based
**Staging/Prod:** PostgreSQL (future)

**Backups:**
- Daily snapshots
- Store in S3/GCS
- Retention: 90 days

### Monitoring (Phase 5+)

```
Prometheus metrics:
├─ api_fetch_duration (by service)
├─ edge_detection_rate (per hour)
├─ alert_send_success_rate
└─ calculation_time (Poisson model)

Alerting:
├─ App crash: PagerDuty
├─ API error rate >5%: Slack
└─ Calculation timeout: Email
```

## Scalability Considerations

### Current Design (Phase 1)
- ~5-10 matches per hour (EPL season)
- <1 GB memory
- No concurrent API calls (sequential fetch)
- Single-threaded processing

### Phase 3+ Enhancements
- **Async I/O:** Parallel API fetches (50ms vs 2s)
- **Caching layer:** Redis for frequently accessed data
- **Database indexing:** On match_id, market_type, timestamp
- **Batch processing:** Store 5+ match calculations in single DB transaction

### Phase 5+ (If needed)
- **Horizontal scaling:** Multiple workers (K8s)
- **Message queue:** Kafka for alert backpressure
- **Data warehouse:** BigQuery for historical analysis
- **CDN:** Cache static weight profiles globally

## Security Architecture

### API Keys & Secrets
```
.env (not committed) ──> Pydantic validation ──> In-memory settings
                         ├─ Length checks
                         └─ Pattern validation

Never:
- Log secrets
- Print settings to console
- Store in database
- Commit to Git
```

### SSL/TLS
```
Betfair ──> client-2048.crt + client-2048.key (in ./certs/)
            ├─ Verified by Betfair servers
            └─ Expires annually (auto-renew recommended)

Telegram ──> HTTPS only (python-telegram-bot handles)
Polymarket ──> Ethereum chain (self-custodial, no API secret)
```

### Data Validation
```
All external inputs ──> Pydantic models ──> Type-safe in code
├─ API responses
├─ Configuration
└─ User input (future)
```

## Error & Failure Modes

### Recoverable Errors
```
Network timeout ──> Retry with backoff (3 attempts, max 10s total)
                    ├─ If still failed: Use cached data
                    └─ Log warning

API rate limit ──> Wait 60s, retry
                  └─ Degrade gracefully (skip market temporarily)

Parse error ──> Log error, continue with other matches
               └─ Does not crash scheduler
```

### Non-Recoverable Errors
```
Config validation error ──> Fail fast at startup
                           └─ Admin must fix .env

Database corruption ──> Crash, alert ops
                       └─ Restore from backup

Telegram auth failure ──> Disable alerts, continue processing
                         └─ Email fallback
```

### Graceful Shutdown
```
SIGTERM ──> APScheduler stops taking new jobs
            ├─ Wait for current job to finish (30s timeout)
            ├─ Flush any pending alerts
            └─ Close database connection gracefully
```

## Performance Targets

| Metric | Target | Current | Phase |
|--------|--------|---------|-------|
| Fetch latency | <2s | TBD | 2 |
| Calculation time | <1s (5 matches) | TBD | 3 |
| Alert latency | <5 min | TBD | 5 |
| Memory usage | <1 GB | TBD | 3 |
| Database size | <100MB/year | TBD | 4 |
| API calls/hour | <200 | TBD | 2 |

---

**Architecture Review:** Occurs after Phase 2 data fetchers complete. Will refine based on actual performance metrics.
