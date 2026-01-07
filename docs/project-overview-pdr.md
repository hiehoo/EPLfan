# EPL Bet Indicator - Project Overview & PDR

**Version:** 2.0 (Phase 5)
**Status:** Phase 5 Complete - Full Stack Implementation
**Last Updated:** January 7, 2026

## Executive Summary

EPL Bet Indicator is a quantitative sports betting system that:
- Monitors 3 market types on Polymarket (1X2 match results, Over/Under goals, Both Teams to Score)
- Aggregates signals from 4 data sources (Betfair exchange odds, expected goals metrics, team ratings, market prices)
- Applies unified Poisson probability model with time-adaptive weights
- Triggers alerts when probabilistic edge exceeds configured thresholds
- Targets ~5-15% annualized return through disciplined edge identification

**Primary Market:** Polymarket (Ethereum-based prediction markets)
**Target Users:** Quant traders, serious sports bettors with technical background

## Product Development Requirements (PDR)

### Vision & Objectives

**Vision:** Democratize sophisticated sports betting through transparent, quantifiable probability models.

**Objectives:**
1. Provide reliable edge identification across multiple EPL betting markets
2. Automate monitoring 5+ simultaneous matches 24/7
3. Integrate diverse data sources into coherent probability estimates
4. Deliver sub-5-minute alert latency for actionable opportunities
5. Enable transparent backtesting & performance tracking

### Functional Requirements

#### 1. Data Ingestion Layer
- **Betfair Integration:** Fetch implied odds from EPL market
- **Understat Metrics:** xG/xGA per team for current season
- **Club ELO Ratings:** Current team strength ratings
- **Polymarket Orderbook:** Real-time bid/ask for 1X2/OU/BTTS markets
- **Update Frequency:** Every 15 minutes (configurable)

#### 2. Probability Calculation
- **Model:** Modified Poisson (matches per team per game)
- **Inputs:** Home/away goals distribution from xG + ELO
- **Outputs:** P(Home Win), P(Draw), P(Away Win), P(Over 1.5/2.5/3.5), P(BTTS)
- **Weights:** Adaptive based on hours to kickoff
  - Early (>96h): Trust analytics (xG 35%, ELO 25%)
  - Mid (24-96h): Balanced approach
  - Late (<24h): Trust market (Betfair 55%)

#### 3. Edge Detection
- **Formula:** `Edge = (Our Prob - Implied Prob) / Implied Prob`
- **Thresholds:**
  - 1X2: 5% minimum edge
  - Over/Under: 7% minimum edge
  - BTTS: 7% minimum edge
- **Triggers:** Alert when any market exceeds threshold

#### 4. Alerting System
- **Telegram:** Real-time notifications with match details
- **Format:** `[MATCH] Team A vs Team B | Market: 1X2 | Edge: 8.3% | Our: 52% | Market: 48%`
- **Latency:** <5 minutes from edge detection to user notification

#### 5. Dashboard (UI)
- **Streamlit:** Visual analytics for trader monitoring
- **Displays:**
  - Active alerts with edge breakdown
  - Historical edge tracking (7-day, 30-day)
  - Win rate by market type
  - ROI projections based on edge history

#### 6. Persistence Layer
- **Database:** SQLite (local) or PostgreSQL (production)
- **Tracks:** Alerts sent, edges detected, match outcomes
- **Purpose:** Backtesting, performance analysis, audit trail

### Non-Functional Requirements

#### Performance
- **Latency:** <5 min from new data to alert
- **Throughput:** Monitor 5+ simultaneous matches
- **Uptime:** 99%+ availability during EPL season
- **Resource:** <1GB RAM, <10% CPU on modern hardware

#### Reliability
- **Data Fallback:** Use cached values if API fails (max 30 min staleness)
- **Alert Retry:** Reattempt failed Telegram sends up to 3 times
- **Validation:** Verify weight profiles sum to 1.0, probability bounds [0,1]

#### Maintainability
- **Type Safety:** 100% type hints (mypy strict)
- **Test Coverage:** >95% critical paths
- **Documentation:** Inline comments for complex probability logic
- **Configurability:** All thresholds & weights via env + YAML

#### Security
- **API Keys:** Via .env (never committed)
- **SSL Certs:** Betfair client certs in ./certs
- **Validation:** Pydantic models validate all inputs

### Success Metrics

#### Phase 1 (Complete)
- [x] Configuration system with Pydantic
- [x] Weight profile specification (YAML)
- [x] Logging infrastructure
- [x] 97%+ test coverage on config
- [x] README + environment setup docs

#### Phase 2 (Data Layer) - COMPLETE
- [x] Fetch Betfair odds with SSL auth
- [x] Parse Understat xG/xGA metrics
- [x] Load Club ELO ratings
- [x] Query Polymarket orderbook (py-clob-client)
- [x] Unit tests for each fetcher (90%+ coverage)

#### Phase 3 (Processing Layer) - COMPLETE
- [x] Implement modified Poisson model
- [x] Aggregate weighted signals
- [x] Calculate edges (all 3 markets)
- [x] Filter by minimum edge threshold
- [x] Performance tests: <1 sec calculation on 5 matches

#### Phase 4 (Dashboard) - COMPLETE
- [x] Streamlit UI with real-time data (4 pages: live signals, match analysis, historical, settings)
- [x] Historical edge tracking
- [x] Edge badge component with color coding
- [x] Probability bar visualizations
- [x] Real-time market data display

#### Phase 5 (Alerts & Scheduler) - COMPLETE
- [x] Telegram bot integration with formatted alerts
- [x] APScheduler job runner (15 min configurable interval)
- [x] Graceful shutdown with SIGTERM handling
- [x] Retry logic for failed sends (3 attempts)
- [x] CLI commands (run, dashboard, fetch, analyze, scan, status, init_db, test_telegram)

#### Phase 6 (Testing & Validation) - IN PROGRESS
- [x] 137 passing unit & integration tests
- [x] Test coverage: 2350 LOC tests
- [x] Edge detection validation across all markets
- [ ] Backtesting framework (planned)
- [ ] ROI calculation (planned)

### Constraints & Assumptions

**Constraints:**
- Python 3.11+ only (async support, modern typing)
- Betfair API requires UK residency + account
- Polymarket contracts on Ethereum mainnet
- Telegram bot requires manual start

**Assumptions:**
- EPL matches have sufficient liquidity on Polymarket
- Betfair odds reflect true probability (efficient market hypothesis)
- xG metrics from Understat are reliable proxy for team strength
- ELO ratings stable week-to-week
- <5 min latency acceptable for manual execution (not algo trading)

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| API downtime | Medium | High | Cache data locally, implement fallback ordering |
| Model calibration error | Medium | Medium | Backtest against 2+ seasons, adjust weights iteratively |
| Telegram rate limits | Low | Medium | Batch alerts, implement exponential backoff |
| Polymarket liquidity | Low | High | Monitor spread, skip low-volume markets |
| Betfair auth failure | Low | High | Graceful degradation, email alerts on auth error |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    External APIs                            │
│  Betfair  Understat  Club ELO  Polymarket  Telegram        │
└────┬──────────┬──────────────┬──────────────┬──────────────┘
     │          │              │              │
     ▼          ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Fetchers Layer                         │
│  ├─ BetfairClient  ├─ UnderstatScraper                     │
│  ├─ ELORatingsFetcher  ├─ PolymarketClient                │
└────┬──────────────────────────────────────────────────────┬──┘
     │                                                       │
     ▼                                                       ▼
┌──────────────────────────────┐      ┌────────────────────────┐
│  Processing Layer            │      │  Config Management    │
│  ├─ Poisson Model          │      │  ├─ Settings (Env)    │
│  ├─ Edge Calculator        │      │  ├─ Weight Profiles   │
│  ├─ Signal Aggregator      │      │  └─ Validators        │
│  └─ Alert Filter            │      └────────────────────────┘
└────┬─────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                Storage Layer (SQLite)                        │
│  Alerts  Edges  Outcomes  Performance Stats               │
└────┬──────────────────────────────────────────────────────┬──┘
     │                                                       │
     └───────────┬───────────────┬──────────────┬──────────────┘
                 ▼               ▼              ▼
            ┌─────────┐    ┌───────────┐  ┌──────────┐
            │Streamlit│    │ Telegram  │  │Backtest  │
            │Dashboard│    │Bot (UI)   │  │Framework │
            └─────────┘    └───────────┘  └──────────┘
```

## Timeline & Milestones

| Phase | Deliverable | Target | Actual | Status |
|-------|-------------|--------|--------|--------|
| 1 | Config system + tests | Dec 6 | Jan 6 | ✓ Complete |
| 2 | Data fetchers | Jan 12 | Jan 6 | ✓ Complete |
| 3 | Processing & models | Jan 19 | Jan 6 | ✓ Complete |
| 4 | Dashboard UI | Jan 26 | Jan 6 | ✓ Complete |
| 5 | Alerts & scheduler | Feb 2 | Jan 6 | ✓ Complete |
| 6 | Testing & validation | Feb 9 | In Progress | ⧖ Ongoing |

**Total Dev Time:** <1 month (accelerated delivery)

## Success Criteria

**Phase 1 (✓ COMPLETE):**
- Pydantic settings with 6+ validators
- Weight profiles for 3 markets with time-based selection
- 12 unit tests, 97% coverage
- Documentation & setup guide

**Phase 2-3 (✓ COMPLETE):**
- [x] Fetch data from 4 sources with error handling & retry logic
- [x] Poisson model with 90%+ test coverage
- [x] <1 sec calculation time for 5 matches
- [x] 971 LOC fetchers + 876 LOC models

**Phase 4-5 (✓ COMPLETE):**
- [x] Dashboard deployed & responsive (4 pages, 1155 LOC)
- [x] Telegram alerts with formatted templates (<1 min latency)
- [x] Scheduler running continuously (APScheduler daemon)
- [x] 8 CLI commands for automation
- [x] 362 LOC alerts module + 344 LOC services

**Phase 6 (IN PROGRESS):**
- [x] 137 passing tests validating all workflows
- [x] Edge detection across 1X2, Over/Under, BTTS markets
- [x] Alert generation & persistence
- [ ] Historical backtesting with win rate calculation
- [ ] ROI projection based on edge history

## Glossary

| Term | Definition |
|------|-----------|
| **Edge** | (Our Probability - Market Probability) / Market Probability |
| **xG** | Expected Goals - probability-based goal prediction model |
| **Poisson Model** | Statistical model for goal distributions in football |
| **ELO Rating** | Relative strength metric for teams based on historical results |
| **BTTS** | Both Teams To Score - market predicting if both teams score |
| **1X2** | Match result: 1=Home Win, X=Draw, 2=Away Win |
| **OU** | Over/Under goals (1.5, 2.5, 3.5 thresholds) |
| **Implied Odds** | Probability inferred from betting odds |
| **Betfair** | Betting exchange (player vs player) |
| **Polymarket** | Ethereum-based prediction market |

---

**Document Version History:**
- v1.0 (Jan 6, 2026): Phase 1 baseline with complete requirements
- v2.0 (Jan 7, 2026): Phase 5 complete - full stack implementation (4400 LOC, 137 tests)
