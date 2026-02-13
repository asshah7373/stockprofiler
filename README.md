# FinAgent - Autonomous Financial Multi-Agent System for Indian Markets

An AI-powered stock analysis and recommendation system built for Indian equity markets (NSE/BSE). Uses multi-agent architecture with strict separation of concerns, real-time news sentiment analysis, and SEBI-compliant risk profiling.

## Features

### Multi-Agent Architecture

Three specialized agents ensure reliability and prevent hallucination:

- **Perceiver Agent** - Data ingestion only. Fetches market data, documents, and news. Never analyzes or recommends.
- **Analyst Agent** - Reasoning engine. Generates Chain-of-Thought analysis with mandatory source citations. Uses only data from Perceiver.
- **Reviewer Agent** - Fact-checker. Validates cited figures, checks calculations, flags unsupported claims. Prevents hallucination.

### Real-Time News & Sentiment Analysis

Live news fetching from multiple sources with NLP-based sentiment scoring:

- **Google News RSS** - Search-based financial news
- **Yahoo Finance** - Stock-specific news via yfinance
- **NSE India** - Corporate announcements, circulars, press releases
- **BSE India** - Corporate announcements, filings, results

Sentiment analysis pipeline:

| Method | Description | Priority |
|--------|-------------|----------|
| **FinBERT** | Transformer model fine-tuned on financial text | Highest |
| **VADER** | Rule-based sentiment for social/news text | Medium |
| **Keyword** | Pattern-based fallback (bullish/bearish keywords) | Fallback |

News sentiment is integrated into the combined scoring: `combined_score = (technical * 0.7) + (fundamental * 0.3)` with a 15% boost when technical and fundamental signals align.

### Technical Analysis

Computed deterministically using TA-Lib (with numpy fallback):

- RSI (14-period), MACD, Bollinger Bands
- SMA 20/50/200, EMA 12/26
- Golden Cross / Death Cross detection
- ATR, Stochastic, ADX
- Volume trend analysis

### Fundamental Analysis

Scoring across four dimensions (0-100):

- **Valuation** - P/E, Forward P/E, P/B, P/S, PEG, EV/EBITDA
- **Quality** - Profit margin, ROE, ROA, debt-to-equity
- **Growth** - Revenue growth, earnings growth, free cash flow
- **Financial Health** - Liquidity and solvency ratios

### Signal Combination Engine

Combines 6+ signal types with time-horizon weighted approach:

| Signal Type | Description |
|-------------|-------------|
| Technical | TA-Lib indicator signals |
| Fundamental | Valuation and quality scores |
| Institutional | FII/DII daily net flows |
| Bulk/Block | Large institutional trades |
| Insider | SAST filings (promoter/director trades) |
| News/Catalyst | Sentiment and event-driven signals |
| PEAD | Post-Earnings Announcement Drift |

Signal weights vary by time horizon:

- **Intraday** - 40% technical, 25% institutional, 20% news
- **Short-term (1d-1w)** - 30% technical, 25% catalyst, 15% news
- **Medium-term (1w-1m)** - 25% fundamental, 20% institutional, 15% news
- **Long-term (1m+)** - 35% fundamental, 20% insider, 20% institutional

### Stock Universe

Expanded coverage of Indian markets:

| Universe | Stocks | Description |
|----------|--------|-------------|
| `nifty50` | 50 | Nifty 50 index constituents |
| `nifty200` | ~200 | Nifty 50 + Next 50 + Nifty 200 |
| `fno` | ~230 | F&O eligible stocks |
| `broad` | ~500 | All major NSE stocks |
| `midcap` | ~900 | Mid-cap and small-cap extension |
| `nsebse` | ~1500 | Comprehensive NSE + BSE coverage |

140+ company name mappings for accurate news search across Nifty 50, banking, IT, pharma, auto, FMCG, metals, energy, and infrastructure sectors.

### Backtesting

Fast vectorized backtesting via VectorBT (100x faster than traditional):

- **Strategies** - RSI, MACD, Bollinger Bands, SMA crossover, Combined
- **Metrics** - Total return, CAGR, Sharpe/Sortino ratio, max drawdown, win rate
- **Optimization** - Grid search parameter optimization
- **Risk management** - Configurable stop-loss and take-profit

### SEBI Compliance

- **Risk Profiling** - 10-question SEBI-mandated assessment (Conservative/Moderate/Aggressive)
- **Suitability Check** - Recommendations filtered by risk profile constraints
- **Audit Trail** - 5-year compliance logging of all interactions and recommendations
- **Data Provenance** - All data sources tracked and cited

### RAG System

Retrieval-Augmented Generation for document analysis:

- ChromaDB vector store with persistent storage
- Sentence transformer embeddings (`all-MiniLM-L6-v2`)
- Multi-collection support (circulars, results, news, annual reports)
- Semantic search for relevant document retrieval

### Institutional Data Tracking

- **FII/DII Activity** - Daily net flows (buy-sell in Crores)
- **Bulk/Block Deals** - Trades >0.5% shares or >5 lakh shares
- **Insider Trading** - SAST filings from promoters, directors, KMP

### Data Pipelines

| Pipeline | Source | Data |
|----------|--------|------|
| Structured | yfinance, brokers, nsepython | OHLCV, fundamentals, live quotes |
| Unstructured | BSE/NSE APIs | Circulars, filings, board meetings |
| Press Releases | Company websites | Corporate announcements |
| Institutional | NSE/BSE/SEBI | FII/DII, bulk deals, insider trades |
| Sentiment | News APIs | Aggregated sentiment scores |

## Installation

### Prerequisites

- Python 3.9+
- TA-Lib C library (for technical analysis)

### Install

```bash
# Clone the repository
git clone https://github.com/asshah7373/stockprofiler.git
cd stockprofiler

# Install TA-Lib C library (required)
# Ubuntu/Debian:
sudo apt-get install libta-lib-dev
# macOS:
brew install ta-lib

# Install the package
pip install -e .

# With backtesting support
pip install -e ".[backtest]"

# With all optional dependencies
pip install -e ".[all]"
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for AI-powered analysis
ANTHROPIC_API_KEY=your_key_here

# Optional - for enhanced news
NEWSAPI_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Optional - broker integration
ZERODHA_API_KEY=your_key_here
UPSTOX_API_KEY=your_key_here
```

## Usage

### Stock Analysis

Full analysis pipeline (Perceiver -> Analyst -> Reviewer):

```bash
# Analyze a single stock
finagent analyze RELIANCE.NS

# Detailed analysis with verbose output
finagent analyze INFY.NS --detailed --verbose

# JSON output for programmatic use
finagent analyze HDFCBANK.NS --json
```

### Stock Scanner

Scan the market for top opportunities using 8 advanced technical indicators combined with real-time news sentiment analysis.

**Technical Indicators Used:**
- MACD (trend momentum)
- Williams %R Trend Exhaustion (tops/bottoms)
- Williams VixFix (volatility bottoms)
- Hull Moving Average (trend identification)
- Laguerre RSI (smoothed momentum)
- Supertrend (trailing stop levels)
- RSI with Divergence (reversal detection)
- Ichimoku Cloud (comprehensive trend analysis)

**Fundamental/News Integration:**
- Live news from Google News, Yahoo Finance, NSE, BSE
- Sentiment scored via FinBERT/VADER/Keywords
- Catalyst detection: contract wins, earnings, expansions, dividends
- PEAD (Post-Earnings Announcement Drift) signals
- Combined score: 70% technical + 30% fundamental, 15% alignment boost

**All Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--universe` | `-u` | `nifty50` | Stock universe to scan |
| `--hold` | `-h` | `7` | Holding period in days |
| `--min-confidence` | `-c` | `40.0` | Minimum signal confidence (0-100) |
| `--count` | `-n` | `20` | Number of results to show |
| `--direction` | `-d` | `all` | Filter: `buy`, `sell`, or `all` |
| `--file` | `-f` | | Custom ticker file (one per line) |
| `--ticker` | `-t` | | Scan a single ticker |
| `--fundamentals` | | `True` | Include news/sentiment (`--no-fundamentals` to disable) |
| `--news-days` | | `30` | Days to look back for news/circulars |
| `--json` | `-j` | `False` | Output as JSON |
| `--verbose` | `-v` | `False` | Show debug output (news fetching, sentiment) |

**Stock Universes:**

| Universe | Stocks | Description |
|----------|--------|-------------|
| `nifty50` | 50 | Nifty 50 index constituents |
| `nifty100` | ~83 | Nifty 50 + Nifty Next 50 |
| `nifty200` | ~200 | Nifty 50 + Next 50 + Nifty 200 |
| `fno` | ~230 | All F&O eligible stocks |
| `midcap` | ~400 | Mid-cap and small-cap NSE stocks |
| `all` | ~500 | All major NSE stocks (broad market) |
| `comprehensive` / `nsebse` | ~1500 | All NSE + BSE stocks |
| `custom` | varies | Use `--file` to provide your own list |

**Examples:**

```bash
# Scan Nifty 50 with 7-day holding (default)
finagent scan --universe nifty50 --hold 7

# Scan midcap stocks
finagent scan --universe midcap --hold 7

# Scan comprehensive NSE+BSE universe (1500+ stocks)
finagent scan --universe nsebse --hold 14

# Only show buy signals with high confidence
finagent scan --universe nifty200 --direction buy --min-confidence 60

# Scan single stock
finagent scan --ticker RELIANCE.NS

# Scan custom list from file
finagent scan --file my_watchlist.txt --hold 14

# Fast scan without news/sentiment (technical only)
finagent scan --universe all --no-fundamentals

# Recent news only (7-day lookback instead of default 30)
finagent scan --universe nifty50 --news-days 7

# Show top 50 results
finagent scan --universe fno --count 50

# Verbose mode (debug news fetching and sentiment)
finagent scan --universe nifty50 --hold 7 --verbose

# JSON output for programmatic use
finagent scan --universe nifty50 --json
```

**Output includes:**
- Ticker, signal direction (Strong Buy/Buy/Sell/Strong Sell)
- Technical confidence score (%)
- News sentiment (BULL/BEAR/NEUT) with method used (FinBERT/VADER/Keyword)
- Combined score (technical + fundamental)
- Current price, target price, expected return, risk:reward ratio
- Detailed breakdown for top pick: entry, stop loss, 3 targets, hold days
- Indicator signals, bullish/bearish factors, catalysts, recent headlines

### News-Driven Suggestions

Screen ingested news for positive catalysts:

```bash
# Get top 5 suggestions
finagent suggest --count 5

# Filter by sector
finagent suggest --sector PHARMA

# Filter by time horizon
finagent suggest --horizon 1w

# Commodity-driven plays
finagent suggest --commodity gold

# Minimum score threshold
finagent suggest --min-score 20.0
```

Detected catalyst types: contract wins, product launches, expansions, partnerships, acquisitions, strong results, dividends, buybacks, regulatory approvals, rating upgrades, sector tailwinds.

### Test News Fetching

Debug and verify news fetching for a specific ticker:

```bash
# Test news for a specific stock
finagent test-news RELIANCE.NS

# Custom lookback period
finagent test-news INFY.NS --days 14
```

### Risk Profiling

Interactive SEBI-mandated risk assessment:

```bash
finagent profile
```

Assesses age, income, net worth, experience, knowledge, time horizon, risk tolerance. Produces Conservative/Moderate/Aggressive profile valid for 12 months.

### Signal Generation

Combine multiple signal sources into a unified recommendation:

```bash
# Generate signals for a stock
finagent signals RELIANCE.NS

# With specific time horizon
finagent signals INFY.NS --horizon 1w

# JSON output
finagent signals HDFCBANK.NS --json
```

### Backtesting

Test strategies against historical data:

```bash
# RSI strategy with default settings
finagent backtest RELIANCE.NS --strategy rsi

# MACD strategy with custom capital and risk
finagent backtest INFY.NS --strategy macd --capital 500000 --stop-loss 0.05

# Combined strategy with optimization
finagent backtest TCS.NS --strategy combined --optimize

# Available strategies: rsi, macd, bollinger, sma, combined
finagent backtest HDFCBANK.NS --strategy bollinger --days 365
```

### Data Ingestion

Batch ingest BSE/NSE documents:

```bash
# Ingest circulars from both exchanges
finagent ingest --exchange BOTH --days 7

# Ingest specific ticker
finagent ingest --ticker RELIANCE --type all

# Force re-processing
finagent ingest --exchange NSE --force

# Ingest all data for a single company
finagent ingest-company RELIANCE --days 365 --results --press
```

### Institutional Data

Track institutional flows and insider activity:

```bash
# FII/DII daily activity
finagent institutional fii-dii

# Bulk and block deals
finagent institutional bulk --days 7

# Insider trading (SAST filings)
finagent institutional insider --ticker RELIANCE

# All institutional data
finagent institutional all --json
```

### Search Filings

Search ingested circulars and documents:

```bash
finagent search-filings "quarterly results" --ticker RELIANCE --days 30
finagent search-filings "board meeting" --exchange NSE --type result
```

### System Status

```bash
# System health check
finagent status

# Ingestion statistics
finagent ingestion-stats

# SEBI compliance report
finagent compliance-report --output report.json --days 30
```

### Update Data

```bash
# Refresh circular database and market data
finagent update-data --days 7 --exchange BOTH
```

## Architecture

```
                    ┌─────────────────┐
                    │   CLI (Typer)   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌─────▼─────┐ ┌────────▼────────┐
    │   Perceiver   │ │  Analyst  │ │    Reviewer     │
    │  (Data Only)  │ │ (Reason)  │ │ (Fact-Check)    │
    └───────┬───────┘ └─────┬─────┘ └────────┬────────┘
            │               │                │
    ┌───────▼───────────────▼────────────────▼────────┐
    │              Data & Analysis Layer               │
    ├──────────┬──────────┬──────────┬────────────────┤
    │ Signals  │ Analysis │ Pipelines│     RAG        │
    │ Combiner │ Tech/Fun │ Struct/  │  ChromaDB +    │
    │ 6+ types │ Sentimnt │ Unstruct │  Embeddings    │
    └──────────┴──────────┴──────────┴────────────────┘
            │               │                │
    ┌───────▼───────────────▼────────────────▼────────┐
    │              External Data Sources               │
    ├────────┬────────┬──────────┬─────────┬──────────┤
    │yfinance│NSE API │ BSE API  │ Google  │ Brokers  │
    │        │        │          │  News   │ (Zerodha)│
    └────────┴────────┴──────────┴─────────┴──────────┘
```

## Project Structure

```
stockprofiler/
├── finagent/
│   ├── agents/                    # Multi-agent system
│   │   ├── perceiver.py          # Data fetching (read-only)
│   │   ├── analyst.py            # Analysis & recommendations
│   │   └── reviewer.py           # Fact-checking & validation
│   ├── analysis/                 # Analysis modules
│   │   ├── technical.py          # TA-Lib technical indicators
│   │   ├── fundamental.py        # Valuation & quality scoring
│   │   ├── sentiment_analyzer.py # NLP sentiment (FinBERT/VADER)
│   │   ├── news_screener.py      # Catalyst detection
│   │   ├── live_news_fetcher.py  # Real-time news from 4 sources
│   │   └── synthesizer.py        # Chain-of-thought synthesis
│   ├── pipelines/                # Data pipelines
│   │   ├── ingestion_orchestrator.py  # Batch document processing
│   │   ├── structured_data.py         # OHLCV market data
│   │   ├── unstructured_data.py       # BSE/NSE circulars
│   │   ├── press_releases.py          # Press release pipeline
│   │   ├── sentiment_data.py          # Sentiment scoring
│   │   └── institutional_data.py      # FII/DII, insider trades
│   ├── signals/                  # Signal generation
│   │   ├── signal_combiner.py    # Multi-source combination
│   │   ├── advanced_signals.py   # Advanced signal generation
│   │   └── pead_strategy.py      # Post-earnings drift
│   ├── strategies/               # Backtesting
│   │   └── vectorbt_framework.py # VectorBT backtesting
│   ├── rag/                      # Retrieval-Augmented Generation
│   │   ├── vector_store.py       # ChromaDB vector store
│   │   ├── embeddings.py         # Sentence transformers
│   │   └── retriever.py          # Document retrieval
│   ├── risk_profile/             # SEBI risk profiling
│   │   ├── questionnaire.py      # Risk assessment
│   │   └── constraints.py        # Risk-based constraints
│   ├── config/
│   │   └── settings.py           # Central configuration
│   ├── utils/
│   │   ├── compliance_logger.py  # SEBI audit trail (5-year)
│   │   ├── cache.py              # Multi-level caching
│   │   └── rate_limiter.py       # API rate limiting
│   ├── mcp_servers/              # Model Context Protocol
│   │   ├── market_data_server.py
│   │   ├── news_server.py
│   │   └── document_parser_server.py
│   └── main.py                   # CLI entry point
├── pyproject.toml
└── README.md
```

## Configuration

Central configuration in `finagent/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| Cache TTL | 24 hours | Market data cache duration |
| Price TTL | 5 minutes | Live quote cache duration |
| RSI Oversold | 30 | RSI buy signal threshold |
| RSI Overbought | 70 | RSI sell signal threshold |
| Rate limit (NSE) | 1 req/s | NSE API rate limit |
| Rate limit (BSE) | 1 req/s | BSE API rate limit |
| Rate limit (yfinance) | 2 req/s | Yahoo Finance rate limit |
| Compliance retention | 5 years | SEBI audit trail retention |
| News cache | 30 minutes | Live news cache TTL |
| Embedding model | all-MiniLM-L6-v2 | Sentence transformer model |

## Data Sources (Priority Order)

1. **Broker APIs** - Zerodha Kite, ICICI Breeze, Upstox (if configured)
2. **yfinance** - Historical OHLCV, company fundamentals
3. **nsepython** - Live quotes, option chains, index data
4. **BSE/NSE Official APIs** - Circulars, corporate actions (rate-limited)
5. **Google News RSS** - General financial news
6. **SEBI Website** - SAST insider trading filings

## Requirements

- Python >= 3.9
- TA-Lib C library
- Dependencies: typer, rich, pandas, numpy, yfinance, httpx, chromadb, sentence-transformers, beautifulsoup4, pydantic

## License

MIT
