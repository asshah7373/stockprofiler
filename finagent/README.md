# FinAgent - Autonomous Financial Multi-Agent System for Indian Markets

A dual-agent financial analysis system for Indian equity markets that provides investment recommendations with SEBI compliance.

## Features

- **Multi-Agent Architecture**: Perceiver (data fetching), Analyst (reasoning), Reviewer (fact-checking)
- **BSE/NSE Integration**: Direct API integration for circulars, announcements, and filings
- **RAG System**: ChromaDB vector store for document retrieval and citation
- **Technical Analysis**: TA-Lib indicators (RSI, MACD, Bollinger Bands, etc.)
- **SEBI Compliance**: Risk profiling, audit trail, mandatory disclaimers
- **No Hallucination**: All calculations are programmatic, all claims are cited

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd stockprofiler/finagent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the `finagent` directory:

```bash
# Required for news sentiment (optional but recommended)
NEWSAPI_KEY=your_newsapi_key

# Optional: For advanced PDF parsing
LLAMA_CLOUD_API_KEY=your_llama_parse_key

# Optional: For Finnhub news
FINNHUB_API_KEY=your_finnhub_key
```

Or create `config/local_settings.py`:

```python
# config/local_settings.py
NEWSAPI_KEY = "your_key_here"
LOG_LEVEL = "INFO"
```

### 3. Initialize the System

```bash
# Run risk profiling (required before any recommendations)
python -m finagent.main profile

# Ingest recent circulars and press releases
python -m finagent.main ingest --days 7

# Analyze a stock
python -m finagent.main analyze RELIANCE.NS --detailed
```

## CLI Commands

### Risk Profiling

```bash
# Interactive risk assessment (SEBI requirement)
python -m finagent.main profile
```

### Stock Analysis

```bash
# Basic analysis
python -m finagent.main analyze RELIANCE.NS

# Detailed analysis with technical indicators
python -m finagent.main analyze TCS.NS --detailed

# JSON output
python -m finagent.main analyze INFY.NS --json

# Verbose mode for debugging
python -m finagent.main analyze HDFCBANK.NS --verbose
```

### Document Ingestion

```bash
# Ingest all circulars and press releases (last 7 days)
python -m finagent.main ingest --days 7

# Ingest from specific exchange
python -m finagent.main ingest --exchange NSE --days 30

# Ingest only circulars (skip press releases)
python -m finagent.main ingest --type circulars

# Ingest for a specific company (comprehensive, last year)
python -m finagent.main ingest-company RELIANCE --days 365

# Skip downloading files (metadata only)
python -m finagent.main ingest --no-download

# Skip parsing content
python -m finagent.main ingest --no-parse
```

### Search Ingested Documents

```bash
# Search all filings
python -m finagent.main search-filings "quarterly results"

# Filter by ticker
python -m finagent.main search-filings --ticker TCS

# Filter by document type
python -m finagent.main search-filings --type quarterly_results

# Filter by exchange
python -m finagent.main search-filings --exchange BSE

# Limit results
python -m finagent.main search-filings --limit 50
```

### System Status

```bash
# View system status
python -m finagent.main status

# View ingestion statistics
python -m finagent.main ingestion-stats

# Generate compliance report
python -m finagent.main compliance-report --output report.json --days 30
```

### Stock Suggestions

```bash
# Get stock suggestions based on risk profile
python -m finagent.main suggest --count 5

# Filter by sector
python -m finagent.main suggest --sector IT
python -m finagent.main suggest --sector BANKING
```

## Project Structure

```
finagent/
├── CLAUDE.md                    # Project constitution and rules
├── agents/
│   ├── perceiver.py             # Data Ingestion Agent
│   ├── analyst.py               # Reasoning Engine Agent
│   └── reviewer.py              # Fact-checking Agent
├── pipelines/
│   ├── structured_data.py       # Market data (OHLCV, company info)
│   ├── unstructured_data.py     # BSE/NSE circulars and filings
│   ├── sentiment_data.py        # News sentiment analysis
│   ├── press_releases.py        # Press release ingestion
│   └── ingestion_orchestrator.py # Batch ingestion coordinator
├── rag/
│   ├── vector_store.py          # ChromaDB vector database
│   ├── embeddings.py            # Document embedding
│   └── retriever.py             # RAG retrieval with citations
├── analysis/
│   ├── technical.py             # TA-Lib indicators
│   ├── fundamental.py           # Financial metrics
│   └── synthesizer.py           # Chain-of-Thought reasoning
├── risk_profile/
│   ├── questionnaire.py         # SEBI risk profiling
│   └── constraints.py           # Investment constraints
├── utils/
│   ├── cache.py                 # SQLite caching
│   ├── rate_limiter.py          # API rate limiting
│   └── compliance_logger.py     # 5-year audit trail
├── config/
│   ├── settings.py              # Configuration
│   └── mcp_config.json          # MCP server config
├── data/                        # Local data storage
│   ├── circulars/               # Downloaded PDFs
│   ├── press_releases/          # Press release files
│   ├── cache/                   # SQLite databases
│   └── vectors/                 # ChromaDB storage
├── logs/                        # Compliance audit logs
├── tests/                       # Test suite
├── requirements.txt
└── main.py                      # CLI orchestrator
```

## Data Sources

### Market Data
- **yfinance**: Historical OHLCV data (use `.NS` for NSE, `.BO` for BSE)
- **nsepython**: Live quotes, option chains, index data

### Corporate Filings
- **BSE API**: Corporate announcements, financial results, board meetings
- **NSE API**: Corporate filings, shareholding patterns, corporate actions

### News & Sentiment
- **NewsAPI**: General news articles
- **Finnhub**: Financial news with sentiment

## Document Types Ingested

| Type | Description |
|------|-------------|
| `quarterly_results` | Q1/Q2/Q3/Q4 financial results |
| `annual_report` | Annual reports |
| `board_meeting` | Board meeting outcomes |
| `shareholding_pattern` | Shareholding disclosures |
| `corporate_action` | Dividends, splits, bonus, rights |
| `press_release` | Company press releases |
| `investor_presentation` | Investor/analyst presentations |
| `credit_rating` | Credit rating updates |
| `insider_trading` | SAST/insider trading disclosures |

## Risk Profiles

After completing the questionnaire, users are categorized into:

| Profile | Max Beta | Min Market Cap | Excluded Instruments |
|---------|----------|----------------|---------------------|
| Conservative | 0.8 | Large Cap | F&O, Penny Stocks, SME |
| Moderate | 1.2 | Mid Cap | Penny Stocks |
| Aggressive | 2.0 | Small Cap | None |

## SEBI Compliance

### Mandatory Requirements Implemented

1. **Risk Profiling**: Complete questionnaire before any recommendations
2. **Suitability**: Match recommendations to user risk profile
3. **Disclosure**: Mandatory disclaimer on all recommendations
4. **Audit Trail**: 5-year retention of all interactions
5. **Source Citation**: All claims must cite source documents

### Disclaimer

Every recommendation includes:
```
This recommendation is AI-generated by FinAgent.
This is NOT advice from a SEBI-registered Investment Adviser.
Past performance does not guarantee future results.
Conduct your own due diligence before investing.
```

## Testing

```bash
# Run all tests
pytest finagent/tests/ -v

# Run specific test file
pytest finagent/tests/test_pipelines.py -v

# Run with coverage
pytest finagent/tests/ --cov=finagent
```

## Troubleshooting

### NSE API Returns 401

The NSE website requires session cookies. The system handles this automatically, but if you see 401 errors:

```python
# The session is reset automatically, but you can force it:
from finagent.pipelines.unstructured_data import CircularPipeline
pipeline = CircularPipeline()
pipeline._reset_nse_session()
```

### PDF Parsing Fails

Install optional dependencies for better PDF parsing:

```bash
# For table extraction
pip install camelot-py[cv] ghostscript

# For complex PDFs
pip install llama-parse
```

### Rate Limiting

Both BSE and NSE enforce rate limits. The system respects 1 request/second by default. To adjust:

```python
pipeline = CircularPipeline()
pipeline.rate_limit = 2.0  # 2 seconds between requests
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWSAPI_KEY` | No | NewsAPI.org API key for news sentiment |
| `FINNHUB_API_KEY` | No | Finnhub API key for financial news |
| `LLAMA_CLOUD_API_KEY` | No | LlamaParse API key for PDF parsing |

## Development

### Adding New Document Types

1. Add the type to `DocumentType` enum in `unstructured_data.py`
2. Add categorization logic in `_categorize_bse_document()` or `_categorize_nse_document()`
3. Update the vector store collection in `rag/vector_store.py`

### Adding New Data Sources

1. Create a new pipeline class in `pipelines/`
2. Implement `fetch_*`, `parse_*`, and `save_*` methods
3. Add to `IngestionOrchestrator` if batch processing is needed
4. Export from `pipelines/__init__.py`

## License

This project is for educational and informational purposes only. Not financial advice.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.
