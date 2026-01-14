# FinAgent Setup Guide

This guide provides detailed instructions for setting up and configuring FinAgent.

## System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended for PDF parsing)
- 2GB disk space for document storage
- Internet connection for API access

## Installation

### Step 1: Python Environment

```bash
# Check Python version
python --version  # Should be 3.9+

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
.\venv\Scripts\activate
```

### Step 2: Install Package

```bash
# Navigate to the project root (where pyproject.toml is located)
cd /path/to/stockprofiler

# Install in development mode (recommended)
pip install -e .

# Or install from requirements.txt only (not recommended - won't enable `python -m finagent.main`)
# pip install -r finagent/requirements.txt
```

### Step 3: Install Optional Dependencies

#### For Technical Analysis (TA-Lib)

TA-Lib requires system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Windows: Download from https://www.ta-lib.org/
# Then:
pip install TA-Lib
```

#### For PDF Table Extraction (Camelot)

```bash
# Install Ghostscript first
# Ubuntu/Debian:
sudo apt-get install ghostscript

# macOS:
brew install ghostscript

# Then install camelot
pip install camelot-py[cv]
```

#### For Advanced PDF Parsing (LlamaParse)

```bash
pip install llama-parse
```

### Step 4: Verify Installation

```bash
# Test the installation
python -c "from finagent.main import app; print('Installation successful!')"

# Or run directly
finagent --help

# Or using python -m
python -m finagent.main --help
```

## Configuration

### Environment Variables

Create a `.env` file in the `finagent` directory:

```bash
# .env file

# News API (get key from https://newsapi.org)
NEWSAPI_KEY=your_newsapi_key_here

# Finnhub API (get key from https://finnhub.io)
FINNHUB_API_KEY=your_finnhub_key_here

# LlamaParse API (get key from https://cloud.llamaindex.ai)
LLAMA_CLOUD_API_KEY=your_llama_key_here

# Logging level
LOG_LEVEL=INFO
```

### Configuration File

For more advanced configuration, create `config/local_settings.py`:

```python
# config/local_settings.py

# API Keys
NEWSAPI_KEY = "your_key"
FINNHUB_API_KEY = "your_key"

# Data directories
DATA_DIR = "data"
LOGS_DIR = "logs"

# Rate limiting (seconds between requests)
RATE_LIMIT_BSE = 1.0
RATE_LIMIT_NSE = 1.0
RATE_LIMIT_NEWS = 0.5

# Cache settings
CACHE_TTL_MINUTES = 15
MARKET_DATA_CACHE_DAYS = 1

# Vector store settings
VECTOR_STORE_PATH = "data/vectors"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Compliance settings
AUDIT_LOG_RETENTION_YEARS = 5

# Risk profile settings
RISK_PROFILE_EXPIRY_DAYS = 365
```

## Directory Setup

The system creates directories automatically, but you can pre-create them:

```bash
mkdir -p data/circulars
mkdir -p data/press_releases
mkdir -p data/cache
mkdir -p data/vectors
mkdir -p logs
```

## Database Initialization

Databases are created automatically on first use:

| Database | Location | Purpose |
|----------|----------|---------|
| `market_data.db` | `data/cache/` | OHLCV data cache |
| `circulars.db` | `data/cache/` | Circular metadata |
| `press_releases.db` | `data/cache/` | Press release metadata |
| `sentiment.db` | `data/cache/` | News sentiment cache |
| `ingestion.db` | `data/cache/` | Ingestion job tracking |
| `compliance.db` | `logs/` | Audit trail |
| `risk_profiles.db` | `data/cache/` | User risk profiles |

## First Run

### 1. Complete Risk Profile

```bash
python -m finagent.main profile
```

Answer the 7-10 questions to establish your risk tolerance. This is required before getting recommendations.

### 2. Ingest Initial Data

```bash
# Ingest last 7 days of circulars
python -m finagent.main ingest --days 7

# Or for a specific company
python -m finagent.main ingest-company RELIANCE --days 90
```

### 3. Verify Setup

```bash
# Check system status
python -m finagent.main status

# Check ingestion stats
python -m finagent.main ingestion-stats
```

### 4. Run First Analysis

```bash
python -m finagent.main analyze RELIANCE.NS --detailed
```

## API Access Requirements

### BSE API

- **URL**: `https://api.bseindia.com/`
- **Authentication**: None required
- **Rate Limit**: 1 request/second (enforced by system)
- **No API key needed**

### NSE API

- **URL**: `https://www.nseindia.com/api/`
- **Authentication**: Session-based (cookies)
- **Rate Limit**: 1 request/second
- **No API key needed**

The system automatically:
1. Visits the NSE homepage to get session cookies
2. Uses cookies for subsequent API calls
3. Refreshes session when expired

### NewsAPI

- **URL**: `https://newsapi.org/`
- **Authentication**: API key required
- **Free tier**: 100 requests/day
- **Get key**: https://newsapi.org/register

### Finnhub

- **URL**: `https://finnhub.io/`
- **Authentication**: API key required
- **Free tier**: 60 requests/minute
- **Get key**: https://finnhub.io/register

## Troubleshooting

### Import Errors / ModuleNotFoundError

```bash
# If you see "ModuleNotFoundError: No module named 'finagent'"
# You need to install the package properly:

# 1. Navigate to the project root (where pyproject.toml is located)
cd /path/to/stockprofiler

# 2. Install in development mode
pip install -e .

# 3. Now you can run from anywhere:
python -m finagent.main status

# Or use the finagent command directly:
finagent status
```

### TA-Lib Installation Issues

```bash
# If TA-Lib fails to install, try:
pip install numpy  # Install numpy first
pip install TA-Lib

# On Windows, use pre-built wheel:
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

### NSE API Errors

```bash
# If you see 401 errors from NSE
# The system auto-retries, but you can also:
python -c "
from finagent.pipelines.unstructured_data import CircularPipeline
p = CircularPipeline()
p._reset_nse_session()
print('Session reset')
"
```

### PDF Parsing Errors

```bash
# Install all PDF dependencies
pip install pypdf camelot-py[cv] ghostscript

# If Ghostscript is not found:
# Ubuntu: sudo apt-get install ghostscript
# macOS: brew install ghostscript
# Windows: Download and install from https://ghostscript.com/
```

### Vector Store Issues

```bash
# If ChromaDB fails:
pip install chromadb --upgrade

# Reset vector store if corrupted:
rm -rf data/vectors
python -m finagent.main ingest --days 7
```

## Performance Optimization

### For Large-Scale Ingestion

```python
# In your script:
from finagent.pipelines.ingestion_orchestrator import IngestionOrchestrator

orchestrator = IngestionOrchestrator(max_workers=8)  # Increase workers
```

### For Memory-Constrained Systems

```python
# Process documents in smaller batches
from finagent.pipelines.unstructured_data import CircularPipeline

pipeline = CircularPipeline()
# Fetch in smaller chunks
for i in range(0, 30, 7):
    docs = pipeline.fetch_recent_circulars(days_back=7)
    # Process docs...
```

### Caching Optimization

```python
# Increase cache TTL for less frequent updates
from finagent.config.settings import Settings

settings = Settings()
settings.cache_ttl = 60  # 60 minutes
```

## Security Considerations

1. **API Keys**: Never commit `.env` or API keys to version control
2. **Data Storage**: User data is stored locally in SQLite
3. **Compliance Logs**: 5-year retention as per SEBI requirements
4. **No Cloud Storage**: All data remains on local machine

## Updating

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# If database schema changes, backup first:
cp -r data/cache data/cache_backup
```

## Uninstallation

```bash
# Remove virtual environment
deactivate
rm -rf venv

# Remove data (optional)
rm -rf data logs
```

## Getting Help

- Check `python -m finagent.main --help` for CLI help
- Review `CLAUDE.md` for system rules and constraints
- Open an issue on GitHub for bugs
