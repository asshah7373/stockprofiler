# FinAgent API Reference

This document provides detailed API reference for using FinAgent programmatically.

## Pipelines

### MarketDataPipeline

Fetches structured market data from yfinance.

```python
from finagent.pipelines import MarketDataPipeline

pipeline = MarketDataPipeline(cache_db="data/cache/market_data.db")

# Get historical OHLCV data
df = pipeline.get_historical_data(
    ticker="RELIANCE.NS",
    period="1y",       # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
    interval="1d"      # 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
)
# Returns: DataFrame with Open, High, Low, Close, Adj Close, Volume

# Get live quote
quote = pipeline.get_live_quote("TCS.NS")
# Returns: {"price": 3500.00, "change": 25.50, "change_pct": 0.73, ...}

# Get company info
info = pipeline.get_company_info("INFY.NS")
# Returns: {"name": "Infosys Limited", "sector": "Technology", "pe_ratio": 25.5, ...}

# Check if market is open
is_open = pipeline.is_market_open()
```

### CircularPipeline

Fetches and parses BSE/NSE circulars and filings.

```python
from finagent.pipelines import CircularPipeline, DocumentType

pipeline = CircularPipeline(
    data_dir="data/circulars",
    db_path="data/cache/circulars.db"
)

# Fetch recent circulars from both exchanges
circulars = pipeline.fetch_recent_circulars(
    exchange="BOTH",      # "BSE", "NSE", or "BOTH"
    ticker="RELIANCE",    # Optional: filter by ticker
    days_back=7,          # Lookback period
    doc_types=None        # Optional: filter by document types
)
# Returns: List[CircularDocument]

# Fetch BSE announcements
bse_docs = pipeline.fetch_bse_announcements(
    ticker="TCS",
    days_back=30,
    category="Result"     # Optional: "Result", "Board Meeting", etc.
)

# Fetch NSE financial results
nse_results = pipeline.fetch_nse_financial_results(
    ticker="INFY",
    period="Quarterly"    # "Quarterly", "Half-Yearly", "Annual"
)

# Fetch NSE board meetings
meetings = pipeline.fetch_nse_board_meetings(
    ticker="HDFCBANK",
    days_back=60
)

# Fetch shareholding pattern
shareholding = pipeline.fetch_nse_shareholding("RELIANCE")

# Fetch all filings for a company
filings = pipeline.fetch_company_filings(
    ticker="TCS",
    days_back=365
)
# Returns: {
#     "quarterly_results": [...],
#     "board_meetings": [...],
#     "shareholding": [...],
#     "announcements": [...],
#     "all": [...]
# }

# Download a document
from pathlib import Path
local_path = pipeline.download_document(circular_doc)
# Returns: Path to downloaded file or None

# Parse a PDF
content = pipeline.parse_pdf(Path("data/circulars/doc.pdf"))
# Returns: {
#     "text": "Full extracted text...",
#     "tables": [{"headers": [...], "data": [[...]]}],
#     "key_figures": {"revenue": 1234.56, "net_profit": 567.89},
#     "metadata": {"pages": 5, "parsed_at": "..."}
# }

# Chunk for RAG
chunks = pipeline.chunk_for_rag(
    content=content,
    doc_id="abc123",
    metadata={"ticker": "TCS"},
    chunk_size=500,
    overlap=50
)
# Returns: List[{"text": "...", "metadata": {...}, "chunk_index": 0}]

# Process a document (download + parse + chunk + save)
doc, content, chunks = pipeline.process_document(
    doc=circular_doc,
    download=True,
    parse=True
)

# Search stored circulars
results = pipeline.search_circulars(
    query="quarterly results",
    ticker="TCS",
    doc_type="quarterly_results",
    exchange="NSE",
    days_back=90
)
# Returns: List[Dict]

# Get ingestion stats
stats = pipeline.get_ingestion_stats()
# Returns: {
#     "total_documents": 150,
#     "by_exchange": {"BSE": 80, "NSE": 70},
#     "by_type": {"quarterly_results": 20, ...},
#     "processed": 120,
#     "unprocessed": 30,
#     "last_7_days": 25
# }
```

### PressReleasePipeline

Fetches and parses company press releases.

```python
from finagent.pipelines import PressReleasePipeline

pipeline = PressReleasePipeline(
    data_dir="data/press_releases",
    db_path="data/cache/press_releases.db"
)

# Fetch from BSE
bse_releases = pipeline.fetch_bse_press_releases(
    ticker="RELIANCE",
    days_back=30
)
# Returns: List[PressRelease]

# Fetch from NSE
nse_releases = pipeline.fetch_nse_press_releases(
    ticker="TCS",
    days_back=30
)

# Fetch from all sources
all_releases = pipeline.fetch_all_press_releases(
    ticker="INFY",
    days_back=30
)

# Fetch from company website (experimental)
company_releases = pipeline.fetch_company_press_releases(
    company_website="https://www.reliance.com",
    ticker="RELIANCE",
    company_name="Reliance Industries Limited",
    max_releases=20
)

# Fetch full content of a press release
pr_with_content = pipeline.fetch_press_release_content(press_release)
# Returns: PressRelease with content, summary, categories, key_topics

# Process a press release
pr, chunks = pipeline.process_press_release(
    pr=press_release,
    fetch_content=True
)

# Search press releases
results = pipeline.search_press_releases(
    query="acquisition",
    ticker="TCS",
    source="BSE",      # "BSE", "NSE", "Company"
    days_back=90
)

# Get statistics
stats = pipeline.get_stats()
# Returns: {"total": 50, "by_source": {...}, "parsed": 40, "unparsed": 10}
```

### SentimentPipeline

Fetches and analyzes news sentiment.

```python
from finagent.pipelines import SentimentPipeline

pipeline = SentimentPipeline(
    newsapi_key="your_key",      # Optional
    finnhub_key="your_key",      # Optional
    db_path="data/cache/sentiment.db"
)

# Get news for a ticker
articles = pipeline.get_ticker_news(
    ticker="RELIANCE",
    days_back=7,
    max_articles=20
)
# Returns: List[{
#     "headline": "...",
#     "source": "Economic Times",
#     "url": "...",
#     "published_at": "...",
#     "sentiment_score": 0.65
# }]

# Get sector sentiment
sector = pipeline.get_sector_sentiment("IT")
# Returns: {
#     "sector": "IT",
#     "sentiment_score": 0.42,
#     "article_count": 25,
#     "top_headlines": [...],
#     "tickers_analyzed": ["TCS", "INFY", ...]
# }

# Get market sentiment
market = pipeline.get_market_sentiment()
# Returns: {
#     "overall_sentiment": 0.35,
#     "mood": "SLIGHTLY_BULLISH",
#     "sectors": {...},
#     "total_articles_analyzed": 150
# }

# Calculate aggregate sentiment
score = pipeline.aggregate_sentiment(articles)
# Returns: float (-1 to 1, weighted by recency and source)

# Get comprehensive summary
summary = pipeline.get_sentiment_summary("TCS", days_back=7)
# Returns: {
#     "ticker": "TCS",
#     "overall_sentiment": 0.45,
#     "sentiment_trend": "POSITIVE",
#     "positive_count": 8,
#     "negative_count": 3,
#     "neutral_count": 4,
#     "recent_headlines": [...]
# }
```

### IngestionOrchestrator

Coordinates batch ingestion jobs.

```python
from finagent.pipelines import IngestionOrchestrator

orchestrator = IngestionOrchestrator(
    data_dir="data",
    db_path="data/cache/ingestion.db",
    max_workers=4
)

# Ingest circulars
result = orchestrator.ingest_circulars(
    exchange="BOTH",
    ticker=None,           # All tickers
    days_back=7,
    download=True,
    parse=True,
    progress_callback=lambda cur, tot, msg: print(f"{cur}/{tot}: {msg}")
)
# Returns: IngestionResult

print(f"Found: {result.job.documents_found}")
print(f"Processed: {result.job.documents_processed}")
print(f"Chunks: {result.chunks_generated}")
print(f"Duration: {result.duration_seconds}s")

# Ingest press releases
result = orchestrator.ingest_press_releases(
    ticker="TCS",
    days_back=30,
    fetch_content=True
)

# Ingest everything
results = orchestrator.ingest_all(
    exchange="BOTH",
    ticker=None,
    days_back=7
)
# Returns: {"circulars": IngestionResult, "press_releases": IngestionResult}

# Comprehensive company ingestion
company_data = orchestrator.ingest_company(
    ticker="RELIANCE",
    days_back=365,
    include_results=True,
    include_announcements=True,
    include_press_releases=True
)
# Returns: {
#     "ticker": "RELIANCE",
#     "quarterly_results": [...],
#     "board_meetings": [...],
#     "shareholding": [...],
#     "announcements": [...],
#     "press_releases": [...],
#     "total_documents": 150,
#     "total_chunks": 450,
#     "errors": [...]
# }

# Get job details
job = orchestrator.get_job("abc123")

# Get recent jobs
jobs = orchestrator.get_recent_jobs(limit=10)

# Get statistics
stats = orchestrator.get_stats()

# Generate report
report = orchestrator.generate_report()
print(report)
```

## Analysis

### TechnicalAnalyzer

Calculates technical indicators using TA-Lib.

```python
from finagent.analysis import TechnicalAnalyzer
import pandas as pd

analyzer = TechnicalAnalyzer()

# Analyze OHLCV DataFrame
result = analyzer.analyze(df)
# Returns: {
#     "rsi_14": 55.3,
#     "macd": {"value": 12.5, "signal": 10.2, "histogram": 2.3},
#     "bollinger": {"upper": 3600, "middle": 3500, "lower": 3400},
#     "sma_50": 3450.0,
#     "sma_200": 3300.0,
#     "golden_cross": True,
#     "death_cross": False,
#     "beta": 1.15,
#     "volatility_30d": 0.25,
#     "signals": {
#         "rsi_signal": "NEUTRAL",
#         "macd_signal": "BULLISH",
#         "trend": "UPTREND"
#     }
# }

# Individual indicators
rsi = analyzer.calculate_rsi(df['Close'], period=14)
macd = analyzer.calculate_macd(df['Close'])
beta = analyzer.calculate_beta(stock_returns, benchmark_returns)
```

### FundamentalAnalyzer

Analyzes company fundamentals.

```python
from finagent.analysis import FundamentalAnalyzer

analyzer = FundamentalAnalyzer()

# Analyze company data
result = analyzer.analyze(company_info)
# Returns: {
#     "valuation": {
#         "pe_ratio": 25.5,
#         "pb_ratio": 4.2,
#         "ev_ebitda": 15.3,
#         "assessment": "FAIRLY_VALUED"
#     },
#     "profitability": {
#         "roe": 22.5,
#         "roa": 8.3,
#         "profit_margin": 15.2,
#         "assessment": "STRONG"
#     },
#     "growth": {
#         "revenue_growth": 12.5,
#         "profit_growth": 15.3,
#         "eps_growth": 14.2,
#         "assessment": "GROWING"
#     },
#     "leverage": {
#         "debt_to_equity": 45.2,
#         "interest_coverage": 8.5,
#         "assessment": "HEALTHY"
#     }
# }
```

### ChainOfThoughtSynthesizer

Generates recommendations using structured reasoning.

```python
from finagent.analysis import ChainOfThoughtSynthesizer

synthesizer = ChainOfThoughtSynthesizer()

# Generate recommendation
recommendation = synthesizer.synthesize(
    ticker="RELIANCE.NS",
    market_data={"price": 2500.0, ...},
    technical_data={...},
    fundamental_data={...},
    sentiment_data={...},
    rag_context={...},
    user_profile={"risk_category": "MODERATE", ...}
)
# Returns: Recommendation dataclass

print(recommendation.action)           # "BUY", "SELL", "HOLD"
print(recommendation.confidence_score) # 0.75
print(recommendation.time_horizon)     # "MEDIUM"
print(recommendation.rationale)        # Dict with reasoning
print(recommendation.sources_cited)    # List of source IDs
print(recommendation.suitability)      # Profile suitability

# Format as report
report = synthesizer.format_recommendation_report(recommendation)
print(report)

# Get as JSON
json_output = recommendation.to_json()
```

## Agents

### PerceiverAgent

Fetches all required data for analysis.

```python
from finagent.agents import PerceiverAgent

perceiver = PerceiverAgent()

# Perceive all data for a ticker
data = perceiver.perceive("RELIANCE.NS")
# Returns: {
#     "ticker": "RELIANCE.NS",
#     "market_data": {...},
#     "company_info": {...},
#     "technical_data": {"ohlcv": {...}},
#     "sentiment_data": {...},
#     "rag_context": {...},
#     "sources": [...]
# }

# Update circular database
result = perceiver.update_circular_database(
    exchange="BOTH",
    days_back=7
)

# Get data status
status = perceiver.get_data_status()
```

### AnalystAgent

Analyzes data and generates recommendations.

```python
from finagent.agents import AnalystAgent

analyst = AnalystAgent()

# Analyze perceived data
recommendation = analyst.analyze(
    perceived_data=data,
    technical_analysis=technical_data,
    user_profile=profile_dict
)

# Format report
report = analyst.format_analysis_report(recommendation)
print(report)
```

### ReviewerAgent

Validates analyst recommendations.

```python
from finagent.agents import ReviewerAgent

reviewer = ReviewerAgent()

# Review recommendation
result = reviewer.review(
    recommendation=recommendation.to_dict(),
    source_documents=rag_context,
    calculated_technicals=technical_data,
    perceived_data=perceived_data
)

print(result.is_valid)        # True/False
print(result.errors)          # List of errors
print(result.warnings)        # List of warnings
print(result.verified_claims) # List of verified claims

# Check if regeneration needed
if reviewer.requires_regeneration(result):
    guidance = reviewer.get_regeneration_guidance(result)
    print(guidance)

# Generate review report
report = reviewer.generate_review_report(result)
print(report)
```

## Risk Profile

### RiskProfiler

Conducts SEBI-mandated risk profiling.

```python
from finagent.risk_profile import RiskProfiler

profiler = RiskProfiler(db_path="data/cache/risk_profiles.db")

# Get questions
questions = profiler.get_questions()
# Returns: List of question dicts

# Conduct profiling
profile = profiler.conduct_profiling(responses={
    "age": 1,           # Index of selected option
    "income": 2,
    "experience": 1,
    "horizon": 2,
    "risk_scenario": 2,
    "capital": 2,
    "goal": 2
})
# Returns: RiskProfile

print(profile.profile_id)     # UUID
print(profile.risk_category)  # "CONSERVATIVE", "MODERATE", "AGGRESSIVE"
print(profile.risk_score)     # 0-21
print(profile.constraints)    # Investment constraints dict
print(profile.expiry_date)    # Profile expiry

# Get latest profile
profile = profiler.get_latest_profile()

# Check if expired
is_expired = profiler.is_profile_expired(profile)

# Get summary
summary = profiler.get_profile_summary(profile)
print(summary)
```

### ConstraintApplier

Applies risk profile constraints.

```python
from finagent.risk_profile import ConstraintApplier

applier = ConstraintApplier()

# Check stock suitability
result = applier.check_suitability(
    stock_data={"beta": 1.5, "market_cap_category": "SMALL", ...},
    profile=risk_profile
)

print(result.is_suitable)    # True/False
print(result.violations)     # List of constraint violations
print(result.warnings)       # List of warnings

# Filter recommendations
filtered = applier.filter_recommendations(
    recommendations=recommendations_list,
    profile=risk_profile
)

# Get max allocation
max_alloc = applier.get_allocation_limit(
    stock_data=stock_data,
    profile=risk_profile,
    portfolio_size=1000000
)
```

## RAG System

### VectorStore

ChromaDB-based vector storage.

```python
from finagent.rag import VectorStore

store = VectorStore(persist_dir="data/vectors")

# Add document
store.add_document(
    collection="circulars",
    doc_id="abc123",
    chunks=[
        {"text": "...", "metadata": {...}, "chunk_index": 0},
        ...
    ],
    metadata={"ticker": "TCS", "filing_date": "2024-01-15"}
)

# Search
results = store.search(
    collection="circulars",
    query="quarterly revenue growth",
    filters={"ticker": "TCS"},
    top_k=5
)
# Returns: List of matching chunks with relevance scores

# Get document chunks
chunks = store.get_document_chunks("circulars", "abc123")

# Get collection stats
stats = store.get_collection_stats("circulars")
```

### RAGRetriever

High-level retrieval with citation support.

```python
from finagent.rag import RAGRetriever, VectorStore

retriever = RAGRetriever(VectorStore())

# Retrieve context for analysis
context = retriever.retrieve_for_analysis(
    ticker="TCS",
    query="quarterly revenue and profit",
    doc_types=["quarterly_results", "circulars"],
    lookback_days=90,
    top_k=10
)
# Returns: {
#     "chunks": [...],
#     "sources": [{"doc_id": "...", "title": "...", "date": "..."}],
#     "retrieval_metadata": {...}
# }

# Format for LLM
formatted = retriever.format_context_for_llm(context, max_tokens=4000)
print(formatted)

# Verify a citation
verification = retriever.verify_citation(
    claim="Revenue grew 15%",
    doc_id="abc123",
    collection="circulars"
)
print(verification["verified"])  # True/False
print(verification["supporting_text"])
```

## Compliance

### ComplianceLogger

SEBI-mandated audit trail.

```python
from finagent.utils import ComplianceLogger

logger = ComplianceLogger(db_path="logs/compliance.db")

# Log interaction
interaction_id = logger.log_interaction(
    interaction_type="stock_analysis",
    query="Analyze RELIANCE.NS",
    response="Analysis report...",
    recommendation={"action": "BUY", ...},
    sources=["yfinance", "bse_circular_123"],
    risk_profile={"profile_id": "...", "risk_category": "MODERATE"},
    user_id="user123",
    session_id="session456"
)

# Log data access
logger.log_data_access(
    data_source="BSE",
    data_type="circular",
    ticker="RELIANCE",
    interaction_id=interaction_id,
    success=True
)

# Get audit trail
trail = logger.get_audit_trail(
    user_id="user123",
    start_date="2024-01-01",
    end_date="2024-12-31",
    interaction_type="stock_analysis",
    limit=100
)

# Verify integrity
is_valid = logger.verify_integrity(interaction_id)

# Get compliance summary
summary = logger.get_compliance_summary()

# Export audit report
logger.export_audit_report(
    output_path="compliance_report.json",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

## Data Types

### CircularDocument

```python
from finagent.pipelines import CircularDocument

doc = CircularDocument(
    id="abc123",
    ticker="RELIANCE",
    company_name="Reliance Industries Limited",
    title="Quarterly Results Q3 FY24",
    url="https://...",
    doc_type="quarterly_results",
    exchange="BSE",
    filing_date="2024-01-15T00:00:00",
    category="Results",
    subcategory="Quarterly",
    description="...",
    attachment_name="results.pdf",
    file_size=1024000,
    local_path="/path/to/file.pdf",
    parsed=True
)

# Convert to dict
doc_dict = doc.to_dict()
```

### PressRelease

```python
from finagent.pipelines import PressRelease

pr = PressRelease(
    id="xyz789",
    ticker="TCS",
    company_name="Tata Consultancy Services",
    title="TCS announces strategic partnership",
    url="https://...",
    source="BSE",
    release_date="2024-01-20T10:30:00",
    content="Full content...",
    summary="Summary...",
    categories=["partnership", "expansion"],
    key_topics=["Amount: Rs 1000 Cr", "Entity: Microsoft"],
    sentiment_keywords=["+growth", "+partnership"],
    is_parsed=True
)
```

### Recommendation

```python
from finagent.analysis import Recommendation, Action, TimeHorizon

rec = Recommendation(
    ticker="RELIANCE.NS",
    company_name="Reliance Industries Limited",
    action="BUY",
    time_horizon="MEDIUM",
    confidence_score=0.75,
    target_price=None,
    stop_loss=None,
    current_price=2500.0,
    rationale={
        "macro_context": "...",
        "fundamental_analysis": "...",
        "technical_analysis": "...",
        "risk_factors": "...",
        "catalyst": "..."
    },
    sources_cited=["bse_123", "nse_456"],
    suitability="Suitable for Moderate profiles",
    risk_level="MODERATE",
    disclaimer="..."
)

# Convert to dict/json
rec_dict = rec.to_dict()
rec_json = rec.to_json()
```

### RiskProfile

```python
from finagent.risk_profile import RiskProfile, RiskCategory

profile = RiskProfile(
    profile_id="uuid-here",
    created_at="2024-01-15T10:30:00",
    expiry_date="2025-01-15T10:30:00",
    risk_category="MODERATE",
    risk_score=12,
    constraints={
        "max_beta": 1.2,
        "min_market_cap": "MID",
        "allowed_sectors": "ALL",
        "excluded_instruments": ["Penny Stocks"],
        "max_single_stock_allocation": 0.10,
        "prefer_dividend": False,
        "min_credit_rating": "A"
    },
    responses={...}  # Original questionnaire responses
)
```
