"""
FinAgent Orchestrator
Main entry point and workflow coordinator.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Optional
import logging
import json
from datetime import datetime

from .config.settings import get_settings, Settings
from .agents.perceiver import PerceiverAgent
from .agents.analyst import AnalystAgent
from .agents.reviewer import ReviewerAgent
from .analysis.technical import TechnicalAnalyzer
from .analysis.fundamental import FundamentalAnalyzer
from .analysis.synthesizer import ChainOfThoughtSynthesizer
from .risk_profile.questionnaire import RiskProfiler
from .risk_profile.constraints import ConstraintApplier
from .utils.compliance_logger import ComplianceLogger, DISCLAIMER_TEXT

app = typer.Typer(
    name="finagent",
    help="FinAgent - Autonomous Financial Analysis System for Indian Markets"
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@app.command()
def profile():
    """
    Run interactive risk profiling questionnaire.

    SEBI-mandated risk assessment before any recommendations.
    """
    console.print(Panel(
        "[bold cyan]Risk Profile Assessment[/bold cyan]\n\n"
        "This questionnaire helps determine your investment risk tolerance.\n"
        "Your answers will be used to tailor recommendations to your profile.",
        title="FinAgent"
    ))

    profiler = RiskProfiler()
    questions = profiler.get_questions()
    responses = {}

    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold]Question {i}/{len(questions)}[/bold]")
        console.print(f"[yellow]{q['question']}[/yellow]\n")

        for j, option in enumerate(q['options']):
            console.print(f"  {j + 1}. {option}")

        while True:
            try:
                answer = typer.prompt("Your choice (1-" + str(len(q['options'])) + ")")
                answer_idx = int(answer) - 1
                if 0 <= answer_idx < len(q['options']):
                    responses[q['id']] = answer_idx
                    break
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")

    # Create profile
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Creating your risk profile...", total=None)
        profile = profiler.conduct_profiling(responses)

    # Display result
    console.print("\n")
    console.print(Panel(
        f"[bold green]Risk Profile Created![/bold green]\n\n"
        f"Profile ID: {profile.profile_id}\n"
        f"Risk Category: [bold]{profile.risk_category}[/bold]\n"
        f"Risk Score: {profile.risk_score}/21\n"
        f"Valid Until: {profile.expiry_date[:10]}",
        title="Your Risk Profile"
    ))

    console.print(profiler.get_profile_summary(profile))


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., RELIANCE.NS)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed analysis"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")
):
    """
    Run full analysis on a single stock.
    """
    setup_logging(verbose)

    # Check for risk profile
    profiler = RiskProfiler()
    profile = profiler.get_latest_profile()

    if not profile:
        console.print("[yellow]Warning: No risk profile found. Running analysis without profile constraints.[/yellow]")
        console.print("Run 'finagent profile' to create a risk profile first.\n")
        user_profile = None
    else:
        if profiler.is_profile_expired(profile):
            console.print("[yellow]Warning: Risk profile has expired. Consider updating it.[/yellow]\n")
        user_profile = profile.to_dict()

    console.print(Panel(
        f"[bold cyan]Analyzing: {ticker}[/bold cyan]",
        title="FinAgent Stock Analysis"
    ))

    # Initialize components
    perceiver = PerceiverAgent()
    analyst = AnalystAgent()
    reviewer = ReviewerAgent()
    technical_analyzer = TechnicalAnalyzer()
    compliance_logger = ComplianceLogger()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Step 1: Perceive data
            task = progress.add_task("Fetching market data...", total=None)
            perceived_data = perceiver.perceive(ticker)
            progress.update(task, description="[green]Market data fetched[/green]")

            # Step 2: Technical analysis
            task = progress.add_task("Running technical analysis...", total=None)
            if perceived_data.get("technical_data", {}).get("ohlcv"):
                import pandas as pd
                ohlcv = perceived_data["technical_data"]["ohlcv"]
                df = pd.DataFrame.from_dict(ohlcv, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                technical_data = technical_analyzer.analyze(df)
            else:
                technical_data = {"error": "No historical data available"}
            progress.update(task, description="[green]Technical analysis complete[/green]")

            # Step 3: Generate recommendation
            task = progress.add_task("Generating recommendation...", total=None)
            recommendation = analyst.analyze(
                perceived_data=perceived_data,
                technical_analysis=technical_data,
                user_profile=user_profile
            )
            progress.update(task, description="[green]Recommendation generated[/green]")

            # Step 4: Review
            task = progress.add_task("Reviewing analysis...", total=None)
            rag_context = perceived_data.get("rag_context", {})
            review_result = reviewer.review(
                recommendation=recommendation.to_dict() if hasattr(recommendation, 'to_dict') else recommendation.__dict__,
                source_documents=rag_context,
                calculated_technicals=technical_data,
                perceived_data=perceived_data
            )
            progress.update(task, description="[green]Review complete[/green]")

        # Log for compliance
        compliance_logger.log_interaction(
            interaction_type="stock_analysis",
            query=f"Analyze {ticker}",
            response=analyst.format_analysis_report(recommendation),
            recommendation=recommendation.to_dict() if hasattr(recommendation, 'to_dict') else recommendation.__dict__,
            sources=perceived_data.get("sources", []),
            risk_profile=user_profile
        )

        # Display results
        if output_json:
            result = {
                "recommendation": recommendation.to_dict() if hasattr(recommendation, 'to_dict') else recommendation.__dict__,
                "review": {
                    "is_valid": review_result.is_valid,
                    "warnings": review_result.warnings,
                    "errors": review_result.errors
                }
            }
            console.print_json(json.dumps(result, indent=2, default=str))
        else:
            console.print(analyst.format_analysis_report(recommendation))

            if review_result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for w in review_result.warnings:
                    console.print(f"  • {w}")

            if review_result.errors:
                console.print("\n[red]Errors detected in analysis:[/red]")
                for e in review_result.errors:
                    console.print(f"  • {e}")

            if detailed:
                console.print("\n[bold]Technical Indicators:[/bold]")
                table = Table()
                table.add_column("Indicator", style="cyan")
                table.add_column("Value", style="green")
                table.add_column("Signal", style="yellow")

                signals = technical_data.get("signals", {})
                table.add_row("RSI (14)", f"{technical_data.get('rsi_14', 'N/A'):.1f}" if technical_data.get('rsi_14') else "N/A", signals.get("rsi_signal", "N/A"))
                table.add_row("MACD", f"{technical_data.get('macd', {}).get('value', 'N/A'):.2f}" if technical_data.get('macd', {}).get('value') else "N/A", signals.get("macd_signal", "N/A"))
                table.add_row("Trend", "-", signals.get("trend", "N/A"))
                table.add_row("50-DMA", f"{technical_data.get('sma_50', 'N/A'):.2f}" if technical_data.get('sma_50') else "N/A", "-")
                table.add_row("200-DMA", f"{technical_data.get('sma_200', 'N/A'):.2f}" if technical_data.get('sma_200') else "N/A", "-")

                console.print(table)

    except Exception as e:
        console.print(f"[red]Error analyzing {ticker}: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def suggest(
    count: int = typer.Option(5, "--count", "-n", help="Number of suggestions"),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector (IT, BANKING, PHARMA, AUTO, FMCG, INFRA, METALS, OIL_GAS, TELECOM)"),
    horizon: str = typer.Option("1w", "--horizon", "-h", help="Time horizon (intraday, 1d, 1w, 1m, 3m, 1y)"),
    commodity: Optional[str] = typer.Option(None, "--commodity", "-c", help="Commodity play (gold, oil, steel, copper, agriculture)"),
    min_score: float = typer.Option(20.0, "--min-score", help="Minimum recommendation score (0-100)")
):
    """
    Generate stock recommendations based on news catalysts and time horizon.

    Analyzes ingested news/announcements to find stocks with positive catalysts:
    - Contract wins
    - Product launches
    - Strong results
    - Regulatory approvals
    - Strategic partnerships
    - And more...

    Examples:
        finagent suggest --horizon intraday     # Today's opportunities
        finagent suggest --horizon 1w -n 10     # Weekly picks
        finagent suggest --commodity gold       # Gold price beneficiaries
        finagent suggest --sector PHARMA        # Pharma sector only
    """
    from .analysis.news_screener import NewsScreener, TimeHorizon
    from rich.table import Table

    # Map horizon string to enum
    horizon_map = {
        'intraday': TimeHorizon.INTRADAY,
        '1d': TimeHorizon.SHORT,
        '1w': TimeHorizon.WEEKLY,
        '1m': TimeHorizon.MONTHLY,
        '3m': TimeHorizon.QUARTERLY,
        '1y': TimeHorizon.LONG,
    }

    time_horizon = horizon_map.get(horizon.lower(), TimeHorizon.WEEKLY)

    horizon_display = {
        TimeHorizon.INTRADAY: "Intraday (same day)",
        TimeHorizon.SHORT: "Short-term (1 day)",
        TimeHorizon.WEEKLY: "Weekly (1 week)",
        TimeHorizon.MONTHLY: "Monthly (1 month)",
        TimeHorizon.QUARTERLY: "Quarterly (3 months)",
        TimeHorizon.LONG: "Long-term (1 year+)",
    }

    console.print(Panel(
        f"[bold cyan]News-Driven Stock Suggestions[/bold cyan]\n\n"
        f"Time Horizon: {horizon_display.get(time_horizon, horizon)}\n"
        f"Count: {count}" +
        (f"\nSector: {sector.upper()}" if sector else "") +
        (f"\nCommodity Play: {commodity}" if commodity else ""),
        title="FinAgent"
    ))

    # Check for risk profile
    profiler = RiskProfiler()
    profile = profiler.get_latest_profile()

    if profile:
        console.print(f"\nUsing profile: [bold]{profile.risk_category}[/bold] (ID: {profile.profile_id[:8]}...)")
    else:
        console.print("\n[yellow]Note: No risk profile. Run 'finagent profile' for personalized recommendations.[/yellow]")

    # Initialize screener
    screener = NewsScreener()

    # Check data availability
    summary = screener.get_screening_summary()
    if summary['total_documents'] == 0:
        console.print("\n[red]No news data available for screening.[/red]")
        console.print("Run 'finagent ingest --days 7' first to fetch recent announcements.")
        raise typer.Exit(1)

    console.print(f"\n[dim]Analyzing {summary['total_documents']} documents...[/dim]")

    # Handle commodity play request
    if commodity:
        console.print(f"\n[bold]Stocks benefiting from {commodity.upper()} price movement:[/bold]\n")
        commodity_stocks = screener.get_commodity_plays(commodity, trend="bullish")

        if not commodity_stocks:
            console.print(f"[yellow]No {commodity}-related stocks found in recent news.[/yellow]")
        else:
            for i, stock in enumerate(commodity_stocks[:count], 1):
                console.print(f"  {i}. [bold]{stock['ticker']}[/bold] - {stock['company_name']}")
                console.print(f"     [dim]{stock['headline'][:80]}...[/dim]")

        console.print(f"\n[dim]Run 'finagent analyze <ticker>' for detailed analysis.[/dim]")
        return

    # Get news-based recommendations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Screening for catalysts...", total=None)
        recommendations = screener.screen_stocks(
            time_horizon=time_horizon,
            sector=sector.upper() if sector else None,
            min_confidence=0.4,
            limit=count
        )
        progress.update(task, description="[green]Screening complete[/green]")

    if not recommendations:
        console.print("\n[yellow]No strong recommendations found for the given criteria.[/yellow]")
        console.print("Try:")
        console.print("  - Different time horizon (--horizon 1m)")
        console.print("  - Removing sector filter")
        console.print("  - Running 'finagent ingest' to fetch more data")

        # Fallback to popular stocks
        console.print("\n[bold]Popular stocks for manual analysis:[/bold]")
        fallback = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'][:count]
        for i, ticker in enumerate(fallback, 1):
            console.print(f"  {i}. {ticker}")
        return

    # Display recommendations
    console.print(f"\n[bold green]Top {len(recommendations)} Recommendations ({horizon_display.get(time_horizon, horizon)}):[/bold green]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Ticker", style="bold")
    table.add_column("Company", width=25)
    table.add_column("Score", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("Catalysts", width=40)

    for i, rec in enumerate(recommendations, 1):
        # Format catalysts summary
        catalyst_summary = []
        for c in rec.catalysts[:2]:
            cat_type = c.catalyst_type.value.replace('_', ' ').title()
            catalyst_summary.append(cat_type)
        catalyst_str = ", ".join(catalyst_summary) if catalyst_summary else "Multiple"

        # Risk color
        risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(rec.risk_level, "white")

        table.add_row(
            str(i),
            rec.ticker,
            (rec.company_name[:23] + "..") if len(rec.company_name) > 25 else rec.company_name,
            f"{rec.score:.1f}",
            f"[{risk_color}]{rec.risk_level}[/{risk_color}]",
            catalyst_str
        )

    console.print(table)

    # Show detailed reasoning for top 3
    console.print("\n[bold]Key Catalysts:[/bold]\n")
    for rec in recommendations[:3]:
        console.print(f"[bold]{rec.ticker}[/bold] ({rec.sector})")
        console.print(f"  {rec.reasoning}")
        if rec.catalysts:
            for c in rec.catalysts[:2]:
                headline = c.headline[:70] if len(c.headline) > 70 else c.headline
                console.print(f"  - [dim]{headline}...[/dim]")
        console.print()

    console.print(f"[dim]Run 'finagent analyze <ticker>' for detailed analysis.[/dim]")


@app.command()
def update_data(
    days: int = typer.Option(7, "--days", "-d", help="Fetch circulars from last N days"),
    exchange: str = typer.Option("BOTH", "--exchange", "-e", help="Exchange (NSE, BSE, or BOTH)")
):
    """
    Update circular database and market data cache.
    """
    console.print(Panel(
        f"[bold cyan]Updating Data[/bold cyan]\n\n"
        f"Fetching circulars from {exchange} for last {days} days",
        title="FinAgent"
    ))

    perceiver = PerceiverAgent()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Fetching {exchange} circulars...", total=None)
        result = perceiver.update_circular_database(exchange=exchange, days_back=days)
        progress.update(task, description="[green]Complete[/green]")

    console.print(f"\n[green]Update complete![/green]")
    console.print(f"  • Circulars found: {result.get('circulars_found', 0)}")
    console.print(f"  • Processed: {result.get('processed', 0)}")
    console.print(f"  • Failed: {result.get('failed', 0)}")


@app.command()
def ingest(
    exchange: str = typer.Option("BOTH", "--exchange", "-e", help="Exchange (NSE, BSE, or BOTH)"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Specific ticker to ingest"),
    days: int = typer.Option(7, "--days", "-d", help="Days to look back"),
    doc_type: str = typer.Option("all", "--type", help="Document type (circulars, press, all)"),
    no_download: bool = typer.Option(False, "--no-download", help="Skip downloading files"),
    no_parse: bool = typer.Option(False, "--no-parse", help="Skip parsing content"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-process already ingested documents")
):
    """
    Ingest circulars and press releases from BSE/NSE.

    This command fetches, downloads, and processes regulatory filings
    and press releases for storage in the RAG system.

    By default, already-ingested documents are skipped. Use --force to re-process them.
    """
    from .pipelines.ingestion_orchestrator import IngestionOrchestrator
    from rich.progress import BarColumn, TaskProgressColumn

    console.print(Panel(
        f"[bold cyan]Document Ingestion[/bold cyan]\n\n"
        f"Exchange: {exchange}\n"
        f"Ticker: {ticker or 'All'}\n"
        f"Lookback: {days} days\n"
        f"Type: {doc_type}",
        title="FinAgent Ingestion"
    ))

    orchestrator = IngestionOrchestrator()

    def progress_callback(current: int, total: int, message: str):
        if total > 0:
            pct = (current / total) * 100
            console.print(f"  [{current}/{total}] {message}")

    try:
        if doc_type in ["all", "circulars"]:
            console.print("\n[bold]Ingesting Circulars...[/bold]")
            result = orchestrator.ingest_circulars(
                exchange=exchange,
                ticker=ticker,
                days_back=days,
                download=not no_download,
                parse=not no_parse,
                skip_existing=not force,
                progress_callback=progress_callback
            )

            console.print(f"\n[green]Circular Ingestion Complete![/green]")
            console.print(f"  • New documents: {result.job.documents_found}")
            console.print(f"  • Processed: {result.job.documents_processed}")
            console.print(f"  • Failed: {result.job.documents_failed}")
            console.print(f"  • Chunks generated: {result.chunks_generated}")
            console.print(f"  • Duration: {result.duration_seconds:.1f}s")

        if doc_type in ["all", "press"]:
            console.print("\n[bold]Ingesting Press Releases...[/bold]")
            result = orchestrator.ingest_press_releases(
                ticker=ticker,
                days_back=days,
                fetch_content=not no_parse,
                skip_existing=not force,
                progress_callback=progress_callback
            )

            console.print(f"\n[green]Press Release Ingestion Complete![/green]")
            console.print(f"  • New documents: {result.job.documents_found}")
            console.print(f"  • Processed: {result.job.documents_processed}")
            console.print(f"  • Failed: {result.job.documents_failed}")
            console.print(f"  • Chunks generated: {result.chunks_generated}")
            console.print(f"  • Duration: {result.duration_seconds:.1f}s")

    except Exception as e:
        console.print(f"[red]Ingestion error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ingest_company(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., RELIANCE)"),
    days: int = typer.Option(365, "--days", "-d", help="Days to look back"),
    include_results: bool = typer.Option(True, "--results/--no-results", help="Include quarterly results"),
    include_press: bool = typer.Option(True, "--press/--no-press", help="Include press releases"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-process already ingested documents")
):
    """
    Comprehensive ingestion for a specific company.

    Fetches all available filings including:
    - Quarterly and annual results
    - Board meeting outcomes
    - Shareholding patterns
    - Corporate announcements
    - Press releases

    By default, already-ingested documents are skipped. Use --force to re-process them.
    """
    from .pipelines.ingestion_orchestrator import IngestionOrchestrator

    console.print(Panel(
        f"[bold cyan]Company Ingestion: {ticker}[/bold cyan]\n\n"
        f"Lookback: {days} days\n"
        f"Include Results: {include_results}\n"
        f"Include Press: {include_press}",
        title="FinAgent"
    ))

    orchestrator = IngestionOrchestrator()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Ingesting data for {ticker}...", total=None)

            result = orchestrator.ingest_company(
                ticker=ticker,
                days_back=days,
                include_results=include_results,
                include_press_releases=include_press,
                skip_existing=not force
            )

            progress.update(task, description="[green]Complete[/green]")

        console.print(f"\n[green]Company Ingestion Complete![/green]")
        console.print(f"\n[bold]Documents Found:[/bold]")
        console.print(f"  • Quarterly Results: {len(result.get('quarterly_results', []))}")
        console.print(f"  • Board Meetings: {len(result.get('board_meetings', []))}")
        console.print(f"  • Shareholding: {len(result.get('shareholding', []))}")
        console.print(f"  • Announcements: {len(result.get('announcements', []))}")
        console.print(f"  • Press Releases: {len(result.get('press_releases', []))}")
        console.print(f"\n  Total Documents: {result.get('total_documents', 0)}")
        console.print(f"  Total Chunks: {result.get('total_chunks', 0)}")

        if result.get('errors'):
            console.print(f"\n[yellow]Errors ({len(result['errors'])}):[/yellow]")
            for err in result['errors'][:5]:
                console.print(f"  • {err}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def search_filings(
    query: Optional[str] = typer.Argument(None, help="Search query"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    doc_type: Optional[str] = typer.Option(None, "--type", help="Filter by document type"),
    exchange: Optional[str] = typer.Option(None, "--exchange", "-e", help="Filter by exchange"),
    days: int = typer.Option(30, "--days", "-d", help="Days to look back"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results to show")
):
    """
    Search ingested circulars and press releases.
    """
    from .pipelines.unstructured_data import CircularPipeline
    from .pipelines.press_releases import PressReleasePipeline

    circular_pipeline = CircularPipeline()
    press_pipeline = PressReleasePipeline()

    console.print(Panel(
        f"[bold cyan]Search Filings[/bold cyan]\n\n"
        f"Query: {query or 'All'}\n"
        f"Ticker: {ticker or 'All'}\n"
        f"Type: {doc_type or 'All'}",
        title="FinAgent"
    ))

    # Search circulars
    circulars = circular_pipeline.search_circulars(
        query=query,
        ticker=ticker,
        doc_type=doc_type,
        exchange=exchange,
        days_back=days
    )

    # Search press releases
    press_releases = press_pipeline.search_press_releases(
        query=query,
        ticker=ticker,
        days_back=days
    )

    # Display results
    if circulars:
        console.print(f"\n[bold]Circulars ({len(circulars)}):[/bold]")
        table = Table()
        table.add_column("Date", style="cyan", width=12)
        table.add_column("Ticker", style="green", width=12)
        table.add_column("Type", style="yellow", width=15)
        table.add_column("Title", width=50)

        for doc in circulars[:limit]:
            date = doc.get('filing_date', '')[:10] if doc.get('filing_date') else 'N/A'
            table.add_row(
                date,
                doc.get('ticker', 'N/A'),
                doc.get('doc_type', 'N/A'),
                (doc.get('title', 'N/A')[:47] + '...') if len(doc.get('title', '')) > 50 else doc.get('title', 'N/A')
            )

        console.print(table)

    if press_releases:
        console.print(f"\n[bold]Press Releases ({len(press_releases)}):[/bold]")
        table = Table()
        table.add_column("Date", style="cyan", width=12)
        table.add_column("Ticker", style="green", width=12)
        table.add_column("Source", style="yellow", width=10)
        table.add_column("Title", width=50)

        for pr in press_releases[:limit]:
            date = pr.get('release_date', '')[:10] if pr.get('release_date') else 'N/A'
            table.add_row(
                date,
                pr.get('ticker', 'N/A'),
                pr.get('source', 'N/A'),
                (pr.get('title', 'N/A')[:47] + '...') if len(pr.get('title', '')) > 50 else pr.get('title', 'N/A')
            )

        console.print(table)

    if not circulars and not press_releases:
        console.print("[yellow]No results found.[/yellow]")


@app.command()
def ingestion_stats():
    """
    Show ingestion statistics and status.
    """
    from .pipelines.ingestion_orchestrator import IngestionOrchestrator

    orchestrator = IngestionOrchestrator()
    stats = orchestrator.get_stats()

    console.print(Panel(
        "[bold cyan]Ingestion Statistics[/bold cyan]",
        title="FinAgent"
    ))

    # Jobs
    console.print("\n[bold]Ingestion Jobs:[/bold]")
    console.print(f"  • Total jobs: {stats['jobs'].get('total', 0)}")
    console.print(f"  • Documents processed: {stats['jobs'].get('total_processed', 0)}")
    console.print(f"  • Documents failed: {stats['jobs'].get('total_failed', 0)}")

    # Circulars
    console.print("\n[bold]Circulars:[/bold]")
    console.print(f"  • Total: {stats['circulars'].get('total_documents', 0)}")
    console.print(f"  • Processed: {stats['circulars'].get('processed', 0)}")
    console.print(f"  • Pending: {stats['circulars'].get('unprocessed', 0)}")
    console.print(f"  • Last 7 days: {stats['circulars'].get('last_7_days', 0)}")

    by_exchange = stats['circulars'].get('by_exchange', {})
    if by_exchange:
        console.print("  • By Exchange:")
        for ex, count in by_exchange.items():
            console.print(f"      {ex}: {count}")

    by_type = stats['circulars'].get('by_type', {})
    if by_type:
        console.print("  • By Type:")
        for t, count in list(by_type.items())[:5]:
            console.print(f"      {t}: {count}")

    # Press Releases
    console.print("\n[bold]Press Releases:[/bold]")
    console.print(f"  • Total: {stats['press_releases'].get('total', 0)}")
    console.print(f"  • Parsed: {stats['press_releases'].get('parsed', 0)}")
    console.print(f"  • Pending: {stats['press_releases'].get('unparsed', 0)}")

    by_source = stats['press_releases'].get('by_source', {})
    if by_source:
        console.print("  • By Source:")
        for src, count in by_source.items():
            console.print(f"      {src}: {count}")

    # Recent jobs
    recent_jobs = orchestrator.get_recent_jobs(limit=5)
    if recent_jobs:
        console.print("\n[bold]Recent Jobs:[/bold]")
        table = Table()
        table.add_column("Job ID", style="cyan", width=12)
        table.add_column("Type", style="green", width=15)
        table.add_column("Status", width=10)
        table.add_column("Processed", width=10)
        table.add_column("Created", width=20)

        for job in recent_jobs:
            status_color = "green" if job['status'] == 'completed' else "yellow" if job['status'] == 'partial' else "red"
            table.add_row(
                job['job_id'],
                job['job_type'],
                f"[{status_color}]{job['status']}[/{status_color}]",
                str(job.get('documents_processed', 0)),
                job['created_at'][:19] if job.get('created_at') else 'N/A'
            )

        console.print(table)


@app.command()
def status():
    """
    Show system status and data statistics.
    """
    settings = get_settings()
    perceiver = PerceiverAgent()
    profiler = RiskProfiler()

    console.print(Panel(
        "[bold cyan]System Status[/bold cyan]",
        title="FinAgent"
    ))

    # Data status
    data_status = perceiver.get_data_status()

    console.print("\n[bold]Vector Store:[/bold]")
    for name, stats in data_status.get("vector_store", {}).items():
        console.print(f"  • {name}: {stats.get('document_count', 0)} documents")

    # Risk profile
    profile = profiler.get_latest_profile()
    console.print("\n[bold]Risk Profile:[/bold]")
    if profile:
        console.print(f"  • Category: {profile.risk_category}")
        console.print(f"  • Created: {profile.created_at[:10]}")
        console.print(f"  • Expires: {profile.expiry_date[:10]}")
    else:
        console.print("  • No profile found. Run 'finagent profile' to create one.")

    # Settings
    console.print("\n[bold]Configuration:[/bold]")
    config = settings.to_dict()
    console.print(f"  • Log level: {config.get('log_level')}")
    console.print(f"  • News API: {'Configured' if config.get('has_newsapi_key') else 'Not configured'}")


@app.command()
def compliance_report(
    output: str = typer.Option("compliance_report.json", "--output", "-o", help="Output file"),
    days: int = typer.Option(30, "--days", "-d", help="Report period in days")
):
    """
    Generate compliance audit report.
    """
    from datetime import timedelta

    logger = ComplianceLogger()

    start_date = (datetime.now() - timedelta(days=days)).isoformat()
    end_date = datetime.now().isoformat()

    console.print(f"Generating compliance report for last {days} days...")

    logger.export_audit_report(
        output_path=output,
        start_date=start_date,
        end_date=end_date
    )

    console.print(f"[green]Report saved to: {output}[/green]")

    # Show summary
    summary = logger.get_compliance_summary()
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  • Total interactions: {summary.get('total_interactions', 0)}")
    console.print(f"  • Recommendations (30d): {summary.get('recommendations_last_30_days', 0)}")


@app.command()
def institutional(
    data_type: str = typer.Argument("fii-dii", help="Data type: fii-dii, bulk, block, all"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    days: int = typer.Option(7, "--days", "-d", help="Days to look back"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    Fetch and display institutional investment data.

    Data types:
    - fii-dii: FII/DII daily activity (highest alpha for market direction)
    - bulk: Bulk deals (>0.5% of shares)
    - block: Block deals (>5 lakh shares or Rs 10 crore)
    - insider: Insider trading (SAST filings)
    - all: All institutional data

    Examples:
        finagent institutional fii-dii           # FII/DII flow last 7 days
        finagent institutional bulk -t RELIANCE  # Bulk deals for RELIANCE
        finagent institutional insider --days 30 # Insider trading last 30 days
    """
    from .pipelines.institutional_data import InstitutionalDataPipeline

    console.print(Panel(
        f"[bold cyan]Institutional Data: {data_type.upper()}[/bold cyan]\n\n"
        f"Lookback: {days} days" +
        (f"\nTicker: {ticker}" if ticker else ""),
        title="FinAgent"
    ))

    pipeline = InstitutionalDataPipeline()

    try:
        if data_type.lower() in ['fii-dii', 'fiidii', 'fii', 'dii']:
            # FII/DII Activity
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Fetching FII/DII data...", total=None)
                trend = pipeline.get_fii_dii_trend(days=days)
                progress.update(task, description="[green]Complete[/green]")

            if output_json:
                console.print_json(json.dumps(trend, indent=2, default=str))
            else:
                console.print("\n[bold]FII Activity:[/bold]")
                fii = trend.get('fii', {})
                fii_color = "green" if fii.get('sentiment') == 'bullish' else "red" if fii.get('sentiment') == 'bearish' else "yellow"
                console.print(f"  Net Flow: [{fii_color}]₹{fii.get('net_total_cr', 0):,.0f} Cr[/{fii_color}]")
                console.print(f"  Positive Days: {fii.get('positive_days', 0)}/{fii.get('total_days', 0)}")
                console.print(f"  Sentiment: [{fii_color}]{fii.get('sentiment', 'N/A').upper()}[/{fii_color}]")

                console.print("\n[bold]DII Activity:[/bold]")
                dii = trend.get('dii', {})
                dii_color = "green" if dii.get('sentiment') == 'bullish' else "red" if dii.get('sentiment') == 'bearish' else "yellow"
                console.print(f"  Net Flow: [{dii_color}]₹{dii.get('net_total_cr', 0):,.0f} Cr[/{dii_color}]")
                console.print(f"  Positive Days: {dii.get('positive_days', 0)}/{dii.get('total_days', 0)}")
                console.print(f"  Sentiment: [{dii_color}]{dii.get('sentiment', 'N/A').upper()}[/{dii_color}]")

                combined = trend.get('combined_sentiment', 'neutral')
                combined_color = "green" if 'bullish' in combined else "red" if 'bearish' in combined else "yellow"
                console.print(f"\n[bold]Combined Institutional Sentiment:[/bold] [{combined_color}]{combined.upper().replace('_', ' ')}[/{combined_color}]")

        elif data_type.lower() in ['bulk', 'block', 'deals']:
            # Bulk/Block Deals
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Fetching bulk/block deals...", total=None)
                deals = pipeline.fetch_all_bulk_block_deals(days_back=days, ticker=ticker)
                progress.update(task, description="[green]Complete[/green]")

            if output_json:
                console.print_json(json.dumps([d.to_dict() for d in deals], indent=2, default=str))
            else:
                if not deals:
                    console.print("[yellow]No bulk/block deals found.[/yellow]")
                else:
                    console.print(f"\n[bold]Recent Bulk/Block Deals ({len(deals)}):[/bold]\n")

                    table = Table(show_header=True, header_style="bold cyan")
                    table.add_column("Date", width=12)
                    table.add_column("Ticker", style="bold", width=12)
                    table.add_column("Type", width=6)
                    table.add_column("Client", width=25)
                    table.add_column("Trade", width=6)
                    table.add_column("Value (Cr)", justify="right")

                    for deal in deals[:20]:
                        trade_color = "green" if deal.trade_type == 'buy' else "red" if deal.trade_type == 'sell' else "white"
                        table.add_row(
                            deal.date[:10] if deal.date else 'N/A',
                            deal.ticker,
                            deal.deal_type.upper()[:5],
                            (deal.client_name[:23] + '..') if len(deal.client_name) > 25 else deal.client_name,
                            f"[{trade_color}]{deal.trade_type.upper()[:4]}[/{trade_color}]",
                            f"{deal.value:.1f}"
                        )

                    console.print(table)

        elif data_type.lower() in ['insider', 'sast']:
            # Insider Trading
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Fetching insider trading data...", total=None)

                if ticker:
                    sentiment = pipeline.get_insider_sentiment(ticker, days_back=days)
                    progress.update(task, description="[green]Complete[/green]")

                    if output_json:
                        console.print_json(json.dumps(sentiment, indent=2, default=str))
                    else:
                        console.print(f"\n[bold]Insider Trading Analysis: {sentiment['ticker']}[/bold]\n")
                        console.print(f"  Total Filings: {sentiment.get('total_filings', 0)}")

                        buys = sentiment.get('buys', {})
                        sells = sentiment.get('sells', {})
                        console.print(f"\n  [bold]Buys:[/bold]")
                        console.print(f"    Count: {buys.get('count', 0)}")
                        console.print(f"    Value: ₹{buys.get('value_lakhs', 0):.2f} Lakhs")
                        console.print(f"    Promoter Buys: {buys.get('promoter_buys', 0)}")

                        console.print(f"\n  [bold]Sells:[/bold]")
                        console.print(f"    Count: {sells.get('count', 0)}")
                        console.print(f"    Value: ₹{sells.get('value_lakhs', 0):.2f} Lakhs")
                        console.print(f"    Promoter Sells: {sells.get('promoter_sells', 0)}")

                        sent = sentiment.get('sentiment', 'neutral')
                        sent_color = "green" if 'bullish' in sent else "red" if 'bearish' in sent else "yellow"
                        console.print(f"\n  [bold]Sentiment:[/bold] [{sent_color}]{sent.upper().replace('_', ' ')}[/{sent_color}]")
                else:
                    filings = pipeline.fetch_nse_insider_trading(days_back=days)
                    progress.update(task, description="[green]Complete[/green]")

                    if output_json:
                        console.print_json(json.dumps([f.to_dict() for f in filings], indent=2, default=str))
                    else:
                        if not filings:
                            console.print("[yellow]No insider trading filings found.[/yellow]")
                        else:
                            console.print(f"\n[bold]Recent Insider Trading ({len(filings)}):[/bold]\n")

                            table = Table(show_header=True, header_style="bold cyan")
                            table.add_column("Date", width=12)
                            table.add_column("Ticker", style="bold", width=12)
                            table.add_column("Acquirer", width=25)
                            table.add_column("Type", width=10)
                            table.add_column("Trade", width=6)
                            table.add_column("% After", justify="right")

                            for filing in filings[:20]:
                                trade_color = "green" if filing.is_bullish else "red"
                                table.add_row(
                                    filing.date[:10] if filing.date else 'N/A',
                                    filing.ticker,
                                    (filing.acquirer_name[:23] + '..') if len(filing.acquirer_name) > 25 else filing.acquirer_name,
                                    filing.acquirer_type[:10] if filing.acquirer_type else 'N/A',
                                    f"[{trade_color}]{filing.trade_type.upper()[:4]}[/{trade_color}]",
                                    f"{filing.percent_after:.2f}%"
                                )

                            console.print(table)

        elif data_type.lower() == 'all':
            # All data summary
            console.print("\n[bold]Fetching all institutional data...[/bold]")

            trend = pipeline.get_fii_dii_trend(days=days)
            deals = pipeline.fetch_all_bulk_block_deals(days_back=days, ticker=ticker)

            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  FII/DII Sentiment: {trend.get('combined_sentiment', 'N/A').upper()}")
            console.print(f"  Total Bulk/Block Deals: {len(deals)}")

            stats = pipeline.get_stats()
            console.print(f"\n[bold]Database Stats:[/bold]")
            console.print(f"  FII/DII Records: {stats.get('fii_dii_records', 0)}")
            console.print(f"  Bulk/Block Deals: {stats.get('bulk_block_deals', 0)}")
            console.print(f"  SAST Filings: {stats.get('sast_filings', 0)}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def signals(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., RELIANCE.NS)"),
    horizon: str = typer.Option("1w", "--horizon", "-h", help="Time horizon (intraday, 1d, 1w, 1m, 3m, 1y)"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    Generate combined signals and recommendation for a stock.

    Combines multiple signal sources:
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Fundamental analysis (P/E, ROE, debt)
    - Institutional flow (FII/DII activity)
    - Bulk/Block deals
    - Insider trading (SAST)
    - News sentiment

    Example:
        finagent signals RELIANCE.NS --horizon 1w
    """
    from .signals.signal_combiner import SignalCombiner, TimeHorizon as SignalTimeHorizon
    from .pipelines.institutional_data import InstitutionalDataPipeline
    from .analysis.technical import TechnicalAnalyzer
    from .pipelines.sentiment_data import SentimentPipeline

    horizon_map = {
        'intraday': SignalTimeHorizon.INTRADAY,
        '1d': SignalTimeHorizon.SHORT,
        '1w': SignalTimeHorizon.WEEKLY,
        '1m': SignalTimeHorizon.MONTHLY,
        '3m': SignalTimeHorizon.QUARTERLY,
        '1y': SignalTimeHorizon.LONG,
    }

    time_horizon = horizon_map.get(horizon.lower(), SignalTimeHorizon.WEEKLY)

    console.print(Panel(
        f"[bold cyan]Combined Signal Analysis: {ticker}[/bold cyan]\n\n"
        f"Time Horizon: {horizon}",
        title="FinAgent"
    ))

    # Initialize components
    combiner = SignalCombiner()
    institutional = InstitutionalDataPipeline()
    technical_analyzer = TechnicalAnalyzer()
    perceiver = PerceiverAgent()

    try:
        signals = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # 1. Technical signals
            task = progress.add_task("Fetching price data...", total=None)
            perceived_data = perceiver.perceive(ticker)
            progress.update(task, description="[green]Price data fetched[/green]")

            current_price = None
            company_name = perceived_data.get("overview", {}).get("company_name", ticker)

            if perceived_data.get("technical_data", {}).get("ohlcv"):
                import pandas as pd
                ohlcv = perceived_data["technical_data"]["ohlcv"]
                df = pd.DataFrame.from_dict(ohlcv, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                task = progress.add_task("Calculating technical indicators...", total=None)
                technical_data = technical_analyzer.analyze(df)
                progress.update(task, description="[green]Technical analysis complete[/green]")

                current_price = float(df['close'].iloc[-1]) if 'close' in df.columns else None

                # Generate technical signal
                indicators = {
                    'rsi': technical_data.get('rsi_14'),
                    'macd_histogram': technical_data.get('macd', {}).get('histogram'),
                    'bb_position': technical_data.get('bollinger', {}).get('position'),
                    'sma_trend': technical_data.get('signals', {}).get('trend'),
                    'volume_trend': technical_data.get('signals', {}).get('volume_trend')
                }
                tech_signal = combiner.generate_technical_signal(ticker, indicators)
                if tech_signal:
                    signals.append(tech_signal)

            # 2. Institutional signals
            task = progress.add_task("Fetching institutional data...", total=None)
            fii_dii_trend = institutional.get_fii_dii_trend(days=5)
            inst_signal = combiner.generate_institutional_signal(ticker, fii_dii_trend)
            if inst_signal:
                signals.append(inst_signal)
            progress.update(task, description="[green]Institutional data fetched[/green]")

            # 3. Bulk/Block deal signals
            task = progress.add_task("Checking bulk/block deals...", total=None)
            ticker_clean = ticker.replace('.NS', '').replace('.BO', '').upper()
            deal_data = institutional.get_stock_deal_activity(ticker_clean, days_back=30)
            deal_signal = combiner.generate_bulk_block_signal(ticker, deal_data)
            if deal_signal:
                signals.append(deal_signal)
            progress.update(task, description="[green]Deals checked[/green]")

            # 4. Insider trading signals
            task = progress.add_task("Checking insider activity...", total=None)
            insider_data = institutional.get_insider_sentiment(ticker_clean, days_back=90)
            insider_signal = combiner.generate_insider_signal(ticker, insider_data)
            if insider_signal:
                signals.append(insider_signal)
            progress.update(task, description="[green]Insider activity checked[/green]")

            # 5. News sentiment
            task = progress.add_task("Fetching news sentiment...", total=None)
            try:
                sentiment_pipeline = SentimentPipeline()
                sentiment_summary = sentiment_pipeline.get_sentiment_summary(ticker_clean)
                news_data = {
                    'sentiment_score': sentiment_summary.get('overall_sentiment', 0),
                    'article_count': sentiment_summary.get('article_count', 0)
                }
                news_signal = combiner.generate_news_signal(ticker, news_data)
                if news_signal:
                    signals.append(news_signal)
            except Exception:
                pass  # News data optional
            progress.update(task, description="[green]News sentiment fetched[/green]")

            # Combine signals
            task = progress.add_task("Combining signals...", total=None)
            recommendation = combiner.combine_signals(
                ticker=ticker,
                signals=signals,
                time_horizon=time_horizon,
                current_price=current_price,
                company_name=company_name
            )
            progress.update(task, description="[green]Signals combined[/green]")

        # Display results
        if output_json:
            console.print_json(json.dumps(recommendation.to_dict(), indent=2, default=str))
        else:
            # Action with color
            action_colors = {
                'STRONG_BUY': 'bold green',
                'BUY': 'green',
                'HOLD': 'yellow',
                'SELL': 'red',
                'STRONG_SELL': 'bold red'
            }
            action_color = action_colors.get(recommendation.action, 'white')

            console.print(f"\n[bold]Stock:[/bold] {recommendation.ticker} ({recommendation.company_name})")
            if current_price:
                console.print(f"[bold]Current Price:[/bold] ₹{current_price:,.2f}")
            console.print(f"\n[bold]RECOMMENDATION:[/bold] [{action_color}]{recommendation.action}[/{action_color}]")
            console.print(f"[bold]Score:[/bold] {recommendation.score:.1f}/100")
            console.print(f"[bold]Confidence:[/bold] {recommendation.confidence*100:.0f}%")

            if recommendation.target_price and recommendation.stop_loss:
                console.print(f"\n[bold]Target Price:[/bold] ₹{recommendation.target_price:,.2f}")
                console.print(f"[bold]Stop Loss:[/bold] ₹{recommendation.stop_loss:,.2f}")
                if recommendation.expected_return:
                    console.print(f"[bold]Expected Return:[/bold] {recommendation.expected_return:.1f}%")

            console.print(f"\n[bold]Key Factors:[/bold]")
            for factor in recommendation.key_factors:
                console.print(f"  • {factor}")

            if recommendation.risks:
                console.print(f"\n[bold]Risks:[/bold]")
                for risk in recommendation.risks:
                    console.print(f"  • {risk}")

            console.print(f"\n[bold]Individual Signals ({len(signals)}):[/bold]")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Source", width=15)
            table.add_column("Signal", width=12)
            table.add_column("Confidence", justify="right")
            table.add_column("Detail", width=40)

            for sig in signals:
                signal_color = "green" if sig.strength.value > 0 else "red" if sig.strength.value < 0 else "yellow"
                table.add_row(
                    sig.signal_type.value.title(),
                    f"[{signal_color}]{sig.strength.name}[/{signal_color}]",
                    f"{sig.confidence*100:.0f}%",
                    (sig.source[:38] + '..') if len(sig.source) > 40 else sig.source
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def backtest(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., RELIANCE.NS)"),
    strategy: str = typer.Option("rsi", "--strategy", "-s", help="Strategy: rsi, macd, bollinger, sma, combined"),
    days: int = typer.Option(365, "--days", "-d", help="Days of historical data"),
    capital: float = typer.Option(100000, "--capital", "-c", help="Initial capital (INR)"),
    stop_loss: float = typer.Option(0.05, "--stop-loss", help="Stop loss percentage (0.05 = 5%)"),
    take_profit: float = typer.Option(0.10, "--take-profit", help="Take profit percentage"),
    optimize: bool = typer.Option(False, "--optimize", "-o", help="Optimize strategy parameters"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    Backtest trading strategies using VectorBT.

    Available strategies:
    - rsi: RSI mean reversion (buy oversold, sell overbought)
    - macd: MACD crossover (trend following)
    - bollinger: Bollinger Bands mean reversion
    - sma: SMA crossover (trend following)
    - combined: RSI + MACD + SMA filter

    Example:
        finagent backtest RELIANCE.NS --strategy rsi --days 365
        finagent backtest TCS.NS --strategy combined --optimize
    """
    from .strategies.vectorbt_framework import VectorBTFramework
    import pandas as pd

    console.print(Panel(
        f"[bold cyan]Backtest: {ticker}[/bold cyan]\n\n"
        f"Strategy: {strategy.upper()}\n"
        f"Period: {days} days\n"
        f"Capital: ₹{capital:,.0f}",
        title="FinAgent Backtest"
    ))

    # Fetch historical data
    perceiver = PerceiverAgent()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching historical data...", total=None)
            perceived_data = perceiver.perceive(ticker)

            if not perceived_data.get("technical_data", {}).get("ohlcv"):
                console.print("[red]Error: No historical data available for backtesting.[/red]")
                raise typer.Exit(1)

            ohlcv = perceived_data["technical_data"]["ohlcv"]
            df = pd.DataFrame.from_dict(ohlcv, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Ensure we have enough data
            if len(df) < 50:
                console.print(f"[red]Error: Insufficient data ({len(df)} days). Need at least 50 days.[/red]")
                raise typer.Exit(1)

            progress.update(task, description=f"[green]Loaded {len(df)} days of data[/green]")

            # Initialize framework
            framework = VectorBTFramework(
                initial_capital=capital,
                commission_pct=0.001,  # 0.1% typical for Indian brokers
                slippage_pct=0.001
            )

            # Run backtest
            task = progress.add_task(f"Running {strategy.upper()} backtest...", total=None)

            if optimize:
                progress.update(task, description=f"Optimizing {strategy.upper()} parameters...")

                if strategy.lower() == 'rsi':
                    opt_result = framework.optimize_rsi_strategy(df, ticker=ticker)
                elif strategy.lower() == 'sma':
                    opt_result = framework.optimize_sma_crossover(df, ticker=ticker)
                else:
                    console.print(f"[yellow]Optimization not available for {strategy}. Running default.[/yellow]")
                    opt_result = None

                if opt_result and opt_result.get('best_result'):
                    result = opt_result['best_result']
                    best_params = opt_result.get('best_params', {})
                    progress.update(task, description="[green]Optimization complete[/green]")

                    console.print(f"\n[bold]Optimized Parameters:[/bold]")
                    for k, v in best_params.items():
                        console.print(f"  {k}: {v}")
                else:
                    console.print("[yellow]Optimization found no improvement. Using default parameters.[/yellow]")
                    result = None
            else:
                opt_result = None
                result = None

            # Run standard backtest if no optimization result
            if result is None:
                if strategy.lower() == 'rsi':
                    result = framework.backtest_rsi_strategy(
                        df, ticker=ticker,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                elif strategy.lower() == 'macd':
                    result = framework.backtest_macd_strategy(
                        df, ticker=ticker,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                elif strategy.lower() == 'bollinger':
                    result = framework.backtest_bollinger_strategy(
                        df, ticker=ticker,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                elif strategy.lower() == 'sma':
                    result = framework.backtest_sma_crossover(
                        df, ticker=ticker,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                elif strategy.lower() == 'combined':
                    result = framework.backtest_combined_strategy(
                        df, ticker=ticker,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                else:
                    console.print(f"[red]Unknown strategy: {strategy}[/red]")
                    raise typer.Exit(1)

            progress.update(task, description="[green]Backtest complete[/green]")

        # Display results
        if output_json:
            console.print_json(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            return_color = "green" if result.total_return > 0 else "red"
            sharpe_color = "green" if result.sharpe_ratio > 1 else "yellow" if result.sharpe_ratio > 0 else "red"
            dd_color = "green" if result.max_drawdown > -10 else "yellow" if result.max_drawdown > -20 else "red"

            console.print(f"\n[bold]Strategy:[/bold] {result.strategy_name}")
            console.print(f"[bold]Period:[/bold] {result.start_date} to {result.end_date}")

            console.print(f"\n[bold]Performance:[/bold]")
            console.print(f"  Initial Capital: ₹{result.initial_capital:,.0f}")
            console.print(f"  Final Value: ₹{result.final_value:,.0f}")
            console.print(f"  Total Return: [{return_color}]{result.total_return:.2f}%[/{return_color}]")
            if result.cagr:
                console.print(f"  CAGR: [{return_color}]{result.cagr:.2f}%[/{return_color}]")

            console.print(f"\n[bold]Risk Metrics:[/bold]")
            console.print(f"  Sharpe Ratio: [{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]")
            if result.sortino_ratio:
                console.print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
            console.print(f"  Max Drawdown: [{dd_color}]{result.max_drawdown:.2f}%[/{dd_color}]")

            console.print(f"\n[bold]Trade Statistics:[/bold]")
            console.print(f"  Total Trades: {result.total_trades}")
            console.print(f"  Win Rate: {result.win_rate:.1f}%")
            console.print(f"  Profit Factor: {result.profit_factor:.2f}")
            if result.avg_holding_days:
                console.print(f"  Avg Holding: {result.avg_holding_days:.1f} days")

            # Recent trades
            if result.trades:
                console.print(f"\n[bold]Recent Trades:[/bold]")
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Entry", width=12)
                table.add_column("Exit", width=12)
                table.add_column("Entry Price", justify="right")
                table.add_column("Exit Price", justify="right")
                table.add_column("Return", justify="right")

                for trade in result.trades[-10:]:
                    ret_color = "green" if trade.get('return_pct', 0) > 0 else "red"
                    table.add_row(
                        str(trade.get('entry_date', 'N/A'))[:10],
                        str(trade.get('exit_date', 'N/A'))[:10],
                        f"₹{trade.get('entry_price', 0):,.2f}",
                        f"₹{trade.get('exit_price', 0):,.2f}",
                        f"[{ret_color}]{trade.get('return_pct', 0):.2f}%[/{ret_color}]"
                    )

                console.print(table)

            # Compare with buy-and-hold
            if len(df) > 0:
                close_col = 'close' if 'close' in df.columns else 'Close'
                buy_hold_return = (float(df[close_col].iloc[-1]) / float(df[close_col].iloc[0]) - 1) * 100
                console.print(f"\n[bold]Benchmark:[/bold]")
                bh_color = "green" if buy_hold_return > 0 else "red"
                console.print(f"  Buy & Hold Return: [{bh_color}]{buy_hold_return:.2f}%[/{bh_color}]")

                alpha = result.total_return - buy_hold_return
                alpha_color = "green" if alpha > 0 else "red"
                console.print(f"  Strategy Alpha: [{alpha_color}]{alpha:.2f}%[/{alpha_color}]")

    except ImportError as e:
        console.print(f"[red]Error: VectorBT not installed. Install with: pip install vectorbt[/red]")
        console.print(f"[dim]Details: {e}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def forecast(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., RELIANCE.NS, INFY.NS)"),
    days: int = typer.Option(180, "--days", "-d", help="Days of historical data for analysis"),
    stop_loss: float = typer.Option(0.05, "--stop-loss", "-sl", help="Stop loss percentage (default: 5%)"),
    target1: float = typer.Option(0.08, "--target1", "-t1", help="First target percentage (default: 8%)"),
    target2: float = typer.Option(0.15, "--target2", "-t2", help="Second target percentage (default: 15%)"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    Generate forward-looking trading signal with ENTER/HOLD/EXIT recommendation.

    Analyzes current market conditions using RSI, MACD, and SMA indicators
    to provide actionable trading signals with entry points, stop loss, and targets.

    Example:
        finagent forecast INFY.NS
        finagent forecast RELIANCE.NS --stop-loss 0.03 --target1 0.10
    """
    import yfinance as yf
    from rich.panel import Panel
    from rich.table import Table

    console.print(Panel(
        f"[bold cyan]Trading Forecast: {ticker}[/bold cyan]\n\n"
        f"Analysis Period: {days} days\n"
        f"Stop Loss: {stop_loss*100:.1f}%\n"
        f"Target 1: {target1*100:.1f}% | Target 2: {target2*100:.1f}%",
        title="FinAgent Forecast"
    ))

    try:
        from .strategies.vectorbt_framework import VectorBTFramework, SignalAction

        # Fetch data
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Fetching price data...", total=None)

            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{days}d")

            if len(df) < 50:
                console.print(f"[red]Error: Insufficient data for {ticker}. Got {len(df)} days, need at least 50.[/red]")
                raise typer.Exit(1)

            progress.update(task, description=f"Loaded {len(df)} days of data")

            # Generate forecast
            progress.update(task, description="Analyzing indicators...")
            framework = VectorBTFramework()
            signal = framework.generate_forecast(
                df,
                ticker=ticker,
                stop_loss_pct=stop_loss,
                target_pct_1=target1,
                target_pct_2=target2
            )
            progress.update(task, description="[green]Analysis complete[/green]")

        if output_json:
            import json
            console.print(json.dumps(signal.to_dict(), indent=2))
            return

        # Display signal
        action_colors = {
            SignalAction.STRONG_BUY: "bold green",
            SignalAction.BUY: "green",
            SignalAction.HOLD: "yellow",
            SignalAction.SELL: "red",
            SignalAction.STRONG_SELL: "bold red"
        }

        action_emoji = {
            SignalAction.STRONG_BUY: "BUY",
            SignalAction.BUY: "BUY",
            SignalAction.HOLD: "HOLD",
            SignalAction.SELL: "SELL",
            SignalAction.STRONG_SELL: "SELL"
        }

        color = action_colors.get(signal.action, "white")

        console.print(f"\n[bold]Signal: [{color}]{signal.action.value}[/{color}][/bold]")
        console.print(f"Confidence: {signal.confidence:.0f}%")
        console.print(f"Current Price: ₹{signal.current_price:,.2f}")
        console.print(f"As of: {signal.timestamp}")

        # Price Targets Table
        if signal.action in [SignalAction.STRONG_BUY, SignalAction.BUY]:
            console.print(f"\n[bold green]Entry/Exit Levels:[/bold green]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("Level", style="cyan")
            table.add_column("Price", justify="right")
            table.add_column("% Move", justify="right")

            if signal.entry_price:
                entry_pct = ((signal.entry_price / signal.current_price) - 1) * 100
                table.add_row("Entry", f"₹{signal.entry_price:,.2f}", f"{entry_pct:+.1f}%")
            if signal.stop_loss:
                sl_pct = ((signal.stop_loss / signal.current_price) - 1) * 100
                table.add_row("Stop Loss", f"₹{signal.stop_loss:,.2f}", f"[red]{sl_pct:+.1f}%[/red]")
            if signal.target_1:
                t1_pct = ((signal.target_1 / signal.current_price) - 1) * 100
                table.add_row("Target 1", f"₹{signal.target_1:,.2f}", f"[green]+{t1_pct:.1f}%[/green]")
            if signal.target_2:
                t2_pct = ((signal.target_2 / signal.current_price) - 1) * 100
                table.add_row("Target 2", f"₹{signal.target_2:,.2f}", f"[green]+{t2_pct:.1f}%[/green]")

            console.print(table)

        elif signal.action in [SignalAction.STRONG_SELL, SignalAction.SELL]:
            console.print(f"\n[bold red]Downside Targets:[/bold red]")
            if signal.target_1:
                t1_pct = ((signal.target_1 / signal.current_price) - 1) * 100
                console.print(f"  Support 1: ₹{signal.target_1:,.2f} ({t1_pct:+.1f}%)")
            if signal.target_2:
                t2_pct = ((signal.target_2 / signal.current_price) - 1) * 100
                console.print(f"  Support 2: ₹{signal.target_2:,.2f} ({t2_pct:+.1f}%)")

        # Indicators Table
        console.print(f"\n[bold]Technical Indicators:[/bold]")
        ind_table = Table(show_header=True, header_style="bold")
        ind_table.add_column("Indicator")
        ind_table.add_column("Value", justify="right")
        ind_table.add_column("Signal")

        # RSI
        rsi_color = "green" if signal.rsi < 30 else ("red" if signal.rsi > 70 else "yellow")
        rsi_signal = "Oversold" if signal.rsi < 30 else ("Overbought" if signal.rsi > 70 else "Neutral")
        ind_table.add_row("RSI (14)", f"{signal.rsi:.1f}", f"[{rsi_color}]{rsi_signal}[/{rsi_color}]")

        # MACD
        macd_color = "green" if "BULLISH" in signal.macd_signal else ("red" if "BEARISH" in signal.macd_signal else "yellow")
        ind_table.add_row("MACD", f"{signal.macd_histogram:.4f}", f"[{macd_color}]{signal.macd_signal}[/{macd_color}]")

        # Trend
        trend_color = "green" if signal.sma_trend == "UPTREND" else ("red" if signal.sma_trend == "DOWNTREND" else "yellow")
        ind_table.add_row("Trend (SMA50)", f"{signal.price_vs_sma:+.1f}%", f"[{trend_color}]{signal.sma_trend}[/{trend_color}]")

        console.print(ind_table)

        # Reasons and Risks
        if signal.reasons:
            console.print(f"\n[bold green]Bullish Factors:[/bold green]")
            for reason in signal.reasons:
                console.print(f"  [green]+[/green] {reason}")

        if signal.risks:
            console.print(f"\n[bold red]Risk Factors:[/bold red]")
            for risk in signal.risks:
                console.print(f"  [red]-[/red] {risk}")

        # Disclaimer
        console.print(f"\n[dim]Disclaimer: This is algorithmic analysis, not financial advice. Always do your own research.[/dim]")

    except ImportError as e:
        console.print(f"[red]Error: VectorBT not installed. Install with: pip install vectorbt[/red]")
        console.print(f"[dim]Details: {e}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def pead(
    ticker: Optional[str] = typer.Argument(None, help="Stock ticker (optional, shows all if not specified)"),
    days: int = typer.Option(60, "--days", "-d", help="Days to look back for earnings (7-90)"),
    min_score: float = typer.Option(30.0, "--min-score", help="Minimum PEAD score (0-100)"),
    count: int = typer.Option(10, "--count", "-n", help="Number of results to show"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    Generate PEAD (Post-Earnings Announcement Drift) signals.

    PEAD is a market anomaly where stocks continue to drift in the direction
    of their earnings surprise for 30-60 days after the announcement.

    The strategy:
    - Identifies stocks with recent positive/negative earnings surprises
    - Calculates expected drift based on surprise magnitude
    - Scores opportunities by timing and surprise strength

    Examples:
        finagent pead                    # Show all PEAD opportunities (60 days)
        finagent pead --days 7           # Only last 7 days of earnings
        finagent pead RELIANCE           # Check specific stock
        finagent pead --min-score 50     # Only strong signals
    """
    from rich.panel import Panel
    from rich.table import Table
    from .signals.pead_strategy import PEADStrategy, EarningsSurprise

    console.print(Panel(
        f"[bold cyan]PEAD Strategy Scanner[/bold cyan]\n\n"
        f"Post-Earnings Announcement Drift Analysis\n"
        f"Lookback: {days} days | Min Score: {min_score}",
        title="FinAgent PEAD"
    ))

    try:
        strategy = PEADStrategy()

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Scanning earnings announcements...", total=None)

            signals = strategy.generate_signals(
                days_back=days,
                min_score=min_score,
                ticker=ticker
            )
            progress.update(task, description=f"[green]Found {len(signals)} opportunities[/green]")

        if output_json:
            import json
            console.print(json.dumps([s.to_dict() for s in signals[:count]], indent=2))
            return

        if not signals:
            console.print("\n[yellow]No PEAD opportunities found.[/yellow]")
            console.print("[dim]Try lowering --min-score or run 'finagent ingest' to fetch more data.[/dim]")
            return

        # Summary
        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total Signals: {len(signals)}")
        console.print(f"  [green]Buy Signals: {len(buy_signals)}[/green]")
        console.print(f"  [red]Sell/Avoid Signals: {len(sell_signals)}[/red]")

        # Display signals table
        table = Table(title=f"\nTop {min(count, len(signals))} PEAD Opportunities", show_header=True, header_style="bold")
        table.add_column("Ticker", style="cyan", width=12)
        table.add_column("Action", justify="center", width=8)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Surprise", justify="right", width=10)
        table.add_column("Days Ago", justify="right", width=10)
        table.add_column("Window Left", justify="right", width=12)
        table.add_column("Expected Drift", justify="right", width=14)

        for signal in signals[:count]:
            event = signal.earnings_event

            # Action color
            action_color = "green" if signal.action == "BUY" else "red"

            # Surprise formatting
            surprise_str = f"{event.surprise_magnitude:+.1f}%"
            surprise_color = "green" if event.surprise_magnitude > 0 else "red"

            # Drift formatting
            drift_str = f"{signal.expected_drift:+.1f}%"
            drift_color = "green" if signal.expected_drift > 0 else "red"

            table.add_row(
                event.ticker,
                f"[{action_color}]{signal.action}[/{action_color}]",
                f"{event.pead_score:.0f}",
                f"[{surprise_color}]{surprise_str}[/{surprise_color}]",
                str(event.days_since_announcement),
                f"{event.drift_window_remaining}d",
                f"[{drift_color}]{drift_str}[/{drift_color}]"
            )

        console.print(table)

        # Show detailed view for top signal
        if signals:
            top = signals[0]
            console.print(f"\n[bold]Top Opportunity: {top.ticker}[/bold]")
            console.print(f"  Company: {top.company_name}")
            console.print(f"  Quarter: {top.earnings_event.quarter}")
            console.print(f"  Announced: {top.earnings_event.announcement_date.strftime('%Y-%m-%d')}")

            if top.action == "BUY" and top.entry_price:
                console.print(f"\n  [green]Suggested Entry:[/green] ₹{top.entry_price:,.2f}")
                if top.stop_loss:
                    console.print(f"  [red]Stop Loss:[/red] ₹{top.stop_loss:,.2f}")
                if top.target_price:
                    console.print(f"  [green]Target:[/green] ₹{top.target_price:,.2f}")
                console.print(f"  [dim]Hold Period: ~{top.holding_period_days} days[/dim]")

            console.print(f"\n  [bold green]Reasons:[/bold green]")
            for reason in top.reasons:
                console.print(f"    + {reason}")

            console.print(f"\n  [bold red]Risks:[/bold red]")
            for risk in top.risks:
                console.print(f"    - {risk}")

        # Disclaimer
        console.print(f"\n[dim]Note: PEAD is a statistical tendency, not a guarantee. Always do your own research.[/dim]")

    except FileNotFoundError as e:
        console.print(f"\n[yellow]No earnings data available.[/yellow]")
        console.print(f"[dim]{e}[/dim]")
        console.print("\nTo fetch earnings data, run:")
        console.print("  [cyan]finagent ingest --days 60[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def scan(
    universe: str = typer.Option("nifty50", "--universe", "-u", help="Stock universe (nifty50, nifty100, nifty200, fno, midcap, all, comprehensive/nsebse, custom)"),
    hold_days: int = typer.Option(7, "--hold", "-h", help="Holding period in days"),
    min_confidence: float = typer.Option(40.0, "--min-confidence", "-c", help="Minimum signal confidence (0-100)"),
    count: int = typer.Option(20, "--count", "-n", help="Number of results to show"),
    direction: str = typer.Option("all", "--direction", "-d", help="Filter by direction (buy, sell, all)"),
    tickers_file: Optional[str] = typer.Option(None, "--file", "-f", help="File with tickers (one per line)"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Scan single ticker"),
    fundamentals: bool = typer.Option(True, "--fundamentals/--no-fundamentals", help="Include NSE circulars/news data"),
    news_days: int = typer.Option(30, "--news-days", help="Days to look back for news/circulars"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output including news fetching")
):
    """
    Scan Indian stocks using advanced technical indicators + fundamental data.

    Technical Indicators:
    - MACD (trend momentum)
    - Williams %R Trend Exhaustion (tops/bottoms)
    - Williams VixFix (volatility bottoms)
    - Hull Moving Average (trend identification)
    - Laguerre RSI (better moving averages)
    - Supertrend (trailing stop levels)
    - RSI with Divergence (reversal detection)
    - Ichimoku Cloud (comprehensive analysis)

    Fundamental Data (from NSE circulars & news):
    - Corporate announcements and results
    - Contract wins, expansion news
    - Earnings surprises (PEAD signals)
    - Dividends, buybacks, partnerships
    - Rating changes and analyst views

    Stock Universes:
    - nifty50: Top 50 stocks by market cap
    - nifty100: Top 100 stocks
    - nifty200: Top 200 stocks
    - fno: All F&O stocks (~200)
    - midcap: Mid-cap and Small-cap NSE stocks (~400)
    - all: All major NSE stocks (~500)
    - comprehensive/nsebse: All NSE + BSE stocks (~1500+)
    - custom: Use --file to provide your own list

    Examples:
        finagent scan                           # Scan Nifty 50 with fundamentals
        finagent scan --universe all            # Scan all ~500 stocks
        finagent scan --no-fundamentals         # Technical only (faster)
        finagent scan --ticker RELIANCE.NS      # Scan single stock
        finagent scan --news-days 7             # Only recent news (7 days)
    """
    from rich.panel import Panel
    from rich.table import Table
    from .signals.advanced_signals import (
        AdvancedSignalGenerator, SignalDirection,
        NIFTY_50, NIFTY_NEXT_50, ALL_INDIAN_STOCKS,
        NIFTY_200, FNO_STOCKS, BROAD_MARKET,
        NSE_MIDCAP_SMALLCAP, ALL_NSE_BSE
    )

    # Select universe
    if ticker:
        # Single ticker scan
        tickers = [ticker if ".NS" in ticker or ".BO" in ticker else f"{ticker}.NS"]
        universe_name = f"Single Stock: {ticker}"
    elif tickers_file:
        # Load from file
        try:
            with open(tickers_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
                # Add .NS suffix if missing
                tickers = [t if ".NS" in t or ".BO" in t else f"{t}.NS" for t in tickers]
            universe_name = f"Custom ({len(tickers)} from file)"
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {tickers_file}[/red]")
            raise typer.Exit(1)
    elif universe.lower() == "nifty50":
        tickers = NIFTY_50
        universe_name = "Nifty 50"
    elif universe.lower() == "nifty100":
        tickers = NIFTY_50 + NIFTY_NEXT_50
        universe_name = "Nifty 100"
    elif universe.lower() == "nifty200":
        tickers = NIFTY_200
        universe_name = "Nifty 200"
    elif universe.lower() == "fno":
        tickers = FNO_STOCKS
        universe_name = "F&O Stocks"
    elif universe.lower() == "midcap":
        tickers = NSE_MIDCAP_SMALLCAP
        universe_name = "NSE Mid/Small Cap"
    elif universe.lower() == "all":
        tickers = BROAD_MARKET
        universe_name = "All NSE Stocks"
    elif universe.lower() == "comprehensive" or universe.lower() == "nsebse":
        tickers = ALL_NSE_BSE
        universe_name = "All NSE + BSE Stocks"
    else:
        tickers = NIFTY_50
        universe_name = "Nifty 50"

    # Setup logging if verbose
    if verbose:
        setup_logging(verbose=True)
        logging.getLogger('finagent').setLevel(logging.DEBUG)
        console.print("[dim]Verbose mode enabled - showing debug output[/dim]\n")

    fund_status = f"[green]Enabled[/green] ({news_days}-day lookback)" if fundamentals else "[dim]Disabled[/dim]"
    console.print(Panel(
        f"[bold cyan]Advanced Technical + Fundamental Scanner[/bold cyan]\n\n"
        f"Universe: {universe_name} ({len(tickers)} stocks)\n"
        f"Holding Period: {hold_days} days\n"
        f"Min Confidence: {min_confidence}%\n"
        f"Direction Filter: {direction.upper()}\n"
        f"Fundamental Data: {fund_status}",
        title="FinAgent Scanner"
    ))

    try:
        generator = AdvancedSignalGenerator(use_fundamentals=fundamentals, days_lookback=news_days)

        # Show news fetcher status in verbose mode
        if verbose and fundamentals and generator.fundamental_enhancer:
            fe = generator.fundamental_enhancer
            console.print(f"[dim]News fetcher: {'Enabled' if fe.news_fetcher else 'Disabled'}[/dim]")
            console.print(f"[dim]Sentiment analyzer: {'Enabled' if fe.sentiment_analyzer else 'Disabled'}[/dim]")
            console.print(f"[dim]Database: {fe.circulars_db or 'Not found (using live news)'}[/dim]\n")

        signals = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Scanning {len(tickers)} stocks...", total=len(tickers))

            def progress_callback(current, total, ticker):
                progress.update(task, completed=current, description=f"Scanning {ticker}...")

            signals = generator.scan_stocks(
                tickers=tickers,
                hold_days=hold_days,
                min_confidence=min_confidence,
                progress_callback=progress_callback
            )

        # Filter by direction
        if direction.lower() == "buy":
            signals = [s for s in signals if s.direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY]]
        elif direction.lower() == "sell":
            signals = [s for s in signals if s.direction in [SignalDirection.STRONG_SELL, SignalDirection.SELL]]

        if output_json:
            import json
            console.print(json.dumps([s.to_dict() for s in signals[:count]], indent=2))
            return

        if not signals:
            console.print("\n[yellow]No signals found matching criteria.[/yellow]")
            console.print("[dim]Try lowering --min-confidence or changing --direction[/dim]")
            return

        # Summary
        buy_signals = [s for s in signals if s.direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY]]
        sell_signals = [s for s in signals if s.direction in [SignalDirection.STRONG_SELL, SignalDirection.SELL]]

        console.print(f"\n[bold]Scan Results:[/bold]")
        console.print(f"  Total Signals: {len(signals)}")
        console.print(f"  [green]Buy Signals: {len(buy_signals)}[/green]")
        console.print(f"  [red]Sell Signals: {len(sell_signals)}[/red]")

        # Main results table
        table = Table(title=f"\nTop {min(count, len(signals))} Opportunities", show_header=True, header_style="bold")
        table.add_column("Ticker", style="cyan", width=14)
        table.add_column("Signal", justify="center", width=12)
        table.add_column("Tech", justify="right", width=5)
        if fundamentals:
            table.add_column("News", justify="center", width=6)
            table.add_column("FII", justify="center", width=5)
            table.add_column("Pol", justify="center", width=4)
            table.add_column("Score", justify="right", width=6)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Target", justify="right", width=10)
        table.add_column("Exp.Ret", justify="right", width=8)
        table.add_column("R:R", justify="right", width=5)

        for signal in signals[:count]:
            dir_color = "green" if signal.direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY] else "red"
            dir_str = signal.direction.value.replace("_", " ")

            row = [
                signal.ticker.replace(".NS", ""),
                f"[{dir_color}]{dir_str}[/{dir_color}]",
                f"{signal.confidence:.0f}%",
            ]

            if fundamentals:
                # News sentiment
                if signal.fundamental and signal.fundamental.has_recent_news:
                    sentiment = signal.fundamental.news_sentiment
                    sent_color = "green" if sentiment == "BULLISH" else ("red" if sentiment == "BEARISH" else "yellow")
                    row.append(f"[{sent_color}]{sentiment[:4]}[/{sent_color}]")
                else:
                    row.append("[dim]--[/dim]")

                # FII sentiment (institutional)
                if signal.fundamental:
                    fii = getattr(signal.fundamental, 'fii_sentiment', 'NEUTRAL')
                    fii_color = "green" if fii == "BULLISH" else ("red" if fii == "BEARISH" else "dim")
                    fii_icon = "↑" if fii == "BULLISH" else ("↓" if fii == "BEARISH" else "-")
                    row.append(f"[{fii_color}]{fii_icon}[/{fii_color}]")
                else:
                    row.append("[dim]-[/dim]")

                # Policy boost indicator
                if signal.fundamental:
                    has_policy = getattr(signal.fundamental, 'has_policy_boost', False)
                    policy_score = getattr(signal.fundamental, 'policy_score', 0)
                    if has_policy:
                        row.append("[green]✓[/green]")
                    elif policy_score < -10:
                        row.append("[red]✗[/red]")
                    else:
                        row.append("[dim]-[/dim]")
                else:
                    row.append("[dim]-[/dim]")

                # Combined score
                row.append(f"{signal.combined_score:.0f}")

            row.extend([
                f"₹{signal.current_price:,.0f}",
                f"₹{signal.target_2:,.0f}",
                f"[{dir_color}]{signal.expected_return_pct:+.1f}%[/{dir_color}]",
                f"{signal.risk_reward_ratio:.1f}",
            ])

            table.add_row(*row)

        console.print(table)

        # Detailed view for top signal
        if signals:
            top = signals[0]
            console.print(f"\n[bold]Top Pick: {top.ticker.replace('.NS', '')}[/bold]")

            dir_color = "green" if top.direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY] else "red"
            console.print(f"  Signal: [{dir_color}]{top.direction.value}[/{dir_color}] ({top.confidence:.0f}% confidence)")
            console.print(f"  Current Price: ₹{top.current_price:,.2f}")

            console.print(f"\n  [bold]Trade Setup:[/bold]")
            console.print(f"    Entry: ₹{top.entry_price:,.2f}")
            console.print(f"    Stop Loss: ₹{top.stop_loss:,.2f} ([red]{((top.stop_loss/top.entry_price)-1)*100:+.1f}%[/red])")
            console.print(f"    Target 1: ₹{top.target_1:,.2f} ([green]{((top.target_1/top.entry_price)-1)*100:+.1f}%[/green])")
            console.print(f"    Target 2: ₹{top.target_2:,.2f} ([green]{((top.target_2/top.entry_price)-1)*100:+.1f}%[/green])")
            console.print(f"    Target 3: ₹{top.target_3:,.2f} ([green]{((top.target_3/top.entry_price)-1)*100:+.1f}%[/green])")
            console.print(f"    Suggested Hold: {top.suggested_hold_days} days")
            console.print(f"    Risk/Reward: {top.risk_reward_ratio:.2f}")

            # Indicator breakdown
            console.print(f"\n  [bold]Indicator Signals:[/bold]")
            for name, ind in top.indicators.items():
                ind_signal = ind.get('signal', 'N/A')
                ind_strength = ind.get('strength', 0)
                color = "green" if ind_signal in ['BUY', 'BULLISH', 'STRONG_BUY', 'OVERSOLD'] else (
                    "red" if ind_signal in ['SELL', 'BEARISH', 'STRONG_SELL', 'OVERBOUGHT'] else "yellow"
                )
                console.print(f"    {name.upper():12} [{color}]{ind_signal:12}[/{color}] (strength: {ind_strength})")

            if top.reasons:
                console.print(f"\n  [bold green]Bullish Factors:[/bold green]")
                for reason in top.reasons[:5]:
                    console.print(f"    + {reason}")

            if top.risks:
                console.print(f"\n  [bold red]Risk Factors:[/bold red]")
                for risk in top.risks[:5]:
                    console.print(f"    - {risk}")

            # Fundamental data section
            if fundamentals and top.fundamental and top.fundamental.has_recent_news:
                console.print(f"\n  [bold magenta]Fundamental Data (from NSE Circulars/News):[/bold magenta]")

                sentiment = top.fundamental.news_sentiment
                sent_color = "green" if sentiment == "BULLISH" else ("red" if sentiment == "BEARISH" else "yellow")
                method = getattr(top.fundamental, 'sentiment_method', 'Keyword')
                method_color = "cyan" if method == "FinBERT" else ("blue" if method == "VADER" else "dim")
                console.print(f"    News Sentiment: [{sent_color}]{sentiment}[/{sent_color}] (score: {top.fundamental.news_score:+.0f}) via [{method_color}]{method}[/{method_color}]")

                if top.fundamental.earnings_surprise:
                    ear_color = "green" if top.fundamental.earnings_surprise == "BEAT" else ("red" if top.fundamental.earnings_surprise == "MISS" else "yellow")
                    console.print(f"    Earnings: [{ear_color}]{top.fundamental.earnings_surprise}[/{ear_color}] (PEAD score: {top.fundamental.pead_score:+.0f})")

                if top.fundamental.catalysts:
                    console.print(f"\n    [bold]Recent Catalysts:[/bold]")
                    for catalyst in top.fundamental.catalysts[:3]:
                        cat_color = "green" if catalyst['sentiment'] == 'BULLISH' else "red"
                        impact = catalyst.get('impact', 'MEDIUM')
                        console.print(f"      [{cat_color}]{catalyst['type'].replace('_', ' ').title()}[/{cat_color}] [{impact}]")
                        console.print(f"        {catalyst['headline'][:70]}...")

                if top.fundamental.recent_headlines:
                    console.print(f"\n    [bold]Recent Headlines:[/bold]")
                    for headline in top.fundamental.recent_headlines[:3]:
                        console.print(f"      • {headline[:80]}...")

                # Institutional Data Section
                console.print(f"\n  [bold blue]Institutional & Macro Data:[/bold blue]")
                console.print(f"    [dim](Note: FII/DII is MARKET-WIDE sentiment, not stock-specific)[/dim]")

                # FII/DII sentiment (market-wide)
                fii_sent = getattr(top.fundamental, 'fii_sentiment', 'NEUTRAL')
                dii_sent = getattr(top.fundamental, 'dii_sentiment', 'NEUTRAL')
                fii_color = "green" if fii_sent == "BULLISH" else ("red" if fii_sent == "BEARISH" else "yellow")
                dii_color = "green" if dii_sent == "BULLISH" else ("red" if dii_sent == "BEARISH" else "yellow")
                inst_score = getattr(top.fundamental, 'institutional_score', 0)

                console.print(f"    Market FII Flow: [{fii_color}]{fii_sent}[/{fii_color}] | "
                            f"Market DII Flow: [{dii_color}]{dii_sent}[/{dii_color}] "
                            f"(weight: {inst_score:+.0f})")

                # Policy impact
                has_policy = getattr(top.fundamental, 'has_policy_boost', False)
                policy_score = getattr(top.fundamental, 'policy_score', 0)
                sectors = getattr(top.fundamental, 'affected_sectors', [])
                policies = getattr(top.fundamental, 'relevant_policies', [])

                if has_policy:
                    console.print(f"    [green]Policy Boost: YES[/green] (score: {policy_score:+.0f})")
                    if sectors:
                        console.print(f"    Affected Sectors: {', '.join(sectors[:3])}")
                    if policies:
                        console.print(f"\n    [bold]Relevant Govt Policies/News:[/bold]")
                        for policy in policies[:3]:
                            pol_sent = policy.get('sentiment', 'NEUTRAL')
                            pol_color = "green" if pol_sent == "POSITIVE" else "red"
                            console.print(f"      [{pol_color}]{policy.get('keyword', 'policy').title()}[/{pol_color}]: {policy.get('headline', '')[:60]}...")
                elif policy_score < -10:
                    console.print(f"    [red]Policy Impact: NEGATIVE[/red] (score: {policy_score:+.0f})")
                else:
                    console.print(f"    Policy Impact: Neutral")

                console.print(f"\n    [dim]Combined Score (70% tech + 30% fundamental): {top.combined_score:.0f}[/dim]")
                console.print(f"    [dim]Breakdown: News({top.fundamental.news_score:+.0f}) + PEAD({top.fundamental.pead_score:+.0f}) + Inst({inst_score*0.2:+.0f}) + Policy({policy_score*0.2:+.0f})[/dim]")

        # Disclaimer
        console.print(f"\n[dim]Disclaimer: Technical analysis is not financial advice. Always do your own research.[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("test-news")
def test_news(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., RELIANCE.NS)"),
    days: int = typer.Option(7, "--days", "-d", help="Days to look back for news"),
):
    """
    Test live news fetching for a single ticker.

    Use this to diagnose why news isn't showing in the scanner.

    Examples:
        finagent test-news RELIANCE.NS
        finagent test-news INFY.NS --days 14
    """
    from rich.panel import Panel

    # Ensure ticker has suffix
    if ".NS" not in ticker and ".BO" not in ticker:
        ticker = f"{ticker}.NS"

    console.print(Panel(
        f"[bold cyan]Testing News Fetching[/bold cyan]\n\n"
        f"Ticker: {ticker}\n"
        f"Days Back: {days}",
        title="FinAgent News Test"
    ))

    try:
        from .analysis.live_news_fetcher import LiveNewsFetcher

        fetcher = LiveNewsFetcher()
        console.print(f"\n[bold]Fetching news for {ticker}...[/bold]")

        # Enable debug logging for this test
        import logging
        logging.basicConfig(level=logging.DEBUG)

        news_items = fetcher.fetch_news(ticker, days_back=days)

        if not news_items:
            console.print(f"\n[yellow]No news found for {ticker}[/yellow]")
            console.print("\nPossible reasons:")
            console.print("  1. No recent news published about this company")
            console.print("  2. Google News RSS may be blocked in your network")
            console.print("  3. Yahoo Finance may not return news for this ticker")
        else:
            console.print(f"\n[green]Found {len(news_items)} news items:[/green]\n")

            for i, item in enumerate(news_items[:10], 1):
                console.print(f"[bold]{i}. [{item.source}][/bold]")
                console.print(f"   {item.title[:80]}")
                if item.published_date:
                    console.print(f"   [dim]Date: {item.published_date.strftime('%Y-%m-%d %H:%M')}[/dim]")
                console.print()

            # Test sentiment analysis
            console.print("\n[bold]Testing Sentiment Analysis:[/bold]")
            try:
                from .analysis.sentiment_analyzer import FinancialSentimentAnalyzer

                analyzer = FinancialSentimentAnalyzer()
                methods = analyzer.get_available_methods()
                console.print(f"  Available methods: {', '.join(methods)}")

                # Analyze first 3 headlines
                console.print("\n  Sample sentiment analysis:")
                for item in news_items[:3]:
                    result = analyzer.analyze(item.title)
                    color = "green" if result.score > 0.1 else ("red" if result.score < -0.1 else "yellow")
                    console.print(f"    [{color}]{result.label.value}[/{color}] ({result.score:+.2f}): {item.title[:50]}...")

            except Exception as e:
                console.print(f"  [red]Sentiment analyzer error: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version")
):
    """
    FinAgent - Autonomous Financial Analysis System for Indian Markets

    A multi-agent system for investment analysis with SEBI compliance.
    """
    if version:
        console.print("FinAgent v1.0.0")
        raise typer.Exit()


if __name__ == "__main__":
    app()
