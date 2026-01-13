"""
FinAgent Orchestrator
Main entry point and workflow coordinator.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
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
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector")
):
    """
    Generate top N stock recommendations based on profile.
    """
    console.print(Panel(
        f"[bold cyan]Stock Suggestions[/bold cyan]\n\n"
        f"Generating top {count} recommendations" +
        (f" for {sector} sector" if sector else ""),
        title="FinAgent"
    ))

    # Check for risk profile
    profiler = RiskProfiler()
    profile = profiler.get_latest_profile()

    if not profile:
        console.print("[red]Error: No risk profile found.[/red]")
        console.print("Please run 'finagent profile' first to create a risk profile.")
        raise typer.Exit(1)

    console.print(f"\nUsing profile: {profile.risk_category} (ID: {profile.profile_id[:8]}...)")
    console.print("\n[yellow]Note: Stock suggestions feature requires market screening data.[/yellow]")
    console.print("Consider running 'finagent analyze <ticker>' for specific stocks.\n")

    # Sector tickers for demonstration
    sector_tickers = {
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
        'BANKING': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
        'PHARMA': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
        'AUTO': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS'],
        'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS']
    }

    if sector and sector.upper() in sector_tickers:
        suggestions = sector_tickers[sector.upper()][:count]
    else:
        # Mix of sectors
        suggestions = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'][:count]

    console.print(f"Suggested tickers for analysis:")
    for i, ticker in enumerate(suggestions, 1):
        console.print(f"  {i}. {ticker}")

    console.print(f"\nRun 'finagent analyze <ticker>' to get detailed analysis.")


@app.command()
def update_data(
    days: int = typer.Option(7, "--days", "-d", help="Fetch circulars from last N days"),
    exchange: str = typer.Option("NSE", "--exchange", "-e", help="Exchange (NSE or BSE)")
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
