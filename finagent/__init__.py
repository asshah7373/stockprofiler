"""
FinAgent - Autonomous Financial Multi-Agent System for Indian Markets

A comprehensive financial analysis system that combines:
- Multi-agent architecture (Perceiver, Analyst, Reviewer)
- BSE/NSE data ingestion and analysis
- RAG-based document retrieval
- SEBI-compliant recommendations
- Institutional data tracking (FII/DII, Bulk/Block deals, SAST)
- Signal combination and backtesting framework
"""

__version__ = "1.0.0"
__author__ = "FinAgent Team"

# Expose main modules for convenient imports
from . import agents
from . import pipelines
from . import analysis
from . import signals
from . import strategies
from . import rag
from . import config
