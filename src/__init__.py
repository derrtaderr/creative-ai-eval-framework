"""
Creative AI Evaluation Framework

The open-source standard for evaluating creative AI systems.
Comprehensive, production-ready framework for assessing AI-generated content quality, authenticity, and performance.
"""

__version__ = "0.1.0"
__author__ = "Jason Derr"
__email__ = "jason@example.com"
__license__ = "MIT"
__description__ = "The open-source standard for evaluating creative AI systems"

# Core imports for easy access
from .evaluators import (
    ContentContextEvaluator,
    AuthenticityPerformanceEvaluator,
    TemporalEvaluator,
    MultiModalEvaluator,
)

__all__ = [
    "ContentContextEvaluator",
    "AuthenticityPerformanceEvaluator", 
    "TemporalEvaluator",
    "MultiModalEvaluator",
] 