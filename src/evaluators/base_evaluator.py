"""
Base Evaluator Class

Provides the common interface and functionality that all evaluators inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime
import json
import os


class BaseEvaluator(ABC):
    """
    Abstract base class for all content evaluators.
    
    This class defines the common interface and shared functionality that all
    evaluators must implement or can leverage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base evaluator.
        
        Args:
            config: Optional configuration dictionary for the evaluator
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        self.evaluation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the evaluator."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def evaluate(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main evaluation method that must be implemented by all evaluators.
        
        Args:
            content: The content to evaluate
            context: Optional context information for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def batch_evaluate(self, contents: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple pieces of content in batch.
        
        Args:
            contents: List of content pieces to evaluate
            contexts: Optional list of context dictionaries
            
        Returns:
            List of evaluation results
        """
        if contexts is None:
            contexts = [None] * len(contents)
        
        results = []
        for i, content in enumerate(contents):
            try:
                result = self.evaluate(content, contexts[i])
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating content {i}: {str(e)}")
                results.append({"error": str(e), "content_index": i})
        
        return results
    
    def validate_content(self, content: str) -> bool:
        """
        Validate that content meets basic requirements.
        
        Args:
            content: Content to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        if not content:
            return False
        
        if not isinstance(content, str):
            return False
        
        # Check for minimum length
        min_length = self.config.get('min_content_length', 1)
        if len(content.strip()) < min_length:
            return False
        
        # Check for maximum length
        max_length = self.config.get('max_content_length', 10000)
        if len(content) > max_length:
            return False
        
        return True
    
    def validate_context(self, context: Optional[Dict[str, Any]]) -> bool:
        """
        Validate that context meets basic requirements.
        
        Args:
            context: Context to validate
            
        Returns:
            True if context is valid, False otherwise
        """
        if context is None:
            return True
        
        if not isinstance(context, dict):
            return False
        
        # Check for required context fields if specified
        required_fields = self.config.get('required_context_fields', [])
        for field in required_fields:
            if field not in context:
                return False
        
        return True
    
    def _record_evaluation(self, content: str, context: Optional[Dict[str, Any]], 
                          result: Dict[str, Any], execution_time: float) -> None:
        """
        Record evaluation in history for performance tracking.
        
        Args:
            content: The evaluated content
            context: Context used for evaluation
            result: Evaluation result
            execution_time: Time taken for evaluation
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'evaluator': self.__class__.__name__,
            'content_length': len(content),
            'context_keys': list(context.keys()) if context else [],
            'result_keys': list(result.keys()),
            'execution_time': execution_time,
            'success': 'error' not in result
        }
        
        self.evaluation_history.append(record)
        
        # Keep only last 1000 evaluations to prevent memory issues
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this evaluator.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.evaluation_history:
            return {"message": "No evaluations recorded yet"}
        
        execution_times = [r['execution_time'] for r in self.evaluation_history]
        success_rate = sum(1 for r in self.evaluation_history if r['success']) / len(self.evaluation_history)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'success_rate': success_rate,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'evaluations_last_hour': len([
                r for r in self.evaluation_history 
                if (datetime.now() - datetime.fromisoformat(r['timestamp'])).total_seconds() < 3600
            ])
        }
    
    def save_evaluation_history(self, filepath: str) -> None:
        """
        Save evaluation history to file.
        
        Args:
            filepath: Path to save the history file
        """
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        self.logger.info(f"Evaluation history saved to {filepath}")
    
    def load_evaluation_history(self, filepath: str) -> None:
        """
        Load evaluation history from file.
        
        Args:
            filepath: Path to load the history file from
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.evaluation_history = json.load(f)
            
            self.logger.info(f"Evaluation history loaded from {filepath}")
        else:
            self.logger.warning(f"History file not found: {filepath}")
    
    def reset_history(self) -> None:
        """Reset evaluation history."""
        self.evaluation_history = []
        self.logger.info("Evaluation history reset")
    
    def __call__(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make the evaluator callable.
        
        Args:
            content: Content to evaluate
            context: Optional context for evaluation
            
        Returns:
            Evaluation result
        """
        return self.evaluate(content, context) 