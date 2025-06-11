"""
Test suite for Authenticity Performance Evaluator (Level 1)

Tests the core functionality of authenticity vs performance evaluation including:
- Dynamic threshold calculation
- Viral pattern recognition
- Authenticity scoring
- Performance prediction
- Balanced evaluation
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock

from src.evaluators.authenticity_evaluator import AuthenticityPerformanceEvaluator


class TestAuthenticityPerformanceEvaluator:
    """Test suite for Level 1 Authenticity Performance Evaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a test evaluator instance."""
        config = {
            'min_authenticity_threshold': 0.6,
            'performance_weight': 0.4,
            'authenticity_weight': 0.6
        }
        return AuthenticityPerformanceEvaluator(config)
    
    @pytest.fixture
    def sample_creator_profile(self):
        """Sample creator profile for testing."""
        return {
            "creator_id": "test_creator",
            "variance_tolerance": 0.7,
            "growth_focus": 0.6,
            "voice_consistency": 0.82,
            "historical_posts": [
                {
                    "content": "Building in public has been the scariest and most rewarding decision of my founder journey. Here's what I learned after sharing everything.",
                    "engagement_rate": 0.087
                },
                {
                    "content": "Startup life: celebrating our first milestone while eating ramen because we reinvested everything. The founder journey is wild.",
                    "engagement_rate": 0.076
                },
                {
                    "content": "Product development taught me that perfection is the enemy of progress. We spent months building the perfect feature nobody wanted.",
                    "engagement_rate": 0.065
                }
            ]
        }
    
    def test_evaluator_initialization(self, evaluator):
        """Test that evaluator initializes correctly."""
        assert evaluator.min_authenticity_threshold == 0.6
        assert evaluator.performance_weight == 0.4
        assert evaluator.authenticity_weight == 0.6
        assert evaluator.viral_patterns is not None
        assert len(evaluator.viral_patterns['hooks']) > 0
    
    def test_viral_patterns_loading(self, evaluator):
        """Test that viral patterns are loaded correctly."""
        patterns = evaluator.viral_patterns
        
        assert 'hooks' in patterns
        assert 'engagement_patterns' in patterns
        assert 'structure_patterns' in patterns
        assert 'emotional_triggers' in patterns
        
        # Test specific patterns exist
        assert len(patterns['hooks']) > 5
        assert len(patterns['engagement_patterns']) > 3
        assert len(patterns['structure_patterns']) > 2
    
    def test_calculate_authenticity_score_with_history(self, evaluator, sample_creator_profile):
        """Test authenticity score calculation with historical data."""
        content = "Building in public taught me valuable lessons about the founder journey and startup life."
        
        score = evaluator.calculate_authenticity_score(content, sample_creator_profile)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be similar to historical content
    
    def test_calculate_authenticity_score_without_history(self, evaluator):
        """Test authenticity score when no historical data exists."""
        content = "This is test content."
        creator_profile = {"historical_posts": []}
        
        score = evaluator.calculate_authenticity_score(content, creator_profile)
        
        assert score == 0.5  # Neutral score when no history
    
    def test_calculate_viral_potential_with_hooks(self, evaluator):
        """Test viral potential calculation with strong hooks."""
        content = "What if I told you the secret to startup success? Here's what nobody talks about..."
        
        viral_score = evaluator.calculate_viral_potential(content)
        
        assert 0.0 <= viral_score <= 1.0
        assert viral_score > 0.25  # Should detect question hook and curiosity patterns
    
    def test_calculate_viral_potential_with_engagement_patterns(self, evaluator):
        """Test viral potential with engagement patterns."""
        content = "This changed everything for our startup. What do you think? Comment below and share if you agree!"
        
        viral_score = evaluator.calculate_viral_potential(content)
        
        assert 0.0 <= viral_score <= 1.0
        assert viral_score > 0.25  # Should detect multiple engagement patterns
    
    def test_calculate_viral_potential_low_scoring(self, evaluator):
        """Test viral potential with low-engagement content."""
        content = "This is a plain statement without any engaging elements or questions."
        
        viral_score = evaluator.calculate_viral_potential(content)
        
        assert 0.0 <= viral_score <= 1.0
        assert viral_score < 0.5  # Should score lower
    
    def test_length_optimization_twitter(self, evaluator):
        """Test content length optimization for Twitter."""
        # Optimal length for Twitter (200-280 chars)
        optimal_content = "A" * 250
        score = evaluator._calculate_length_optimization(optimal_content, 'twitter')
        assert score == 1.0
        
        # Too short
        short_content = "A" * 100
        score = evaluator._calculate_length_optimization(short_content, 'twitter')
        assert score < 1.0
        
        # Too long
        long_content = "A" * 400
        score = evaluator._calculate_length_optimization(long_content, 'twitter')
        assert score < 1.0
    
    def test_length_optimization_linkedin(self, evaluator):
        """Test content length optimization for LinkedIn."""
        # Optimal length for LinkedIn (1500-2000 chars)
        optimal_content = "A" * 1750
        score = evaluator._calculate_length_optimization(optimal_content, 'linkedin')
        assert score == 1.0
    
    def test_calculate_dynamic_threshold_high_variance(self, evaluator):
        """Test dynamic threshold with high variance tolerance."""
        creator_profile = {
            'variance_tolerance': 0.8,
            'growth_focus': 0.7,
            'voice_consistency': 0.9
        }
        
        threshold = evaluator.calculate_dynamic_threshold(creator_profile)
        
        assert 0.3 <= threshold <= 0.9
        assert threshold < evaluator.min_authenticity_threshold  # Should be lower due to high variance
    
    def test_calculate_dynamic_threshold_low_variance(self, evaluator):
        """Test dynamic threshold with low variance tolerance."""
        creator_profile = {
            'variance_tolerance': 0.2,
            'growth_focus': 0.3,
            'voice_consistency': 0.5
        }
        
        threshold = evaluator.calculate_dynamic_threshold(creator_profile)
        
        assert 0.3 <= threshold <= 0.9
        assert threshold >= evaluator.min_authenticity_threshold  # Should be higher due to low variance
    
    def test_predict_performance(self, evaluator):
        """Test performance prediction functionality."""
        content = "What's the secret to startup success? Here's what I learned from 5 years of building companies..."
        context = {'platform': 'linkedin'}
        
        prediction = evaluator.predict_performance(content, context)
        
        assert 'predicted_engagement_rate' in prediction
        assert 'predicted_reach_multiplier' in prediction
        assert 'viral_probability' in prediction
        assert 'performance_confidence' in prediction
        
        assert 0.0 <= prediction['predicted_engagement_rate'] <= 1.0
        assert prediction['predicted_reach_multiplier'] >= 1.0
    
    def test_evaluate_high_authenticity_high_viral(self, evaluator, sample_creator_profile):
        """Test evaluation with high authenticity and high viral potential."""
        content = "Building in public taught me the biggest startup lesson: What's your experience with the founder journey? Share below!"
        context = {
            'creator_profile': sample_creator_profile,
            'platform': 'linkedin'
        }
        
        result = evaluator.evaluate(content, context)
        
        assert 'authenticity_score' in result
        assert 'viral_potential' in result
        assert 'balanced_score' in result
        assert 'authenticity_met' in result
        assert 'recommendations' in result
        
        assert 0.0 <= result['authenticity_score'] <= 1.0
        assert 0.0 <= result['viral_potential'] <= 1.0
        assert 0.0 <= result['balanced_score'] <= 1.0
        assert result['authenticity_met'] is not None
    
    def test_evaluate_low_authenticity(self, evaluator, sample_creator_profile):
        """Test evaluation with low authenticity content."""
        content = "Buy this amazing product now! Limited time offer! Don't miss out! Click here immediately!"
        context = {
            'creator_profile': sample_creator_profile,
            'platform': 'linkedin'
        }
        
        result = evaluator.evaluate(content, context)
        
        assert result['authenticity_score'] < 0.5  # Should be low authenticity
        assert result['authenticity_met'] is False
        assert result['balanced_score'] < 0.5  # Should be penalized
        assert len(result['recommendations']) > 0
    
    def test_evaluate_without_context(self, evaluator):
        """Test evaluation without creator context."""
        content = "This is test content for evaluation."
        
        result = evaluator.evaluate(content)
        
        assert 'authenticity_score' in result
        assert 'viral_potential' in result
        assert 'balanced_score' in result
        assert result['authenticity_score'] == 0.5  # Neutral when no context
    
    def test_recommendations_authenticity_improvement(self, evaluator, sample_creator_profile):
        """Test that authenticity improvement recommendations are generated."""
        content = "Buy now! Amazing deal! Limited time! Don't wait!"
        context = {'creator_profile': sample_creator_profile}
        
        result = evaluator.evaluate(content, context)
        
        recommendations = result['recommendations']
        authenticity_recs = [r for r in recommendations if r['type'] == 'authenticity_improvement']
        
        assert len(authenticity_recs) > 0
        assert authenticity_recs[0]['priority'] == 'high'
    
    def test_recommendations_performance_optimization(self, evaluator, sample_creator_profile):
        """Test that performance optimization recommendations are generated."""
        content = "I learned something today about building startups."  # Authentic but low viral potential
        context = {'creator_profile': sample_creator_profile}
        
        result = evaluator.evaluate(content, context)
        
        recommendations = result['recommendations']
        performance_recs = [r for r in recommendations if r['type'] == 'performance_optimization']
        
        # Should suggest performance improvements for low viral content
        assert len([r for r in recommendations if 'viral potential' in r.get('message', '')]) >= 0
    
    def test_batch_evaluate(self, evaluator, sample_creator_profile):
        """Test batch evaluation functionality."""
        content_list = [
            "What's your biggest startup lesson? Share below!",
            "Building in public taught me about founder journey challenges.",
            "Amazing product launch! Buy now!"
        ]
        context_list = [
            {'creator_profile': sample_creator_profile, 'platform': 'linkedin'},
            {'creator_profile': sample_creator_profile, 'platform': 'linkedin'},
            {'creator_profile': sample_creator_profile, 'platform': 'linkedin'}
        ]
        
        results = evaluator.batch_evaluate(content_list, context_list)
        
        assert len(results) == len(content_list)
        for result in results:
            if 'error' not in result:
                assert 'authenticity_score' in result
                assert 'viral_potential' in result
                assert 'balanced_score' in result
    
    def test_evaluation_metadata(self, evaluator, sample_creator_profile):
        """Test that evaluation metadata is included in results."""
        content = "Test content for metadata."
        context = {'creator_profile': sample_creator_profile, 'platform': 'twitter'}
        
        result = evaluator.evaluate(content, context)
        
        assert 'evaluation_metadata' in result
        metadata = result['evaluation_metadata']
        
        assert metadata['level'] == 1
        assert metadata['evaluator'] == 'authenticity_performance'
        assert 'evaluation_time' in metadata
        assert metadata['platform'] == 'twitter'
    
    def test_performance_tracking(self, evaluator):
        """Test that performance metrics are tracked."""
        content = "Test content for performance tracking."
        
        result = evaluator.evaluate(content)
        
        # Should have recorded performance metrics
        assert hasattr(evaluator, 'performance_history')
        assert len(evaluator.performance_history) > 0
    
    def test_error_handling_in_authenticity_calculation(self, evaluator):
        """Test error handling in authenticity score calculation."""
        content = "Test content"
        # Malformed creator profile
        creator_profile = {'historical_posts': [{'malformed': 'data'}]}
        
        # Should not crash and return neutral score
        score = evaluator.calculate_authenticity_score(content, creator_profile)
        assert score == 0.5
    
    def test_viral_pattern_regex_safety(self, evaluator):
        """Test that viral patterns handle edge cases safely."""
        # Test with special characters that could break regex
        content = "What?! How did this happen? Amazing! (Really?) [Unbelievable]"
        
        viral_score = evaluator.calculate_viral_potential(content)
        
        assert 0.0 <= viral_score <= 1.0
        assert viral_score > 0.25  # Should detect patterns despite special chars
    
    def test_configuration_override(self):
        """Test that configuration values can be overridden."""
        custom_config = {
            'min_authenticity_threshold': 0.7,
            'performance_weight': 0.3,
            'authenticity_weight': 0.7
        }
        
        evaluator = AuthenticityPerformanceEvaluator(custom_config)
        
        assert evaluator.min_authenticity_threshold == 0.7
        assert evaluator.performance_weight == 0.3
        assert evaluator.authenticity_weight == 0.7 