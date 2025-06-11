"""
Unit tests for ContentContextEvaluator (Level 0).
"""

import unittest
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluators import ContentContextEvaluator


class TestContentContextEvaluator(unittest.TestCase):
    """Test cases for ContentContextEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ContentContextEvaluator()
        
        # Sample creator profile
        self.sample_profile = {
            "creator_id": "test_creator",
            "name": "Test Creator",
            "platforms": ["twitter", "linkedin"],
            "voice_characteristics": {
                "tone": "professional_casual",
                "expertise_areas": ["AI", "testing"],
                "authenticity_tolerance": 0.75,
                "brand_keywords": ["innovation", "testing", "quality"]
            },
            "historical_content": [
                {
                    "post_id": "test_001",
                    "text": "Testing is crucial for building quality AI systems. We focus on comprehensive evaluation.",
                    "platform": "linkedin",
                    "engagement": {"likes": 25, "comments": 5, "shares": 3},
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                {
                    "post_id": "test_002",
                    "text": "Innovation in AI testing requires systematic approaches. Quality matters more than speed.",
                    "platform": "twitter",
                    "engagement": {"likes": 15, "retweets": 3, "replies": 2},
                    "timestamp": "2024-01-12T14:22:00Z"
                }
            ],
            "engagement_patterns": {
                "peak_hours": [9, 17, 20],
                "best_content_types": ["insights", "technical"],
                "audience_segments": ["developers", "ai_researchers"]
            }
        }

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ContentContextEvaluator()
        self.assertIsNotNone(evaluator)
        self.assertIsNotNone(evaluator.platform_configs)
        self.assertIn('twitter', evaluator.platform_configs)
        self.assertIn('linkedin', evaluator.platform_configs)

    def test_validate_content(self):
        """Test content validation."""
        # Valid content
        self.assertTrue(self.evaluator.validate_content("This is valid content"))
        
        # Invalid content
        self.assertFalse(self.evaluator.validate_content(""))
        self.assertFalse(self.evaluator.validate_content(None))
        self.assertFalse(self.evaluator.validate_content(123))

    def test_platform_configs(self):
        """Test platform configuration loading."""
        configs = self.evaluator.platform_configs
        
        # Check Twitter config
        twitter_config = configs['twitter']
        self.assertEqual(twitter_config['character_limit'], 280)
        self.assertIn('optimal_hashtags', twitter_config)
        
        # Check LinkedIn config
        linkedin_config = configs['linkedin']
        self.assertEqual(linkedin_config['character_limit'], 3000)
        self.assertIn('professional_keywords', linkedin_config)

    def test_assess_platform_optimization_twitter(self):
        """Test platform optimization assessment for Twitter."""
        # Good Twitter content
        good_content = "Great insights on AI testing! Quality matters. #AI #testing"
        result = self.evaluator.assess_platform_optimization(good_content, "twitter")
        
        self.assertIn('platform_score', result)
        self.assertIn('component_scores', result)
        self.assertIn('details', result)
        self.assertGreater(result['platform_score'], 0.5)

    def test_assess_platform_optimization_linkedin(self):
        """Test platform optimization assessment for LinkedIn."""
        # Good LinkedIn content
        good_content = """
        Leadership in AI development requires strategic thinking and innovation. 
        Our team focuses on building quality systems that deliver real value.
        What's your experience with AI leadership challenges? #leadership #AI #innovation
        """
        result = self.evaluator.assess_platform_optimization(good_content, "linkedin")
        
        self.assertIn('platform_score', result)
        self.assertGreater(result['platform_score'], 0.5)

    def test_assess_platform_optimization_overlimit(self):
        """Test platform optimization with content over character limit."""
        # Content over Twitter limit
        long_content = "x" * 300  # Over 280 characters
        result = self.evaluator.assess_platform_optimization(long_content, "twitter")
        
        self.assertEqual(result['component_scores']['character_optimization'], 0.0)

    def test_hashtag_optimization(self):
        """Test hashtag optimization scoring."""
        # Twitter - optimal hashtags (1-2)
        content_with_hashtags = "Great content! #AI #testing"
        result = self.evaluator.assess_platform_optimization(content_with_hashtags, "twitter")
        
        self.assertEqual(result['details']['hashtag_count'], 2)
        self.assertGreaterEqual(result['component_scores']['hashtag_optimization'], 0.9)

    def test_evaluate_trend_relevance(self):
        """Test trend relevance evaluation."""
        content = "AI innovation is trending in tech startups"
        result = self.evaluator.evaluate_trend_relevance(content, "twitter")
        
        self.assertIn('trend_score', result)
        self.assertIn('matching_trends', result)
        self.assertGreaterEqual(result['trend_score'], 0.0)
        self.assertLessEqual(result['trend_score'], 1.0)

    def test_calculate_voice_consistency_no_model(self):
        """Test voice consistency calculation when no model is available."""
        # Mock the voice model to be None
        original_model = self.evaluator.voice_model
        self.evaluator.voice_model = None
        
        content = "Test content for voice consistency"
        score = self.evaluator.calculate_voice_consistency(content, self.sample_profile)
        
        self.assertEqual(score, 0.5)  # Default score when no model
        
        # Restore original model
        self.evaluator.voice_model = original_model

    def test_generate_context_score(self):
        """Test comprehensive context score generation."""
        content = "Testing our new AI evaluation framework! Quality and innovation matter. #AI #testing"
        
        result = self.evaluator.generate_context_score(content, self.sample_profile, "twitter")
        
        # Check required fields
        self.assertIn('context_score', result)
        self.assertIn('voice_consistency', result)
        self.assertIn('platform_optimization', result)
        self.assertIn('trend_relevance', result)
        self.assertIn('recommendations', result)
        self.assertIn('execution_time', result)
        self.assertIn('timestamp', result)
        
        # Check score ranges
        self.assertGreaterEqual(result['context_score'], 0.0)
        self.assertLessEqual(result['context_score'], 1.0)

    def test_evaluate_method(self):
        """Test the main evaluate method."""
        content = "Innovation in AI testing is essential for quality products. #AI"
        context = {
            'creator_profile': self.sample_profile,
            'platform': 'linkedin'
        }
        
        result = self.evaluator.evaluate(content, context)
        
        self.assertIn('context_score', result)
        self.assertIsInstance(result['context_score'], float)

    def test_evaluate_content_convenience_method(self):
        """Test the convenience evaluate_content method."""
        content = "Building quality AI systems requires systematic testing approaches."
        
        result = self.evaluator.evaluate_content(content, self.sample_profile, "linkedin")
        
        self.assertIn('context_score', result)
        self.assertIn('recommendations', result)

    def test_load_creator_profile_file_not_found(self):
        """Test loading creator profile when file doesn't exist."""
        profile = self.evaluator.load_creator_profile("nonexistent_file.json")
        self.assertEqual(profile, {})

    def test_load_creator_profile_success(self):
        """Test successful creator profile loading."""
        # Create temporary profile file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_profile, f)
            temp_file = f.name
        
        try:
            profile = self.evaluator.load_creator_profile(temp_file)
            self.assertEqual(profile['creator_id'], 'test_creator')
            self.assertIn('voice_embedding', profile)  # Should be added during loading
        finally:
            os.unlink(temp_file)

    def test_batch_evaluate(self):
        """Test batch evaluation functionality."""
        contents = [
            "First test content #AI",
            "Second test content #testing",
            "Third test content #innovation"
        ]
        
        contexts = [{'creator_profile': self.sample_profile, 'platform': 'twitter'}] * 3
        
        results = self.evaluator.batch_evaluate(contents, contexts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('context_score', result)

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Reset history
        self.evaluator.reset_history()
        
        # Run evaluation
        content = "Test content for performance tracking"
        self.evaluator.evaluate_content(content, self.sample_profile)
        
        # Check performance stats
        stats = self.evaluator.get_performance_stats()
        self.assertEqual(stats['total_evaluations'], 1)
        self.assertEqual(stats['success_rate'], 1.0)

    def test_unsupported_platform(self):
        """Test evaluation with unsupported platform."""
        content = "Test content"
        result = self.evaluator.assess_platform_optimization(content, "unsupported_platform")
        
        self.assertIn('error', result)

    def test_empty_content_evaluation(self):
        """Test evaluation with empty content."""
        result = self.evaluator.generate_context_score("", self.sample_profile)
        self.assertIn('error', result)

    def test_recommendations_generation(self):
        """Test that recommendations are properly generated."""
        # Content that should trigger recommendations
        bad_content = "x" * 300 + " #test #AI #innovation #startup #tech #viral #amazing #great #wow #cool"
        
        result = self.evaluator.generate_context_score(bad_content, self.sample_profile, "twitter")
        
        self.assertGreater(len(result['recommendations']), 0)


if __name__ == '__main__':
    unittest.main() 