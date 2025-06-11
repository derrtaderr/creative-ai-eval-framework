"""
Tests for Enhanced Level 2 Temporal Evaluator

Tests the enhanced temporal evaluation capabilities including:
- Viral template sustainability scoring
- Creator growth trajectory forecasting
- Optimal content calendar generation
- Enhanced recommendations
"""

import pytest
import json
from datetime import datetime, timedelta, date
from src.evaluators.enhanced_temporal_evaluator import EnhancedTemporalEvaluator


class TestEnhancedTemporalEvaluator:
    """Test Enhanced Level 2 Temporal Evaluator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = EnhancedTemporalEvaluator()
        
        # Sample viral data
        self.sample_viral_data = {
            'pattern_type': 'question_hooks',
            'usage_history': [
                {'date': '2024-12-01', 'engagement_rate': 0.045, 'comment_sentiment': 0.8},
                {'date': '2024-12-03', 'engagement_rate': 0.042, 'comment_sentiment': 0.75},
                {'date': '2024-12-05', 'engagement_rate': 0.038, 'comment_sentiment': 0.72},
                {'date': '2024-12-08', 'engagement_rate': 0.041, 'comment_sentiment': 0.70},
                {'date': '2024-12-10', 'engagement_rate': 0.036, 'comment_sentiment': 0.68}
            ]
        }
        
        # Sample creator profile
        self.sample_creator_profile = {
            'creator_id': 'test_creator',
            'followers': 10000,
            'engagement_rate': 0.05,
            'platform_focus': 'linkedin',
            'historical_posts': [f'post_{i}' for i in range(20)],
            'historical_data_points': 20
        }
        
        # Sample template strategy
        self.sample_template_strategy = {
            'templates_per_month': 6,
            'quality_score': 0.8,
            'consistency_target': '4_per_week',
            'viral_focus': True
        }
        
        # Sample baseline metrics
        self.sample_baseline = {
            'followers': 10000,
            'engagement_rate': 0.05,
            'platform': 'linkedin',
            'historical_data_points': 20
        }
    
    def test_enhanced_evaluator_initialization(self):
        """Test enhanced evaluator initializes correctly."""
        assert self.evaluator is not None
        assert hasattr(self.evaluator, 'extended_time_windows')
        assert hasattr(self.evaluator, 'viral_patterns_db')
        assert hasattr(self.evaluator, 'growth_models')
        assert hasattr(self.evaluator, 'calendar_config')
        
        # Check extended time windows include new ones
        assert 720 in self.evaluator.extended_time_windows  # 30 days
        assert 4320 in self.evaluator.extended_time_windows  # 6 months
    
    def test_viral_patterns_database_structure(self):
        """Test viral patterns database has correct structure."""
        patterns_db = self.evaluator.viral_patterns_db
        
        # Check required pattern types exist
        required_patterns = ['question_hooks', 'emotional_triggers', 'social_proof', 'curiosity_patterns', 'list_formats']
        for pattern in required_patterns:
            assert pattern in patterns_db
            
            # Check each pattern has required fields
            pattern_data = patterns_db[pattern]
            assert 'sustainability_weeks' in pattern_data
            assert 'fatigue_threshold' in pattern_data
            assert 'refresh_indicators' in pattern_data
            
            # Validate data types
            assert isinstance(pattern_data['sustainability_weeks'], int)
            assert isinstance(pattern_data['fatigue_threshold'], float)
            assert isinstance(pattern_data['refresh_indicators'], list)
    
    def test_growth_models_structure(self):
        """Test growth models have correct structure."""
        growth_models = self.evaluator.growth_models
        
        # Check required model types
        required_models = ['follower_growth', 'engagement_lift', 'confidence_scoring']
        for model in required_models:
            assert model in growth_models
        
        # Check follower growth model structure
        follower_model = growth_models['follower_growth']
        assert 'base_rate' in follower_model
        assert 'viral_multiplier' in follower_model
        assert 'engagement_factor' in follower_model
        assert 'platform_coefficients' in follower_model
        
        # Check platform coefficients
        platforms = follower_model['platform_coefficients']
        required_platforms = ['twitter', 'linkedin', 'instagram']
        for platform in required_platforms:
            assert platform in platforms
    
    def test_predict_viral_lifecycle(self):
        """Test viral lifecycle prediction."""
        result = self.evaluator.predict_viral_lifecycle(
            self.sample_viral_data,
            self.sample_creator_profile
        )
        
        # Check result structure
        required_fields = [
            'sustainability_score', 'optimal_usage_weeks', 'audience_fatigue_risk',
            'refresh_date', 'refresh_indicators', 'pattern_type', 'current_usage_rate',
            'recommendations'
        ]
        for field in required_fields:
            assert field in result
        
        # Validate data types and ranges
        assert isinstance(result['sustainability_score'], float)
        assert 0.0 <= result['sustainability_score'] <= 1.0
        
        assert isinstance(result['optimal_usage_weeks'], int)
        assert result['optimal_usage_weeks'] > 0
        
        assert isinstance(result['audience_fatigue_risk'], float)
        assert 0.0 <= result['audience_fatigue_risk'] <= 1.0
        
        assert isinstance(result['refresh_date'], str)
        # Validate date format
        datetime.strptime(result['refresh_date'], '%Y-%m-%d')
        
        assert result['pattern_type'] == 'question_hooks'
    
    def test_forecast_growth_trajectory(self):
        """Test growth trajectory forecasting."""
        result = self.evaluator.forecast_growth_trajectory(
            self.sample_baseline,
            self.sample_template_strategy
        )
        
        # Check result structure
        required_fields = [
            'forecast_period', 'total_follower_growth', 'percentage_growth',
            'average_monthly_growth_rate', 'peak_growth_month', 'engagement_lift_percentage',
            'confidence_score', 'monthly_projections', 'final_metrics',
            'growth_drivers', 'risk_factors'
        ]
        for field in required_fields:
            assert field in result
        
        # Validate data types and ranges
        assert result['forecast_period'] == '6_months'
        assert isinstance(result['total_follower_growth'], int)
        assert result['total_follower_growth'] >= 0
        
        assert isinstance(result['percentage_growth'], float)
        assert result['percentage_growth'] >= 0
        
        assert isinstance(result['peak_growth_month'], int)
        assert 1 <= result['peak_growth_month'] <= 6
        
        assert isinstance(result['confidence_score'], float)
        assert 0.0 <= result['confidence_score'] <= 1.0
        
        # Check monthly projections
        assert len(result['monthly_projections']) == 6
        for proj in result['monthly_projections']:
            assert 'month' in proj
            assert 'projected_followers' in proj
            assert 'growth_rate' in proj
            assert 'projected_engagement_rate' in proj
        
        # Check final metrics
        final_metrics = result['final_metrics']
        assert 'projected_followers' in final_metrics
        assert 'projected_engagement_rate' in final_metrics
        assert final_metrics['projected_followers'] > self.sample_baseline['followers']
    
    def test_generate_content_calendar(self):
        """Test content calendar generation."""
        # First get predictions
        lifecycle_pred = self.evaluator.predict_viral_lifecycle(
            self.sample_viral_data,
            self.sample_creator_profile
        )
        growth_pred = self.evaluator.forecast_growth_trajectory(
            self.sample_baseline,
            self.sample_template_strategy
        )
        
        predictions = {
            'lifecycle': lifecycle_pred,
            'growth': growth_pred
        }
        
        platform_prefs = {
            'platform': 'linkedin',
            'weeks': 4
        }
        
        result = self.evaluator.generate_content_calendar(predictions, platform_prefs)
        
        # Check result structure
        required_fields = [
            'calendar_period', 'platform', 'weekly_schedule',
            'template_rotation_strategy', 'strategic_timing',
            'optimization_insights', 'performance_predictions'
        ]
        for field in required_fields:
            assert field in result
        
        # Validate calendar structure
        assert result['calendar_period'] == '4_weeks'
        assert result['platform'] == 'linkedin'
        assert len(result['weekly_schedule']) == 4
        
        # Check weekly schedule structure
        for week in result['weekly_schedule']:
            required_week_fields = [
                'week', 'week_start', 'optimal_frequency',
                'daily_schedule', 'focus_strategy'
            ]
            for field in required_week_fields:
                assert field in week
            
            assert len(week['daily_schedule']) == 7  # 7 days
            
            # Check daily schedule structure
            for day in week['daily_schedule']:
                assert 'date' in day
                assert 'day' in day
                assert 'post_scheduled' in day
                
                if day['post_scheduled']:
                    assert 'optimal_time' in day
                    assert 'recommended_content_type' in day
        
        # Check template rotation strategy
        rotation = result['template_rotation_strategy']
        rotation_fields = [
            'rotation_frequency', 'templates_per_rotation',
            'total_template_variations_needed', 'priority_patterns',
            'refresh_schedule'
        ]
        for field in rotation_fields:
            assert field in rotation
    
    def test_evaluate_with_lifecycle(self):
        """Test enhanced evaluation with lifecycle analysis."""
        content = "Here's the counterintuitive truth about AI adoption: Most companies fail because of change management, not technology."
        
        context = {
            'viral_data': self.sample_viral_data,
            'creator_profile': self.sample_creator_profile,
            'template_strategy': self.sample_template_strategy,
            'platform': 'linkedin',
            'generate_calendar': True,
            'calendar_weeks': 4
        }
        
        result = self.evaluator.evaluate_with_lifecycle(content, context)
        
        # Check that base temporal evaluation is included
        assert 'temporal_score' in result
        assert 'immediate_metrics' in result
        assert 'delayed_metrics' in result
        assert 'lifecycle_prediction' in result
        
        # Check enhanced features are included
        assert 'viral_lifecycle_analysis' in result
        assert 'growth_trajectory_forecast' in result
        assert 'enhanced_recommendations' in result
        assert 'content_calendar' in result
        
        # Check evaluation metadata
        metadata = result['evaluation_metadata']
        assert metadata['enhanced_level'] == '2+'
        assert metadata['lifecycle_analysis'] is True
        assert metadata['growth_forecasting'] is True
    
    def test_usage_rate_calculation(self):
        """Test usage rate calculation helper method."""
        usage_history = [
            {'date': '2024-12-01', 'engagement_rate': 0.045},
            {'date': '2024-12-15', 'engagement_rate': 0.042},
            {'date': '2024-11-20', 'engagement_rate': 0.038}  # Outside 30-day window
        ]
        
        rate = self.evaluator._calculate_usage_rate(usage_history, days=30)
        
        # Should count posts within 30 days
        assert isinstance(rate, float)
        assert rate >= 0.0
        
        # Test with empty history
        empty_rate = self.evaluator._calculate_usage_rate([], days=30)
        assert empty_rate == 0.0
    
    def test_sustainability_score_calculation(self):
        """Test sustainability score calculation."""
        # Test below threshold (should be high sustainability)
        low_usage = 0.05
        threshold = 0.15
        score = self.evaluator._calculate_sustainability_score(low_usage, threshold)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relatively high
        
        # Test above threshold (should be lower sustainability)
        high_usage = 0.25
        score = self.evaluator._calculate_sustainability_score(high_usage, threshold)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be lower
    
    def test_refresh_date_calculation(self):
        """Test refresh date calculation."""
        optimal_weeks = 4
        fatigue_risk = 0.3
        
        refresh_date = self.evaluator._calculate_refresh_date(optimal_weeks, fatigue_risk)
        
        # Should be a date in the future
        assert isinstance(refresh_date, date)
        assert refresh_date > datetime.now().date()
        
        # Test with high fatigue risk (should be sooner)
        high_fatigue_date = self.evaluator._calculate_refresh_date(optimal_weeks, 0.8)
        low_fatigue_date = self.evaluator._calculate_refresh_date(optimal_weeks, 0.2)
        
        # High fatigue should result in earlier refresh
        assert high_fatigue_date <= low_fatigue_date
    
    def test_template_boost_calculation(self):
        """Test template boost calculation for growth forecasting."""
        template_strategy = {
            'templates_per_month': 6,
            'quality_score': 0.8
        }
        
        boost = self.evaluator._calculate_template_boost(template_strategy, 2, 1.5)
        
        assert isinstance(boost, float)
        assert boost >= 0.0
        
        # Higher quality should yield higher boost
        high_quality_strategy = {**template_strategy, 'quality_score': 0.9}
        high_boost = self.evaluator._calculate_template_boost(high_quality_strategy, 2, 1.5)
        assert high_boost > boost
    
    def test_engagement_lift_calculation(self):
        """Test engagement lift calculation."""
        template_strategy = {'quality_score': 0.8}
        current_rate = 0.05
        config = {
            'template_boost': 0.15,
            'quality_multiplier': 1.3
        }
        
        lift = self.evaluator._calculate_engagement_lift(template_strategy, current_rate, config)
        
        assert isinstance(lift, float)
        assert lift >= 0.0
    
    def test_forecast_confidence_calculation(self):
        """Test forecast confidence calculation."""
        baseline = {
            'historical_data_points': 25
        }
        template_strategy = {
            'templates_per_month': 6,
            'quality_score': 0.8,
            'consistency_target': '4_per_week'
        }
        platform = 'linkedin'
        
        confidence = self.evaluator._calculate_forecast_confidence(baseline, template_strategy, platform)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # More data points should increase confidence
        high_data_baseline = {**baseline, 'historical_data_points': 50}
        high_confidence = self.evaluator._calculate_forecast_confidence(
            high_data_baseline, template_strategy, platform
        )
        assert high_confidence >= confidence
    
    def test_default_predictions(self):
        """Test default prediction generation when data is unavailable."""
        # Test default lifecycle prediction
        content = "Test content"
        default_lifecycle = self.evaluator._generate_default_lifecycle_prediction(content)
        
        required_fields = [
            'sustainability_score', 'optimal_usage_weeks', 'audience_fatigue_risk',
            'refresh_date', 'refresh_indicators', 'pattern_type', 'current_usage_rate',
            'recommendations'
        ]
        for field in required_fields:
            assert field in default_lifecycle
        
        # Test default growth forecast
        default_growth = self.evaluator._generate_default_growth_forecast()
        
        growth_fields = [
            'forecast_period', 'total_follower_growth', 'percentage_growth',
            'average_monthly_growth_rate', 'peak_growth_month', 'engagement_lift_percentage',
            'confidence_score', 'monthly_projections', 'final_metrics',
            'growth_drivers', 'risk_factors'
        ]
        for field in growth_fields:
            assert field in default_growth
    
    def test_baseline_metrics_extraction(self):
        """Test baseline metrics extraction from creator profile."""
        creator_profile = {
            'followers': 15000,
            'engagement_rate': 0.045,
            'platform_focus': 'linkedin',
            'historical_posts': [f'post_{i}' for i in range(30)]
        }
        
        baseline = self.evaluator._extract_baseline_metrics(creator_profile)
        
        assert baseline['followers'] == 15000
        assert baseline['engagement_rate'] == 0.045
        assert baseline['platform'] == 'linkedin'
        assert baseline['historical_data_points'] == 30
    
    def test_enhanced_recommendations(self):
        """Test enhanced recommendations generation."""
        base_result = {
            'recommendations': [
                {'type': 'base', 'priority': 'medium', 'message': 'Base recommendation'}
            ]
        }
        
        lifecycle = {
            'audience_fatigue_risk': 0.6,
            'sustainability_score': 0.4
        }
        
        growth = {
            'confidence_score': 0.85,
            'total_follower_growth': 2500,
            'peak_growth_month': 3
        }
        
        recommendations = self.evaluator._generate_enhanced_recommendations(
            base_result, lifecycle, growth
        )
        
        # Should include base recommendations plus enhanced ones
        assert len(recommendations) >= 1
        
        # Check for enhanced recommendation types
        recommendation_types = [rec['type'] for rec in recommendations]
        assert 'growth_optimization' in recommendation_types
    
    def test_calendar_optimization_features(self):
        """Test calendar optimization features."""
        # Test optimal frequency calculation
        predictions = {
            'lifecycle': {'audience_fatigue_risk': 0.3}
        }
        config = {
            'viral_templates': {'min': 2, 'max': 4}
        }
        
        frequency = self.evaluator._calculate_optimal_frequency(1, predictions, config)
        assert isinstance(frequency, int)
        assert frequency >= config['viral_templates']['min']
        assert frequency <= config['viral_templates']['max']
        
        # Test with high fatigue risk
        high_fatigue_predictions = {
            'lifecycle': {'audience_fatigue_risk': 0.8}
        }
        high_fatigue_frequency = self.evaluator._calculate_optimal_frequency(
            1, high_fatigue_predictions, config
        )
        assert high_fatigue_frequency <= frequency
    
    def test_optimal_days_selection(self):
        """Test optimal days selection for posting."""
        peak_windows = [[9, 11], [14, 16], [18, 20]]
        
        # Test different frequencies
        days_3 = self.evaluator._select_optimal_days(3, peak_windows)
        assert len(days_3) == 3
        assert all(day < 7 for day in days_3)
        
        days_5 = self.evaluator._select_optimal_days(5, peak_windows)
        assert len(days_5) == 5
        assert all(day < 7 for day in days_5)
    
    def test_content_type_recommendations(self):
        """Test content type recommendations."""
        predictions = {
            'lifecycle': {'audience_fatigue_risk': 0.3}
        }
        
        # Test different days
        content_type_1 = self.evaluator._recommend_content_type(1, predictions)  # Tuesday
        content_type_2 = self.evaluator._recommend_content_type(2, predictions)  # Wednesday
        
        valid_types = ['viral_template', 'engagement_content', 'brand_content']
        assert content_type_1 in valid_types
        assert content_type_2 in valid_types
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with minimal data
        minimal_viral_data = {'pattern_type': 'general', 'usage_history': []}
        minimal_creator_profile = {}
        
        result = self.evaluator.predict_viral_lifecycle(minimal_viral_data, minimal_creator_profile)
        assert 'sustainability_score' in result
        assert 'audience_fatigue_risk' in result
        
        # Test with minimal baseline data
        minimal_baseline = {}
        minimal_strategy = {}
        
        growth_result = self.evaluator.forecast_growth_trajectory(minimal_baseline, minimal_strategy)
        assert 'total_follower_growth' in growth_result
        assert 'confidence_score' in growth_result
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for enhanced evaluation."""
        import time
        
        content = "Test content for performance evaluation"
        context = {
            'viral_data': self.sample_viral_data,
            'creator_profile': self.sample_creator_profile,
            'template_strategy': self.sample_template_strategy,
            'platform': 'linkedin'
        }
        
        # Test evaluation performance
        start_time = time.time()
        result = self.evaluator.evaluate_with_lifecycle(content, context)
        eval_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 100ms)
        assert eval_time < 0.1
        assert result is not None
    
    def test_integration_with_base_evaluator(self):
        """Test integration with base temporal evaluator."""
        content = "Integration test content"
        context = {
            'platform': 'linkedin',
            'creator_profile': {'followers': 5000}
        }
        
        result = self.evaluator.evaluate_with_lifecycle(content, context)
        
        # Should include all base temporal evaluator features
        assert 'temporal_score' in result
        assert 'immediate_metrics' in result
        assert 'delayed_metrics' in result
        assert 'lifecycle_prediction' in result
        
        # Plus enhanced features
        assert 'viral_lifecycle_analysis' in result
        assert 'growth_trajectory_forecast' in result
        assert 'enhanced_recommendations' in result
        
        # Metadata should indicate enhanced level
        assert result['evaluation_metadata']['enhanced_level'] == '2+'