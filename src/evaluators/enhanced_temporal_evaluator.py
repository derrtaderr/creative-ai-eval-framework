"""
Enhanced Temporal Evaluator (Level 2+)

Extends the base TemporalEvaluator with viral lifecycle prediction capabilities
that transform temporal evaluation from performance tracking to growth forecasting.

Key Additions:
1. Viral Template Sustainability Scoring
2. Creator Growth Trajectory Forecasting 
3. Optimal Content Calendar Generation
"""

import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from .temporal_evaluator import TemporalEvaluator


class EnhancedTemporalEvaluator(TemporalEvaluator):
    """
    Enhanced Level 2: Temporal Evaluation + Viral Lifecycle Analysis
    
    Extends TemporalEvaluator with predictive growth capabilities:
    - Viral template sustainability scoring (predict pattern fatigue)
    - Creator growth trajectory forecasting (6-month predictions)
    - Optimal content calendar generation (strategic posting schedules)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Enhanced Temporal Evaluator."""
        super().__init__(config)
        
        # Extended time windows for lifecycle analysis
        self.extended_time_windows = self.time_windows + [720, 4320]  # 30 days, 6 months
        
        # Viral pattern sustainability data
        self.viral_patterns_db = self._load_viral_patterns_database()
        
        # Growth forecasting models
        self.growth_models = self._initialize_growth_models()
        
        # Content calendar optimization settings
        self.calendar_config = self._load_calendar_config()
        
        self.logger.info("EnhancedTemporalEvaluator initialized with viral lifecycle capabilities")
    
    def _load_viral_patterns_database(self) -> Dict[str, Any]:
        """Load viral patterns database with sustainability metrics."""
        return {
            'question_hooks': {
                'sustainability_weeks': 4,
                'fatigue_threshold': 0.15,  # 15% usage in 30 days
                'refresh_indicators': ['declining_engagement', 'comment_fatigue', 'low_shares']
            },
            'emotional_triggers': {
                'sustainability_weeks': 3,
                'fatigue_threshold': 0.20,
                'refresh_indicators': ['emotional_desensitization', 'negative_sentiment']
            },
            'curiosity_patterns': {
                'sustainability_weeks': 5,
                'fatigue_threshold': 0.12,
                'refresh_indicators': ['click_rate_decline', 'scroll_past_increase']
            },
            'social_proof': {
                'sustainability_weeks': 6,
                'fatigue_threshold': 0.10,
                'refresh_indicators': ['trust_decline', 'skepticism_increase']
            },
            'list_formats': {
                'sustainability_weeks': 8,
                'fatigue_threshold': 0.08,
                'refresh_indicators': ['format_saturation', 'engagement_plateau']
            }
        }
    
    def _initialize_growth_models(self) -> Dict[str, Any]:
        """Initialize growth forecasting models."""
        return {
            'follower_growth': {
                'base_rate': 0.02,  # 2% monthly base growth
                'viral_multiplier': 1.5,  # 50% boost from viral content
                'engagement_factor': 0.003,  # 0.3% per engagement point
                'platform_coefficients': {
                    'twitter': 1.2,
                    'linkedin': 0.8,
                    'instagram': 1.0
                }
            },
            'engagement_lift': {
                'template_boost': 0.15,  # 15% average template boost
                'consistency_factor': 0.05,  # 5% per week of consistency
                'quality_multiplier': 1.3,  # High quality content boost
                'audience_size_decay': 0.98  # Slight decay with larger audiences
            },
            'confidence_scoring': {
                'data_points_minimum': 10,
                'historical_accuracy': 0.85,
                'volatility_penalty': 0.1,
                'platform_reliability': {
                    'twitter': 0.75,
                    'linkedin': 0.85,
                    'instagram': 0.70
                }
            }
        }
    
    def _load_calendar_config(self) -> Dict[str, Any]:
        """Load content calendar optimization configuration."""
        return {
            'optimal_frequency': {
                'viral_templates': {'min': 2, 'max': 4, 'unit': 'per_week'},
                'standard_content': {'min': 3, 'max': 7, 'unit': 'per_week'},
                'refresh_content': {'min': 1, 'max': 2, 'unit': 'per_month'}
            },
            'spacing_rules': {
                'same_template': 48,  # hours between same template usage
                'similar_patterns': 24,  # hours between similar patterns
                'high_engagement': 12   # hours after high-engagement post
            },
            'peak_windows': {
                'twitter': [[7, 9], [12, 14], [17, 20]],
                'linkedin': [[8, 10], [12, 13], [17, 19]],
                'instagram': [[6, 8], [11, 13], [17, 21]]
            }
        }
    
    def predict_viral_lifecycle(self, viral_data: Dict[str, Any], 
                               creator_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict viral template lifecycle and sustainability.
        
        Args:
            viral_data: Information about the viral pattern/template
            creator_profile: Creator's profile and usage history
            
        Returns:
            Viral lifecycle prediction with sustainability metrics
        """
        pattern_type = viral_data.get('pattern_type', 'general')
        usage_history = viral_data.get('usage_history', [])
        
        # Get pattern sustainability data
        pattern_config = self.viral_patterns_db.get(pattern_type, self.viral_patterns_db['question_hooks'])
        
        # Calculate current usage frequency
        current_usage_rate = self._calculate_usage_rate(usage_history, days=30)
        
        # Determine sustainability score
        sustainability_score = self._calculate_sustainability_score(
            current_usage_rate, pattern_config['fatigue_threshold']
        )
        
        # Calculate optimal usage window
        optimal_weeks = pattern_config['sustainability_weeks']
        if sustainability_score < 0.5:
            optimal_weeks = max(2, optimal_weeks - 2)  # Reduce if overused
        
        # Predict audience fatigue risk
        fatigue_risk = min(1.0, current_usage_rate / pattern_config['fatigue_threshold'])
        
        # Generate refresh date
        refresh_date = self._calculate_refresh_date(optimal_weeks, fatigue_risk)
        
        # Identify refresh indicators
        refresh_indicators = self._analyze_refresh_indicators(
            usage_history, pattern_config['refresh_indicators']
        )
        
        return {
            'sustainability_score': round(sustainability_score, 3),
            'optimal_usage_weeks': optimal_weeks,
            'audience_fatigue_risk': round(fatigue_risk, 3),
            'refresh_date': refresh_date.strftime('%Y-%m-%d'),
            'refresh_indicators': refresh_indicators,
            'pattern_type': pattern_type,
            'current_usage_rate': round(current_usage_rate, 3),
            'recommendations': self._generate_sustainability_recommendations(
                sustainability_score, fatigue_risk, optimal_weeks
            )
        }
    
    def forecast_growth_trajectory(self, baseline: Dict[str, Any], 
                                 template_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast creator growth trajectory over 6 months.
        
        Args:
            baseline: Current creator metrics (followers, engagement, etc.)
            template_strategy: Planned template usage strategy
            
        Returns:
            6-month growth forecast with confidence metrics
        """
        current_followers = baseline.get('followers', 1000)
        current_engagement_rate = baseline.get('engagement_rate', 0.03)
        platform = baseline.get('platform', 'general')
        
        # Get growth model parameters
        growth_config = self.growth_models['follower_growth']
        engagement_config = self.growth_models['engagement_lift']
        
        # Calculate monthly projections
        monthly_projections = []
        projected_followers = current_followers
        projected_engagement = current_engagement_rate
        
        for month in range(1, 7):
            # Base growth rate
            base_growth = growth_config['base_rate']
            
            # Template strategy impact
            template_boost = self._calculate_template_boost(
                template_strategy, month, growth_config['viral_multiplier']
            )
            
            # Platform-specific adjustment
            platform_multiplier = growth_config['platform_coefficients'].get(platform, 1.0)
            
            # Engagement factor
            engagement_factor = projected_engagement * growth_config['engagement_factor'] * 100
            
            # Calculate month growth
            month_growth_rate = (base_growth + template_boost + engagement_factor) * platform_multiplier
            
            # Apply growth
            projected_followers = int(projected_followers * (1 + month_growth_rate))
            
            # Update engagement (with slight decay for larger audiences)
            projected_engagement = min(0.15, projected_engagement * engagement_config['audience_size_decay'])
            
            monthly_projections.append({
                'month': month,
                'projected_followers': projected_followers,
                'growth_rate': round(month_growth_rate, 4),
                'projected_engagement_rate': round(projected_engagement, 4),
                'cumulative_growth': projected_followers - current_followers
            })
        
        # Calculate overall metrics
        total_growth = projected_followers - current_followers
        avg_growth_rate = (projected_followers / current_followers) ** (1/6) - 1
        peak_growth_month = max(monthly_projections, key=lambda x: x['growth_rate'])['month']
        
        # Calculate engagement lift
        engagement_lift = self._calculate_engagement_lift(
            template_strategy, current_engagement_rate, engagement_config
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_forecast_confidence(
            baseline, template_strategy, platform
        )
        
        return {
            'forecast_period': '6_months',
            'total_follower_growth': total_growth,
            'percentage_growth': round(((projected_followers / current_followers) - 1) * 100, 1),
            'average_monthly_growth_rate': round(avg_growth_rate * 100, 2),
            'peak_growth_month': peak_growth_month,
            'engagement_lift_percentage': round(engagement_lift * 100, 1),
            'confidence_score': round(confidence_score, 3),
            'monthly_projections': monthly_projections,
            'final_metrics': {
                'projected_followers': projected_followers,
                'projected_engagement_rate': round(projected_engagement, 4)
            },
            'growth_drivers': self._identify_growth_drivers(template_strategy),
            'risk_factors': self._identify_risk_factors(baseline, template_strategy)
        }
    
    def generate_content_calendar(self, predictions: Dict[str, Any], 
                                platform_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate optimal content calendar based on predictions.
        
        Args:
            predictions: Growth and sustainability predictions
            platform_prefs: Platform-specific preferences and constraints
            
        Returns:
            Optimized content calendar with strategic recommendations
        """
        platform = platform_prefs.get('platform', 'general')
        calendar_weeks = platform_prefs.get('weeks', 4)
        
        # Get platform-specific configuration
        frequency_config = self.calendar_config['optimal_frequency']
        spacing_config = self.calendar_config['spacing_rules']
        peak_windows = self.calendar_config['peak_windows'].get(platform, [[9, 11], [14, 16], [18, 20]])
        
        # Generate weekly schedule
        weekly_schedule = []
        current_date = datetime.now().date()
        
        for week in range(calendar_weeks):
            week_start = current_date + timedelta(weeks=week)
            
            # Determine optimal posting frequency for this week
            optimal_frequency = self._calculate_optimal_frequency(
                week, predictions, frequency_config
            )
            
            # Generate daily recommendations
            week_schedule = self._generate_week_schedule(
                week_start, optimal_frequency, peak_windows, spacing_config, predictions
            )
            
            weekly_schedule.append({
                'week': week + 1,
                'week_start': week_start.strftime('%Y-%m-%d'),
                'optimal_frequency': optimal_frequency,
                'daily_schedule': week_schedule,
                'focus_strategy': self._determine_week_focus(week, predictions)
            })
        
        # Generate template rotation strategy
        template_rotation = self._generate_template_rotation(
            predictions, calendar_weeks, platform
        )
        
        # Calculate strategic timing
        strategic_timing = self._calculate_strategic_timing(
            predictions, platform_prefs
        )
        
        return {
            'calendar_period': f'{calendar_weeks}_weeks',
            'platform': platform,
            'weekly_schedule': weekly_schedule,
            'template_rotation_strategy': template_rotation,
            'strategic_timing': strategic_timing,
            'optimization_insights': self._generate_calendar_insights(
                weekly_schedule, predictions, platform
            ),
            'performance_predictions': self._predict_calendar_performance(
                weekly_schedule, predictions
            )
        }
    
    def evaluate_with_lifecycle(self, content: str, 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform enhanced temporal evaluation with viral lifecycle analysis.
        
        Args:
            content: Content to evaluate
            context: Enhanced context including viral data and creator history
            
        Returns:
            Complete temporal + lifecycle evaluation
        """
        # Perform base temporal evaluation
        base_result = super().evaluate(content, context)
        
        # Extract viral and creator data from context
        viral_data = context.get('viral_data', {}) if context else {}
        creator_profile = context.get('creator_profile', {}) if context else {}
        template_strategy = context.get('template_strategy', {}) if context else {}
        
        # Generate viral lifecycle prediction
        if viral_data:
            lifecycle_prediction = self.predict_viral_lifecycle(viral_data, creator_profile)
        else:
            lifecycle_prediction = self._generate_default_lifecycle_prediction(content)
        
        # Generate growth forecast
        if creator_profile and template_strategy:
            growth_forecast = self.forecast_growth_trajectory(
                self._extract_baseline_metrics(creator_profile),
                template_strategy
            )
        else:
            growth_forecast = self._generate_default_growth_forecast()
        
        # Generate content calendar if requested
        calendar_generation = None
        if context and context.get('generate_calendar', False):
            platform_prefs = {
                'platform': context.get('platform', 'general'),
                'weeks': context.get('calendar_weeks', 4)
            }
            predictions = {
                'lifecycle': lifecycle_prediction,
                'growth': growth_forecast
            }
            calendar_generation = self.generate_content_calendar(predictions, platform_prefs)
        
        # Combine all results
        enhanced_result = {
            **base_result,
            'viral_lifecycle_analysis': lifecycle_prediction,
            'growth_trajectory_forecast': growth_forecast,
            'enhanced_recommendations': self._generate_enhanced_recommendations(
                base_result, lifecycle_prediction, growth_forecast
            ),
            'evaluation_metadata': {
                **base_result['evaluation_metadata'],
                'enhanced_level': '2+',
                'lifecycle_analysis': True,
                'growth_forecasting': True
            }
        }
        
        if calendar_generation:
            enhanced_result['content_calendar'] = calendar_generation
        
        return enhanced_result
    
    # Helper methods for lifecycle analysis
    
    def _calculate_usage_rate(self, usage_history: List[Dict], days: int = 30) -> float:
        """Calculate usage rate over specified days."""
        if not usage_history:
            return 0.0
        
        recent_date = datetime.now() - timedelta(days=days)
        recent_usage = [u for u in usage_history if datetime.fromisoformat(u.get('date', '2024-01-01')) >= recent_date]
        
        return len(recent_usage) / days if days > 0 else 0.0
    
    def _calculate_sustainability_score(self, current_rate: float, threshold: float) -> float:
        """Calculate sustainability score based on usage rate vs threshold."""
        if current_rate >= threshold:
            return max(0.0, 1.0 - (current_rate - threshold) / threshold)
        else:
            return min(1.0, 1.0 - (current_rate / threshold) * 0.3)
    
    def _calculate_refresh_date(self, optimal_weeks: int, fatigue_risk: float) -> date:
        """Calculate when pattern should be refreshed."""
        base_date = datetime.now().date() + timedelta(weeks=optimal_weeks)
        
        # Adjust based on fatigue risk
        if fatigue_risk > 0.7:
            # High fatigue - refresh sooner
            adjustment_days = -int(fatigue_risk * 14)
        elif fatigue_risk < 0.3:
            # Low fatigue - can wait longer
            adjustment_days = int((1 - fatigue_risk) * 7)
        else:
            adjustment_days = 0
        
        return base_date + timedelta(days=adjustment_days)
    
    def _analyze_refresh_indicators(self, usage_history: List[Dict], 
                                  indicators: List[str]) -> List[Dict[str, Any]]:
        """Analyze indicators for when pattern needs refreshing."""
        detected_indicators = []
        
        for indicator in indicators:
            if indicator == 'declining_engagement':
                trend = self._analyze_engagement_trend(usage_history)
                if trend < -0.1:  # 10% decline
                    detected_indicators.append({
                        'indicator': indicator,
                        'severity': 'medium',
                        'trend': round(trend, 3)
                    })
            elif indicator == 'comment_fatigue':
                comment_sentiment = self._analyze_comment_sentiment(usage_history)
                if comment_sentiment < 0.5:
                    detected_indicators.append({
                        'indicator': indicator,
                        'severity': 'high',
                        'sentiment_score': round(comment_sentiment, 3)
                    })
        
        return detected_indicators
    
    def _generate_sustainability_recommendations(self, sustainability: float, 
                                               fatigue_risk: float, 
                                               optimal_weeks: int) -> List[Dict[str, str]]:
        """Generate sustainability-focused recommendations."""
        recommendations = []
        
        if fatigue_risk > 0.7:
            recommendations.append({
                'type': 'immediate_action',
                'priority': 'high',
                'message': f'High fatigue risk ({fatigue_risk:.1%}) - reduce template usage immediately',
                'action': 'Scale back usage by 50% this week'
            })
        
        if sustainability < 0.4:
            recommendations.append({
                'type': 'pattern_refresh',
                'priority': 'medium',
                'message': f'Low sustainability ({sustainability:.1%}) - prepare pattern variations',
                'action': f'Create 3-5 variations within {optimal_weeks} weeks'
            })
        
        if sustainability > 0.8 and fatigue_risk < 0.3:
            recommendations.append({
                'type': 'optimization',
                'priority': 'low',
                'message': f'High sustainability - optimize usage frequency',
                'action': 'Increase usage frequency by 25%'
            })
        
        return recommendations
    
    def _calculate_template_boost(self, template_strategy: Dict[str, Any], 
                                month: int, viral_multiplier: float) -> float:
        """Calculate template strategy boost for specific month."""
        templates_per_month = template_strategy.get('templates_per_month', 4)
        template_quality = template_strategy.get('quality_score', 0.7)
        consistency_bonus = min(month * 0.1, 0.5)  # Builds up over time
        
        return (templates_per_month / 10) * template_quality * viral_multiplier * (1 + consistency_bonus)
    
    def _calculate_engagement_lift(self, template_strategy: Dict[str, Any], 
                                 current_rate: float, 
                                 config: Dict[str, Any]) -> float:
        """Calculate expected engagement lift from template strategy."""
        base_lift = config['template_boost']
        quality_multiplier = template_strategy.get('quality_score', 0.7) * config['quality_multiplier']
        
        return base_lift * quality_multiplier
    
    def _calculate_forecast_confidence(self, baseline: Dict[str, Any], 
                                     template_strategy: Dict[str, Any], 
                                     platform: str) -> float:
        """Calculate confidence score for growth forecast."""
        confidence_config = self.growth_models['confidence_scoring']
        
        # Base confidence from platform reliability
        base_confidence = confidence_config['platform_reliability'].get(platform, 0.75)
        
        # Data points factor
        data_points = baseline.get('historical_data_points', 5)
        data_factor = min(1.0, data_points / confidence_config['data_points_minimum'])
        
        # Strategy clarity factor
        strategy_clarity = len(template_strategy) / 5.0  # Assume 5 key strategy components
        
        return base_confidence * data_factor * strategy_clarity
    
    def _generate_default_lifecycle_prediction(self, content: str) -> Dict[str, Any]:
        """Generate default lifecycle prediction when viral data unavailable."""
        return {
            'sustainability_score': 0.6,
            'optimal_usage_weeks': 4,
            'audience_fatigue_risk': 0.3,
            'refresh_date': (datetime.now().date() + timedelta(weeks=4)).strftime('%Y-%m-%d'),
            'refresh_indicators': [],
            'pattern_type': 'general',
            'current_usage_rate': 0.05,
            'recommendations': []
        }
    
    def _generate_default_growth_forecast(self) -> Dict[str, Any]:
        """Generate default growth forecast when creator data unavailable."""
        return {
            'forecast_period': '6_months',
            'total_follower_growth': 500,
            'percentage_growth': 25.0,
            'average_monthly_growth_rate': 3.8,
            'peak_growth_month': 3,
            'engagement_lift_percentage': 15.0,
            'confidence_score': 0.6,
            'monthly_projections': [],
            'final_metrics': {
                'projected_followers': 2500,
                'projected_engagement_rate': 0.035
            },
            'growth_drivers': ['consistent_posting', 'quality_content'],
            'risk_factors': ['limited_data', 'algorithm_changes']
        }
    
    def _extract_baseline_metrics(self, creator_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract baseline metrics from creator profile."""
        return {
            'followers': creator_profile.get('followers', 1000),
            'engagement_rate': creator_profile.get('engagement_rate', 0.03),
            'platform': creator_profile.get('platform_focus', 'general'),
            'historical_data_points': len(creator_profile.get('historical_posts', []))
        }
    
    def _generate_enhanced_recommendations(self, base_result: Dict[str, Any], 
                                         lifecycle: Dict[str, Any], 
                                         growth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced recommendations combining all analyses."""
        recommendations = list(base_result.get('recommendations', []))
        
        # Add lifecycle-based recommendations
        if lifecycle['audience_fatigue_risk'] > 0.5:
            recommendations.append({
                'type': 'lifecycle_management',
                'priority': 'high',
                'message': f'Audience fatigue risk at {lifecycle["audience_fatigue_risk"]:.1%} - implement pattern rotation',
                'suggestions': [
                    'Reduce template usage frequency by 30%',
                    'Introduce 2-3 pattern variations',
                    'Monitor engagement closely for next 2 weeks'
                ]
            })
        
        # Add growth-based recommendations
        if growth['confidence_score'] > 0.8:
            recommendations.append({
                'type': 'growth_optimization',
                'priority': 'medium',
                'message': f'High-confidence growth forecast ({growth["confidence_score"]:.1%}) - optimize for scale',
                'suggestions': [
                    f'Expect {growth["total_follower_growth"]:,} followers over 6 months',
                    f'Peak growth predicted in month {growth["peak_growth_month"]}',
                    'Prepare content pipeline for increased demand'
                ]
            })
        
        return recommendations
    
    # Additional helper methods for calendar generation
    
    def _calculate_optimal_frequency(self, week: int, predictions: Dict[str, Any], 
                                   config: Dict[str, Any]) -> int:
        """Calculate optimal posting frequency for a specific week."""
        base_frequency = config['viral_templates']['max']
        
        # Adjust based on lifecycle prediction
        lifecycle = predictions.get('lifecycle', {})
        fatigue_risk = lifecycle.get('audience_fatigue_risk', 0.3)
        
        if fatigue_risk > 0.7:
            return max(config['viral_templates']['min'], base_frequency - 2)
        elif fatigue_risk < 0.3:
            return min(config['viral_templates']['max'], base_frequency + 1)
        else:
            return base_frequency
    
    def _generate_week_schedule(self, week_start: date, frequency: int, 
                              peak_windows: List[List[int]], 
                              spacing_config: Dict[str, int],
                              predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate daily schedule for a week."""
        schedule = []
        
        # Distribute posts across the week
        post_days = self._select_optimal_days(frequency, peak_windows)
        
        for day in range(7):
            current_date = week_start + timedelta(days=day)
            day_name = current_date.strftime('%A')
            
            if day in post_days:
                optimal_time = self._select_optimal_time(day_name, peak_windows)
                content_type = self._recommend_content_type(day, predictions)
                
                schedule.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day': day_name,
                    'post_scheduled': True,
                    'optimal_time': optimal_time,
                    'recommended_content_type': content_type,
                    'template_usage': 'recommended' if content_type == 'viral_template' else 'optional'
                })
            else:
                schedule.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day': day_name,
                    'post_scheduled': False,
                    'engagement_focus': True,
                    'recommended_activities': ['respond_to_comments', 'engage_with_community']
                })
        
        return schedule
    
    def _select_optimal_days(self, frequency: int, peak_windows: List[List[int]]) -> List[int]:
        """Select optimal days of week for posting."""
        # Prioritize mid-week days for better engagement
        optimal_days = [1, 2, 3, 4]  # Tuesday through Friday
        
        if frequency <= 3:
            return optimal_days[:frequency]
        elif frequency <= 5:
            return optimal_days + [0, 6][:frequency-4]  # Add Monday and Saturday
        else:
            return list(range(min(7, frequency)))
    
    def _select_optimal_time(self, day_name: str, peak_windows: List[List[int]]) -> str:
        """Select optimal posting time for a day."""
        # Choose middle of peak window
        window = peak_windows[0]  # Use first peak window by default
        optimal_hour = (window[0] + window[1]) // 2
        
        return f"{optimal_hour:02d}:00"
    
    def _recommend_content_type(self, day: int, predictions: Dict[str, Any]) -> str:
        """Recommend content type for specific day."""
        lifecycle = predictions.get('lifecycle', {})
        fatigue_risk = lifecycle.get('audience_fatigue_risk', 0.3)
        
        if day in [1, 3] and fatigue_risk < 0.5:  # Tuesday, Thursday
            return 'viral_template'
        elif day in [2, 4]:  # Wednesday, Friday
            return 'engagement_content'
        else:
            return 'brand_content'
    
    def _analyze_engagement_trend(self, usage_history: List[Dict]) -> float:
        """Analyze engagement trend from usage history."""
        if len(usage_history) < 2:
            return 0.0
        
        # Simple trend calculation (would be more sophisticated in production)
        recent_engagement = np.mean([u.get('engagement_rate', 0.03) for u in usage_history[-5:]])
        older_engagement = np.mean([u.get('engagement_rate', 0.03) for u in usage_history[-10:-5]])
        
        if older_engagement == 0:
            return 0.0
        
        return (recent_engagement - older_engagement) / older_engagement
    
    def _analyze_comment_sentiment(self, usage_history: List[Dict]) -> float:
        """Analyze comment sentiment from usage history."""
        # Simplified sentiment analysis (would use real NLP in production)
        if not usage_history:
            return 0.6
        
        # Mock sentiment based on engagement patterns
        recent_posts = usage_history[-5:]
        avg_sentiment = np.mean([p.get('comment_sentiment', 0.6) for p in recent_posts])
        
        return avg_sentiment
    
    def _determine_week_focus(self, week: int, predictions: Dict[str, Any]) -> str:
        """Determine strategic focus for a specific week."""
        growth = predictions.get('growth', {})
        peak_month = growth.get('peak_growth_month', 3)
        
        if week <= 2:
            return 'foundation_building'
        elif week <= peak_month * 4:
            return 'growth_acceleration'
        else:
            return 'engagement_optimization'
    
    def _generate_template_rotation(self, predictions: Dict[str, Any], 
                                  weeks: int, platform: str) -> Dict[str, Any]:
        """Generate template rotation strategy."""
        lifecycle = predictions.get('lifecycle', {})
        sustainability = lifecycle.get('sustainability_score', 0.6)
        
        if sustainability > 0.7:
            rotation_frequency = 'weekly'
            templates_per_rotation = 3
        elif sustainability > 0.4:
            rotation_frequency = 'bi-weekly'
            templates_per_rotation = 2
        else:
            rotation_frequency = 'daily'
            templates_per_rotation = 5
        
        return {
            'rotation_frequency': rotation_frequency,
            'templates_per_rotation': templates_per_rotation,
            'total_template_variations_needed': templates_per_rotation * (weeks // 2),
            'priority_patterns': ['question_hooks', 'social_proof', 'list_formats'],
            'refresh_schedule': self._generate_refresh_schedule(weeks, sustainability)
        }
    
    def _generate_refresh_schedule(self, weeks: int, sustainability: float) -> List[Dict[str, str]]:
        """Generate schedule for template refreshes."""
        refresh_interval = max(2, int(sustainability * 8))  # 2-8 weeks based on sustainability
        
        schedule = []
        for week in range(refresh_interval, weeks + 1, refresh_interval):
            schedule.append({
                'week': week,
                'action': 'template_refresh',
                'description': f'Introduce new pattern variations in week {week}'
            })
        
        return schedule
    
    def _calculate_strategic_timing(self, predictions: Dict[str, Any], 
                                  platform_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate strategic timing recommendations."""
        growth = predictions.get('growth', {})
        peak_month = growth.get('peak_growth_month', 3)
        
        return {
            'peak_growth_period': f'Month {peak_month}',
            'pre_peak_preparation': f'Weeks {max(1, peak_month*4-4)} to {peak_month*4}',
            'content_pipeline_requirements': f'{growth.get("total_follower_growth", 500) // 100} pieces per week during peak',
            'key_milestones': [
                {'week': 2, 'milestone': 'Establish baseline metrics'},
                {'week': peak_month * 4 - 2, 'milestone': 'Prepare for peak growth period'},
                {'week': peak_month * 4, 'milestone': 'Execute peak growth strategy'},
                {'week': peak_month * 4 + 4, 'milestone': 'Optimize post-peak retention'}
            ]
        }
    
    def _generate_calendar_insights(self, weekly_schedule: List[Dict[str, Any]], 
                                  predictions: Dict[str, Any], 
                                  platform: str) -> List[Dict[str, str]]:
        """Generate insights about the generated calendar."""
        total_posts = sum(len([day for day in week['daily_schedule'] if day.get('post_scheduled', False)]) 
                         for week in weekly_schedule)
        
        insights = [
            {
                'type': 'posting_frequency',
                'insight': f'Optimized for {total_posts} posts over {len(weekly_schedule)} weeks',
                'recommendation': f'Average {total_posts/len(weekly_schedule):.1f} posts per week for {platform}'
            },
            {
                'type': 'growth_optimization',
                'insight': f'Calendar aligned with predicted peak growth in month {predictions.get("growth", {}).get("peak_growth_month", 3)}',
                'recommendation': 'Increase content quality and frequency during peak periods'
            }
        ]
        
        return insights
    
    def _predict_calendar_performance(self, weekly_schedule: List[Dict[str, Any]], 
                                    predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance of the generated calendar."""
        growth = predictions.get('growth', {})
        
        return {
            'expected_follower_growth': growth.get('total_follower_growth', 500),
            'expected_engagement_lift': f"{growth.get('engagement_lift_percentage', 15)}%",
            'confidence_level': f"{growth.get('confidence_score', 0.6)*100:.0f}%",
            'key_performance_weeks': [2, 4, 6, 8]  # Predicted high-performance weeks
        }
    
    def _identify_growth_drivers(self, template_strategy: Dict[str, Any]) -> List[str]:
        """Identify key growth drivers from template strategy."""
        drivers = ['consistent_posting']
        
        if template_strategy.get('quality_score', 0) > 0.7:
            drivers.append('high_quality_templates')
        
        if template_strategy.get('templates_per_month', 0) > 6:
            drivers.append('high_frequency_viral_content')
        
        return drivers
    
    def _identify_risk_factors(self, baseline: Dict[str, Any], 
                             template_strategy: Dict[str, Any]) -> List[str]:
        """Identify risk factors for growth predictions."""
        risks = []
        
        if baseline.get('historical_data_points', 0) < 10:
            risks.append('limited_historical_data')
        
        if template_strategy.get('quality_score', 1) < 0.5:
            risks.append('low_template_quality')
        
        risks.append('algorithm_changes')
        risks.append('market_saturation')
        
        return risks