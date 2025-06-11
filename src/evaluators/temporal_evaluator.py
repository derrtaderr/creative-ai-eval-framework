"""
Temporal Evaluator (Level 2)

Evaluates content performance across multiple time windows to understand
temporal engagement patterns, predict content lifecycle, and optimize timing strategies.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from .base_evaluator import BaseEvaluator


class TemporalEvaluator(BaseEvaluator):
    """
    Level 2: Temporal Assessment
    
    This evaluator analyzes content performance across time windows to:
    1. Track engagement evolution from immediate to delayed metrics
    2. Predict content lifecycle and viral trajectory
    3. Identify optimal posting times and platform patterns
    4. Generate temporal optimization recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Temporal Evaluator."""
        super().__init__(config)
        
        # Time windows for analysis (in hours)
        self.time_windows = config.get('time_windows', [0, 1, 6, 24, 72, 168]) if config else [0, 1, 6, 24, 72, 168]
        
        # Platform-specific engagement decay patterns
        self.platform_decay_patterns = self._load_platform_patterns()
        
        # Temporal models for prediction
        self.immediate_model = LinearRegression()
        self.delayed_model = LinearRegression()
        self.lifecycle_model = PolynomialFeatures(degree=2)
        self._models_trained = False
        
        # Historical temporal data storage
        self.temporal_history = []
        
        self.logger.info("TemporalEvaluator initialized with time windows: {}".format(self.time_windows))
    
    def _load_platform_patterns(self) -> Dict[str, Any]:
        """Load platform-specific temporal engagement patterns."""
        return {
            'twitter': {
                'peak_hours': [7, 8, 9, 12, 17, 18, 19, 20],
                'engagement_half_life': 18,  # minutes
                'viral_threshold_window': 4,  # hours
                'decay_rate': 0.85,
                'weekend_multiplier': 0.7,
                'optimal_follow_up': 2  # hours
            },
            'linkedin': {
                'peak_hours': [8, 9, 10, 11, 12, 17, 18],
                'engagement_half_life': 120,  # minutes  
                'viral_threshold_window': 24,  # hours
                'decay_rate': 0.65,
                'weekend_multiplier': 0.3,
                'optimal_follow_up': 8  # hours
            },
            'instagram': {
                'peak_hours': [6, 7, 8, 11, 17, 18, 19, 20, 21],
                'engagement_half_life': 60,  # minutes
                'viral_threshold_window': 12,  # hours
                'decay_rate': 0.75,
                'weekend_multiplier': 1.2,
                'optimal_follow_up': 4  # hours
            },
            'general': {
                'peak_hours': [8, 9, 12, 17, 18, 19],
                'engagement_half_life': 90,  # minutes
                'viral_threshold_window': 12,  # hours
                'decay_rate': 0.7,
                'weekend_multiplier': 0.8,
                'optimal_follow_up': 6  # hours
            }
        }
    
    def calculate_immediate_metrics(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate immediate engagement potential (T+0 to T+1).
        
        Args:
            content: Content to analyze
            context: Context including platform and timing info
            
        Returns:
            Dictionary of immediate engagement metrics
        """
        platform = context.get('platform', 'general') if context else 'general'
        post_time = context.get('post_time', datetime.now()) if context else datetime.now()
        
        # Extract posting hour
        if isinstance(post_time, str):
            post_time = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
        
        post_hour = post_time.hour
        is_weekend = post_time.weekday() >= 5
        
        # Get platform patterns
        patterns = self.platform_decay_patterns.get(platform, self.platform_decay_patterns['general'])
        
        # Calculate timing optimization score
        timing_score = 1.0 if post_hour in patterns['peak_hours'] else 0.6
        
        # Weekend adjustment
        if is_weekend:
            timing_score *= patterns['weekend_multiplier']
        
        # Content characteristics that drive immediate engagement
        immediate_factors = {
            'timing_optimization': timing_score,
            'content_urgency': self._calculate_urgency_score(content),
            'hook_strength': self._calculate_hook_strength(content),
            'platform_alignment': self._calculate_platform_alignment(content, platform)
        }
        
        # Calculate weighted immediate score
        immediate_score = (
            immediate_factors['timing_optimization'] * 0.3 +
            immediate_factors['content_urgency'] * 0.25 +
            immediate_factors['hook_strength'] * 0.25 +
            immediate_factors['platform_alignment'] * 0.2
        )
        
        return {
            'immediate_score': min(1.0, immediate_score),
            'timing_optimization': immediate_factors['timing_optimization'],
            'content_urgency': immediate_factors['content_urgency'],
            'hook_strength': immediate_factors['hook_strength'],
            'platform_alignment': immediate_factors['platform_alignment'],
            'predicted_1h_engagement': immediate_score * 0.8,
            'viral_window_probability': self._calculate_viral_window_probability(immediate_score, platform)
        }
    
    def calculate_delayed_metrics(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate delayed engagement potential (T+24 to T+168).
        
        Args:
            content: Content to analyze
            context: Context for evaluation
            
        Returns:
            Dictionary of delayed engagement metrics
        """
        platform = context.get('platform', 'general') if context else 'general'
        
        # Factors that contribute to sustained engagement
        delayed_factors = {
            'content_depth': self._calculate_content_depth(content),
            'shareability': self._calculate_shareability(content),
            'evergreen_potential': self._calculate_evergreen_potential(content),
            'discussion_driver': self._calculate_discussion_potential(content)
        }
        
        # Calculate weighted delayed score
        delayed_score = (
            delayed_factors['content_depth'] * 0.3 +
            delayed_factors['shareability'] * 0.25 +
            delayed_factors['evergreen_potential'] * 0.25 +
            delayed_factors['discussion_driver'] * 0.2
        )
        
        # Platform-specific adjustments
        patterns = self.platform_decay_patterns.get(platform, self.platform_decay_patterns['general'])
        platform_adjustment = 1.0 - patterns['decay_rate']
        
        return {
            'delayed_score': min(1.0, delayed_score * platform_adjustment),
            'content_depth': delayed_factors['content_depth'],
            'shareability': delayed_factors['shareability'],
            'evergreen_potential': delayed_factors['evergreen_potential'],
            'discussion_driver': delayed_factors['discussion_driver'],
            'predicted_24h_engagement': delayed_score * 0.6,
            'predicted_72h_engagement': delayed_score * 0.4,
            'predicted_168h_engagement': delayed_score * 0.2
        }
    
    def calculate_lifecycle_prediction(self, immediate_metrics: Dict[str, float], 
                                     delayed_metrics: Dict[str, float],
                                     platform: str = 'general') -> Dict[str, Any]:
        """
        Predict content lifecycle and engagement trajectory.
        
        Args:
            immediate_metrics: Immediate engagement metrics
            delayed_metrics: Delayed engagement metrics
            platform: Target platform
            
        Returns:
            Lifecycle prediction with trajectory points
        """
        patterns = self.platform_decay_patterns.get(platform, self.platform_decay_patterns['general'])
        
        # Create engagement trajectory
        trajectory_points = []
        
        # T+0 (immediate)
        t0_engagement = immediate_metrics['immediate_score']
        trajectory_points.append({'time': 0, 'engagement': t0_engagement})
        
        # T+1 hour
        t1_engagement = immediate_metrics['predicted_1h_engagement']
        trajectory_points.append({'time': 1, 'engagement': t1_engagement})
        
        # T+6 hours (early decay)
        decay_factor = patterns['decay_rate'] ** 0.25  # Quarter decay
        t6_engagement = t1_engagement * decay_factor
        trajectory_points.append({'time': 6, 'engagement': t6_engagement})
        
        # T+24 hours
        t24_engagement = delayed_metrics['predicted_24h_engagement']
        trajectory_points.append({'time': 24, 'engagement': t24_engagement})
        
        # T+72 hours
        t72_engagement = delayed_metrics['predicted_72h_engagement']
        trajectory_points.append({'time': 72, 'engagement': t72_engagement})
        
        # T+168 hours (1 week)
        t168_engagement = delayed_metrics['predicted_168h_engagement']
        trajectory_points.append({'time': 168, 'engagement': t168_engagement})
        
        # Calculate lifecycle characteristics
        peak_time = self._find_peak_engagement_time(trajectory_points)
        total_lifetime_value = sum(point['engagement'] for point in trajectory_points)
        engagement_persistence = t168_engagement / t0_engagement if t0_engagement > 0 else 0
        
        # Determine content type based on trajectory
        content_type = self._classify_content_type(trajectory_points, patterns)
        
        return {
            'trajectory_points': trajectory_points,
            'peak_engagement_time': peak_time,
            'total_lifetime_value': total_lifetime_value,
            'engagement_persistence': engagement_persistence,
            'content_type': content_type,
            'viral_probability': self._calculate_viral_probability(trajectory_points, patterns),
            'optimization_opportunities': self._identify_optimization_opportunities(trajectory_points, patterns)
        }
    
    def _calculate_urgency_score(self, content: str) -> float:
        """Calculate content urgency that drives immediate engagement."""
        content_lower = content.lower()
        urgency_indicators = [
            'breaking', 'urgent', 'now', 'today', 'just', 'happening', 'live',
            'limited time', 'don\'t miss', 'act fast', 'ends soon', 'last chance'
        ]
        
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in content_lower)
        return min(1.0, urgency_count * 0.3)
    
    def _calculate_hook_strength(self, content: str) -> float:
        """Calculate strength of opening hook for immediate attention."""
        content_words = content.split()
        if len(content_words) < 3:
            return 0.3
        
        first_few_words = ' '.join(content_words[:5]).lower()
        
        strong_hooks = [
            'what if', 'imagine if', 'here\'s what', 'this is why', 'you won\'t believe',
            'shocking', 'amazing', 'incredible', 'unbelievable', 'secret', 'truth',
            'everyone is wrong', 'nobody talks about', 'the real reason'
        ]
        
        hook_score = 0.5  # Base score
        for hook in strong_hooks:
            if hook in first_few_words:
                hook_score += 0.3
                break
        
        # Questions at the start get bonus
        if content.strip().startswith(('What', 'How', 'Why', 'When', 'Where', 'Who')):
            hook_score += 0.2
        
        return min(1.0, hook_score)
    
    def _calculate_platform_alignment(self, content: str, platform: str) -> float:
        """Calculate how well content aligns with platform characteristics."""
        content_length = len(content)
        
        platform_preferences = {
            'twitter': {'ideal_length': (200, 280), 'hashtag_boost': True, 'brevity_bonus': True},
            'linkedin': {'ideal_length': (1500, 2000), 'professional_bonus': True, 'story_bonus': True},
            'instagram': {'ideal_length': (2000, 2200), 'visual_bonus': True, 'hashtag_boost': True},
            'general': {'ideal_length': (150, 300), 'hashtag_boost': False, 'brevity_bonus': False}
        }
        
        prefs = platform_preferences.get(platform, platform_preferences['general'])
        min_len, max_len = prefs['ideal_length']
        
        # Length alignment
        if min_len <= content_length <= max_len:
            length_score = 1.0
        elif content_length < min_len:
            length_score = content_length / min_len
        else:
            length_score = max(0.2, 1.0 - (content_length - max_len) / max_len)
        
        return length_score
    
    def _calculate_viral_window_probability(self, immediate_score: float, platform: str) -> float:
        """Calculate probability of content going viral within platform's viral window."""
        patterns = self.platform_decay_patterns.get(platform, self.platform_decay_patterns['general'])
        viral_threshold = 0.7  # Threshold for viral consideration
        
        if immediate_score >= viral_threshold:
            return min(1.0, immediate_score * 1.2)
        else:
            return immediate_score * 0.3
    
    def _calculate_content_depth(self, content: str) -> float:
        """Calculate content depth that contributes to sustained engagement."""
        # Look for indicators of substantial content
        depth_indicators = [
            'learned', 'lesson', 'experience', 'insight', 'analysis', 'research',
            'study', 'data', 'statistics', 'because', 'therefore', 'however',
            'first', 'second', 'third', 'steps', 'process', 'framework'
        ]
        
        content_lower = content.lower()
        depth_count = sum(1 for indicator in depth_indicators if indicator in content_lower)
        
        # Length contributes to depth perception
        length_factor = min(1.0, len(content) / 500)
        
        return min(1.0, (depth_count * 0.15) + (length_factor * 0.4))
    
    def _calculate_shareability(self, content: str) -> float:
        """Calculate how shareable content is for extended reach."""
        content_lower = content.lower()
        
        shareable_elements = [
            'tips', 'advice', 'guide', 'how to', 'step by step', 'framework',
            'template', 'checklist', 'mistakes', 'lessons', 'secrets',
            'everyone should', 'share if', 'tag someone', 'thoughts?'
        ]
        
        share_score = 0.3  # Base shareability
        for element in shareable_elements:
            if element in content_lower:
                share_score += 0.2
        
        return min(1.0, share_score)
    
    def _calculate_evergreen_potential(self, content: str) -> float:
        """Calculate potential for content to remain relevant over time."""
        content_lower = content.lower()
        
        # Time-sensitive words reduce evergreen potential
        time_sensitive = [
            'today', 'yesterday', 'tomorrow', 'this week', 'last week', 'now',
            'currently', 'recently', 'just', 'breaking', 'update', 'news'
        ]
        
        # Timeless words increase evergreen potential
        timeless_words = [
            'always', 'never', 'timeless', 'classic', 'fundamental', 'principle',
            'truth', 'wisdom', 'lesson', 'strategy', 'approach', 'mindset'
        ]
        
        evergreen_score = 0.5  # Base score
        
        # Penalty for time-sensitive content
        time_penalty = sum(1 for word in time_sensitive if word in content_lower)
        evergreen_score -= time_penalty * 0.1
        
        # Bonus for timeless content
        timeless_bonus = sum(1 for word in timeless_words if word in content_lower)
        evergreen_score += timeless_bonus * 0.15
        
        return max(0.0, min(1.0, evergreen_score))
    
    def _calculate_discussion_potential(self, content: str) -> float:
        """Calculate potential for generating ongoing discussion."""
        content_lower = content.lower()
        
        discussion_drivers = [
            '?', 'what do you think', 'thoughts', 'opinion', 'agree', 'disagree',
            'controversial', 'debate', 'discuss', 'comment', 'perspective',
            'experience', 'story', 'mistake', 'failure', 'success'
        ]
        
        discussion_score = 0.2  # Base score
        for driver in discussion_drivers:
            if driver in content_lower:
                discussion_score += 0.2
        
        return min(1.0, discussion_score)
    
    def _find_peak_engagement_time(self, trajectory_points: List[Dict[str, float]]) -> int:
        """Find the time when engagement peaks."""
        max_engagement = max(point['engagement'] for point in trajectory_points)
        for point in trajectory_points:
            if point['engagement'] == max_engagement:
                return point['time']
        return 0
    
    def _classify_content_type(self, trajectory_points: List[Dict[str, float]], 
                             patterns: Dict[str, Any]) -> str:
        """Classify content type based on engagement trajectory."""
        engagements = [point['engagement'] for point in trajectory_points]
        
        # Find peak and analyze pattern
        peak_index = engagements.index(max(engagements))
        
        if peak_index == 0:
            # Immediate peak, fast decay
            return 'flash_viral'
        elif peak_index <= 2:
            # Early peak with moderate decay
            return 'trending'
        elif engagements[-1] > engagements[0] * 0.5:
            # Sustained engagement
            return 'evergreen'
        elif sum(engagements[2:]) > sum(engagements[:2]):
            # Slow burn, builds over time
            return 'slow_burn'
        else:
            # Standard decay pattern
            return 'standard'
    
    def _calculate_viral_probability(self, trajectory_points: List[Dict[str, float]], 
                                   patterns: Dict[str, Any]) -> float:
        """Calculate overall viral probability based on trajectory."""
        # Viral content typically has high immediate engagement and sustained momentum
        immediate_engagement = trajectory_points[0]['engagement'] if trajectory_points else 0
        early_momentum = trajectory_points[1]['engagement'] if len(trajectory_points) > 1 else 0
        
        viral_score = (immediate_engagement * 0.6) + (early_momentum * 0.4)
        
        # Viral threshold varies by platform
        viral_threshold = patterns.get('viral_threshold', 0.7)
        
        if viral_score >= viral_threshold:
            return min(1.0, viral_score * 1.3)
        else:
            return viral_score * 0.5
    
    def _identify_optimization_opportunities(self, trajectory_points: List[Dict[str, float]], 
                                           patterns: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify specific optimization opportunities based on trajectory."""
        opportunities = []
        
        if len(trajectory_points) < 2:
            return opportunities
        
        immediate_engagement = trajectory_points[0]['engagement']
        hour_1_engagement = trajectory_points[1]['engagement']
        
        # Low immediate engagement
        if immediate_engagement < 0.4:
            opportunities.append({
                'type': 'timing_optimization',
                'message': 'Consider posting during peak hours for better immediate engagement',
                'priority': 'high'
            })
        
        # Poor 1-hour retention
        if hour_1_engagement < immediate_engagement * 0.6:
            opportunities.append({
                'type': 'hook_improvement',
                'message': 'Strengthen opening hook to maintain early engagement',
                'priority': 'medium'
            })
        
        # Low sustained engagement
        if len(trajectory_points) > 3 and trajectory_points[3]['engagement'] < 0.2:
            opportunities.append({
                'type': 'content_depth',
                'message': 'Add more substantial content for sustained engagement',
                'priority': 'medium'
            })
        
        return opportunities
    
    def evaluate(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform Level 2 Temporal evaluation.
        
        Args:
            content: Content to evaluate
            context: Evaluation context
            
        Returns:
            Comprehensive temporal analysis
        """
        start_time = time.time()
        
        # Extract context
        platform = context.get('platform', 'general') if context else 'general'
        post_time = context.get('post_time') if context else None
        
        # Calculate immediate and delayed metrics
        immediate_metrics = self.calculate_immediate_metrics(content, context)
        delayed_metrics = self.calculate_delayed_metrics(content, context)
        
        # Generate lifecycle prediction
        lifecycle_prediction = self.calculate_lifecycle_prediction(
            immediate_metrics, delayed_metrics, platform
        )
        
        # Calculate overall temporal score
        temporal_score = (
            immediate_metrics['immediate_score'] * 0.4 +
            delayed_metrics['delayed_score'] * 0.4 +
            lifecycle_prediction['engagement_persistence'] * 0.2
        )
        
        # Generate temporal recommendations
        recommendations = self._generate_temporal_recommendations(
            immediate_metrics, delayed_metrics, lifecycle_prediction, platform
        )
        
        # Performance tracking
        evaluation_time = (time.time() - start_time) * 1000
        
        result = {
            'temporal_score': round(temporal_score, 3),
            'immediate_metrics': {k: round(v, 3) if isinstance(v, float) else v 
                                for k, v in immediate_metrics.items()},
            'delayed_metrics': {k: round(v, 3) if isinstance(v, float) else v 
                              for k, v in delayed_metrics.items()},
            'lifecycle_prediction': lifecycle_prediction,
            'recommendations': recommendations,
            'evaluation_metadata': {
                'level': 2,
                'evaluator': 'temporal',
                'evaluation_time': evaluation_time,
                'platform': platform,
                'time_windows': self.time_windows
            }
        }
        
        # Store performance history
        if not hasattr(self, 'performance_history'):
            self.performance_history = []
        self.performance_history.append(evaluation_time)
        
        return result
    
    def _generate_temporal_recommendations(self, immediate_metrics: Dict[str, float],
                                         delayed_metrics: Dict[str, float],
                                         lifecycle_prediction: Dict[str, Any],
                                         platform: str) -> List[Dict[str, Any]]:
        """Generate temporal optimization recommendations."""
        recommendations = []
        
        # Timing optimization
        if immediate_metrics['timing_optimization'] < 0.7:
            recommendations.append({
                'type': 'timing_optimization',
                'priority': 'high',
                'message': f'Timing score ({immediate_metrics["timing_optimization"]:.2f}) suggests posting during peak hours',
                'suggestions': [
                    f'Post during peak hours for {platform}',
                    'Consider time zone differences for your audience',
                    'Test different posting times and track performance'
                ]
            })
        
        # Hook improvement for immediate engagement
        if immediate_metrics['hook_strength'] < 0.5:
            recommendations.append({
                'type': 'hook_improvement',
                'priority': 'medium',
                'message': f'Hook strength ({immediate_metrics["hook_strength"]:.2f}) could be improved for better immediate engagement',
                'suggestions': [
                    'Start with a compelling question or statement',
                    'Use curiosity-driven opening lines',
                    'Create urgency in the first few words'
                ]
            })
        
        # Content depth for sustained engagement
        if delayed_metrics['content_depth'] < 0.4:
            recommendations.append({
                'type': 'content_depth',
                'priority': 'medium',
                'message': f'Content depth ({delayed_metrics["content_depth"]:.2f}) could be enhanced for sustained engagement',
                'suggestions': [
                    'Add more substantial insights or analysis',
                    'Include specific examples or case studies',
                    'Provide actionable steps or frameworks'
                ]
            })
        
        # Viral optimization
        if lifecycle_prediction['viral_probability'] > 0.6:
            recommendations.append({
                'type': 'viral_amplification',
                'priority': 'high',
                'message': f'High viral probability ({lifecycle_prediction["viral_probability"]:.2f}) - optimize for maximum reach',
                'suggestions': [
                    'Prepare follow-up content to capitalize on momentum',
                    'Engage actively with early comments and shares',
                    'Consider cross-platform promotion'
                ]
            })
        
        return recommendations
    
    def batch_evaluate(self, content_list: List[str], context_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple content pieces efficiently.
        
        Args:
            content_list: List of content to evaluate
            context_list: Optional list of contexts
            
        Returns:
            List of temporal evaluation results
        """
        if context_list is None:
            context_list = [None] * len(content_list)
        
        results = []
        for content, context in zip(content_list, context_list):
            try:
                result = self.evaluate(content, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in temporal evaluation: {e}")
                results.append({
                    'error': str(e),
                    'content_preview': content[:50] + '...' if len(content) > 50 else content
                })
        
        return results