"""
Authenticity Performance Evaluator (Level 1)

Evaluates the balance between brand authenticity and viral performance potential.
This is the core Level 1 evaluation that helps creators maintain their authentic voice
while maximizing engagement and reach.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from .base_evaluator import BaseEvaluator


class AuthenticityPerformanceEvaluator(BaseEvaluator):
    """
    Level 1: Authenticity vs Performance Evaluation
    
    This evaluator balances brand authenticity with viral potential by:
    1. Calculating dynamic authenticity thresholds for each creator
    2. Recognizing viral patterns and structures
    3. Predicting performance potential
    4. Providing calibrated recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Authenticity Performance Evaluator."""
        super().__init__(config)
        
        # Configuration
        self.min_authenticity_threshold = config.get('min_authenticity_threshold', 0.6) if config else 0.6
        self.performance_weight = config.get('performance_weight', 0.4) if config else 0.4
        self.authenticity_weight = config.get('authenticity_weight', 0.6) if config else 0.6
        
        # Viral pattern library
        self.viral_patterns = self._load_viral_patterns()
        
        # Performance prediction models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.performance_model = LinearRegression()
        self._is_trained = False
        
        self.logger.info("AuthenticityPerformanceEvaluator initialized with dynamic thresholds")
    
    def _load_viral_patterns(self) -> Dict[str, Any]:
        """Load viral content patterns and structures."""
        return {
            'hooks': [
                # Question hooks
                r'^\s*[?!]*\s*(what|how|why|when|where|who|which)\s+',
                r'.*[?].*',
                
                # Emotional hooks
                r'^\s*(amazing|incredible|shocking|unbelievable)',
                r'^\s*(stop|wait|hold|listen)',
                r'^\s*(this\s+will|you\s+won\'t\s+believe)',
                
                # Curiosity hooks
                r'^\s*(here\'s\s+what|here\'s\s+how|here\'s\s+why)',
                r'^\s*(the\s+secret|the\s+truth|the\s+real)',
                r'^\s*(nobody\s+talks\s+about|everyone\s+is\s+wrong)',
                
                # Personal hooks
                r'^\s*(i\s+used\s+to|i\s+thought|i\s+learned)',
                r'^\s*(my\s+biggest\s+mistake|my\s+secret)',
            ],
            
            'engagement_patterns': [
                # Call to action patterns
                r'(what\s+do\s+you\s+think|thoughts|agree|disagree)',
                r'(comment\s+below|let\s+me\s+know|tell\s+me)',
                r'(share\s+if|share\s+this|retweet\s+if)',
                r'(tag\s+someone|tag\s+a\s+friend)',
                
                # Social proof patterns
                r'(\d+%|\d+\s+out\s+of\s+\d+|majority\s+of)',
                r'(studies\s+show|research\s+shows|experts\s+say)',
                r'(most\s+people|everyone\s+knows|we\s+all\s+know)',
            ],
            
            'structure_patterns': {
                'problem_solution': r'(problem|issue|challenge).+(solution|answer|fix)',
                'before_after': r'(before|used\s+to).+(after|now|today)',
                'list_format': r'(\d+\s+(ways|reasons|tips|secrets|mistakes))',
                'story_arc': r'(once|story|happened|experience).+(learned|realized|discovered)',
            },
            
            'emotional_triggers': [
                'fear', 'excitement', 'curiosity', 'urgency', 'exclusivity',
                'social_proof', 'authority', 'scarcity', 'reciprocity'
            ]
        }
    
    def calculate_authenticity_score(self, content: str, creator_profile: Dict[str, Any]) -> float:
        """
        Calculate how authentic the content is to the creator's voice.
        
        Args:
            content: Content to evaluate
            creator_profile: Creator's historical profile and preferences
            
        Returns:
            Authenticity score (0.0 to 1.0)
        """
        if not creator_profile.get('historical_posts'):
            return 0.5  # Neutral if no history
        
        # Get creator's historical content
        historical_content = [post.get('content', '') for post in creator_profile['historical_posts']]
        
        if not historical_content:
            return 0.5
        
        try:
            # Create corpus with historical content + new content
            corpus = historical_content + [content]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate similarity between new content and historical average
            historical_vectors = tfidf_matrix[:-1]
            new_content_vector = tfidf_matrix[-1]
            
            # Calculate average historical vector
            avg_historical = np.mean(historical_vectors.toarray(), axis=0).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(new_content_vector, avg_historical)[0][0]
            
            # Normalize to 0-1 range (cosine similarity can be negative)
            authenticity_score = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return authenticity_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating authenticity score: {e}")
            return 0.5
    
    def calculate_viral_potential(self, content: str, platform: str = 'general') -> float:
        """
        Calculate viral potential based on pattern recognition.
        
        Args:
            content: Content to analyze
            platform: Target platform for optimization
            
        Returns:
            Viral potential score (0.0 to 1.0)
        """
        content_lower = content.lower()
        viral_score = 0.0
        total_factors = 0
        
        # Check for viral hooks (weighted more heavily)
        hook_score = 0
        for pattern in self.viral_patterns['hooks']:
            if re.search(pattern, content_lower):
                hook_score += 1
        
        if hook_score > 0:
            # Give higher weight to multiple hooks found
            hook_weight = min(1.0, hook_score * 0.3)  # More generous scoring
            viral_score += hook_weight * 0.3
        total_factors += 0.3
        
        # Check for engagement patterns
        engagement_score = 0
        for pattern in self.viral_patterns['engagement_patterns']:
            if re.search(pattern, content_lower):
                engagement_score += 1
        
        if engagement_score > 0:
            # Give higher weight to engagement patterns
            engagement_weight = min(1.0, engagement_score * 0.4)  # More generous scoring
            viral_score += engagement_weight * 0.3
        total_factors += 0.3
        
        # Check for structure patterns
        structure_score = 0
        for pattern_name, pattern in self.viral_patterns['structure_patterns'].items():
            if re.search(pattern, content_lower):
                structure_score += 1
        
        if structure_score > 0:
            structure_weight = min(1.0, structure_score * 0.5)  # More generous scoring
            viral_score += structure_weight * 0.2
        total_factors += 0.2
        
        # Content length optimization (platform-specific)
        length_score = self._calculate_length_optimization(content, platform)
        viral_score += length_score * 0.2
        total_factors += 0.2
        
        # Normalize
        if total_factors > 0:
            viral_score = viral_score / total_factors
        
        return max(0.0, min(1.0, viral_score))
    
    def _calculate_length_optimization(self, content: str, platform: str) -> float:
        """Calculate content length optimization score."""
        content_length = len(content)
        
        optimal_ranges = {
            'twitter': (200, 280),
            'linkedin': (1500, 2000),
            'instagram': (2000, 2200),
            'general': (150, 300)
        }
        
        min_len, max_len = optimal_ranges.get(platform, optimal_ranges['general'])
        
        if min_len <= content_length <= max_len:
            return 1.0
        elif content_length < min_len:
            return content_length / min_len
        else:
            # Penalty for being too long
            return max(0.1, 1.0 - (content_length - max_len) / max_len)
    
    def calculate_dynamic_threshold(self, creator_profile: Dict[str, Any]) -> float:
        """
        Calculate dynamic authenticity threshold based on creator's profile.
        
        Args:
            creator_profile: Creator's profile data
            
        Returns:
            Dynamic authenticity threshold
        """
        base_threshold = self.min_authenticity_threshold
        
        # Adjust based on creator's variance tolerance
        variance_tolerance = creator_profile.get('variance_tolerance', 0.5)
        
        # Adjust based on creator's growth goals
        growth_focus = creator_profile.get('growth_focus', 0.5)  # 0 = authenticity focus, 1 = growth focus
        
        # Adjust based on creator's established voice strength
        voice_strength = creator_profile.get('voice_consistency', 0.5)
        
        # Calculate dynamic threshold
        threshold_adjustment = 0.0
        
        # Higher variance tolerance = lower threshold (more flexibility)
        threshold_adjustment -= (variance_tolerance - 0.5) * 0.2
        
        # Higher growth focus = lower threshold (more performance optimization)
        threshold_adjustment -= (growth_focus - 0.5) * 0.15
        
        # Stronger voice = can afford lower threshold (more experimentation)
        if voice_strength > 0.7:
            threshold_adjustment -= 0.05
        
        dynamic_threshold = max(0.3, min(0.9, base_threshold + threshold_adjustment))
        
        return dynamic_threshold
    
    def predict_performance(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Predict performance metrics for the content.
        
        Args:
            content: Content to analyze
            context: Additional context
            
        Returns:
            Predicted performance metrics
        """
        viral_potential = self.calculate_viral_potential(
            content, 
            context.get('platform', 'general') if context else 'general'
        )
        
        # Basic performance prediction based on viral patterns
        engagement_prediction = viral_potential * 0.8 + 0.1  # Add baseline
        reach_multiplier = 1.0 + (viral_potential * 2.0)  # Viral content can double reach
        
        return {
            'predicted_engagement_rate': min(1.0, engagement_prediction),
            'predicted_reach_multiplier': reach_multiplier,
            'viral_probability': viral_potential,
            'performance_confidence': min(1.0, viral_potential + 0.3)
        }
    
    def evaluate(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform Level 1 Authenticity vs Performance evaluation.
        
        Args:
            content: Content to evaluate
            context: Evaluation context including creator profile
            
        Returns:
            Comprehensive authenticity vs performance analysis
        """
        import time
        start_time = time.time()
        
        # Extract context
        creator_profile = context.get('creator_profile', {}) if context else {}
        platform = context.get('platform', 'general') if context else 'general'
        
        # Calculate core scores
        authenticity_score = self.calculate_authenticity_score(content, creator_profile)
        viral_potential = self.calculate_viral_potential(content, platform)
        performance_prediction = self.predict_performance(content, context)
        
        # Calculate dynamic threshold
        dynamic_threshold = self.calculate_dynamic_threshold(creator_profile)
        
        # Determine if content meets authenticity requirements
        authenticity_met = authenticity_score >= dynamic_threshold
        
        # Calculate balanced score
        if authenticity_met:
            # If authenticity is met, optimize for performance
            balanced_score = (authenticity_score * self.authenticity_weight + 
                            viral_potential * self.performance_weight)
        else:
            # If authenticity is not met, heavily penalize
            balanced_score = authenticity_score * 0.5
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            content, authenticity_score, viral_potential, dynamic_threshold, creator_profile
        )
        
        # Performance tracking
        evaluation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        result = {
            'authenticity_score': round(authenticity_score, 3),
            'viral_potential': round(viral_potential, 3),
            'dynamic_threshold': round(dynamic_threshold, 3),
            'authenticity_met': authenticity_met,
            'balanced_score': round(balanced_score, 3),
            'performance_prediction': {
                k: round(v, 3) for k, v in performance_prediction.items()
            },
            'recommendations': recommendations,
            'evaluation_metadata': {
                'level': 1,
                'evaluator': 'authenticity_performance',
                'evaluation_time': evaluation_time,
                'platform': platform
            }
        }
        
        # Store in history 
        if not hasattr(self, 'performance_history'):
            self.performance_history = []
        self.performance_history.append(evaluation_time)
        
        return result
    
    def _generate_recommendations(self, content: str, authenticity_score: float, 
                                viral_potential: float, threshold: float, 
                                creator_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        if authenticity_score < threshold:
            recommendations.append({
                'type': 'authenticity_improvement',
                'priority': 'high',
                'message': f'Content authenticity ({authenticity_score:.2f}) is below your threshold ({threshold:.2f})',
                'suggestions': [
                    'Use more of your characteristic vocabulary and phrases',
                    'Incorporate your typical speaking patterns',
                    'Reference your usual topics and interests',
                    'Maintain your established tone and style'
                ]
            })
        
        if viral_potential < 0.4:
            recommendations.append({
                'type': 'performance_optimization',
                'priority': 'medium',
                'message': f'Viral potential ({viral_potential:.2f}) could be improved',
                'suggestions': [
                    'Add a compelling hook at the beginning',
                    'Include a clear call-to-action',
                    'Use more engaging structural patterns',
                    'Optimize content length for platform'
                ]
            })
        
        if authenticity_score > 0.8 and viral_potential < 0.5:
            recommendations.append({
                'type': 'balance_optimization',
                'priority': 'medium',
                'message': 'Great authenticity! You can safely experiment with more viral elements',
                'suggestions': [
                    'Try adding trending topics to your authentic voice',
                    'Experiment with popular content formats',
                    'Include more engagement-driving questions',
                    'Test different hook styles while maintaining your voice'
                ]
            })
        
        return recommendations
    
    def batch_evaluate(self, content_list: List[str], context_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple content pieces efficiently.
        
        Args:
            content_list: List of content to evaluate
            context_list: Optional list of contexts for each content piece
            
        Returns:
            List of evaluation results
        """
        if context_list is None:
            context_list = [None] * len(content_list)
        
        results = []
        for content, context in zip(content_list, context_list):
            try:
                result = self.evaluate(content, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating content: {e}")
                results.append({
                    'error': str(e),
                    'content_preview': content[:50] + '...' if len(content) > 50 else content
                })
        
        return results