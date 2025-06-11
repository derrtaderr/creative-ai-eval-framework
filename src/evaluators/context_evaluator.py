"""
Content Context Evaluator (Level 0)

Evaluates brand voice consistency and platform optimization for AI-generated content.
This is the foundational layer that ensures content maintains creator authenticity
and is optimized for specific social media platforms.
"""

import json
import time
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta

from .base_evaluator import BaseEvaluator


class ContentContextEvaluator(BaseEvaluator):
    """
    Level 0: Context Evaluation
    
    Ensures AI-generated content maintains brand voice and platform optimization.
    
    Key Features:
    - Voice consistency using TF-IDF similarity (fallback when sentence transformers not available)
    - Platform-specific optimization scoring
    - Real-time trend relevance assessment
    - Creator profiling and historical analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Content Context Evaluator.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - voice_model: Model type to use ('tfidf' or 'sentence_transformer')
                - platform_configs: Platform-specific configurations
                - trend_api_key: API key for trend data
                - voice_threshold: Minimum voice consistency threshold
        """
        super().__init__(config)
        
        # Initialize voice similarity method
        voice_model_type = self.config.get('voice_model', 'tfidf')
        
        if voice_model_type == 'sentence_transformer':
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('sentence_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                self.voice_model = SentenceTransformer(model_name)
                self.voice_model_type = 'sentence_transformer'
                self.logger.info(f"Loaded sentence transformer: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}. Falling back to TF-IDF.")
                self.voice_model_type = 'tfidf'
                self.voice_model = None
        else:
            self.voice_model_type = 'tfidf'
            self.voice_model = None
        
        # Initialize TF-IDF vectorizer for keyword analysis and voice similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Platform configurations
        self.platform_configs = self._load_platform_configs()
        
        # Voice consistency threshold
        self.voice_threshold = self.config.get('voice_threshold', 0.7)
        
        # Trend API configuration
        self.trend_api_key = self.config.get('trend_api_key')
        
    def _load_platform_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load platform-specific configuration."""
        default_configs = {
            'twitter': {
                'character_limit': 280,
                'optimal_hashtags': (1, 2),
                'best_posting_times': [9, 13, 17, 20],
                'engagement_weights': {'likes': 1.0, 'retweets': 2.0, 'replies': 1.5},
                'optimal_length': (50, 200),
                'hashtag_boost': 1.1,
                'mention_boost': 1.05
            },
            'linkedin': {
                'character_limit': 3000,
                'optimal_hashtags': (3, 5),
                'best_posting_times': [8, 12, 17, 18],
                'engagement_weights': {'likes': 1.0, 'comments': 3.0, 'shares': 4.0, 'clicks': 2.0},
                'optimal_length': (150, 1000),
                'professional_keywords': ['insights', 'leadership', 'growth', 'strategy'],
                'hashtag_boost': 1.2,
                'professional_tone_boost': 1.15
            },
            'instagram': {
                'character_limit': 2200,
                'optimal_hashtags': (5, 10),
                'best_posting_times': [11, 14, 17, 19],
                'engagement_weights': {'likes': 1.0, 'comments': 2.0, 'shares': 3.0, 'saves': 4.0},
                'optimal_length': (100, 500),
                'hashtag_boost': 1.3,
                'visual_content_required': True
            }
        }
        
        # Override with user-provided configs
        user_configs = self.config.get('platform_configs', {})
        for platform, config in user_configs.items():
            if platform in default_configs:
                default_configs[platform].update(config)
            else:
                default_configs[platform] = config
        
        return default_configs
    
    def load_creator_profile(self, profile_path: str) -> Dict[str, Any]:
        """
        Load and parse creator voice data.
        
        Args:
            profile_path: Path to creator profile JSON file
            
        Returns:
            Creator profile dictionary
        """
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            # Generate voice embedding if historical content exists
            if 'historical_content' in profile:
                historical_texts = [item['text'] for item in profile['historical_content']]
                if historical_texts:
                    profile['voice_embedding'] = self._generate_voice_embedding(historical_texts)
            
            self.logger.info(f"Loaded creator profile: {profile.get('name', 'Unknown')}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to load creator profile: {e}")
            return {}
    
    def _generate_voice_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Generate voice embedding from historical content.
        
        Args:
            texts: List of historical content texts
            
        Returns:
            Voice embedding vector
        """
        if not texts:
            return np.array([])
        
        try:
            if self.voice_model_type == 'sentence_transformer' and self.voice_model:
                embeddings = self.voice_model.encode(texts)
                # Use mean embedding as voice profile
                voice_embedding = np.mean(embeddings, axis=0)
                return voice_embedding
            else:
                # Use TF-IDF approach
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                # Use mean TF-IDF vector as voice profile
                voice_embedding = np.mean(tfidf_matrix.toarray(), axis=0)
                return voice_embedding
                
        except Exception as e:
            self.logger.error(f"Failed to generate voice embedding: {e}")
            return np.array([])
    
    def calculate_voice_consistency(self, content: str, creator_profile: Dict[str, Any]) -> float:
        """
        Calculate voice consistency score using embedding similarity.
        
        Args:
            content: Content to evaluate
            creator_profile: Creator's voice profile
            
        Returns:
            Voice consistency score (0-1)
        """
        if 'voice_embedding' not in creator_profile or len(creator_profile['voice_embedding']) == 0:
            return 0.5  # Default score if no profile
        
        try:
            if self.voice_model_type == 'sentence_transformer' and self.voice_model:
                # Generate embedding for new content
                content_embedding = self.voice_model.encode([content])
                
                # Calculate cosine similarity with creator's voice embedding
                voice_embedding = creator_profile['voice_embedding'].reshape(1, -1)
                similarity = cosine_similarity(content_embedding, voice_embedding)[0][0]
                
                # Convert to 0-1 scale (cosine similarity is -1 to 1)
                voice_score = (similarity + 1) / 2
            else:
                # Use TF-IDF approach
                # Transform new content using the same vectorizer
                try:
                    content_tfidf = self.tfidf_vectorizer.transform([content])
                    voice_embedding = creator_profile['voice_embedding'].reshape(1, -1)
                    
                    # Ensure dimensions match
                    if content_tfidf.shape[1] != voice_embedding.shape[1]:
                        # Pad or truncate to match dimensions
                        content_array = content_tfidf.toarray()[0]
                        voice_array = voice_embedding[0]
                        
                        min_len = min(len(content_array), len(voice_array))
                        content_array = content_array[:min_len]
                        voice_array = voice_array[:min_len]
                        
                        if min_len == 0:
                            return 0.5
                        
                        similarity = cosine_similarity([content_array], [voice_array])[0][0]
                    else:
                        similarity = cosine_similarity(content_tfidf, voice_embedding)[0][0]
                    
                    # Ensure similarity is in valid range
                    if np.isnan(similarity):
                        similarity = 0.0
                    
                    voice_score = max(0.0, min(1.0, similarity))
                    
                except Exception as e:
                    self.logger.warning(f"TF-IDF similarity calculation failed: {e}")
                    # Fallback to simple keyword matching
                    voice_score = self._calculate_keyword_similarity(content, creator_profile)
            
            return float(voice_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate voice consistency: {e}")
            return 0.5
    
    def _calculate_keyword_similarity(self, content: str, creator_profile: Dict[str, Any]) -> float:
        """
        Fallback method for voice similarity using keyword matching.
        
        Args:
            content: Content to evaluate
            creator_profile: Creator profile
            
        Returns:
            Voice similarity score (0-1)
        """
        try:
            # Get brand keywords from profile
            brand_keywords = creator_profile.get('voice_characteristics', {}).get('brand_keywords', [])
            
            if not brand_keywords:
                return 0.5
            
            content_lower = content.lower()
            matching_keywords = sum(1 for keyword in brand_keywords if keyword.lower() in content_lower)
            
            # Calculate similarity based on keyword overlap
            similarity = matching_keywords / len(brand_keywords)
            return min(1.0, similarity * 2)  # Boost score since keyword matching is stricter
            
        except Exception as e:
            self.logger.error(f"Keyword similarity calculation failed: {e}")
            return 0.5
    
    def assess_platform_optimization(self, content: str, platform: str = 'twitter') -> Dict[str, Any]:
        """
        Assess platform-specific optimization scoring.
        
        Args:
            content: Content to evaluate
            platform: Target platform (twitter, linkedin, instagram)
            
        Returns:
            Platform optimization scores and details
        """
        if platform not in self.platform_configs:
            return {'error': f'Platform {platform} not supported'}
        
        config = self.platform_configs[platform]
        scores = {}
        details = {}
        
        # Character count optimization
        char_count = len(content)
        char_limit = config['character_limit']
        if char_count <= char_limit:
            char_score = 1.0 - (char_count / char_limit) * 0.3  # Slight penalty for very long content
        else:
            char_score = 0.0  # Over limit
        
        scores['character_optimization'] = char_score
        details['character_count'] = char_count
        details['character_limit'] = char_limit
        
        # Length optimization
        optimal_min, optimal_max = config.get('optimal_length', (50, 200))
        if optimal_min <= char_count <= optimal_max:
            length_score = 1.0
        elif char_count < optimal_min:
            length_score = char_count / optimal_min
        else:
            length_score = max(0.5, optimal_max / char_count)
        
        scores['length_optimization'] = length_score
        
        # Hashtag optimization
        hashtags = re.findall(r'#\w+', content)
        hashtag_count = len(hashtags)
        optimal_hashtag_min, optimal_hashtag_max = config.get('optimal_hashtags', (1, 3))
        
        if optimal_hashtag_min <= hashtag_count <= optimal_hashtag_max:
            hashtag_score = 1.0
        elif hashtag_count == 0:
            hashtag_score = 0.8  # Not terrible, but suboptimal
        elif hashtag_count < optimal_hashtag_min:
            hashtag_score = 0.9
        else:
            hashtag_score = max(0.5, optimal_hashtag_max / hashtag_count)
        
        scores['hashtag_optimization'] = hashtag_score
        details['hashtag_count'] = hashtag_count
        details['hashtags'] = hashtags
        
        # Platform-specific boosts
        total_boost = 1.0
        
        if hashtag_count > 0:
            total_boost *= config.get('hashtag_boost', 1.0)
        
        # LinkedIn professional tone boost
        if platform == 'linkedin':
            professional_keywords = config.get('professional_keywords', [])
            if any(keyword in content.lower() for keyword in professional_keywords):
                total_boost *= config.get('professional_tone_boost', 1.0)
        
        # Apply boost to relevant scores
        for key in ['hashtag_optimization', 'length_optimization']:
            if key in scores:
                scores[key] = min(1.0, scores[key] * total_boost)
        
        # Calculate overall platform score
        platform_score = np.mean(list(scores.values()))
        
        return {
            'platform_score': platform_score,
            'component_scores': scores,
            'details': details,
            'recommendations': self._generate_platform_recommendations(content, platform, scores, details)
        }
    
    def _generate_platform_recommendations(self, content: str, platform: str, 
                                         scores: Dict[str, float], details: Dict[str, Any]) -> List[str]:
        """Generate platform-specific recommendations."""
        recommendations = []
        
        if scores.get('character_optimization', 1.0) < 0.8:
            if details['character_count'] > details['character_limit']:
                recommendations.append(f"Content exceeds {platform} character limit. Consider shortening.")
            else:
                recommendations.append(f"Content is quite long for {platform}. Consider condensing for better engagement.")
        
        if scores.get('hashtag_optimization', 1.0) < 0.9:
            config = self.platform_configs[platform]
            min_hashtags, max_hashtags = config.get('optimal_hashtags', (1, 3))
            current_hashtags = details.get('hashtag_count', 0)
            
            if current_hashtags < min_hashtags:
                recommendations.append(f"Add {min_hashtags - current_hashtags} more relevant hashtags for better discoverability.")
            elif current_hashtags > max_hashtags:
                recommendations.append(f"Reduce hashtags to {max_hashtags} for optimal {platform} performance.")
        
        if scores.get('length_optimization', 1.0) < 0.8:
            recommendations.append(f"Content length is not optimal for {platform}. Consider adjusting for better engagement.")
        
        return recommendations
    
    def evaluate_trend_relevance(self, content: str, platform: str = 'twitter') -> Dict[str, Any]:
        """
        Evaluate real-time trend alignment.
        
        Args:
            content: Content to evaluate
            platform: Target platform
            
        Returns:
            Trend relevance score and details
        """
        # This is a simplified implementation
        # In production, you'd integrate with Twitter API, Google Trends, etc.
        
        try:
            # Extract potential trending topics/keywords
            words = re.findall(r'\b\w+\b', content.lower())
            
            # Simulate trend scoring (in production, use real API data)
            trend_keywords = self._get_trending_keywords(platform)
            
            matching_trends = [word for word in words if word in trend_keywords]
            trend_score = min(1.0, len(matching_trends) * 0.2)
            
            return {
                'trend_score': trend_score,
                'matching_trends': matching_trends,
                'recommendations': [
                    f"Consider incorporating trending topics: {', '.join(trend_keywords[:3])}"
                ] if trend_score < 0.5 else []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate trend relevance: {e}")
            return {'trend_score': 0.5, 'error': str(e)}
    
    def _get_trending_keywords(self, platform: str) -> List[str]:
        """
        Get trending keywords for platform (simplified implementation).
        
        In production, this would integrate with:
        - Twitter API for trending topics
        - Google Trends API
        - LinkedIn trending content
        - Instagram hashtag trends
        """
        # Simplified mock data
        mock_trends = {
            'twitter': ['ai', 'tech', 'startup', 'innovation', 'productivity'],
            'linkedin': ['leadership', 'career', 'networking', 'growth', 'strategy'],
            'instagram': ['lifestyle', 'inspiration', 'motivation', 'creativity', 'wellness']
        }
        
        return mock_trends.get(platform, [])
    
    def generate_context_score(self, content: str, creator_profile: Dict[str, Any], 
                             platform: str = 'twitter') -> Dict[str, Any]:
        """
        Generate weighted composite context score.
        
        Args:
            content: Content to evaluate
            creator_profile: Creator's profile
            platform: Target platform
            
        Returns:
            Comprehensive context evaluation
        """
        start_time = time.time()
        
        # Validate inputs
        if not self.validate_content(content):
            return {'error': 'Invalid content provided'}
        
        # Component evaluations
        voice_score = self.calculate_voice_consistency(content, creator_profile)
        platform_eval = self.assess_platform_optimization(content, platform)
        trend_eval = self.evaluate_trend_relevance(content, platform)
        
        # Weighted composite score
        weights = self.config.get('context_weights', {
            'voice_consistency': 0.4,
            'platform_optimization': 0.4,
            'trend_relevance': 0.2
        })
        
        context_score = (
            voice_score * weights['voice_consistency'] +
            platform_eval.get('platform_score', 0.5) * weights['platform_optimization'] +
            trend_eval.get('trend_score', 0.5) * weights['trend_relevance']
        )
        
        # Execution time
        execution_time = time.time() - start_time
        
        result = {
            'context_score': context_score,
            'voice_consistency': voice_score,
            'platform_optimization': platform_eval.get('platform_score', 0.5),
            'trend_relevance': trend_eval.get('trend_score', 0.5),
            'platform_details': platform_eval,
            'trend_details': trend_eval,
            'recommendations': [],
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Collect all recommendations
        result['recommendations'].extend(platform_eval.get('recommendations', []))
        result['recommendations'].extend(trend_eval.get('recommendations', []))
        
        # Add voice consistency recommendations
        if voice_score < self.voice_threshold:
            result['recommendations'].append(
                f"Voice consistency ({voice_score:.2f}) is below threshold. "
                f"Consider adjusting tone to match creator's established voice."
            )
        
        # Record evaluation
        self._record_evaluation(content, creator_profile, result, execution_time)
        
        return result
    
    def evaluate(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main evaluation method (implements abstract method from BaseEvaluator).
        
        Args:
            content: Content to evaluate
            context: Context dictionary containing:
                - creator_profile: Creator profile dictionary
                - platform: Target platform (default: 'twitter')
                
        Returns:
            Context evaluation results
        """
        if context is None:
            context = {}
        
        creator_profile = context.get('creator_profile', {})
        platform = context.get('platform', 'twitter')
        
        return self.generate_context_score(content, creator_profile, platform)
    
    def evaluate_content(self, content: str, creator_profile: Dict[str, Any], 
                        platform: str = 'twitter') -> Dict[str, Any]:
        """
        Convenience method for direct content evaluation.
        
        Args:
            content: Content to evaluate
            creator_profile: Creator's profile
            platform: Target platform
            
        Returns:
            Context evaluation results
        """
        return self.generate_context_score(content, creator_profile, platform) 