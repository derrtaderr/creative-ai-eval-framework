"""
Creative AI Content Evaluation Workflow Template
Comprehensive workflow for evaluating creator content across all levels and platforms
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structured result from content evaluation"""
    level: str
    overall_score: float
    component_scores: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime
    platform_optimization: Dict[str, float]
    viral_potential: float
    authenticity_score: float

class ContentEvaluationWorkflow:
    """
    Complete workflow for evaluating creator content using all evaluation levels
    
    This template demonstrates how to integrate:
    - Level 0: Context Evaluation  
    - Level 1: Authenticity vs Performance
    - Level 2: Temporal Evaluation
    - Level 3: Multi-Modal Assessment
    """
    
    def __init__(self, config_path: str = "config/"):
        """Initialize workflow with configuration files"""
        self.config_path = Path(config_path)
        self.creator_profiles = {}
        self.platform_configs = {}
        self.content_templates = {}
        self.evaluation_history = []
        
        # Load configurations
        self._load_configurations()
        
    def _load_configurations(self):
        """Load creator profiles, platform configs, and content templates"""
        try:
            # Load creator profiles
            profiles_path = self.config_path / "creator-profiles"
            if profiles_path.exists():
                for profile_file in profiles_path.glob("*.json"):
                    with open(profile_file, 'r') as f:
                        profile_data = json.load(f)
                        profile_id = profile_data.get('creator_profile', {}).get('creator_id')
                        if profile_id:
                            self.creator_profiles[profile_id] = profile_data
                            
            # Load platform configurations  
            platforms_path = self.config_path / "platform-configs"
            if platforms_path.exists():
                for config_file in platforms_path.glob("*.json"):
                    platform_name = config_file.stem.replace('-optimization-config', '')
                    with open(config_file, 'r') as f:
                        self.platform_configs[platform_name] = json.load(f)
                        
            # Load content templates
            templates_path = self.config_path / "content-types"  
            if templates_path.exists():
                for template_file in templates_path.glob("*.json"):
                    content_type = template_file.stem.replace('-template', '')
                    with open(template_file, 'r') as f:
                        self.content_templates[content_type] = json.load(f)
                        
            logger.info(f"Loaded {len(self.creator_profiles)} creator profiles")
            logger.info(f"Loaded {len(self.platform_configs)} platform configs") 
            logger.info(f"Loaded {len(self.content_templates)} content templates")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            
    def evaluate_content(self, 
                        content: Dict,
                        creator_id: str,
                        platform: str,
                        content_type: str = "general",
                        evaluation_levels: List[str] = None) -> EvaluationResult:
        """
        Comprehensive content evaluation using specified levels
        
        Args:
            content: Content data including text, metadata, media info
            creator_id: ID of creator profile to use
            platform: Target platform for optimization
            content_type: Type of content (educational, entertainment, etc.)
            evaluation_levels: List of evaluation levels to apply
            
        Returns:
            EvaluationResult with comprehensive scores and recommendations
        """
        
        if evaluation_levels is None:
            evaluation_levels = ["level_0", "level_1", "level_2", "level_3"]
            
        logger.info(f"Evaluating content for {creator_id} on {platform}")
        
        # Get configurations
        creator_profile = self.creator_profiles.get(creator_id)
        platform_config = self.platform_configs.get(platform)
        content_template = self.content_templates.get(content_type)
        
        if not creator_profile:
            raise ValueError(f"Creator profile '{creator_id}' not found")
            
        if not platform_config:
            logger.warning(f"Platform config for '{platform}' not found, using defaults")
            
        # Initialize evaluation results
        component_scores = {}
        recommendations = []
        
        # Level 0: Context Evaluation (Foundation)
        if "level_0" in evaluation_levels:
            context_result = self._evaluate_context(content, creator_profile, platform_config)
            component_scores.update(context_result["scores"])
            recommendations.extend(context_result["recommendations"])
            
        # Level 1: Authenticity vs Performance
        if "level_1" in evaluation_levels:
            authenticity_result = self._evaluate_authenticity_performance(
                content, creator_profile, platform_config
            )
            component_scores.update(authenticity_result["scores"])
            recommendations.extend(authenticity_result["recommendations"])
            
        # Level 2: Temporal Evaluation
        if "level_2" in evaluation_levels:
            temporal_result = self._evaluate_temporal_patterns(
                content, creator_profile, platform_config
            )
            component_scores.update(temporal_result["scores"])
            recommendations.extend(temporal_result["recommendations"])
            
        # Level 3: Multi-Modal Assessment
        if "level_3" in evaluation_levels:
            multimodal_result = self._evaluate_multimodal(
                content, creator_profile, platform_config
            )
            component_scores.update(multimodal_result["scores"])
            recommendations.extend(multimodal_result["recommendations"])
            
        # Calculate overall score
        overall_score = self._calculate_overall_score(component_scores)
        
        # Generate platform-specific optimization scores
        platform_optimization = self._calculate_platform_optimization(
            content, platform_config, content_template
        )
        
        # Create result
        result = EvaluationResult(
            level=f"multi_level_{len(evaluation_levels)}",
            overall_score=overall_score,
            component_scores=component_scores,
            recommendations=recommendations,
            timestamp=datetime.now(),
            platform_optimization=platform_optimization,
            viral_potential=component_scores.get("viral_potential", 0.0),
            authenticity_score=component_scores.get("authenticity_score", 0.0)
        )
        
        # Store in history
        self.evaluation_history.append(result)
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.3f}")
        return result
        
    def _evaluate_context(self, content: Dict, creator_profile: Dict, platform_config: Dict) -> Dict:
        """Level 0: Context evaluation implementation"""
        scores = {}
        recommendations = []
        
        # Context clarity analysis
        text_content = content.get("text", "")
        context_clarity = self._analyze_context_clarity(text_content)
        scores["context_clarity"] = context_clarity
        
        if context_clarity < 0.7:
            recommendations.append("Improve content clarity - consider adding context or explanation")
            
        # Relevance to creator profile
        relevance = self._analyze_content_relevance(content, creator_profile)
        scores["content_relevance"] = relevance
        
        if relevance < 0.6:
            recommendations.append("Content may not align well with your established expertise areas")
            
        # Quality thresholds check
        quality_score = self._check_quality_thresholds(content, creator_profile)
        scores["quality_threshold"] = quality_score
        
        return {
            "scores": scores,
            "recommendations": recommendations
        }
        
    def _evaluate_authenticity_performance(self, content: Dict, creator_profile: Dict, platform_config: Dict) -> Dict:
        """Level 1: Authenticity vs Performance evaluation"""
        scores = {}
        recommendations = []
        
        # Voice consistency analysis
        voice_consistency = self._analyze_voice_consistency(content, creator_profile)
        scores["voice_consistency"] = voice_consistency
        
        # Viral pattern analysis
        viral_patterns = self._analyze_viral_patterns(content, creator_profile)
        scores["viral_patterns"] = viral_patterns
        
        # Balance optimization
        balance_score = self._calculate_authenticity_performance_balance(
            voice_consistency, viral_patterns, creator_profile
        )
        scores["authenticity_score"] = balance_score
        
        # Generate recommendations
        if voice_consistency < creator_profile.get("creator_profile", {}).get("authenticity_settings", {}).get("variance_tolerance", 0.7):
            recommendations.append("Content voice differs significantly from your established style")
            
        if viral_patterns > 0.8 and balance_score < 0.6:
            recommendations.append("High viral potential but may compromise authenticity - consider adjustments")
            
        return {
            "scores": scores,
            "recommendations": recommendations
        }
        
    def _evaluate_temporal_patterns(self, content: Dict, creator_profile: Dict, platform_config: Dict) -> Dict:
        """Level 2: Temporal evaluation with viral lifecycle prediction"""
        scores = {}
        recommendations = []
        
        # Predict viral lifecycle
        viral_lifecycle = self._predict_viral_lifecycle(content, platform_config)
        scores["viral_lifecycle"] = viral_lifecycle
        
        # Growth trajectory analysis
        growth_potential = self._analyze_growth_trajectory(content, creator_profile)
        scores["growth_potential"] = growth_potential
        
        # Optimal timing analysis
        timing_score = self._analyze_optimal_timing(content, platform_config)
        scores["timing_optimization"] = timing_score
        
        # Temporal recommendations
        if viral_lifecycle < 48:  # hours
            recommendations.append("Content may have limited viral sustainability - consider evergreen elements")
            
        if timing_score < 0.6:
            recommendations.append("Consider posting at more optimal times for your platform")
            
        return {
            "scores": scores,
            "recommendations": recommendations
        }
        
    def _evaluate_multimodal(self, content: Dict, creator_profile: Dict, platform_config: Dict) -> Dict:
        """Level 3: Multi-modal assessment"""
        scores = {}
        recommendations = []
        
        # Individual modality scores
        if "video" in content:
            video_score = self._analyze_video_quality(content["video"], platform_config)
            scores["video_quality"] = video_score
            
        if "audio" in content:
            audio_score = self._analyze_audio_quality(content["audio"], platform_config)
            scores["audio_quality"] = audio_score
            
        if "image" in content:
            image_score = self._analyze_image_quality(content["image"], platform_config)
            scores["image_quality"] = image_score
            
        # Cross-modal coherence
        coherence_score = self._analyze_cross_modal_coherence(content)
        scores["cross_modal_coherence"] = coherence_score
        
        # Platform-specific optimization
        platform_opt = self._analyze_platform_specific_optimization(content, platform_config)
        scores["platform_optimization"] = platform_opt
        
        # Multi-modal recommendations
        if coherence_score < 0.7:
            recommendations.append("Improve alignment between different content modalities")
            
        if platform_opt < 0.6:
            recommendations.append(f"Optimize content format for {platform_config.get('platform_metadata', {}).get('platform_name', 'platform')} specifications")
            
        return {
            "scores": scores,
            "recommendations": recommendations
        }
        
    def _analyze_context_clarity(self, text: str) -> float:
        """Analyze clarity of content context"""
        if not text:
            return 0.3
            
        # Simple heuristics - replace with more sophisticated NLP
        clarity_indicators = [
            len(text.split()) > 10,  # Sufficient length
            any(word in text.lower() for word in ["how", "what", "why", "learn", "tip"]),  # Instructional words
            text.count("?") <= 2,  # Not too many questions
            text.count("!") <= 3,  # Not overly excited
        ]
        
        return sum(clarity_indicators) / len(clarity_indicators)
        
    def _analyze_content_relevance(self, content: Dict, creator_profile: Dict) -> float:
        """Analyze relevance to creator's expertise"""
        text_content = content.get("text", "").lower()
        
        expertise_areas = creator_profile.get("creator_profile", {}).get("voice_characteristics", {}).get("expertise_areas", [])
        brand_keywords = creator_profile.get("creator_profile", {}).get("voice_characteristics", {}).get("brand_keywords", [])
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in brand_keywords + expertise_areas if keyword.lower() in text_content)
        total_keywords = len(brand_keywords + expertise_areas)
        
        if total_keywords == 0:
            return 0.5  # Neutral if no keywords defined
            
        return min(keyword_matches / total_keywords * 2, 1.0)  # Scale to max 1.0
        
    def _check_quality_thresholds(self, content: Dict, creator_profile: Dict) -> float:
        """Check against creator's quality thresholds"""
        thresholds = creator_profile.get("creator_profile", {}).get("quality_thresholds", {})
        
        # Basic quality checks
        quality_scores = []
        
        # Content length check
        text_length = len(content.get("text", ""))
        if text_length > 0:
            quality_scores.append(min(text_length / 100, 1.0))  # Normalize to 100 chars
            
        # Engagement potential
        engagement_indicators = ["?", "!", "tip", "how", "why", "learn"]
        text_lower = content.get("text", "").lower()
        engagement_score = sum(1 for indicator in engagement_indicators if indicator in text_lower) / len(engagement_indicators)
        quality_scores.append(engagement_score)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
    def _analyze_voice_consistency(self, content: Dict, creator_profile: Dict) -> float:
        """Analyze consistency with creator's established voice"""
        # Simplified voice analysis - replace with more sophisticated NLP
        text_content = content.get("text", "").lower()
        
        voice_chars = creator_profile.get("creator_profile", {}).get("voice_characteristics", {})
        brand_keywords = voice_chars.get("brand_keywords", [])
        
        # Keyword consistency
        keyword_matches = sum(1 for keyword in brand_keywords if keyword.lower() in text_content)
        keyword_score = min(keyword_matches / max(len(brand_keywords), 1) * 2, 1.0)
        
        # Tone consistency (simplified)
        tone = voice_chars.get("tone", "")
        tone_indicators = {
            "professional": ["insights", "analysis", "data", "results"],
            "casual": ["hey", "awesome", "cool", "fun"],
            "inspirational": ["inspire", "amazing", "transform", "growth"]
        }
        
        tone_words = tone_indicators.get(tone.split("_")[0], [])
        tone_score = sum(1 for word in tone_words if word in text_content) / max(len(tone_words), 1)
        
        return (keyword_score + min(tone_score, 1.0)) / 2
        
    def _analyze_viral_patterns(self, content: Dict, creator_profile: Dict) -> float:
        """Analyze viral pattern usage in content"""
        text_content = content.get("text", "").lower()
        
        # Common viral patterns
        viral_patterns = {
            "question_hook": ["what if", "why do", "how many", "?"],
            "curiosity_gap": ["you won't believe", "shocking", "secret", "hidden"],
            "contrarian_take": ["actually", "contrary", "opposite", "wrong about"],
            "transformation": ["before", "after", "changed", "transformed"],
            "list_format": ["ways to", "steps to", "tips for", "secrets"]
        }
        
        pattern_scores = []
        for pattern_type, indicators in viral_patterns.items():
            pattern_score = sum(1 for indicator in indicators if indicator in text_content)
            pattern_scores.append(min(pattern_score, 1))
            
        return sum(pattern_scores) / len(pattern_scores)
        
    def _calculate_authenticity_performance_balance(self, voice_consistency: float, viral_patterns: float, creator_profile: Dict) -> float:
        """Calculate balanced authenticity-performance score"""
        auth_settings = creator_profile.get("creator_profile", {}).get("authenticity_settings", {})
        perf_goals = creator_profile.get("creator_profile", {}).get("performance_goals", {})
        
        # Weight based on creator preferences
        auth_weight = auth_settings.get("voice_consistency_weight", 0.7)
        viral_weight = perf_goals.get("viral_willingness", 0.5)
        
        # Calculate weighted balance
        authenticity_component = voice_consistency * auth_weight
        performance_component = viral_patterns * viral_weight
        
        # Balance factor - penalize extreme imbalance
        balance_factor = 1 - abs(authenticity_component - performance_component) * 0.5
        
        return (authenticity_component + performance_component) * balance_factor / 2
        
    def _predict_viral_lifecycle(self, content: Dict, platform_config: Dict) -> float:
        """Predict viral lifecycle duration in hours"""
        # Simplified lifecycle prediction
        base_lifecycle = 24  # Base 24 hours
        
        # Platform-specific modifiers
        platform_name = platform_config.get("platform_metadata", {}).get("platform_name", "").lower()
        
        platform_modifiers = {
            "tiktok": 0.5,    # Shorter lifecycle
            "instagram": 0.8,  # Medium lifecycle
            "linkedin": 2.0,   # Longer lifecycle
            "youtube": 3.0     # Longest lifecycle
        }
        
        modifier = platform_modifiers.get(platform_name, 1.0)
        return base_lifecycle * modifier
        
    def _analyze_growth_trajectory(self, content: Dict, creator_profile: Dict) -> float:
        """Analyze potential for follower growth"""
        # Simplified growth analysis
        text_content = content.get("text", "").lower()
        
        growth_indicators = [
            "follow for", "subscribe", "share if", "tag someone",
            "save this", "try this", "learn more"
        ]
        
        growth_score = sum(1 for indicator in growth_indicators if indicator in text_content)
        return min(growth_score / 3, 1.0)  # Normalize to max 1.0
        
    def _analyze_optimal_timing(self, content: Dict, platform_config: Dict) -> float:
        """Analyze timing optimization potential"""
        # Simplified timing analysis
        current_hour = datetime.now().hour
        
        peak_hours = platform_config.get("algorithm_optimization", {}).get("posting_optimization", {}).get("peak_hours", [9, 12, 17, 19, 21])
        
        # Calculate distance from peak hours
        distances = [abs(current_hour - peak) for peak in peak_hours]
        min_distance = min(distances)
        
        # Score based on proximity to peak hours (closer = better)
        return max(0, 1 - min_distance / 12)  # 12 hours max distance
        
    def _analyze_video_quality(self, video_data: Dict, platform_config: Dict) -> float:
        """Analyze video quality metrics"""
        # Placeholder for video analysis
        # In real implementation, analyze resolution, duration, frame rate, etc.
        
        required_specs = platform_config.get("technical_specifications", {}).get("video_requirements", {})
        
        quality_score = 0.8  # Placeholder
        return quality_score
        
    def _analyze_audio_quality(self, audio_data: Dict, platform_config: Dict) -> float:
        """Analyze audio quality metrics"""
        # Placeholder for audio analysis
        quality_score = 0.8  # Placeholder
        return quality_score
        
    def _analyze_image_quality(self, image_data: Dict, platform_config: Dict) -> float:
        """Analyze image quality metrics"""
        # Placeholder for image analysis
        quality_score = 0.8  # Placeholder
        return quality_score
        
    def _analyze_cross_modal_coherence(self, content: Dict) -> float:
        """Analyze coherence between different content modalities"""
        # Simplified coherence analysis
        coherence_score = 0.8  # Placeholder
        return coherence_score
        
    def _analyze_platform_specific_optimization(self, content: Dict, platform_config: Dict) -> float:
        """Analyze platform-specific optimization"""
        optimization_score = 0.7  # Placeholder
        return optimization_score
        
    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        if not component_scores:
            return 0.0
            
        # Weight different components
        weights = {
            "context_clarity": 0.15,
            "content_relevance": 0.15,
            "authenticity_score": 0.25,
            "viral_patterns": 0.20,
            "cross_modal_coherence": 0.15,
            "platform_optimization": 0.10
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for component, score in component_scores.items():
            weight = weights.get(component, 0.1)  # Default weight
            weighted_sum += score * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    def _calculate_platform_optimization(self, content: Dict, platform_config: Dict, content_template: Dict) -> Dict[str, float]:
        """Calculate platform-specific optimization scores"""
        optimization_scores = {}
        
        # Analyze against platform requirements
        if platform_config:
            platform_name = platform_config.get("platform_metadata", {}).get("platform_name", "unknown")
            
            # Technical specifications compliance
            tech_score = 0.8  # Placeholder
            optimization_scores[f"{platform_name}_technical"] = tech_score
            
            # Algorithm optimization
            algo_score = 0.7  # Placeholder
            optimization_scores[f"{platform_name}_algorithm"] = algo_score
            
            # Audience preferences
            audience_score = 0.75  # Placeholder
            optimization_scores[f"{platform_name}_audience"] = audience_score
            
        return optimization_scores
        
    def generate_report(self, result: EvaluationResult) -> str:
        """Generate detailed evaluation report"""
        report = f"""
=== CONTENT EVALUATION REPORT ===
Evaluation Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Evaluation Level: {result.level}
Overall Score: {result.overall_score:.3f}/1.000

=== COMPONENT SCORES ===
"""
        
        for component, score in result.component_scores.items():
            report += f"{component.replace('_', ' ').title()}: {score:.3f}\n"
            
        report += f"""
=== PLATFORM OPTIMIZATION ===
"""
        for platform, score in result.platform_optimization.items():
            report += f"{platform.replace('_', ' ').title()}: {score:.3f}\n"
            
        report += f"""
=== KEY METRICS ===
Viral Potential: {result.viral_potential:.3f}
Authenticity Score: {result.authenticity_score:.3f}

=== RECOMMENDATIONS ===
"""
        
        for i, recommendation in enumerate(result.recommendations, 1):
            report += f"{i}. {recommendation}\n"
            
        return report
        
    def batch_evaluate(self, contents: List[Dict]) -> List[EvaluationResult]:
        """Evaluate multiple content pieces in batch"""
        results = []
        
        for i, content in enumerate(contents):
            logger.info(f"Evaluating content {i+1}/{len(contents)}")
            
            # Extract evaluation parameters from content
            creator_id = content.get("creator_id", "default")
            platform = content.get("platform", "general")
            content_type = content.get("content_type", "general")
            
            try:
                result = self.evaluate_content(
                    content=content,
                    creator_id=creator_id,
                    platform=platform,
                    content_type=content_type
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating content {i+1}: {e}")
                continue
                
        return results
        
    def export_results(self, results: List[EvaluationResult], filename: str):
        """Export evaluation results to JSON file"""
        export_data = []
        
        for result in results:
            export_data.append({
                "timestamp": result.timestamp.isoformat(),
                "level": result.level,
                "overall_score": result.overall_score,
                "component_scores": result.component_scores,
                "platform_optimization": result.platform_optimization,
                "viral_potential": result.viral_potential,
                "authenticity_score": result.authenticity_score,
                "recommendations": result.recommendations
            })
            
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(results)} results to {filename}")

# Example usage and testing
def example_usage():
    """Example of how to use the ContentEvaluationWorkflow"""
    
    # Initialize workflow
    workflow = ContentEvaluationWorkflow(config_path="templates/")
    
    # Example content
    sample_content = {
        "text": "Here's a productivity tip that changed my startup journey: Start with user interviews before building anything. This simple habit helped us avoid 6 months of building features nobody wanted. What's one research habit that improved your work?",
        "platform": "linkedin",
        "content_type": "educational",
        "creator_id": "tech_founder_template",
        "metadata": {
            "hashtags": ["#productivity", "#startup", "#userresearch"],
            "mentions": [],
            "links": []
        }
    }
    
    # Evaluate content
    result = workflow.evaluate_content(
        content=sample_content,
        creator_id="tech_founder_template",
        platform="linkedin",
        content_type="educational"
    )
    
    # Generate report
    report = workflow.generate_report(result)
    print(report)
    
    # Batch evaluation example
    batch_contents = [sample_content, sample_content]  # Add more content
    batch_results = workflow.batch_evaluate(batch_contents)
    
    # Export results
    workflow.export_results([result], "evaluation_results.json")

if __name__ == "__main__":
    example_usage()