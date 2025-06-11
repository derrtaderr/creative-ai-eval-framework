"""
Multi-Modal Evaluator (Level 3)

Comprehensive evaluation of video, audio, image, and text content
for advanced creative AI assessment.

Key Capabilities:
1. Video Content Analysis (frame-by-frame, pacing, storytelling)
2. Audio Processing (voice tone, music, sound design)
3. Image Assessment (composition, brand consistency, emotional impact)
4. Cross-Modal Coherence (how elements work together)
5. Platform-Specific Optimization (TikTok/YouTube/Instagram formats)
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import numpy as np
from .base_evaluator import BaseEvaluator


class MultiModalEvaluator(BaseEvaluator):
    """
    Level 3: Multi-Modal Assessment
    
    Evaluates creative content across multiple modalities:
    - Video: Frame analysis, pacing, visual storytelling
    - Audio: Voice tone, music, sound design, engagement
    - Image: Composition, brand consistency, emotional impact
    - Text: Context integration with visual/audio elements
    - Cross-Modal: Coherence and synergy between modalities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Multi-Modal Evaluator."""
        super().__init__(config)
        
        # Multi-modal processing configuration
        self.video_config = self._load_video_config()
        self.audio_config = self._load_audio_config()
        self.image_config = self._load_image_config()
        self.cross_modal_config = self._load_cross_modal_config()
        
        # Platform-specific requirements
        self.platform_requirements = self._load_platform_requirements()
        
        # Analysis models (in production, these would be actual ML models)
        self.analysis_models = self._initialize_analysis_models()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("MultiModalEvaluator initialized with video/audio/image/text analysis")
    
    def _load_video_config(self) -> Dict[str, Any]:
        """Load video analysis configuration."""
        return {
            'frame_analysis': {
                'sample_rate': 1.0,  # Analyze every second
                'quality_metrics': ['composition', 'lighting', 'color_balance', 'sharpness'],
                'motion_detection': True,
                'scene_change_detection': True
            },
            'pacing_analysis': {
                'optimal_shot_length': {'min': 1.5, 'max': 8.0},  # seconds
                'transition_smoothness': True,
                'rhythm_detection': True
            },
            'storytelling_elements': {
                'hook_timing': 3.0,  # First 3 seconds critical
                'narrative_arc': True,
                'emotional_progression': True,
                'call_to_action_placement': True
            },
            'technical_quality': {
                'resolution_standards': {'min': 720, 'optimal': 1080},
                'aspect_ratios': {
                    'tiktok': '9:16',
                    'youtube_shorts': '9:16', 
                    'youtube_standard': '16:9',
                    'instagram_story': '9:16',
                    'instagram_reel': '9:16',
                    'instagram_post': '1:1'
                }
            }
        }
    
    def _load_audio_config(self) -> Dict[str, Any]:
        """Load audio analysis configuration."""
        return {
            'voice_analysis': {
                'tone_detection': ['enthusiastic', 'calm', 'urgent', 'friendly', 'authoritative'],
                'pace_analysis': {'optimal_wpm': 150, 'range': [120, 180]},
                'clarity_metrics': ['pronunciation', 'volume_consistency', 'background_noise'],
                'emotional_markers': ['excitement', 'confidence', 'empathy', 'humor']
            },
            'music_analysis': {
                'genre_detection': True,
                'mood_analysis': ['upbeat', 'dramatic', 'calming', 'energetic', 'inspiring'],
                'volume_balance': {'voice_to_music_ratio': 0.7},
                'timing_sync': True,  # Music sync with visual elements
                'copyright_safe': True
            },
            'sound_design': {
                'sfx_appropriateness': True,
                'audio_branding': True,  # Consistent sound identity
                'silence_usage': True,  # Strategic pause analysis
                'dynamic_range': True   # Audio level variation
            },
            'technical_audio': {
                'sample_rate': 44100,
                'bit_depth': 16,
                'format_requirements': ['mp3', 'aac', 'wav'],
                'loudness_standards': {'lufs': -23}  # Broadcast standard
            }
        }
    
    def _load_image_config(self) -> Dict[str, Any]:
        """Load image analysis configuration."""
        return {
            'composition_analysis': {
                'rule_of_thirds': True,
                'leading_lines': True,
                'symmetry_balance': True,
                'focal_point_clarity': True,
                'depth_of_field': True
            },
            'visual_branding': {
                'color_consistency': True,
                'font_usage': True,
                'logo_placement': True,
                'brand_guidelines_adherence': True,
                'visual_hierarchy': True
            },
            'emotional_impact': {
                'color_psychology': True,
                'facial_expression_analysis': True,
                'body_language_reading': True,
                'scene_mood_detection': True,
                'cultural_sensitivity': True
            },
            'technical_image': {
                'resolution_quality': True,
                'compression_optimization': True,
                'format_standards': ['jpg', 'png', 'webp'],
                'accessibility_features': ['alt_text', 'contrast_ratio']
            },
            'content_analysis': {
                'object_detection': True,
                'text_extraction': True,  # OCR for text in images
                'scene_classification': True,
                'inappropriate_content_detection': True
            }
        }
    
    def _load_cross_modal_config(self) -> Dict[str, Any]:
        """Load cross-modal coherence analysis configuration."""
        return {
            'synchronization': {
                'audio_video_sync': {'tolerance': 0.1},  # 100ms tolerance
                'text_visual_alignment': True,
                'music_pace_matching': True,
                'voice_visual_coherence': True
            },
            'narrative_coherence': {
                'message_consistency': True,
                'emotional_alignment': True,
                'brand_voice_unity': True,
                'story_flow_across_modalities': True
            },
            'engagement_optimization': {
                'modality_balance': True,  # Not overwhelming any single sense
                'attention_distribution': True,
                'cognitive_load_management': True,
                'accessibility_multimodal': True
            },
            'platform_synergy': {
                'format_optimization': True,
                'algorithm_preferences': True,
                'user_behavior_alignment': True,
                'viral_element_integration': True
            }
        }
    
    def _load_platform_requirements(self) -> Dict[str, Any]:
        """Load platform-specific multi-modal requirements."""
        return {
            'tiktok': {
                'video': {'duration': [15, 60], 'aspect_ratio': '9:16', 'fps': 30},
                'audio': {'volume_peaks': True, 'trending_sounds': True},
                'text': {'captions_required': True, 'hashtag_optimization': True},
                'optimization_focus': ['hook_strength', 'visual_interest', 'audio_appeal']
            },
            'youtube_shorts': {
                'video': {'duration': [15, 60], 'aspect_ratio': '9:16', 'fps': 30},
                'audio': {'clear_speech': True, 'music_sync': True},
                'text': {'title_optimization': True, 'description_seo': True},
                'optimization_focus': ['retention_curve', 'click_through_rate']
            },
            'instagram_reels': {
                'video': {'duration': [15, 90], 'aspect_ratio': '9:16', 'fps': 30},
                'audio': {'trending_audio': True, 'original_sound': True},
                'text': {'visual_text_overlay': True, 'story_captions': True},
                'optimization_focus': ['aesthetic_appeal', 'shareability']
            },
            'youtube_standard': {
                'video': {'duration': [60, 600], 'aspect_ratio': '16:9', 'fps': [24, 60]},
                'audio': {'professional_quality': True, 'consistent_levels': True},
                'text': {'thumbnail_text': True, 'chapter_markers': True},
                'optimization_focus': ['watch_time', 'educational_value']
            },
            'linkedin': {
                'video': {'duration': [30, 300], 'aspect_ratio': ['16:9', '1:1'], 'fps': 30},
                'audio': {'professional_tone': True, 'clear_speech': True},
                'text': {'professional_captions': True, 'industry_keywords': True},
                'optimization_focus': ['professional_credibility', 'thought_leadership']
            }
        }
    
    def _initialize_analysis_models(self) -> Dict[str, Any]:
        """Initialize analysis models (mock implementations for demo)."""
        return {
            'video_analyzer': {
                'frame_processor': 'mock_frame_analysis',
                'motion_detector': 'mock_motion_detection',
                'scene_classifier': 'mock_scene_classification'
            },
            'audio_analyzer': {
                'voice_processor': 'mock_voice_analysis',
                'music_detector': 'mock_music_classification',
                'sound_classifier': 'mock_sound_analysis'
            },
            'image_analyzer': {
                'composition_analyzer': 'mock_composition_analysis',
                'object_detector': 'mock_object_detection',
                'emotion_classifier': 'mock_emotion_detection'
            },
            'text_analyzer': {
                'ocr_processor': 'mock_text_extraction',
                'sentiment_analyzer': 'mock_sentiment_analysis',
                'keyword_extractor': 'mock_keyword_extraction'
            },
            'cross_modal_analyzer': {
                'sync_detector': 'mock_synchronization_analysis',
                'coherence_classifier': 'mock_coherence_assessment'
            }
        }
    
    def analyze_video_content(self, video_data: Dict[str, Any], 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze video content for visual storytelling and technical quality.
        
        Args:
            video_data: Video content information (path, metadata, etc.)
            context: Additional context for analysis
            
        Returns:
            Comprehensive video analysis results
        """
        platform = context.get('platform', 'general') if context else 'general'
        duration = video_data.get('duration', 30.0)
        
        # Frame-by-frame analysis
        frame_analysis = self._analyze_video_frames(video_data, platform)
        
        # Pacing and rhythm analysis
        pacing_analysis = self._analyze_video_pacing(video_data, duration)
        
        # Visual storytelling assessment
        storytelling_analysis = self._analyze_visual_storytelling(video_data, context)
        
        # Technical quality evaluation
        technical_analysis = self._analyze_video_technical_quality(video_data, platform)
        
        # Calculate overall video score
        video_score = self._calculate_video_score(
            frame_analysis, pacing_analysis, storytelling_analysis, technical_analysis
        )
        
        return {
            'video_score': round(video_score, 3),
            'frame_analysis': frame_analysis,
            'pacing_analysis': pacing_analysis,
            'storytelling_analysis': storytelling_analysis,
            'technical_analysis': technical_analysis,
            'platform_optimization': self._get_video_platform_optimization(video_data, platform),
            'improvement_suggestions': self._generate_video_improvements(
                frame_analysis, pacing_analysis, storytelling_analysis, technical_analysis
            )
        }
    
    def analyze_audio_content(self, audio_data: Dict[str, Any], 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze audio content for voice, music, and sound design.
        
        Args:
            audio_data: Audio content information
            context: Additional context for analysis
            
        Returns:
            Comprehensive audio analysis results
        """
        platform = context.get('platform', 'general') if context else 'general'
        
        # Voice analysis
        voice_analysis = self._analyze_voice_content(audio_data, context)
        
        # Music and sound analysis
        music_analysis = self._analyze_music_content(audio_data, platform)
        
        # Sound design evaluation
        sound_design_analysis = self._analyze_sound_design(audio_data, context)
        
        # Technical audio quality
        technical_audio_analysis = self._analyze_audio_technical_quality(audio_data)
        
        # Calculate overall audio score
        audio_score = self._calculate_audio_score(
            voice_analysis, music_analysis, sound_design_analysis, technical_audio_analysis
        )
        
        return {
            'audio_score': round(audio_score, 3),
            'voice_analysis': voice_analysis,
            'music_analysis': music_analysis,
            'sound_design_analysis': sound_design_analysis,
            'technical_audio_analysis': technical_audio_analysis,
            'platform_optimization': self._get_audio_platform_optimization(audio_data, platform),
            'improvement_suggestions': self._generate_audio_improvements(
                voice_analysis, music_analysis, sound_design_analysis
            )
        }
    
    def analyze_image_content(self, image_data: Dict[str, Any], 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze image content for composition, branding, and emotional impact.
        
        Args:
            image_data: Image content information
            context: Additional context for analysis
            
        Returns:
            Comprehensive image analysis results
        """
        platform = context.get('platform', 'general') if context else 'general'
        
        # Composition analysis
        composition_analysis = self._analyze_image_composition(image_data)
        
        # Visual branding assessment
        branding_analysis = self._analyze_visual_branding(image_data, context)
        
        # Emotional impact evaluation
        emotional_analysis = self._analyze_image_emotional_impact(image_data, context)
        
        # Technical image quality
        technical_analysis = self._analyze_image_technical_quality(image_data, platform)
        
        # Calculate overall image score
        image_score = self._calculate_image_score(
            composition_analysis, branding_analysis, emotional_analysis, technical_analysis
        )
        
        return {
            'image_score': round(image_score, 3),
            'composition_analysis': composition_analysis,
            'branding_analysis': branding_analysis,
            'emotional_analysis': emotional_analysis,
            'technical_analysis': technical_analysis,
            'platform_optimization': self._get_image_platform_optimization(image_data, platform),
            'improvement_suggestions': self._generate_image_improvements(
                composition_analysis, branding_analysis, emotional_analysis
            )
        }
    
    def analyze_cross_modal_coherence(self, content_data: Dict[str, Any], 
                                    individual_analyses: Dict[str, Any],
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze how well different modalities work together.
        
        Args:
            content_data: Multi-modal content information
            individual_analyses: Results from individual modality analyses
            context: Additional context for analysis
            
        Returns:
            Cross-modal coherence analysis
        """
        platform = context.get('platform', 'general') if context else 'general'
        
        # Synchronization analysis
        sync_analysis = self._analyze_modality_synchronization(content_data, individual_analyses)
        
        # Narrative coherence assessment
        narrative_analysis = self._analyze_narrative_coherence(individual_analyses, context)
        
        # Engagement optimization evaluation
        engagement_analysis = self._analyze_engagement_optimization(
            content_data, individual_analyses, platform
        )
        
        # Platform synergy assessment
        synergy_analysis = self._analyze_platform_synergy(
            content_data, individual_analyses, platform
        )
        
        # Calculate overall coherence score
        coherence_score = self._calculate_coherence_score(
            sync_analysis, narrative_analysis, engagement_analysis, synergy_analysis
        )
        
        return {
            'coherence_score': round(coherence_score, 3),
            'synchronization_analysis': sync_analysis,
            'narrative_coherence': narrative_analysis,
            'engagement_optimization': engagement_analysis,
            'platform_synergy': synergy_analysis,
            'cross_modal_recommendations': self._generate_cross_modal_recommendations(
                sync_analysis, narrative_analysis, engagement_analysis, synergy_analysis
            )
        }
    
    def evaluate(self, content: Union[str, Dict[str, Any]], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform Level 3 Multi-Modal evaluation.
        
        Args:
            content: Multi-modal content (can be text + media references or structured data)
            context: Evaluation context including media data
            
        Returns:
            Comprehensive multi-modal analysis
        """
        start_time = time.time()
        
        # Extract multi-modal content data
        content_data = self._extract_multimodal_data(content, context)
        platform = context.get('platform', 'general') if context else 'general'
        
        # Individual modality analyses
        individual_analyses = {}
        
        # Video analysis (if present)
        if content_data.get('video_data'):
            individual_analyses['video'] = self.analyze_video_content(
                content_data['video_data'], context
            )
        
        # Audio analysis (if present)
        if content_data.get('audio_data'):
            individual_analyses['audio'] = self.analyze_audio_content(
                content_data['audio_data'], context
            )
        
        # Image analysis (if present)
        if content_data.get('image_data'):
            individual_analyses['image'] = self.analyze_image_content(
                content_data['image_data'], context
            )
        
        # Text analysis (always present)
        if content_data.get('text_data'):
            individual_analyses['text'] = self._analyze_text_in_multimodal_context(
                content_data['text_data'], content_data, context
            )
        
        # Cross-modal coherence analysis
        coherence_analysis = self.analyze_cross_modal_coherence(
            content_data, individual_analyses, context
        )
        
        # Calculate overall multi-modal score
        multimodal_score = self._calculate_multimodal_score(
            individual_analyses, coherence_analysis, platform
        )
        
        # Generate platform-specific optimization recommendations
        platform_recommendations = self._generate_platform_recommendations(
            content_data, individual_analyses, coherence_analysis, platform
        )
        
        # Performance tracking
        evaluation_time = (time.time() - start_time) * 1000
        
        result = {
            'multimodal_score': round(multimodal_score, 3),
            'individual_analyses': individual_analyses,
            'coherence_analysis': coherence_analysis,
            'platform_optimization': platform_recommendations,
            'content_modalities': list(individual_analyses.keys()),
            'recommendations': self._generate_multimodal_recommendations(
                individual_analyses, coherence_analysis, platform
            ),
            'evaluation_metadata': {
                'level': 3,
                'evaluator': 'multimodal',
                'evaluation_time': evaluation_time,
                'platform': platform,
                'modalities_analyzed': list(individual_analyses.keys()),
                'analysis_depth': 'comprehensive'
            }
        }
        
        return result 

    # Helper methods for video analysis
    
    def _analyze_video_frames(self, video_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Analyze video frame quality and composition."""
        # Mock implementation - in production would use computer vision models
        duration = video_data.get('duration', 30.0)
        fps = video_data.get('fps', 30)
        total_frames = int(duration * fps)
        
        # Simulate frame analysis
        composition_quality = 0.75 + np.random.normal(0, 0.1)
        lighting_quality = 0.70 + np.random.normal(0, 0.15)
        color_balance = 0.80 + np.random.normal(0, 0.1)
        motion_smoothness = 0.85 + np.random.normal(0, 0.05)
        
        # Platform-specific adjustments
        if platform in ['tiktok', 'instagram_reels']:
            motion_smoothness *= 1.1  # Higher motion tolerance for short-form
        
        return {
            'total_frames_analyzed': total_frames,
            'composition_quality': round(max(0, min(1, composition_quality)), 3),
            'lighting_quality': round(max(0, min(1, lighting_quality)), 3),
            'color_balance': round(max(0, min(1, color_balance)), 3),
            'motion_smoothness': round(max(0, min(1, motion_smoothness)), 3),
            'visual_consistency': round(max(0, min(1, (composition_quality + lighting_quality + color_balance) / 3)), 3),
            'scene_changes': max(1, int(duration / 5)),  # Estimate scene changes
            'frame_quality_score': round(max(0, min(1, (composition_quality + lighting_quality + color_balance + motion_smoothness) / 4)), 3)
        }
    
    def _analyze_video_pacing(self, video_data: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Analyze video pacing and rhythm."""
        # Mock pacing analysis
        shot_count = max(1, int(duration / 3.5))  # Estimate shots
        avg_shot_length = duration / shot_count
        
        # Optimal pacing varies by platform
        optimal_range = self.video_config['pacing_analysis']['optimal_shot_length']
        pacing_score = 1.0
        
        if avg_shot_length < optimal_range['min']:
            pacing_score = 0.6  # Too fast
        elif avg_shot_length > optimal_range['max']:
            pacing_score = 0.7  # Too slow
        
        rhythm_consistency = 0.75 + np.random.normal(0, 0.1)
        transition_quality = 0.80 + np.random.normal(0, 0.1)
        
        return {
            'shot_count': shot_count,
            'average_shot_length': round(avg_shot_length, 2),
            'pacing_score': round(max(0, min(1, pacing_score)), 3),
            'rhythm_consistency': round(max(0, min(1, rhythm_consistency)), 3),
            'transition_quality': round(max(0, min(1, transition_quality)), 3),
            'overall_pacing': round(max(0, min(1, (pacing_score + rhythm_consistency + transition_quality) / 3)), 3)
        }
    
    def _analyze_visual_storytelling(self, video_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze visual storytelling elements."""
        duration = video_data.get('duration', 30.0)
        
        # Hook strength (first 3 seconds)
        hook_strength = 0.70 + np.random.normal(0, 0.15)
        
        # Narrative progression
        narrative_clarity = 0.75 + np.random.normal(0, 0.1)
        emotional_arc = 0.65 + np.random.normal(0, 0.2)
        
        # Call-to-action effectiveness
        has_cta = duration > 15  # Assume longer videos have CTAs
        cta_placement = 0.8 if has_cta else 0.5
        cta_clarity = 0.75 if has_cta else 0.3
        
        visual_hierarchy = 0.80 + np.random.normal(0, 0.1)
        
        return {
            'hook_strength': round(max(0, min(1, hook_strength)), 3),
            'narrative_clarity': round(max(0, min(1, narrative_clarity)), 3),
            'emotional_progression': round(max(0, min(1, emotional_arc)), 3),
            'visual_hierarchy': round(max(0, min(1, visual_hierarchy)), 3),
            'cta_placement': round(max(0, min(1, cta_placement)), 3),
            'cta_clarity': round(max(0, min(1, cta_clarity)), 3),
            'storytelling_score': round(max(0, min(1, (hook_strength + narrative_clarity + emotional_arc + visual_hierarchy) / 4)), 3)
        }
    
    def _analyze_video_technical_quality(self, video_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Analyze technical video quality."""
        resolution = video_data.get('resolution', 1080)
        fps = video_data.get('fps', 30)
        aspect_ratio = video_data.get('aspect_ratio', '16:9')
        
        # Platform requirements
        platform_reqs = self.platform_requirements.get(platform, {}).get('video', {})
        optimal_aspect = platform_reqs.get('aspect_ratio', '16:9')
        
        # Technical scoring
        resolution_score = 1.0 if resolution >= 1080 else 0.7 if resolution >= 720 else 0.4
        fps_score = 1.0 if fps >= 24 else 0.6
        aspect_ratio_score = 1.0 if aspect_ratio == optimal_aspect else 0.7
        
        # File quality simulation
        compression_quality = 0.85 + np.random.normal(0, 0.1)
        audio_video_sync = 0.95 + np.random.normal(0, 0.05)
        
        return {
            'resolution': resolution,
            'fps': fps,
            'aspect_ratio': aspect_ratio,
            'resolution_score': round(resolution_score, 3),
            'fps_score': round(fps_score, 3),
            'aspect_ratio_score': round(aspect_ratio_score, 3),
            'compression_quality': round(max(0, min(1, compression_quality)), 3),
            'audio_video_sync': round(max(0, min(1, audio_video_sync)), 3),
            'technical_score': round(max(0, min(1, (resolution_score + fps_score + aspect_ratio_score + compression_quality + audio_video_sync) / 5)), 3),
            'platform_compliance': aspect_ratio == optimal_aspect
        }
    
    # Helper methods for audio analysis
    
    def _analyze_voice_content(self, audio_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze voice content quality and characteristics."""
        duration = audio_data.get('duration', 30.0)
        
        # Voice quality metrics
        clarity = 0.80 + np.random.normal(0, 0.1)
        volume_consistency = 0.75 + np.random.normal(0, 0.15)
        
        # Tone analysis (mock)
        tone_confidence = np.random.choice(['enthusiastic', 'calm', 'professional', 'friendly'], p=[0.3, 0.2, 0.3, 0.2])
        tone_appropriateness = 0.85 + np.random.normal(0, 0.1)
        
        # Pace analysis
        estimated_words = int(duration * 2.5)  # Rough estimate
        words_per_minute = (estimated_words / duration) * 60
        pace_score = 1.0 if 120 <= words_per_minute <= 180 else 0.7
        
        # Emotional engagement
        emotional_range = 0.70 + np.random.normal(0, 0.15)
        authenticity = 0.80 + np.random.normal(0, 0.1)
        
        return {
            'clarity': round(max(0, min(1, clarity)), 3),
            'volume_consistency': round(max(0, min(1, volume_consistency)), 3),
            'detected_tone': tone_confidence,
            'tone_appropriateness': round(max(0, min(1, tone_appropriateness)), 3),
            'estimated_wpm': round(words_per_minute, 1),
            'pace_score': round(pace_score, 3),
            'emotional_range': round(max(0, min(1, emotional_range)), 3),
            'authenticity': round(max(0, min(1, authenticity)), 3),
            'voice_score': round(max(0, min(1, (clarity + volume_consistency + tone_appropriateness + pace_score + emotional_range + authenticity) / 6)), 3)
        }
    
    def _analyze_music_content(self, audio_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Analyze music and background audio."""
        has_music = audio_data.get('has_music', True)
        
        if not has_music:
            return {
                'has_music': False,
                'music_score': 0.5,
                'mood_alignment': 0.5,
                'volume_balance': 0.7,  # Good for voice-only
                'copyright_safe': 1.0,
                'platform_trending': 0.3
            }
        
        # Music analysis
        mood_appropriateness = 0.75 + np.random.normal(0, 0.15)
        volume_balance = 0.70 + np.random.normal(0, 0.1)  # Voice should dominate
        
        # Platform-specific factors
        trending_factor = 0.8 if platform in ['tiktok', 'instagram_reels'] else 0.6
        copyright_safety = 0.95 + np.random.normal(0, 0.05)
        
        sync_quality = 0.80 + np.random.normal(0, 0.1)
        
        return {
            'has_music': True,
            'detected_mood': np.random.choice(['upbeat', 'calming', 'energetic', 'dramatic']),
            'mood_appropriateness': round(max(0, min(1, mood_appropriateness)), 3),
            'volume_balance': round(max(0, min(1, volume_balance)), 3),
            'sync_quality': round(max(0, min(1, sync_quality)), 3),
            'copyright_safe': round(max(0, min(1, copyright_safety)), 3),
            'platform_trending': round(max(0, min(1, trending_factor)), 3),
            'music_score': round(max(0, min(1, (mood_appropriateness + volume_balance + sync_quality + copyright_safety) / 4)), 3)
        }
    
    def _analyze_sound_design(self, audio_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sound design and audio effects."""
        has_sfx = audio_data.get('has_sound_effects', False)
        
        # Sound effects analysis
        sfx_appropriateness = 0.75 if has_sfx else 0.5
        audio_branding = 0.60 + np.random.normal(0, 0.2)
        
        # Strategic silence usage
        silence_usage = 0.70 + np.random.normal(0, 0.15)
        dynamic_range = 0.75 + np.random.normal(0, 0.1)
        
        # Environmental audio
        background_noise = 0.85 + np.random.normal(0, 0.1)  # Lower is better (less noise)
        
        return {
            'has_sound_effects': has_sfx,
            'sfx_appropriateness': round(max(0, min(1, sfx_appropriateness)), 3),
            'audio_branding': round(max(0, min(1, audio_branding)), 3),
            'silence_usage': round(max(0, min(1, silence_usage)), 3),
            'dynamic_range': round(max(0, min(1, dynamic_range)), 3),
            'background_noise_control': round(max(0, min(1, background_noise)), 3),
            'sound_design_score': round(max(0, min(1, (sfx_appropriateness + audio_branding + silence_usage + dynamic_range + background_noise) / 5)), 3)
        }
    
    def _analyze_audio_technical_quality(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical audio quality."""
        sample_rate = audio_data.get('sample_rate', 44100)
        bit_depth = audio_data.get('bit_depth', 16)
        format_type = audio_data.get('format', 'mp3')
        
        # Technical scoring
        sample_rate_score = 1.0 if sample_rate >= 44100 else 0.7
        bit_depth_score = 1.0 if bit_depth >= 16 else 0.6
        format_score = 1.0 if format_type in ['wav', 'aac'] else 0.8 if format_type == 'mp3' else 0.6
        
        # Audio levels
        peak_levels = 0.85 + np.random.normal(0, 0.1)
        loudness_consistency = 0.80 + np.random.normal(0, 0.1)
        distortion_free = 0.90 + np.random.normal(0, 0.05)
        
        return {
            'sample_rate': sample_rate,
            'bit_depth': bit_depth,
            'format': format_type,
            'sample_rate_score': round(sample_rate_score, 3),
            'bit_depth_score': round(bit_depth_score, 3),
            'format_score': round(format_score, 3),
            'peak_levels': round(max(0, min(1, peak_levels)), 3),
            'loudness_consistency': round(max(0, min(1, loudness_consistency)), 3),
            'distortion_free': round(max(0, min(1, distortion_free)), 3),
            'technical_audio_score': round(max(0, min(1, (sample_rate_score + bit_depth_score + format_score + peak_levels + loudness_consistency + distortion_free) / 6)), 3)
        }
    
    # Helper methods for image analysis
    
    def _analyze_image_composition(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image composition and visual elements."""
        # Mock composition analysis
        rule_of_thirds = 0.75 + np.random.normal(0, 0.15)
        leading_lines = 0.60 + np.random.normal(0, 0.2)
        symmetry_balance = 0.70 + np.random.normal(0, 0.15)
        focal_point_clarity = 0.80 + np.random.normal(0, 0.1)
        depth_of_field = 0.65 + np.random.normal(0, 0.2)
        
        # Color composition
        color_harmony = 0.75 + np.random.normal(0, 0.15)
        contrast_ratio = 0.80 + np.random.normal(0, 0.1)
        
        return {
            'rule_of_thirds': round(max(0, min(1, rule_of_thirds)), 3),
            'leading_lines': round(max(0, min(1, leading_lines)), 3),
            'symmetry_balance': round(max(0, min(1, symmetry_balance)), 3),
            'focal_point_clarity': round(max(0, min(1, focal_point_clarity)), 3),
            'depth_of_field': round(max(0, min(1, depth_of_field)), 3),
            'color_harmony': round(max(0, min(1, color_harmony)), 3),
            'contrast_ratio': round(max(0, min(1, contrast_ratio)), 3),
            'composition_score': round(max(0, min(1, (rule_of_thirds + leading_lines + symmetry_balance + focal_point_clarity + depth_of_field + color_harmony + contrast_ratio) / 7)), 3)
        }
    
    def _analyze_visual_branding(self, image_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze visual branding consistency."""
        brand_colors = context.get('brand_colors', []) if context else []
        has_logo = image_data.get('has_logo', False)
        
        # Branding analysis
        color_consistency = 0.80 if brand_colors else 0.60
        color_consistency += np.random.normal(0, 0.1)
        
        font_consistency = 0.75 + np.random.normal(0, 0.15)
        logo_placement = 0.85 if has_logo else 0.30
        brand_guidelines = 0.70 + np.random.normal(0, 0.15)
        visual_hierarchy = 0.75 + np.random.normal(0, 0.1)
        
        return {
            'color_consistency': round(max(0, min(1, color_consistency)), 3),
            'font_consistency': round(max(0, min(1, font_consistency)), 3),
            'logo_placement': round(max(0, min(1, logo_placement)), 3),
            'brand_guidelines_adherence': round(max(0, min(1, brand_guidelines)), 3),
            'visual_hierarchy': round(max(0, min(1, visual_hierarchy)), 3),
            'has_brand_elements': has_logo or len(brand_colors) > 0,
            'branding_score': round(max(0, min(1, (color_consistency + font_consistency + logo_placement + brand_guidelines + visual_hierarchy) / 5)), 3)
        }
    
    def _analyze_image_emotional_impact(self, image_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional impact of images."""
        # Mock emotional analysis
        detected_emotion = np.random.choice(['positive', 'neutral', 'inspiring', 'energetic', 'calm'])
        emotion_strength = 0.70 + np.random.normal(0, 0.15)
        
        # Facial expressions (if people present)
        has_faces = image_data.get('has_faces', False)
        facial_expression_score = 0.75 if has_faces else 0.50
        
        # Color psychology
        color_psychology = 0.75 + np.random.normal(0, 0.15)
        mood_alignment = 0.70 + np.random.normal(0, 0.15)
        
        # Cultural sensitivity
        cultural_appropriateness = 0.90 + np.random.normal(0, 0.05)
        
        return {
            'detected_emotion': detected_emotion,
            'emotion_strength': round(max(0, min(1, emotion_strength)), 3),
            'has_faces': has_faces,
            'facial_expression_score': round(max(0, min(1, facial_expression_score)), 3),
            'color_psychology': round(max(0, min(1, color_psychology)), 3),
            'mood_alignment': round(max(0, min(1, mood_alignment)), 3),
            'cultural_appropriateness': round(max(0, min(1, cultural_appropriateness)), 3),
            'emotional_impact_score': round(max(0, min(1, (emotion_strength + facial_expression_score + color_psychology + mood_alignment + cultural_appropriateness) / 5)), 3)
        }
    
    def _analyze_image_technical_quality(self, image_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Analyze technical image quality."""
        resolution = image_data.get('resolution', [1080, 1080])
        format_type = image_data.get('format', 'jpg')
        file_size = image_data.get('file_size_mb', 2.0)
        
        # Resolution scoring
        min_dimension = min(resolution)
        resolution_score = 1.0 if min_dimension >= 1080 else 0.8 if min_dimension >= 720 else 0.5
        
        # Format scoring
        format_score = 1.0 if format_type in ['png', 'webp'] else 0.8 if format_type == 'jpg' else 0.6
        
        # File size optimization
        size_score = 1.0 if file_size <= 5.0 else 0.7 if file_size <= 10.0 else 0.5
        
        # Compression quality
        compression_quality = 0.85 + np.random.normal(0, 0.1)
        sharpness = 0.80 + np.random.normal(0, 0.1)
        
        return {
            'resolution': resolution,
            'format': format_type,
            'file_size_mb': file_size,
            'resolution_score': round(resolution_score, 3),
            'format_score': round(format_score, 3),
            'file_size_score': round(size_score, 3),
            'compression_quality': round(max(0, min(1, compression_quality)), 3),
            'sharpness': round(max(0, min(1, sharpness)), 3),
            'technical_image_score': round(max(0, min(1, (resolution_score + format_score + size_score + compression_quality + sharpness) / 5)), 3)
        }
    
    # Cross-modal analysis methods
    
    def _analyze_modality_synchronization(self, content_data: Dict[str, Any], 
                                        individual_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synchronization between different modalities."""
        # Audio-video sync
        audio_video_sync = 0.95 + np.random.normal(0, 0.05) if 'video' in individual_analyses and 'audio' in individual_analyses else 1.0
        
        # Text-visual alignment
        text_visual_alignment = 0.80 + np.random.normal(0, 0.1) if 'text' in individual_analyses and ('video' in individual_analyses or 'image' in individual_analyses) else 1.0
        
        # Music-pace matching (if both exist)
        music_pace_matching = 0.75 + np.random.normal(0, 0.15) if 'audio' in individual_analyses and 'video' in individual_analyses else 1.0
        
        # Voice-visual coherence
        voice_visual_coherence = 0.85 + np.random.normal(0, 0.1) if 'audio' in individual_analyses and ('video' in individual_analyses or 'image' in individual_analyses) else 1.0
        
        return {
            'audio_video_sync': round(max(0, min(1, audio_video_sync)), 3),
            'text_visual_alignment': round(max(0, min(1, text_visual_alignment)), 3),
            'music_pace_matching': round(max(0, min(1, music_pace_matching)), 3),
            'voice_visual_coherence': round(max(0, min(1, voice_visual_coherence)), 3),
            'synchronization_score': round(max(0, min(1, (audio_video_sync + text_visual_alignment + music_pace_matching + voice_visual_coherence) / 4)), 3)
        }
    
    def _analyze_narrative_coherence(self, individual_analyses: Dict[str, Any], 
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze narrative coherence across modalities."""
        # Message consistency across modalities
        message_consistency = 0.80 + np.random.normal(0, 0.1)
        
        # Emotional alignment between modalities
        emotional_alignment = 0.75 + np.random.normal(0, 0.15)
        
        # Brand voice unity
        brand_voice_unity = 0.85 + np.random.normal(0, 0.1)
        
        # Story flow across modalities
        story_flow = 0.70 + np.random.normal(0, 0.15)
        
        # Thematic consistency
        thematic_consistency = 0.75 + np.random.normal(0, 0.1)
        
        return {
            'message_consistency': round(max(0, min(1, message_consistency)), 3),
            'emotional_alignment': round(max(0, min(1, emotional_alignment)), 3),
            'brand_voice_unity': round(max(0, min(1, brand_voice_unity)), 3),
            'story_flow_coherence': round(max(0, min(1, story_flow)), 3),
            'thematic_consistency': round(max(0, min(1, thematic_consistency)), 3),
            'narrative_coherence_score': round(max(0, min(1, (message_consistency + emotional_alignment + brand_voice_unity + story_flow + thematic_consistency) / 5)), 3)
        }
    
    def _analyze_engagement_optimization(self, content_data: Dict[str, Any], 
                                       individual_analyses: Dict[str, Any], 
                                       platform: str) -> Dict[str, Any]:
        """Analyze engagement optimization across modalities."""
        # Modality balance (not overwhelming any single sense)
        modality_count = len(individual_analyses)
        modality_balance = 1.0 if modality_count <= 3 else 0.8  # 3+ modalities can be overwhelming
        
        # Attention distribution
        attention_distribution = 0.75 + np.random.normal(0, 0.1)
        
        # Cognitive load management
        cognitive_load = 0.80 + np.random.normal(0, 0.1)
        if modality_count > 3:
            cognitive_load *= 0.9  # Penalty for too many modalities
        
        # Accessibility considerations
        accessibility_score = 0.70 + np.random.normal(0, 0.15)
        if 'text' in individual_analyses:  # Text helps accessibility
            accessibility_score += 0.1
        
        # Platform-specific engagement factors
        platform_engagement = self._calculate_platform_engagement_factor(platform, individual_analyses)
        
        return {
            'modality_balance': round(max(0, min(1, modality_balance)), 3),
            'attention_distribution': round(max(0, min(1, attention_distribution)), 3),
            'cognitive_load_management': round(max(0, min(1, cognitive_load)), 3),
            'accessibility_score': round(max(0, min(1, accessibility_score)), 3),
            'platform_engagement_factor': round(max(0, min(1, platform_engagement)), 3),
            'engagement_optimization_score': round(max(0, min(1, (modality_balance + attention_distribution + cognitive_load + accessibility_score + platform_engagement) / 5)), 3)
        }
    
    def _analyze_platform_synergy(self, content_data: Dict[str, Any], 
                                 individual_analyses: Dict[str, Any], 
                                 platform: str) -> Dict[str, Any]:
        """Analyze how well modalities work together for specific platform."""
        platform_reqs = self.platform_requirements.get(platform, {})
        
        # Format optimization for platform
        format_optimization = self._calculate_format_optimization(individual_analyses, platform_reqs)
        
        # Algorithm preferences alignment
        algorithm_preferences = self._calculate_algorithm_preferences(individual_analyses, platform)
        
        # User behavior alignment for platform
        user_behavior_alignment = 0.75 + np.random.normal(0, 0.1)
        
        # Viral element integration
        viral_element_integration = self._calculate_viral_elements(individual_analyses, platform)
        
        return {
            'format_optimization': round(max(0, min(1, format_optimization)), 3),
            'algorithm_preferences': round(max(0, min(1, algorithm_preferences)), 3),
            'user_behavior_alignment': round(max(0, min(1, user_behavior_alignment)), 3),
            'viral_element_integration': round(max(0, min(1, viral_element_integration)), 3),
            'platform_synergy_score': round(max(0, min(1, (format_optimization + algorithm_preferences + user_behavior_alignment + viral_element_integration) / 4)), 3)
        }
    
    # Scoring and calculation methods
    
    def _calculate_video_score(self, frame_analysis: Dict[str, Any], 
                             pacing_analysis: Dict[str, Any],
                             storytelling_analysis: Dict[str, Any], 
                             technical_analysis: Dict[str, Any]) -> float:
        """Calculate overall video score."""
        weights = {
            'frame_quality': 0.25,
            'pacing': 0.25,
            'storytelling': 0.30,
            'technical': 0.20
        }
        
        return (
            frame_analysis['frame_quality_score'] * weights['frame_quality'] +
            pacing_analysis['overall_pacing'] * weights['pacing'] +
            storytelling_analysis['storytelling_score'] * weights['storytelling'] +
            technical_analysis['technical_score'] * weights['technical']
        )
    
    def _calculate_audio_score(self, voice_analysis: Dict[str, Any], 
                             music_analysis: Dict[str, Any],
                             sound_design_analysis: Dict[str, Any], 
                             technical_audio_analysis: Dict[str, Any]) -> float:
        """Calculate overall audio score."""
        weights = {
            'voice': 0.35,
            'music': 0.25,
            'sound_design': 0.20,
            'technical': 0.20
        }
        
        return (
            voice_analysis['voice_score'] * weights['voice'] +
            music_analysis['music_score'] * weights['music'] +
            sound_design_analysis['sound_design_score'] * weights['sound_design'] +
            technical_audio_analysis['technical_audio_score'] * weights['technical']
        )
    
    def _calculate_image_score(self, composition_analysis: Dict[str, Any], 
                             branding_analysis: Dict[str, Any],
                             emotional_analysis: Dict[str, Any], 
                             technical_analysis: Dict[str, Any]) -> float:
        """Calculate overall image score."""
        weights = {
            'composition': 0.30,
            'branding': 0.25,
            'emotional': 0.25,
            'technical': 0.20
        }
        
        return (
            composition_analysis['composition_score'] * weights['composition'] +
            branding_analysis['branding_score'] * weights['branding'] +
            emotional_analysis['emotional_impact_score'] * weights['emotional'] +
            technical_analysis['technical_image_score'] * weights['technical']
        )
    
    def _calculate_coherence_score(self, sync_analysis: Dict[str, Any], 
                                 narrative_analysis: Dict[str, Any],
                                 engagement_analysis: Dict[str, Any], 
                                 synergy_analysis: Dict[str, Any]) -> float:
        """Calculate overall cross-modal coherence score."""
        weights = {
            'synchronization': 0.25,
            'narrative': 0.30,
            'engagement': 0.25,
            'synergy': 0.20
        }
        
        return (
            sync_analysis['synchronization_score'] * weights['synchronization'] +
            narrative_analysis['narrative_coherence_score'] * weights['narrative'] +
            engagement_analysis['engagement_optimization_score'] * weights['engagement'] +
            synergy_analysis['platform_synergy_score'] * weights['synergy']
        )
    
    def _calculate_multimodal_score(self, individual_analyses: Dict[str, Any], 
                                  coherence_analysis: Dict[str, Any], 
                                  platform: str) -> float:
        """Calculate overall multi-modal score."""
        # Base scores from individual modalities
        modality_scores = []
        modality_weights = {'video': 0.35, 'audio': 0.30, 'image': 0.20, 'text': 0.15}
        
        total_weight = 0
        weighted_sum = 0
        
        for modality, analysis in individual_analyses.items():
            if modality in modality_weights:
                score_key = f'{modality}_score'
                if score_key in analysis:
                    weighted_sum += analysis[score_key] * modality_weights[modality]
                    total_weight += modality_weights[modality]
        
        # Normalize by actual modalities present
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Apply coherence multiplier
        coherence_multiplier = 0.8 + (coherence_analysis['coherence_score'] * 0.4)  # Range: 0.8 to 1.2
        
        # Calculate final score
        final_score = base_score * coherence_multiplier
        
        return min(1.0, final_score)  # Cap at 1.0
    
    # Utility methods
    
    def _extract_multimodal_data(self, content: Union[str, Dict[str, Any]], 
                                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract multi-modal data from content and context."""
        if isinstance(content, str):
            # Text-only content
            return {
                'text_data': {'content': content},
                'video_data': context.get('video_data') if context else None,
                'audio_data': context.get('audio_data') if context else None,
                'image_data': context.get('image_data') if context else None
            }
        elif isinstance(content, dict):
            # Structured multi-modal content
            return {
                'text_data': content.get('text_data'),
                'video_data': content.get('video_data'),
                'audio_data': content.get('audio_data'),
                'image_data': content.get('image_data')
            }
        else:
            return {'text_data': {'content': str(content)}}
    
    def _analyze_text_in_multimodal_context(self, text_data: Dict[str, Any], 
                                          content_data: Dict[str, Any],
                                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze text content in multi-modal context."""
        content = text_data.get('content', '')
        
        # Basic text analysis
        readability = 0.75 + np.random.normal(0, 0.1)
        engagement_hooks = 0.70 + np.random.normal(0, 0.15)
        
        # Multi-modal context considerations
        visual_text_integration = 0.80 if content_data.get('video_data') or content_data.get('image_data') else 0.5
        audio_text_sync = 0.85 if content_data.get('audio_data') else 0.5
        
        # Platform-specific text optimization
        platform = context.get('platform', 'general') if context else 'general'
        platform_text_optimization = self._calculate_text_platform_optimization(content, platform)
        
        return {
            'text_score': round(max(0, min(1, (readability + engagement_hooks + visual_text_integration + audio_text_sync + platform_text_optimization) / 5)), 3),
            'readability': round(max(0, min(1, readability)), 3),
            'engagement_hooks': round(max(0, min(1, engagement_hooks)), 3),
            'visual_text_integration': round(max(0, min(1, visual_text_integration)), 3),
            'audio_text_sync': round(max(0, min(1, audio_text_sync)), 3),
            'platform_optimization': round(max(0, min(1, platform_text_optimization)), 3)
        }
    
    def _calculate_platform_engagement_factor(self, platform: str, individual_analyses: Dict[str, Any]) -> float:
        """Calculate platform-specific engagement factor."""
        base_factor = 0.75
        
        # Platform-specific bonuses
        if platform == 'tiktok':
            if 'video' in individual_analyses and 'audio' in individual_analyses:
                base_factor += 0.15  # TikTok loves video + audio
        elif platform == 'instagram_reels':
            if 'video' in individual_analyses:
                base_factor += 0.10  # Instagram prioritizes video
        elif platform == 'youtube_standard':
            if 'video' in individual_analyses and 'audio' in individual_analyses:
                base_factor += 0.20  # YouTube rewards high-quality video + audio
        elif platform == 'linkedin':
            if 'text' in individual_analyses:
                base_factor += 0.10  # LinkedIn values text content
        
        return base_factor
    
    def _calculate_format_optimization(self, individual_analyses: Dict[str, Any], platform_reqs: Dict[str, Any]) -> float:
        """Calculate format optimization score for platform."""
        optimization_score = 0.75  # Base score
        
        # Check video format compliance
        if 'video' in individual_analyses and 'video' in platform_reqs:
            video_analysis = individual_analyses['video']
            if video_analysis.get('technical_analysis', {}).get('platform_compliance', False):
                optimization_score += 0.15
        
        # Check audio format compliance
        if 'audio' in individual_analyses and 'audio' in platform_reqs:
            optimization_score += 0.10  # Assume good audio format
        
        return min(1.0, optimization_score)
    
    def _calculate_algorithm_preferences(self, individual_analyses: Dict[str, Any], platform: str) -> float:
        """Calculate algorithm preference alignment."""
        base_score = 0.70
        
        # Platform-specific algorithm preferences
        if platform in ['tiktok', 'instagram_reels']:
            # Short-form platforms prefer high engagement, quick hooks
            if 'video' in individual_analyses:
                video_storytelling = individual_analyses['video'].get('storytelling_analysis', {})
                hook_strength = video_storytelling.get('hook_strength', 0.5)
                base_score += hook_strength * 0.2
        
        elif platform == 'youtube_standard':
            # YouTube prefers watch time, technical quality
            if 'video' in individual_analyses:
                video_technical = individual_analyses['video'].get('technical_analysis', {})
                technical_score = video_technical.get('technical_score', 0.5)
                base_score += technical_score * 0.15
        
        return min(1.0, base_score)
    
    def _calculate_viral_elements(self, individual_analyses: Dict[str, Any], platform: str) -> float:
        """Calculate viral element integration score."""
        viral_score = 0.60  # Base viral potential
        
        # Multi-modal viral elements
        if len(individual_analyses) >= 3:  # Multiple modalities increase viral potential
            viral_score += 0.15
        
        # Platform-specific viral elements
        if platform in ['tiktok', 'instagram_reels']:
            if 'audio' in individual_analyses:
                music_analysis = individual_analyses['audio'].get('music_analysis', {})
                if music_analysis.get('platform_trending', 0) > 0.7:
                    viral_score += 0.20
        
        return min(1.0, viral_score)
    
    def _calculate_text_platform_optimization(self, content: str, platform: str) -> float:
        """Calculate text optimization for specific platform."""
        base_score = 0.70
        
        # Platform-specific text optimizations
        if platform == 'tiktok':
            # TikTok loves hashtags and short, punchy text
            if '#' in content:
                base_score += 0.10
            if len(content) < 150:  # Short text performs better
                base_score += 0.10
        
        elif platform == 'linkedin':
            # LinkedIn prefers professional, longer-form content
            if len(content) > 200:
                base_score += 0.15
        
        elif platform == 'youtube_standard':
            # YouTube values descriptive, SEO-optimized text
            if len(content) > 100:
                base_score += 0.10
        
        return min(1.0, base_score)
    
    # Platform optimization methods
    
    def _get_video_platform_optimization(self, video_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Get video optimization recommendations for platform."""
        platform_reqs = self.platform_requirements.get(platform, {}).get('video', {})
        
        recommendations = []
        current_aspect = video_data.get('aspect_ratio', '16:9')
        optimal_aspect = platform_reqs.get('aspect_ratio', '16:9')
        
        if current_aspect != optimal_aspect:
            recommendations.append(f"Change aspect ratio from {current_aspect} to {optimal_aspect}")
        
        duration = video_data.get('duration', 30)
        duration_range = platform_reqs.get('duration', [15, 60])
        if duration < duration_range[0] or duration > duration_range[1]:
            recommendations.append(f"Adjust duration to {duration_range[0]}-{duration_range[1]} seconds")
        
        return {
            'platform': platform,
            'compliance_score': 0.85 if current_aspect == optimal_aspect else 0.60,
            'recommendations': recommendations,
            'optimal_specs': platform_reqs
        }
    
    def _get_audio_platform_optimization(self, audio_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Get audio optimization recommendations for platform."""
        platform_reqs = self.platform_requirements.get(platform, {}).get('audio', {})
        
        recommendations = []
        optimization_score = 0.80  # Base score
        
        # Platform-specific audio recommendations
        if platform in ['tiktok', 'instagram_reels']:
            if not audio_data.get('has_music', False):
                recommendations.append("Consider adding trending music for better reach")
                optimization_score -= 0.15
        
        elif platform == 'youtube_standard':
            if audio_data.get('sample_rate', 44100) < 44100:
                recommendations.append("Increase audio quality to 44.1kHz or higher")
                optimization_score -= 0.10
        
        return {
            'platform': platform,
            'optimization_score': round(max(0, min(1, optimization_score)), 3),
            'recommendations': recommendations,
            'requirements': platform_reqs
        }
    
    def _get_image_platform_optimization(self, image_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Get image optimization recommendations for platform."""
        recommendations = []
        optimization_score = 0.80
        
        resolution = image_data.get('resolution', [1080, 1080])
        min_res = min(resolution)
        
        if min_res < 1080:
            recommendations.append("Increase resolution to at least 1080p")
            optimization_score -= 0.15
        
        # Platform-specific image recommendations
        if platform == 'instagram_post':
            if resolution[0] != resolution[1]:  # Not square
                recommendations.append("Use square (1:1) aspect ratio for Instagram posts")
                optimization_score -= 0.10
        
        return {
            'platform': platform,
            'optimization_score': round(max(0, min(1, optimization_score)), 3),
            'recommendations': recommendations
        }
    
    # Improvement suggestion methods
    
    def _generate_video_improvements(self, frame_analysis: Dict[str, Any], 
                                   pacing_analysis: Dict[str, Any],
                                   storytelling_analysis: Dict[str, Any], 
                                   technical_analysis: Dict[str, Any]) -> List[str]:
        """Generate video improvement suggestions."""
        suggestions = []
        
        if frame_analysis['composition_quality'] < 0.7:
            suggestions.append("Improve composition using rule of thirds and better framing")
        
        if frame_analysis['lighting_quality'] < 0.7:
            suggestions.append("Enhance lighting quality for better visual appeal")
        
        if pacing_analysis['overall_pacing'] < 0.7:
            suggestions.append("Adjust pacing - vary shot lengths for better rhythm")
        
        if storytelling_analysis['hook_strength'] < 0.7:
            suggestions.append("Strengthen opening hook - first 3 seconds are crucial")
        
        if technical_analysis['technical_score'] < 0.7:
            suggestions.append("Improve technical quality - resolution, audio sync, compression")
        
        return suggestions
    
    def _generate_audio_improvements(self, voice_analysis: Dict[str, Any], 
                                   music_analysis: Dict[str, Any],
                                   sound_design_analysis: Dict[str, Any]) -> List[str]:
        """Generate audio improvement suggestions."""
        suggestions = []
        
        if voice_analysis['voice_score'] < 0.7:
            suggestions.append("Improve voice clarity and consistency")
        
        if voice_analysis['pace_score'] < 0.8:
            suggestions.append(f"Adjust speaking pace - aim for 150 WPM (currently {voice_analysis.get('estimated_wpm', 'unknown')})")
        
        if music_analysis['music_score'] < 0.7:
            suggestions.append("Better music selection and volume balance")
        
        if sound_design_analysis['sound_design_score'] < 0.7:
            suggestions.append("Enhance sound design and reduce background noise")
        
        return suggestions
    
    def _generate_image_improvements(self, composition_analysis: Dict[str, Any], 
                                   branding_analysis: Dict[str, Any],
                                   emotional_analysis: Dict[str, Any]) -> List[str]:
        """Generate image improvement suggestions."""
        suggestions = []
        
        if composition_analysis['composition_score'] < 0.7:
            suggestions.append("Improve image composition and visual balance")
        
        if branding_analysis['branding_score'] < 0.7:
            suggestions.append("Strengthen visual branding consistency")
        
        if emotional_analysis['emotional_impact_score'] < 0.7:
            suggestions.append("Enhance emotional impact through better color psychology and expressions")
        
        return suggestions
    
    def _generate_cross_modal_recommendations(self, sync_analysis: Dict[str, Any], 
                                            narrative_analysis: Dict[str, Any],
                                            engagement_analysis: Dict[str, Any], 
                                            synergy_analysis: Dict[str, Any]) -> List[str]:
        """Generate cross-modal improvement recommendations."""
        recommendations = []
        
        if sync_analysis['synchronization_score'] < 0.8:
            recommendations.append("Improve synchronization between audio and visual elements")
        
        if narrative_analysis['narrative_coherence_score'] < 0.7:
            recommendations.append("Strengthen narrative coherence across all modalities")
        
        if engagement_analysis['engagement_optimization_score'] < 0.7:
            recommendations.append("Optimize engagement balance - avoid overwhelming any single sense")
        
        if synergy_analysis['platform_synergy_score'] < 0.7:
            recommendations.append("Better align content format and style with platform preferences")
        
        return recommendations
    
    def _generate_platform_recommendations(self, content_data: Dict[str, Any], 
                                         individual_analyses: Dict[str, Any],
                                         coherence_analysis: Dict[str, Any], 
                                         platform: str) -> Dict[str, Any]:
        """Generate platform-specific optimization recommendations."""
        recommendations = {
            'format_optimizations': [],
            'content_strategies': [],
            'technical_improvements': []
        }
        
        platform_reqs = self.platform_requirements.get(platform, {})
        optimization_focus = platform_reqs.get('optimization_focus', [])
        
        # Format recommendations based on platform requirements
        if 'video' in individual_analyses and 'video' in platform_reqs:
            video_reqs = platform_reqs['video']
            recommendations['format_optimizations'].extend([
                f"Optimal duration: {video_reqs.get('duration', [15, 60])} seconds",
                f"Required aspect ratio: {video_reqs.get('aspect_ratio', '16:9')}",
                f"Recommended FPS: {video_reqs.get('fps', 30)}"
            ])
        
        # Content strategy recommendations
        for focus_area in optimization_focus:
            if focus_area == 'hook_strength':
                recommendations['content_strategies'].append("Prioritize strong opening hook (first 3 seconds)")
            elif focus_area == 'visual_interest':
                recommendations['content_strategies'].append("Maximize visual variety and movement")
            elif focus_area == 'professional_credibility':
                recommendations['content_strategies'].append("Maintain professional tone and appearance")
            elif focus_area == 'shareability':
                recommendations['content_strategies'].append("Include shareable moments and clear value propositions")
        
        # Technical improvements based on coherence analysis
        if coherence_analysis['coherence_score'] < 0.8:
            recommendations['technical_improvements'].extend([
                "Improve synchronization between modalities",
                "Enhance cross-modal narrative consistency"
            ])
        
        return recommendations
    
    def _generate_multimodal_recommendations(self, individual_analyses: Dict[str, Any], 
                                           coherence_analysis: Dict[str, Any], 
                                           platform: str) -> List[str]:
        """Generate overall multi-modal recommendations."""
        recommendations = []
        
        # Individual modality recommendations
        for modality, analysis in individual_analyses.items():
            score_key = f'{modality}_score'
            if score_key in analysis and analysis[score_key] < 0.7:
                recommendations.append(f"Improve {modality} quality (current score: {analysis[score_key]})")
        
        # Cross-modal recommendations
        if coherence_analysis['coherence_score'] < 0.8:
            recommendations.append("Enhance coherence between different content modalities")
        
        # Platform-specific recommendations
        if platform in ['tiktok', 'instagram_reels']:
            recommendations.append("Optimize for short-form engagement with strong hooks and visual interest")
        elif platform == 'youtube_standard':
            recommendations.append("Focus on technical quality and watch time optimization")
        elif platform == 'linkedin':
            recommendations.append("Maintain professional tone while maximizing educational value")
        
        return recommendations 