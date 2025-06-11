"""
Comprehensive Test Suite for Multi-Modal Evaluator (Level 3)

Tests cover:
1. Individual modality analysis (video, audio, image, text)
2. Cross-modal coherence evaluation
3. Platform-specific optimization
4. Performance benchmarking
5. Edge case handling
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluators.multimodal_evaluator import MultiModalEvaluator


class TestMultiModalEvaluator(unittest.TestCase):
    """Comprehensive test suite for Multi-Modal Evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = MultiModalEvaluator()
        
        # Sample test data
        self.sample_video_data = {
            'duration': 30.0,
            'resolution': 1080,
            'fps': 30,
            'aspect_ratio': '9:16'
        }
        
        self.sample_audio_data = {
            'duration': 30.0,
            'has_music': True,
            'has_voice': True,
            'sample_rate': 44100,
            'bit_depth': 16,
            'format': 'mp3'
        }
        
        self.sample_image_data = {
            'resolution': [1080, 1080],
            'format': 'png',
            'file_size_mb': 1.5,
            'has_logo': True,
            'has_faces': False
        }
        
        self.sample_text_data = {
            'content': 'This is a sample text for testing multi-modal integration.'
        }
        
        self.sample_context = {
            'platform': 'tiktok',
            'brand_colors': ['#FF6B6B', '#4ECDC4']
        }
    
    # Initialization Tests
    
    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        self.assertIsInstance(self.evaluator, MultiModalEvaluator)
        self.assertIn('video_config', dir(self.evaluator))
        self.assertIn('audio_config', dir(self.evaluator))
        self.assertIn('image_config', dir(self.evaluator))
        self.assertIn('platform_requirements', dir(self.evaluator))
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Video config
        self.assertIn('frame_analysis', self.evaluator.video_config)
        self.assertIn('pacing_analysis', self.evaluator.video_config)
        self.assertIn('storytelling_elements', self.evaluator.video_config)
        
        # Audio config
        self.assertIn('voice_analysis', self.evaluator.audio_config)
        self.assertIn('music_analysis', self.evaluator.audio_config)
        self.assertIn('sound_design', self.evaluator.audio_config)
        
        # Image config
        self.assertIn('composition_analysis', self.evaluator.image_config)
        self.assertIn('visual_branding', self.evaluator.image_config)
        self.assertIn('emotional_impact', self.evaluator.image_config)
    
    def test_platform_requirements_loading(self):
        """Test platform requirements configuration."""
        platforms = ['tiktok', 'youtube_shorts', 'instagram_reels', 'youtube_standard', 'linkedin']
        
        for platform in platforms:
            self.assertIn(platform, self.evaluator.platform_requirements)
            platform_req = self.evaluator.platform_requirements[platform]
            self.assertIn('optimization_focus', platform_req)
    
    # Video Analysis Tests
    
    def test_video_content_analysis(self):
        """Test video content analysis."""
        result = self.evaluator.analyze_video_content(self.sample_video_data, self.sample_context)
        
        # Check result structure
        required_keys = ['video_score', 'frame_analysis', 'pacing_analysis', 
                        'storytelling_analysis', 'technical_analysis', 'platform_optimization']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check score ranges
        self.assertGreaterEqual(result['video_score'], 0.0)
        self.assertLessEqual(result['video_score'], 1.0)
        
        # Check frame analysis
        frame_analysis = result['frame_analysis']
        self.assertIn('frame_quality_score', frame_analysis)
        self.assertIn('composition_quality', frame_analysis)
        self.assertIn('lighting_quality', frame_analysis)
        
        # Check pacing analysis
        pacing_analysis = result['pacing_analysis']
        self.assertIn('overall_pacing', pacing_analysis)
        self.assertIn('shot_count', pacing_analysis)
        
        # Check storytelling analysis
        storytelling_analysis = result['storytelling_analysis']
        self.assertIn('storytelling_score', storytelling_analysis)
        self.assertIn('hook_strength', storytelling_analysis)
    
    def test_video_platform_compliance(self):
        """Test video platform compliance checking."""
        tiktok_context = {'platform': 'tiktok'}
        result = self.evaluator.analyze_video_content(self.sample_video_data, tiktok_context)
        
        technical_analysis = result['technical_analysis']
        self.assertIn('platform_compliance', technical_analysis)
        self.assertIsInstance(technical_analysis['platform_compliance'], bool)
        
        platform_opt = result['platform_optimization']
        self.assertIn('platform', platform_opt)
        self.assertEqual(platform_opt['platform'], 'tiktok')
    
    def test_video_different_platforms(self):
        """Test video analysis across different platforms."""
        platforms = ['tiktok', 'youtube_standard', 'instagram_reels']
        
        for platform in platforms:
            context = {'platform': platform}
            result = self.evaluator.analyze_video_content(self.sample_video_data, context)
            
            self.assertIn('video_score', result)
            self.assertGreaterEqual(result['video_score'], 0.0)
            self.assertLessEqual(result['video_score'], 1.0)
    
    # Audio Analysis Tests
    
    def test_audio_content_analysis(self):
        """Test audio content analysis."""
        result = self.evaluator.analyze_audio_content(self.sample_audio_data, self.sample_context)
        
        # Check result structure
        required_keys = ['audio_score', 'voice_analysis', 'music_analysis', 
                        'sound_design_analysis', 'technical_audio_analysis']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check score ranges
        self.assertGreaterEqual(result['audio_score'], 0.0)
        self.assertLessEqual(result['audio_score'], 1.0)
        
        # Check voice analysis
        voice_analysis = result['voice_analysis']
        self.assertIn('voice_score', voice_analysis)
        self.assertIn('clarity', voice_analysis)
        self.assertIn('detected_tone', voice_analysis)
        self.assertIn('estimated_wpm', voice_analysis)
        
        # Check music analysis
        music_analysis = result['music_analysis']
        self.assertIn('music_score', music_analysis)
        self.assertIn('has_music', music_analysis)
        
        # Check technical audio analysis
        technical_analysis = result['technical_audio_analysis']
        self.assertIn('technical_audio_score', technical_analysis)
        self.assertIn('sample_rate', technical_analysis)
    
    def test_audio_without_music(self):
        """Test audio analysis without music."""
        audio_data_no_music = self.sample_audio_data.copy()
        audio_data_no_music['has_music'] = False
        
        result = self.evaluator.analyze_audio_content(audio_data_no_music, self.sample_context)
        
        music_analysis = result['music_analysis']
        self.assertFalse(music_analysis['has_music'])
        self.assertEqual(music_analysis['copyright_safe'], 1.0)  # No music = copyright safe
    
    def test_audio_voice_detection(self):
        """Test voice detection and analysis."""
        result = self.evaluator.analyze_audio_content(self.sample_audio_data, self.sample_context)
        
        voice_analysis = result['voice_analysis']
        
        # Check tone detection
        self.assertIn('detected_tone', voice_analysis)
        self.assertIn(voice_analysis['detected_tone'], 
                     ['enthusiastic', 'calm', 'professional', 'friendly'])
        
        # Check pace analysis
        self.assertIn('estimated_wpm', voice_analysis)
        self.assertGreater(voice_analysis['estimated_wpm'], 0)
    
    # Image Analysis Tests
    
    def test_image_content_analysis(self):
        """Test image content analysis."""
        result = self.evaluator.analyze_image_content(self.sample_image_data, self.sample_context)
        
        # Check result structure
        required_keys = ['image_score', 'composition_analysis', 'branding_analysis', 
                        'emotional_analysis', 'technical_analysis']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check score ranges
        self.assertGreaterEqual(result['image_score'], 0.0)
        self.assertLessEqual(result['image_score'], 1.0)
        
        # Check composition analysis
        composition = result['composition_analysis']
        self.assertIn('composition_score', composition)
        self.assertIn('rule_of_thirds', composition)
        self.assertIn('color_harmony', composition)
        
        # Check branding analysis
        branding = result['branding_analysis']
        self.assertIn('branding_score', branding)
        self.assertIn('has_brand_elements', branding)
        
        # Check emotional analysis
        emotional = result['emotional_analysis']
        self.assertIn('emotional_impact_score', emotional)
        self.assertIn('detected_emotion', emotional)
    
    def test_image_branding_with_colors(self):
        """Test image branding analysis with brand colors."""
        context_with_colors = self.sample_context.copy()
        context_with_colors['brand_colors'] = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        result = self.evaluator.analyze_image_content(self.sample_image_data, context_with_colors)
        
        branding_analysis = result['branding_analysis']
        self.assertTrue(branding_analysis['has_brand_elements'])
        self.assertGreater(branding_analysis['color_consistency'], 0.6)  # Should be higher with brand colors
    
    def test_image_technical_quality(self):
        """Test image technical quality analysis."""
        result = self.evaluator.analyze_image_content(self.sample_image_data, self.sample_context)
        
        technical = result['technical_analysis']
        self.assertIn('technical_image_score', technical)
        self.assertIn('resolution_score', technical)
        self.assertIn('format_score', technical)
        self.assertIn('compression_quality', technical)
        
        # Check resolution handling
        self.assertEqual(technical['resolution'], [1080, 1080])
        self.assertEqual(technical['format'], 'png')
    
    # Cross-Modal Analysis Tests
    
    def test_cross_modal_coherence_analysis(self):
        """Test cross-modal coherence analysis."""
        content_data = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data,
            'audio_data': self.sample_audio_data,
            'image_data': self.sample_image_data
        }
        
        individual_analyses = {
            'video': {'video_score': 0.85},
            'audio': {'audio_score': 0.78},
            'image': {'image_score': 0.82},
            'text': {'text_score': 0.75}
        }
        
        result = self.evaluator.analyze_cross_modal_coherence(
            content_data, individual_analyses, self.sample_context
        )
        
        # Check result structure
        required_keys = ['coherence_score', 'synchronization_analysis', 'narrative_coherence', 
                        'engagement_optimization', 'platform_synergy']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check score ranges
        self.assertGreaterEqual(result['coherence_score'], 0.0)
        self.assertLessEqual(result['coherence_score'], 1.0)
        
        # Check synchronization analysis
        sync_analysis = result['synchronization_analysis']
        self.assertIn('synchronization_score', sync_analysis)
        self.assertIn('audio_video_sync', sync_analysis)
        self.assertIn('text_visual_alignment', sync_analysis)
    
    def test_cross_modal_with_missing_modalities(self):
        """Test cross-modal analysis with missing modalities."""
        content_data = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data
            # Missing audio and image
        }
        
        individual_analyses = {
            'video': {'video_score': 0.85},
            'text': {'text_score': 0.75}
        }
        
        result = self.evaluator.analyze_cross_modal_coherence(
            content_data, individual_analyses, self.sample_context
        )
        
        self.assertIn('coherence_score', result)
        self.assertGreaterEqual(result['coherence_score'], 0.0)
        self.assertLessEqual(result['coherence_score'], 1.0)
    
    def test_modality_synchronization(self):
        """Test modality synchronization analysis."""
        content_data = {
            'video_data': self.sample_video_data,
            'audio_data': self.sample_audio_data
        }
        
        individual_analyses = {
            'video': {'video_score': 0.85},
            'audio': {'audio_score': 0.78}
        }
        
        sync_result = self.evaluator._analyze_modality_synchronization(
            content_data, individual_analyses
        )
        
        self.assertIn('synchronization_score', sync_result)
        self.assertIn('audio_video_sync', sync_result)
        self.assertGreaterEqual(sync_result['audio_video_sync'], 0.0)
        self.assertLessEqual(sync_result['audio_video_sync'], 1.0)
    
    # Comprehensive Multi-Modal Evaluation Tests
    
    def test_full_multimodal_evaluation_text_only(self):
        """Test full multi-modal evaluation with text only."""
        content = "This is a test text content for evaluation."
        context = {'platform': 'linkedin'}
        
        result = self.evaluator.evaluate(content, context)
        
        # Check result structure
        required_keys = ['multimodal_score', 'individual_analyses', 'coherence_analysis', 
                        'content_modalities', 'recommendations']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that text analysis was performed
        self.assertIn('text', result['individual_analyses'])
        self.assertIn('text', result['content_modalities'])
        
        # Check scores
        self.assertGreaterEqual(result['multimodal_score'], 0.0)
        self.assertLessEqual(result['multimodal_score'], 1.0)
    
    def test_full_multimodal_evaluation_structured_content(self):
        """Test full multi-modal evaluation with structured content."""
        content = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data,
            'audio_data': self.sample_audio_data
        }
        
        context = {'platform': 'tiktok'}
        result = self.evaluator.evaluate(content, context)
        
        # Check modalities were analyzed
        expected_modalities = ['text', 'video', 'audio']
        for modality in expected_modalities:
            self.assertIn(modality, result['content_modalities'])
            self.assertIn(modality, result['individual_analyses'])
        
        # Check coherence analysis
        self.assertIn('coherence_analysis', result)
        self.assertIn('coherence_score', result['coherence_analysis'])
    
    def test_all_modalities_evaluation(self):
        """Test evaluation with all modalities present."""
        content = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data,
            'audio_data': self.sample_audio_data,
            'image_data': self.sample_image_data
        }
        
        context = {'platform': 'youtube_standard'}
        result = self.evaluator.evaluate(content, context)
        
        # All modalities should be present
        expected_modalities = ['text', 'video', 'audio', 'image']
        self.assertEqual(set(result['content_modalities']), set(expected_modalities))
        
        # Check individual analyses
        for modality in expected_modalities:
            self.assertIn(modality, result['individual_analyses'])
            analysis = result['individual_analyses'][modality]
            score_key = f'{modality}_score'
            self.assertIn(score_key, analysis)
    
    # Platform-Specific Tests
    
    def test_platform_specific_optimization(self):
        """Test platform-specific optimization recommendations."""
        platforms = ['tiktok', 'youtube_standard', 'instagram_reels', 'linkedin']
        content = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data
        }
        
        for platform in platforms:
            context = {'platform': platform}
            result = self.evaluator.evaluate(content, context)
            
            self.assertIn('platform_optimization', result)
            platform_opt = result['platform_optimization']
            
            # Check optimization structure
            self.assertIn('format_optimizations', platform_opt)
            self.assertIn('content_strategies', platform_opt)
            self.assertIn('technical_improvements', platform_opt)
    
    def test_tiktok_specific_analysis(self):
        """Test TikTok-specific analysis features."""
        content = {
            'text_data': {'content': 'POV: You found the best productivity hack #productivity #viral'},
            'video_data': {**self.sample_video_data, 'aspect_ratio': '9:16', 'duration': 28.0},
            'audio_data': self.sample_audio_data
        }
        
        context = {'platform': 'tiktok'}
        result = self.evaluator.evaluate(content, context)
        
        # Check platform-specific optimizations
        platform_opt = result['platform_optimization']
        self.assertIn('format_optimizations', platform_opt)
        
        # TikTok should have specific recommendations
        recommendations = result['recommendations']
        self.assertIsInstance(recommendations, list)
    
    def test_youtube_specific_analysis(self):
        """Test YouTube-specific analysis features."""
        content = {
            'text_data': {'content': 'In this comprehensive tutorial, we will explore advanced techniques...'},
            'video_data': {**self.sample_video_data, 'aspect_ratio': '16:9', 'duration': 300.0},
            'audio_data': self.sample_audio_data
        }
        
        context = {'platform': 'youtube_standard'}
        result = self.evaluator.evaluate(content, context)
        
        # YouTube should focus on technical quality and watch time
        platform_opt = result['platform_optimization']
        self.assertIn('content_strategies', platform_opt)
    
    # Scoring and Calculation Tests
    
    def test_video_score_calculation(self):
        """Test video score calculation."""
        frame_analysis = {'frame_quality_score': 0.8}
        pacing_analysis = {'overall_pacing': 0.7}
        storytelling_analysis = {'storytelling_score': 0.9}
        technical_analysis = {'technical_score': 0.75}
        
        score = self.evaluator._calculate_video_score(
            frame_analysis, pacing_analysis, storytelling_analysis, technical_analysis
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(score, float)
    
    def test_audio_score_calculation(self):
        """Test audio score calculation."""
        voice_analysis = {'voice_score': 0.8}
        music_analysis = {'music_score': 0.7}
        sound_design_analysis = {'sound_design_score': 0.75}
        technical_analysis = {'technical_audio_score': 0.85}
        
        score = self.evaluator._calculate_audio_score(
            voice_analysis, music_analysis, sound_design_analysis, technical_analysis
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(score, float)
    
    def test_multimodal_score_calculation(self):
        """Test multi-modal score calculation."""
        individual_analyses = {
            'video': {'video_score': 0.85},
            'audio': {'audio_score': 0.78},
            'image': {'image_score': 0.82},
            'text': {'text_score': 0.75}
        }
        
        coherence_analysis = {'coherence_score': 0.80}
        platform = 'tiktok'
        
        score = self.evaluator._calculate_multimodal_score(
            individual_analyses, coherence_analysis, platform
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_multimodal_score_with_missing_modalities(self):
        """Test multi-modal score calculation with missing modalities."""
        individual_analyses = {
            'video': {'video_score': 0.85},
            'text': {'text_score': 0.75}
        }
        
        coherence_analysis = {'coherence_score': 0.80}
        platform = 'youtube_standard'
        
        score = self.evaluator._calculate_multimodal_score(
            individual_analyses, coherence_analysis, platform
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    # Utility and Helper Method Tests
    
    def test_multimodal_data_extraction(self):
        """Test multi-modal data extraction."""
        # Test string content
        string_content = "Test content"
        extracted = self.evaluator._extract_multimodal_data(string_content, None)
        
        self.assertIn('text_data', extracted)
        self.assertEqual(extracted['text_data']['content'], string_content)
        
        # Test structured content
        structured_content = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data
        }
        
        extracted = self.evaluator._extract_multimodal_data(structured_content, None)
        
        self.assertEqual(extracted['text_data'], self.sample_text_data)
        self.assertEqual(extracted['video_data'], self.sample_video_data)
    
    def test_platform_engagement_factor_calculation(self):
        """Test platform engagement factor calculation."""
        individual_analyses = {
            'video': {'video_score': 0.85},
            'audio': {'audio_score': 0.78}
        }
        
        # Test different platforms
        tiktok_factor = self.evaluator._calculate_platform_engagement_factor('tiktok', individual_analyses)
        youtube_factor = self.evaluator._calculate_platform_engagement_factor('youtube_standard', individual_analyses)
        
        self.assertGreaterEqual(tiktok_factor, 0.0)
        self.assertLessEqual(tiktok_factor, 1.0)
        self.assertGreaterEqual(youtube_factor, 0.0)
        self.assertLessEqual(youtube_factor, 1.0)
    
    def test_text_platform_optimization(self):
        """Test text platform optimization."""
        # TikTok text (short with hashtags)
        tiktok_text = "Great content! #viral #trending"
        tiktok_score = self.evaluator._calculate_text_platform_optimization(tiktok_text, 'tiktok')
        
        # LinkedIn text (longer professional)
        linkedin_text = "After 10 years in the industry, I've learned that effective communication is the key to successful leadership. Here are five principles that have guided my approach to team management and stakeholder engagement throughout my career."
        linkedin_score = self.evaluator._calculate_text_platform_optimization(linkedin_text, 'linkedin')
        
        self.assertGreaterEqual(tiktok_score, 0.0)
        self.assertLessEqual(tiktok_score, 1.0)
        self.assertGreaterEqual(linkedin_score, 0.0)
        self.assertLessEqual(linkedin_score, 1.0)
    
    # Performance and Edge Case Tests
    
    def test_evaluation_performance(self):
        """Test evaluation performance and timing."""
        content = {
            'text_data': self.sample_text_data,
            'video_data': self.sample_video_data,
            'audio_data': self.sample_audio_data
        }
        
        start_time = time.time()
        result = self.evaluator.evaluate(content, self.sample_context)
        end_time = time.time()
        
        evaluation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within reasonable time (e.g., 100ms for mock implementation)
        self.assertLess(evaluation_time, 100)
        
        # Check evaluation metadata
        metadata = result['evaluation_metadata']
        self.assertIn('evaluation_time', metadata)
        self.assertIn('level', metadata)
        self.assertEqual(metadata['level'], 3)
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        empty_content = ""
        result = self.evaluator.evaluate(empty_content, self.sample_context)
        
        self.assertIn('multimodal_score', result)
        self.assertIn('text', result['content_modalities'])
    
    def test_invalid_platform_handling(self):
        """Test handling of invalid platform."""
        content = self.sample_text_data
        context = {'platform': 'invalid_platform'}
        
        result = self.evaluator.evaluate(content, context)
        
        # Should not crash, should use general/default handling
        self.assertIn('multimodal_score', result)
    
    def test_missing_context_handling(self):
        """Test handling of missing context."""
        content = self.sample_text_data
        
        result = self.evaluator.evaluate(content, None)
        
        self.assertIn('multimodal_score', result)
        self.assertIn('evaluation_metadata', result)
    
    # Recommendation Generation Tests
    
    def test_video_improvement_suggestions(self):
        """Test video improvement suggestion generation."""
        # Create low-quality frame analysis to trigger suggestions
        frame_analysis = {'composition_quality': 0.6, 'lighting_quality': 0.5}
        pacing_analysis = {'overall_pacing': 0.8}
        storytelling_analysis = {'hook_strength': 0.6}
        technical_analysis = {'technical_score': 0.9}
        
        suggestions = self.evaluator._generate_video_improvements(
            frame_analysis, pacing_analysis, storytelling_analysis, technical_analysis
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)  # Should have suggestions for low scores
    
    def test_cross_modal_recommendations(self):
        """Test cross-modal recommendation generation."""
        # Create analyses with low coherence scores
        sync_analysis = {'synchronization_score': 0.6}
        narrative_analysis = {'narrative_coherence_score': 0.65}
        engagement_analysis = {'engagement_optimization_score': 0.8}
        synergy_analysis = {'platform_synergy_score': 0.6}
        
        recommendations = self.evaluator._generate_cross_modal_recommendations(
            sync_analysis, narrative_analysis, engagement_analysis, synergy_analysis
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)  # Should have recommendations for low scores
    
    def test_platform_recommendations_generation(self):
        """Test platform-specific recommendation generation."""
        content_data = {
            'video_data': self.sample_video_data,
            'audio_data': self.sample_audio_data
        }
        
        individual_analyses = {
            'video': {'video_score': 0.85},
            'audio': {'audio_score': 0.78}
        }
        
        coherence_analysis = {'coherence_score': 0.75}
        
        recommendations = self.evaluator._generate_platform_recommendations(
            content_data, individual_analyses, coherence_analysis, 'tiktok'
        )
        
        self.assertIn('format_optimizations', recommendations)
        self.assertIn('content_strategies', recommendations)
        self.assertIn('technical_improvements', recommendations)


class TestMultiModalEvaluatorIntegration(unittest.TestCase):
    """Integration tests for multi-modal evaluator with real-world scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.evaluator = MultiModalEvaluator()
    
    def test_tiktok_viral_video_scenario(self):
        """Test complete TikTok viral video evaluation scenario."""
        content = {
            'text_data': {
                'content': 'POV: You found the productivity hack that changed everything 🔥 #productivity #lifehack #mindset'
            },
            'video_data': {
                'duration': 28.0,
                'resolution': 1080,
                'fps': 30,
                'aspect_ratio': '9:16'
            },
            'audio_data': {
                'duration': 28.0,
                'has_music': True,
                'has_voice': True,
                'sample_rate': 44100,
                'bit_depth': 16,
                'format': 'aac'
            }
        }
        
        context = {'platform': 'tiktok'}
        result = self.evaluator.evaluate(content, context)
        
        # Comprehensive result validation
        self.assertIn('multimodal_score', result)
        self.assertIn('coherence_analysis', result)
        self.assertEqual(set(result['content_modalities']), {'text', 'video', 'audio'})
        
        # Platform-specific validation
        self.assertEqual(result['evaluation_metadata']['platform'], 'tiktok')
        self.assertIn('recommendations', result)
    
    def test_youtube_educational_content_scenario(self):
        """Test complete YouTube educational content evaluation scenario."""
        content = {
            'text_data': {
                'content': 'In this comprehensive tutorial, I will walk you through the complete process of building a modern web application from scratch.'
            },
            'video_data': {
                'duration': 480.0,  # 8 minutes
                'resolution': 1080,
                'fps': 30,
                'aspect_ratio': '16:9'
            },
            'audio_data': {
                'duration': 480.0,
                'has_music': False,
                'has_voice': True,
                'sample_rate': 48000,
                'bit_depth': 24,
                'format': 'wav'
            },
            'image_data': {
                'resolution': [1920, 1080],
                'format': 'png',
                'file_size_mb': 1.2,
                'has_logo': True,
                'has_faces': True
            }
        }
        
        context = {'platform': 'youtube_standard'}
        result = self.evaluator.evaluate(content, context)
        
        # Should analyze all modalities
        self.assertEqual(set(result['content_modalities']), {'text', 'video', 'audio', 'image'})
        
        # Should have high coherence score for well-structured educational content
        self.assertGreaterEqual(result['coherence_analysis']['coherence_score'], 0.6)
    
    def test_cross_platform_comparison(self):
        """Test same content across different platforms."""
        base_content = {
            'text_data': {'content': 'Great content for social media sharing'},
            'video_data': {
                'duration': 30.0,
                'resolution': 1080,
                'fps': 30,
                'aspect_ratio': '16:9'
            }
        }
        
        platforms = ['tiktok', 'instagram_reels', 'youtube_shorts']
        results = {}
        
        for platform in platforms:
            context = {'platform': platform}
            results[platform] = self.evaluator.evaluate(base_content, context)
        
        # All should have valid scores
        for platform, result in results.items():
            self.assertIn('multimodal_score', result)
            self.assertGreaterEqual(result['multimodal_score'], 0.0)
            self.assertLessEqual(result['multimodal_score'], 1.0)
        
        # Platform-specific recommendations should differ
        tiktok_recs = results['tiktok']['recommendations']
        instagram_recs = results['instagram_reels']['recommendations']
        
        # Should have some different recommendations due to platform differences
        self.assertIsInstance(tiktok_recs, list)
        self.assertIsInstance(instagram_recs, list)


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2) 