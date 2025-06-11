"""
Multi-Modal Evaluator Demo (Level 3)

Comprehensive demonstration of video, audio, image, and text content evaluation
with cross-modal coherence analysis and platform-specific optimization.

Key Demo Scenarios:
1. TikTok Video: Short-form video with trending audio
2. YouTube Tutorial: Long-form educational content with multiple modalities  
3. Instagram Reel: Visual-first content with music
4. LinkedIn Post: Professional image + text combination
5. Multi-Modal Campaign: All modalities working together
"""

import sys
import os
import json
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluators.multimodal_evaluator import MultiModalEvaluator


def create_sample_multimodal_content():
    """Create sample multi-modal content for different scenarios."""
    return {
        'tiktok_video': {
            'text_data': {
                'content': "POV: You discovered the productivity hack that changed everything 🔥 #productivity #lifehack #mindset"
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
            },
            'platform': 'tiktok'
        },
        
        'youtube_tutorial': {
            'text_data': {
                'content': "In this comprehensive tutorial, I'll walk you through the complete process of building a modern web application from scratch. We'll cover everything from setup to deployment, with real-world examples and best practices."
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
            },
            'platform': 'youtube_standard'
        },
        
        'instagram_reel': {
            'text_data': {
                'content': "Morning routine that actually works ✨ Save this for later! #morningroutine #wellness #selfcare"
            },
            'video_data': {
                'duration': 45.0,
                'resolution': 1080,
                'fps': 30,
                'aspect_ratio': '9:16'
            },
            'audio_data': {
                'duration': 45.0,
                'has_music': True,
                'has_voice': False,
                'sample_rate': 44100,
                'bit_depth': 16,
                'format': 'mp3'
            },
            'image_data': {
                'resolution': [1080, 1920],
                'format': 'jpg',
                'file_size_mb': 2.8,
                'has_logo': False,
                'has_faces': True
            },
            'platform': 'instagram_reels'
        },
        
        'linkedin_post': {
            'text_data': {
                'content': "After 10 years in tech leadership, here are the 5 communication patterns I've seen separate great leaders from good ones. Each of these took me years to learn, but you can start implementing them today. Thread below 👇"
            },
            'image_data': {
                'resolution': [1200, 630],
                'format': 'png',
                'file_size_mb': 0.8,
                'has_logo': True,
                'has_faces': False
            },
            'platform': 'linkedin'
        },
        
        'multi_modal_campaign': {
            'text_data': {
                'content': "Introducing our revolutionary new product that's changing how creators work. Experience the future of content creation."
            },
            'video_data': {
                'duration': 60.0,
                'resolution': 1080, 
                'fps': 60,
                'aspect_ratio': '16:9'
            },
            'audio_data': {
                'duration': 60.0,
                'has_music': True,
                'has_voice': True,
                'has_sound_effects': True,
                'sample_rate': 48000,
                'bit_depth': 24,
                'format': 'wav'
            },
            'image_data': {
                'resolution': [1920, 1080],
                'format': 'png',
                'file_size_mb': 3.2,
                'has_logo': True,
                'has_faces': True
            },
            'platform': 'youtube_standard'
        }
    }


def run_individual_modality_demos(evaluator):
    """Demonstrate individual modality analysis capabilities."""
    print("🎯 Individual Modality Analysis Demos")
    print("=" * 60)
    
    # Video Analysis Demo
    print("\n🎥 Video Content Analysis")
    print("-" * 30)
    
    video_data = {
        'duration': 30.0,
        'resolution': 1080,
        'fps': 30,
        'aspect_ratio': '9:16'
    }
    
    video_context = {'platform': 'tiktok'}
    video_result = evaluator.analyze_video_content(video_data, video_context)
    
    print(f"Video Score: {video_result['video_score']}")
    print(f"Frame Quality: {video_result['frame_analysis']['frame_quality_score']}")
    print(f"Storytelling: {video_result['storytelling_analysis']['storytelling_score']}")
    print(f"Technical Quality: {video_result['technical_analysis']['technical_score']}")
    print(f"Platform Compliance: {video_result['technical_analysis']['platform_compliance']}")
    
    # Audio Analysis Demo
    print("\n🎵 Audio Content Analysis")
    print("-" * 30)
    
    audio_data = {
        'duration': 30.0,
        'has_music': True,
        'has_voice': True,
        'sample_rate': 44100,
        'bit_depth': 16,
        'format': 'mp3'
    }
    
    audio_context = {'platform': 'tiktok'}
    audio_result = evaluator.analyze_audio_content(audio_data, audio_context)
    
    print(f"Audio Score: {audio_result['audio_score']}")
    print(f"Voice Quality: {audio_result['voice_analysis']['voice_score']}")
    print(f"Music Integration: {audio_result['music_analysis']['music_score']}")
    print(f"Sound Design: {audio_result['sound_design_analysis']['sound_design_score']}")
    print(f"Detected Tone: {audio_result['voice_analysis']['detected_tone']}")
    
    # Image Analysis Demo
    print("\n🖼️ Image Content Analysis")
    print("-" * 30)
    
    image_data = {
        'resolution': [1080, 1080],
        'format': 'png',
        'file_size_mb': 1.5,
        'has_logo': True,
        'has_faces': True
    }
    
    image_context = {'platform': 'instagram_post', 'brand_colors': ['#FF6B6B', '#4ECDC4']}
    image_result = evaluator.analyze_image_content(image_data, image_context)
    
    print(f"Image Score: {image_result['image_score']}")
    print(f"Composition: {image_result['composition_analysis']['composition_score']}")
    print(f"Branding: {image_result['branding_analysis']['branding_score']}")
    print(f"Emotional Impact: {image_result['emotional_analysis']['emotional_impact_score']}")
    print(f"Detected Emotion: {image_result['emotional_analysis']['detected_emotion']}")


def run_cross_modal_coherence_demo(evaluator):
    """Demonstrate cross-modal coherence analysis."""
    print("\n🔄 Cross-Modal Coherence Analysis")
    print("=" * 60)
    
    # Create sample content with all modalities
    content_data = {
        'text_data': {'content': 'Professional tutorial content'},
        'video_data': {'duration': 60.0, 'resolution': 1080, 'fps': 30, 'aspect_ratio': '16:9'},
        'audio_data': {'duration': 60.0, 'has_music': True, 'has_voice': True},
        'image_data': {'resolution': [1920, 1080], 'format': 'png', 'has_logo': True}
    }
    
    # Mock individual analyses
    individual_analyses = {
        'video': {'video_score': 0.85},
        'audio': {'audio_score': 0.78},
        'image': {'image_score': 0.82},
        'text': {'text_score': 0.75}
    }
    
    context = {'platform': 'youtube_standard'}
    coherence_result = evaluator.analyze_cross_modal_coherence(
        content_data, individual_analyses, context
    )
    
    print(f"Overall Coherence Score: {coherence_result['coherence_score']}")
    print(f"Synchronization: {coherence_result['synchronization_analysis']['synchronization_score']}")
    print(f"Narrative Coherence: {coherence_result['narrative_coherence']['narrative_coherence_score']}")
    print(f"Engagement Optimization: {coherence_result['engagement_optimization']['engagement_optimization_score']}")
    print(f"Platform Synergy: {coherence_result['platform_synergy']['platform_synergy_score']}")
    
    print("\nCross-Modal Recommendations:")
    for rec in coherence_result['cross_modal_recommendations']:
        print(f"  • {rec}")


def run_platform_specific_demos(evaluator, sample_content):
    """Demonstrate platform-specific multi-modal optimization."""
    print("\n📱 Platform-Specific Multi-Modal Optimization")
    print("=" * 60)
    
    platforms = ['tiktok', 'youtube_standard', 'instagram_reels', 'linkedin']
    
    for platform in platforms:
        print(f"\n{platform.upper()} Platform Analysis")
        print("-" * 40)
        
        # Use appropriate content for platform
        if platform == 'tiktok' and 'tiktok_video' in sample_content:
            content = sample_content['tiktok_video']
        elif platform == 'youtube_standard' and 'youtube_tutorial' in sample_content:
            content = sample_content['youtube_tutorial']
        elif platform == 'instagram_reels' and 'instagram_reel' in sample_content:
            content = sample_content['instagram_reel']
        elif platform == 'linkedin' and 'linkedin_post' in sample_content:
            content = sample_content['linkedin_post']
        else:
            continue
        
        # Run evaluation
        context = {'platform': platform}
        result = evaluator.evaluate(content, context)
        
        print(f"Multi-Modal Score: {result['multimodal_score']}")
        print(f"Modalities Analyzed: {', '.join(result['content_modalities'])}")
        print(f"Coherence Score: {result['coherence_analysis']['coherence_score']}")
        
        # Platform optimization
        platform_opt = result['platform_optimization']
        if platform_opt['format_optimizations']:
            print("Format Optimizations:")
            for opt in platform_opt['format_optimizations'][:2]:  # Show first 2
                print(f"  • {opt}")
        
        if platform_opt['content_strategies']:
            print("Content Strategies:")
            for strategy in platform_opt['content_strategies'][:2]:  # Show first 2
                print(f"  • {strategy}")


def run_comprehensive_evaluation_demo(evaluator, sample_content):
    """Run comprehensive multi-modal evaluation scenarios."""
    print("\n🚀 Comprehensive Multi-Modal Evaluation")
    print("=" * 60)
    
    scenarios = [
        ('TikTok Viral Video', 'tiktok_video'),
        ('YouTube Educational Content', 'youtube_tutorial'),
        ('Instagram Lifestyle Reel', 'instagram_reel'),
        ('LinkedIn Professional Post', 'linkedin_post'),
        ('Multi-Modal Marketing Campaign', 'multi_modal_campaign')
    ]
    
    results_summary = []
    
    for scenario_name, content_key in scenarios:
        if content_key not in sample_content:
            continue
            
        print(f"\n📊 {scenario_name}")
        print("-" * 50)
        
        content = sample_content[content_key]
        platform = content.get('platform', 'general')
        context = {'platform': platform}
        
        # Run evaluation
        start_time = time.time()
        result = evaluator.evaluate(content, context)
        evaluation_time = (time.time() - start_time) * 1000
        
        # Display results
        print(f"Platform: {platform.upper()}")
        print(f"Multi-Modal Score: {result['multimodal_score']}")
        print(f"Evaluation Time: {evaluation_time:.1f}ms")
        print(f"Modalities: {', '.join(result['content_modalities'])}")
        
        # Individual modality scores
        individual_scores = {}
        for modality, analysis in result['individual_analyses'].items():
            score_key = f'{modality}_score'
            if score_key in analysis:
                individual_scores[modality] = analysis[score_key]
                print(f"{modality.title()} Score: {analysis[score_key]}")
        
        print(f"Cross-Modal Coherence: {result['coherence_analysis']['coherence_score']}")
        
        # Top recommendations
        if result['recommendations']:
            print("Top Recommendations:")
            for rec in result['recommendations'][:3]:  # Show top 3
                print(f"  • {rec}")
        
        # Store for summary
        results_summary.append({
            'scenario': scenario_name,
            'platform': platform,
            'multimodal_score': result['multimodal_score'],
            'coherence_score': result['coherence_analysis']['coherence_score'],
            'modalities': result['content_modalities'],
            'evaluation_time': evaluation_time
        })
    
    return results_summary


def run_business_insights_analysis(results_summary):
    """Analyze business insights from multi-modal evaluation results."""
    print("\n💼 Business Insights & Performance Analysis")
    print("=" * 60)
    
    if not results_summary:
        print("No results to analyze.")
        return
    
    # Performance metrics
    avg_score = sum(r['multimodal_score'] for r in results_summary) / len(results_summary)
    avg_coherence = sum(r['coherence_score'] for r in results_summary) / len(results_summary)
    avg_eval_time = sum(r['evaluation_time'] for r in results_summary) / len(results_summary)
    
    print(f"📈 Performance Metrics:")
    print(f"  Average Multi-Modal Score: {avg_score:.3f}")
    print(f"  Average Coherence Score: {avg_coherence:.3f}")
    print(f"  Average Evaluation Time: {avg_eval_time:.1f}ms")
    
    # Platform performance comparison
    platform_scores = {}
    for result in results_summary:
        platform = result['platform']
        if platform not in platform_scores:
            platform_scores[platform] = []
        platform_scores[platform].append(result['multimodal_score'])
    
    print(f"\n🎯 Platform Performance Comparison:")
    for platform, scores in platform_scores.items():
        avg_platform_score = sum(scores) / len(scores)
        print(f"  {platform.upper()}: {avg_platform_score:.3f} (based on {len(scores)} evaluation{'s' if len(scores) != 1 else ''})")
    
    # Modality usage analysis
    modality_usage = {}
    for result in results_summary:
        for modality in result['modalities']:
            modality_usage[modality] = modality_usage.get(modality, 0) + 1
    
    print(f"\n🎨 Modality Usage Analysis:")
    total_evaluations = len(results_summary)
    for modality, count in sorted(modality_usage.items()):
        percentage = (count / total_evaluations) * 100
        print(f"  {modality.title()}: {count}/{total_evaluations} scenarios ({percentage:.1f}%)")
    
    # Best performing scenarios
    best_scenario = max(results_summary, key=lambda x: x['multimodal_score'])
    most_coherent = max(results_summary, key=lambda x: x['coherence_score'])
    
    print(f"\n🏆 Top Performers:")
    print(f"  Highest Multi-Modal Score: {best_scenario['scenario']} ({best_scenario['multimodal_score']:.3f})")
    print(f"  Best Cross-Modal Coherence: {most_coherent['scenario']} ({most_coherent['coherence_score']:.3f})")
    
    # Business recommendations
    print(f"\n💡 Business Recommendations:")
    
    if avg_score >= 0.8:
        print("  • Excellent multi-modal content quality - ready for premium tier positioning")
    elif avg_score >= 0.7:
        print("  • Good content quality - focus on coherence improvements for premium features")
    else:
        print("  • Content quality needs improvement - prioritize individual modality optimization")
    
    if avg_coherence >= 0.8:
        print("  • Strong cross-modal synergy - highlight this as competitive advantage")
    else:
        print("  • Improve cross-modal coherence for better user engagement")
    
    if avg_eval_time <= 50:
        print("  • Fast evaluation times - suitable for real-time content optimization")
    else:
        print("  • Consider performance optimization for real-time use cases")


def save_demo_results(results_summary):
    """Save demo results to JSON file."""
    output_data = {
        'demo_metadata': {
            'demo_type': 'Level 3 Multi-Modal Assessment',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.4.0',
            'evaluator': 'MultiModalEvaluator'
        },
        'scenarios': results_summary,
        'summary_stats': {
            'total_scenarios': len(results_summary),
            'average_multimodal_score': sum(r['multimodal_score'] for r in results_summary) / len(results_summary) if results_summary else 0,
            'average_coherence_score': sum(r['coherence_score'] for r in results_summary) / len(results_summary) if results_summary else 0,
            'platforms_tested': list(set(r['platform'] for r in results_summary)),
            'modalities_coverage': {
                'video': sum(1 for r in results_summary if 'video' in r['modalities']),
                'audio': sum(1 for r in results_summary if 'audio' in r['modalities']),
                'image': sum(1 for r in results_summary if 'image' in r['modalities']),
                'text': sum(1 for r in results_summary if 'text' in r['modalities'])
            }
        }
    }
    
    filename = f'demo_multimodal_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Demo results saved to: {filename}")
    return filename


def main():
    """Run comprehensive Multi-Modal Evaluator demo."""
    print("🚀 Creative AI Evaluation Framework - Level 3 Multi-Modal Assessment Demo")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Framework Version: 1.4.0")
    print("Evaluator: MultiModalEvaluator")
    print()
    
    # Initialize evaluator
    print("🔧 Initializing Multi-Modal Evaluator...")
    evaluator = MultiModalEvaluator()
    
    # Create sample content
    print("📝 Creating sample multi-modal content...")
    sample_content = create_sample_multimodal_content()
    
    # Run demo sections
    try:
        # Individual modality demos
        run_individual_modality_demos(evaluator)
        
        # Cross-modal coherence demo
        run_cross_modal_coherence_demo(evaluator)
        
        # Platform-specific demos
        run_platform_specific_demos(evaluator, sample_content)
        
        # Comprehensive evaluation demos
        results_summary = run_comprehensive_evaluation_demo(evaluator, sample_content)
        
        # Business insights analysis
        run_business_insights_analysis(results_summary)
        
        # Save results
        output_file = save_demo_results(results_summary)
        
        print("\n✅ Multi-Modal Assessment Demo Completed Successfully!")
        print(f"📊 Evaluated {len(results_summary)} scenarios across multiple platforms and modalities")
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 