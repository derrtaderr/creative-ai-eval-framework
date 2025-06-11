"""
Creative AI Evaluation Framework - Level 2 Demo
Temporal Evaluation & Lifecycle Prediction

This demo showcases the Level 2 temporal evaluation capabilities including:
- Immediate vs delayed engagement analysis
- Content lifecycle prediction with trajectory mapping
- Platform-specific temporal patterns
- Timing optimization recommendations
- Viral probability forecasting
"""

import json
import os
from datetime import datetime, timedelta
from src.evaluators.temporal_evaluator import TemporalEvaluator


def load_creator_profile():
    """Load the sample creator profile."""
    profile_path = "data/creator_profiles/creator_001_profile.json"
    with open(profile_path, 'r') as f:
        return json.load(f)


def main():
    """Run Level 2 Temporal evaluation demo."""
    
    print("⏱️  Creative AI Evaluation Framework - Level 2 Demo")
    print("=" * 60)
    print("Level 2: Temporal Evaluation & Lifecycle Prediction")
    print("Immediate vs delayed analysis, trajectory mapping, and timing optimization\n")
    
    # Initialize evaluator
    print("🔧 Initializing Temporal Evaluator...")
    evaluator = TemporalEvaluator({
        'time_windows': [0, 1, 6, 24, 72, 168]
    })
    
    # Load creator profile for context
    print("📁 Loading creator profile...")
    creator_profile = load_creator_profile()
    print(f"✅ Loaded profile for: {creator_profile['name']}\n")
    
    # Test content samples with different temporal profiles
    test_scenarios = [
        {
            "name": "Breaking News / Urgent Content",
            "content": "BREAKING: Just announced at the conference - this changes everything for AI startups. Here's what you need to know NOW before your competitors catch up.",
            "post_time": "2024-06-10T09:00:00",  # Peak morning hour
            "platform": "twitter",
            "expected": "High immediate engagement, fast decay typical of news content"
        },
        {
            "name": "Evergreen Educational Content",
            "content": "Here's the fundamental framework I learned after 10 years of building products: The 3-step validation process that prevents wasted development time. This timeless approach works regardless of industry or market conditions.",
            "post_time": "2024-06-10T17:00:00",  # Peak evening hour
            "platform": "linkedin", 
            "expected": "Moderate immediate, strong sustained engagement over time"
        },
        {
            "name": "Story-Driven Personal Content",
            "content": "My biggest failure as a founder taught me the most valuable lesson. I fired our best engineer because I misunderstood their communication style. Here's what I learned about leadership and how you can avoid this mistake.",
            "post_time": "2024-06-10T12:00:00",  # Lunch hour
            "platform": "linkedin",
            "expected": "Strong immediate hook, good discussion potential, sustained engagement"
        },
        {
            "name": "Poor Timing Example",
            "content": "This incredible startup strategy could revolutionize your business approach. What do you think about this framework?",
            "post_time": "2024-06-09T03:00:00",  # Off-peak hour, weekend
            "platform": "linkedin",
            "expected": "Low immediate engagement due to poor timing, moderate content quality"
        },
        {
            "name": "High Viral Potential",
            "content": "What if I told you the secret that 99% of successful entrepreneurs know but won't share? Here's the truth about startup success that nobody talks about. Share this if you think others should know!",
            "post_time": "2024-06-10T19:00:00",  # Peak evening
            "platform": "linkedin",
            "expected": "High viral probability, strong immediate and sustained metrics"
        }
    ]
    
    print("🧪 Running Level 2 Temporal Evaluations...")
    print("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        print(f"Content: \"{scenario['content'][:80]}{'...' if len(scenario['content']) > 80 else ''}\"")
        print(f"Timing: {scenario['post_time']} on {scenario['platform']}")
        print("-" * 50)
        
        # Evaluate the content with temporal context
        context = {
            'creator_profile': creator_profile,
            'platform': scenario['platform'],
            'post_time': scenario['post_time']
        }
        
        result = evaluator.evaluate(scenario['content'], context)
        
        # Display temporal scores
        print(f"⏱️  TEMPORAL SCORES:")
        print(f"   Overall Temporal Score: {result['temporal_score']:.3f}")
        print(f"   Immediate Score:        {result['immediate_metrics']['immediate_score']:.3f}")
        print(f"   Delayed Score:          {result['delayed_metrics']['delayed_score']:.3f}")
        
        print(f"\n🚀 IMMEDIATE METRICS (T+0 to T+1):")
        immediate = result['immediate_metrics']
        print(f"   Timing Optimization:    {immediate['timing_optimization']:.3f}")
        print(f"   Content Urgency:        {immediate['content_urgency']:.3f}")
        print(f"   Hook Strength:          {immediate['hook_strength']:.3f}")
        print(f"   Platform Alignment:     {immediate['platform_alignment']:.3f}")
        print(f"   Viral Window Prob:      {immediate['viral_window_probability']:.3f}")
        
        print(f"\n⏳ DELAYED METRICS (T+24 to T+168):")
        delayed = result['delayed_metrics']
        print(f"   Content Depth:          {delayed['content_depth']:.3f}")
        print(f"   Shareability:           {delayed['shareability']:.3f}")
        print(f"   Evergreen Potential:    {delayed['evergreen_potential']:.3f}")
        print(f"   Discussion Driver:      {delayed['discussion_driver']:.3f}")
        
        print(f"\n📈 LIFECYCLE PREDICTION:")
        lifecycle = result['lifecycle_prediction']
        print(f"   Content Type:           {lifecycle['content_type']}")
        print(f"   Peak Engagement Time:   T+{lifecycle['peak_engagement_time']} hours")
        print(f"   Total Lifetime Value:   {lifecycle['total_lifetime_value']:.3f}")
        print(f"   Engagement Persistence: {lifecycle['engagement_persistence']:.3f}")
        print(f"   Viral Probability:      {lifecycle['viral_probability']:.3f}")
        
        print(f"\n📊 ENGAGEMENT TRAJECTORY:")
        for point in lifecycle['trajectory_points']:
            time_label = f"T+{point['time']}h"
            bar_length = int(point['engagement'] * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {time_label:>6}: {bar} {point['engagement']:.3f}")
        
        print(f"\n💡 RECOMMENDATIONS ({len(result['recommendations'])} total):")
        for rec in result['recommendations']:
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"   {priority_emoji} {rec['type'].replace('_', ' ').title()}")
            print(f"      {rec['message']}")
            if 'suggestions' in rec and len(rec['suggestions']) > 0:
                print(f"      Suggestions:")
                for suggestion in rec['suggestions'][:2]:  # Show first 2
                    print(f"        • {suggestion}")
                if len(rec['suggestions']) > 2:
                    print(f"        • ... and {len(rec['suggestions']) - 2} more")
        
        print(f"\n⚡ Evaluation completed in {result['evaluation_metadata']['evaluation_time']:.3f}ms")
        print("=" * 60)
    
    # Demonstrate temporal comparison across platforms
    print(f"\n🔄 Cross-Platform Temporal Comparison")
    print("-" * 40)
    
    comparison_content = "Here's the startup lesson that changed everything for me: Customer feedback beats perfect planning every time. What's your experience?"
    platforms = ['twitter', 'linkedin', 'instagram']
    
    print(f"Content: \"{comparison_content[:60]}...\"")
    print(f"Posted at optimal time for each platform\n")
    
    platform_results = {}
    for platform in platforms:
        # Use optimal time for each platform
        optimal_times = {
            'twitter': '2024-06-10T09:00:00',
            'linkedin': '2024-06-10T17:00:00', 
            'instagram': '2024-06-10T19:00:00'
        }
        
        context = {
            'creator_profile': creator_profile,
            'platform': platform,
            'post_time': optimal_times[platform]
        }
        
        result = evaluator.evaluate(comparison_content, context)
        platform_results[platform] = result
        
        print(f"📱 {platform.upper()}:")
        print(f"   Temporal Score: {result['temporal_score']:.3f} | "
              f"Immediate: {result['immediate_metrics']['immediate_score']:.3f} | "
              f"Delayed: {result['delayed_metrics']['delayed_score']:.3f}")
        print(f"   Content Type: {result['lifecycle_prediction']['content_type']} | "
              f"Viral Prob: {result['lifecycle_prediction']['viral_probability']:.3f}")
    
    # Show platform-specific insights
    print(f"\n🎯 PLATFORM INSIGHTS:")
    best_immediate = max(platforms, key=lambda p: platform_results[p]['immediate_metrics']['immediate_score'])
    best_sustained = max(platforms, key=lambda p: platform_results[p]['delayed_metrics']['delayed_score'])
    best_viral = max(platforms, key=lambda p: platform_results[p]['lifecycle_prediction']['viral_probability'])
    
    print(f"   Best for immediate engagement: {best_immediate.upper()}")
    print(f"   Best for sustained engagement: {best_sustained.upper()}")
    print(f"   Highest viral probability: {best_viral.upper()}")
    
    # Show evaluator performance stats
    print(f"\n📊 EVALUATOR PERFORMANCE STATS:")
    if hasattr(evaluator, 'performance_history') and evaluator.performance_history:
        avg_time = sum(evaluator.performance_history) / len(evaluator.performance_history)
        print(f"   Average evaluation time: {avg_time:.3f}ms")
        print(f"   Total evaluations: {len(evaluator.performance_history)}")
        print(f"   Fastest evaluation: {min(evaluator.performance_history):.3f}ms")
        print(f"   Slowest evaluation: {max(evaluator.performance_history):.3f}ms")
    
    print(f"\n🎉 Level 2 Demo Complete!")
    print("=" * 60)
    print("✅ Demonstrated immediate vs delayed engagement analysis")
    print("✅ Showed content lifecycle prediction with trajectory mapping")
    print("✅ Validated platform-specific temporal patterns")
    print("✅ Generated timing optimization recommendations")
    print("✅ Tested viral probability forecasting")
    print("✅ Compared cross-platform temporal performance")
    print(f"\n🚀 Ready for Phase 4: Level 3 Multi-Modal Assessment!")


if __name__ == "__main__":
    main() 