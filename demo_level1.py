"""
Creative AI Evaluation Framework - Level 1 Demo
Authenticity vs Performance Evaluation

This demo showcases the Level 1 evaluation capabilities including:
- Dynamic authenticity threshold calculation
- Viral pattern recognition
- Performance prediction
- Balanced authenticity vs performance scoring
- Creator-specific recommendations
"""

import json
import os
from src.evaluators.authenticity_evaluator import AuthenticityPerformanceEvaluator


def load_creator_profile():
    """Load the sample creator profile."""
    profile_path = "data/creator_profiles/creator_001_profile.json"
    with open(profile_path, 'r') as f:
        return json.load(f)


def main():
    """Run Level 1 Authenticity vs Performance evaluation demo."""
    
    print("🚀 Creative AI Evaluation Framework - Level 1 Demo")
    print("=" * 60)
    print("Level 1: Authenticity vs Performance Evaluation")
    print("Dynamic threshold calculation, viral pattern recognition, and balanced scoring\n")
    
    # Initialize evaluator
    print("🔧 Initializing Authenticity Performance Evaluator...")
    evaluator = AuthenticityPerformanceEvaluator({
        'min_authenticity_threshold': 0.6,
        'performance_weight': 0.4,
        'authenticity_weight': 0.6
    })
    
    # Load creator profile
    print("📁 Loading creator profile...")
    creator_profile = load_creator_profile()
    print(f"✅ Loaded profile for: {creator_profile['name']}")
    print(f"   Voice consistency: {creator_profile['authenticity_settings']['voice_consistency']}")
    print(f"   Growth focus: {creator_profile['authenticity_settings']['growth_focus']}")
    print(f"   Variance tolerance: {creator_profile['authenticity_settings']['variance_tolerance']}\n")
    
    # Test content samples with different authenticity/performance profiles
    test_contents = [
        {
            "name": "High Authenticity + High Viral Potential",
            "content": "Building in public has taught me the most valuable startup lesson: What's your biggest founder mistake? Here's mine - I spent 6 months building a feature nobody wanted. Share your story below!",
            "expected": "Should score high on both authenticity and viral potential"
        },
        {
            "name": "High Authenticity + Low Viral Potential", 
            "content": "Today I reflected on my founder journey and the challenges of building a startup. Product development continues to teach me about the importance of customer feedback in our process.",
            "expected": "Should score high authenticity but suggest viral improvements"
        },
        {
            "name": "Low Authenticity + High Viral Potential",
            "content": "🚨 SHOCKING: This ONE SECRET will 10X your business OVERNIGHT! Don't miss out! Click here NOW before it's too late! Limited time offer! What do you think?",
            "expected": "Should fail authenticity threshold despite high viral signals"
        },
        {
            "name": "Balanced Authentic + Viral",
            "content": "Here's what 5 years of startup failures taught me about product-market fit. The secret isn't having the perfect product - it's finding the perfect customer. What's been your biggest product lesson?",
            "expected": "Should demonstrate good balance between authenticity and performance"
        },
        {
            "name": "Platform-Optimized Content",
            "content": "Fundraising reality: Got rejected by 47 investors before our seed round. Each 'no' was a lesson. Here are the 3 biggest mistakes I made and how I fixed them. What's your fundraising story? Tag an entrepreneur who needs to see this!",
            "expected": "Should score well on viral patterns and length optimization"
        }
    ]
    
    print("🧪 Running Level 1 Evaluations...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_contents, 1):
        print(f"\n📝 Test Case {i}: {test_case['name']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Content: \"{test_case['content'][:100]}{'...' if len(test_case['content']) > 100 else ''}\"")
        print("-" * 50)
        
        # Evaluate the content
        context = {
            'creator_profile': creator_profile,
            'platform': 'linkedin'
        }
        
        result = evaluator.evaluate(test_case['content'], context)
        
        # Display results
        print(f"🎯 SCORES:")
        print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
        print(f"   Viral Potential:    {result['viral_potential']:.3f}")
        print(f"   Dynamic Threshold:  {result['dynamic_threshold']:.3f}")
        print(f"   Authenticity Met:   {'✅' if result['authenticity_met'] else '❌'}")
        print(f"   Balanced Score:     {result['balanced_score']:.3f}")
        
        print(f"\n📊 PERFORMANCE PREDICTION:")
        for key, value in result['performance_prediction'].items():
            print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        
        print(f"\n💡 RECOMMENDATIONS ({len(result['recommendations'])} total):")
        for rec in result['recommendations']:
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"   {priority_emoji} {rec['type'].replace('_', ' ').title()}")
            print(f"      {rec['message']}")
            if len(rec['suggestions']) > 0:
                print(f"      Suggestions:")
                for suggestion in rec['suggestions'][:2]:  # Show first 2 suggestions
                    print(f"        • {suggestion}")
                if len(rec['suggestions']) > 2:
                    print(f"        • ... and {len(rec['suggestions']) - 2} more")
        
        print(f"\n⚡ Evaluation completed in {result['evaluation_metadata']['evaluation_time']:.3f}ms")
        print("=" * 60)
    
    # Demonstrate batch evaluation
    print(f"\n🔄 Batch Evaluation Demo")
    print("-" * 30)
    
    batch_content = [tc['content'] for tc in test_contents[:3]]
    batch_contexts = [{'creator_profile': creator_profile, 'platform': 'linkedin'}] * 3
    
    print(f"Evaluating {len(batch_content)} content pieces simultaneously...")
    batch_results = evaluator.batch_evaluate(batch_content, batch_contexts)
    
    print(f"\n📈 BATCH RESULTS SUMMARY:")
    for i, result in enumerate(batch_results):
        if 'error' not in result:
            print(f"   Content {i+1}: Auth={result['authenticity_score']:.2f} | "
                  f"Viral={result['viral_potential']:.2f} | "
                  f"Balanced={result['balanced_score']:.2f} | "
                  f"Met={'✅' if result['authenticity_met'] else '❌'}")
    
    # Show evaluator performance stats
    print(f"\n📊 EVALUATOR PERFORMANCE STATS:")
    if hasattr(evaluator, 'performance_history') and evaluator.performance_history:
        avg_time = sum(evaluator.performance_history) / len(evaluator.performance_history)
        print(f"   Average evaluation time: {avg_time:.3f}ms")
        print(f"   Total evaluations: {len(evaluator.performance_history)}")
        print(f"   Fastest evaluation: {min(evaluator.performance_history):.3f}ms")
        print(f"   Slowest evaluation: {max(evaluator.performance_history):.3f}ms")
    
    print(f"\n🎉 Level 1 Demo Complete!")
    print("=" * 60)
    print("✅ Demonstrated dynamic authenticity thresholds")
    print("✅ Showed viral pattern recognition")
    print("✅ Validated performance predictions")
    print("✅ Generated creator-specific recommendations")
    print("✅ Tested batch evaluation capabilities")
    print(f"\n🚀 Ready for Phase 2 Level 2: Temporal Evaluation!")


if __name__ == "__main__":
    main() 