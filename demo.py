#!/usr/bin/env python3
"""
Creative AI Evaluation Framework - Demo Script

This script demonstrates the basic functionality of the framework.
"""

import sys
import os
import json

# Add src to path
sys.path.append('src')

from evaluators import ContentContextEvaluator

def main():
    print("🚀 Creative AI Evaluation Framework Demo")
    print("=" * 50)
    
    # Initialize evaluator
    print("\n📦 Initializing evaluator...")
    evaluator = ContentContextEvaluator()
    print("✅ Evaluator initialized successfully!")
    
    # Load creator profile
    print("\n👤 Loading creator profile...")
    try:
        creator_profile = evaluator.load_creator_profile('data/creator_profiles/creator_001_profile.json')
        print(f"✅ Loaded profile for: {creator_profile['name']}")
        print(f"🎯 Platforms: {', '.join(creator_profile['platforms'])}")
    except Exception as e:
        print(f"❌ Error loading profile: {e}")
        return
    
    # Sample content to evaluate
    test_contents = [
        {
            "content": "Just launched our new AI-powered content optimization tool! The team has been working tirelessly to bring this vision to life. Excited to see how it helps creators grow their audience. #AI #startup #innovation",
            "platform": "linkedin",
            "description": "High-quality, on-brand content"
        },
        {
            "content": "OMG this new AI thing is CRAZY!!! 🤯🤯🤯 Everyone needs to try it NOW!!! #viral #trending #AI #amazing #wow #mustuse #gamechanging #revolution",
            "platform": "twitter", 
            "description": "Low-quality, off-brand content"
        },
        {
            "content": "Building great products requires understanding your users deeply. We spend 3 hours every week talking directly to customers. What's your approach to user research? #product #startup",
            "platform": "twitter",
            "description": "Medium-quality, somewhat on-brand"
        }
    ]
    
    print(f"\n🧪 Evaluating {len(test_contents)} content samples...")
    print("=" * 50)
    
    for i, sample in enumerate(test_contents, 1):
        print(f"\n📝 Sample {i}: {sample['description']}")
        print(f"📱 Platform: {sample['platform'].title()}")
        print(f"📄 Content: {sample['content'][:100]}{'...' if len(sample['content']) > 100 else ''}")
        
        try:
            result = evaluator.evaluate_content(
                sample['content'], 
                creator_profile, 
                platform=sample['platform']
            )
            
            print("\n📊 Results:")
            print(f"   🎯 Context Score: {result['context_score']:.3f}")
            print(f"   🎭 Voice Consistency: {result['voice_consistency']:.3f}")
            print(f"   📱 Platform Optimization: {result['platform_optimization']:.3f}")
            print(f"   📈 Trend Relevance: {result['trend_relevance']:.3f}")
            print(f"   ⚡ Execution Time: {result['execution_time']:.3f}s")
            
            if result.get('recommendations'):
                print("\n💡 Recommendations:")
                for rec in result['recommendations']:
                    print(f"   • {rec}")
            else:
                print("\n✨ No recommendations - content looks good!")
                
        except Exception as e:
            print(f"❌ Error evaluating content: {e}")
    
    # Performance summary
    print("\n" + "=" * 50)
    print("📈 Performance Summary")
    
    try:
        stats = evaluator.get_performance_stats()
        print(f"📊 Total Evaluations: {stats['total_evaluations']}")
        print(f"✅ Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"⚡ Average Time: {stats['avg_execution_time']:.3f}s")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("🚀 Ready to evaluate creative AI content at scale!")

if __name__ == "__main__":
    main() 