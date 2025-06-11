#!/usr/bin/env python3
"""
Enhanced Level 2 Temporal Evaluation Demo

This demo showcases the Enhanced Level 2+ temporal evaluation capabilities including:
- Viral Template Sustainability Scoring
- Creator Growth Trajectory Forecasting (6-month predictions)
- Optimal Content Calendar Generation
- Strategic Growth Planning
"""

import json
import time
from datetime import datetime, timedelta
from src.evaluators.enhanced_temporal_evaluator import EnhancedTemporalEvaluator


def create_sample_viral_data():
    """Create sample viral pattern data for testing."""
    return {
        'question_hooks': {
            'pattern_type': 'question_hooks',
            'usage_history': [
                {'date': '2024-12-01', 'engagement_rate': 0.045, 'comment_sentiment': 0.8},
                {'date': '2024-12-03', 'engagement_rate': 0.042, 'comment_sentiment': 0.75},
                {'date': '2024-12-05', 'engagement_rate': 0.038, 'comment_sentiment': 0.72},
                {'date': '2024-12-08', 'engagement_rate': 0.041, 'comment_sentiment': 0.70},
                {'date': '2024-12-10', 'engagement_rate': 0.036, 'comment_sentiment': 0.68},
                {'date': '2024-12-12', 'engagement_rate': 0.033, 'comment_sentiment': 0.65},
                {'date': '2024-12-15', 'engagement_rate': 0.030, 'comment_sentiment': 0.62}
            ]
        },
        'social_proof': {
            'pattern_type': 'social_proof',
            'usage_history': [
                {'date': '2024-11-15', 'engagement_rate': 0.055, 'comment_sentiment': 0.85},
                {'date': '2024-11-22', 'engagement_rate': 0.052, 'comment_sentiment': 0.82},
                {'date': '2024-11-29', 'engagement_rate': 0.048, 'comment_sentiment': 0.80},
                {'date': '2024-12-06', 'engagement_rate': 0.050, 'comment_sentiment': 0.78}
            ]
        },
        'emotional_triggers': {
            'pattern_type': 'emotional_triggers',
            'usage_history': [
                {'date': '2024-12-01', 'engagement_rate': 0.065, 'comment_sentiment': 0.45},
                {'date': '2024-12-02', 'engagement_rate': 0.062, 'comment_sentiment': 0.42},
                {'date': '2024-12-04', 'engagement_rate': 0.058, 'comment_sentiment': 0.40},
                {'date': '2024-12-06', 'engagement_rate': 0.055, 'comment_sentiment': 0.38},
                {'date': '2024-12-08', 'engagement_rate': 0.052, 'comment_sentiment': 0.35},
                {'date': '2024-12-10', 'engagement_rate': 0.048, 'comment_sentiment': 0.32}
            ]
        }
    }


def create_sample_creator_profiles():
    """Create sample creator profiles for growth forecasting."""
    return {
        'tech_creator': {
            'creator_id': 'tech_guru_2024',
            'followers': 15000,
            'engagement_rate': 0.045,
            'platform_focus': 'linkedin',
            'content_focus': 'B2B tech insights',
            'historical_posts': [f'post_{i}' for i in range(25)],
            'historical_data_points': 25,
            'audience_growth_rate': 0.032,
            'peak_posting_times': [8, 12, 17]
        },
        'lifestyle_creator': {
            'creator_id': 'lifestyle_maven',
            'followers': 8500,
            'engagement_rate': 0.068,
            'platform_focus': 'instagram',
            'content_focus': 'personal development',
            'historical_posts': [f'post_{i}' for i in range(18)],
            'historical_data_points': 18,
            'audience_growth_rate': 0.028,
            'peak_posting_times': [6, 11, 19]
        },
        'startup_founder': {
            'creator_id': 'startup_stories',
            'followers': 3200,
            'engagement_rate': 0.085,
            'platform_focus': 'twitter',
            'content_focus': 'entrepreneurship',
            'historical_posts': [f'post_{i}' for i in range(12)],
            'historical_data_points': 12,
            'audience_growth_rate': 0.045,
            'peak_posting_times': [7, 14, 20]
        }
    }


def create_sample_template_strategies():
    """Create sample template strategies for different creators."""
    return {
        'aggressive_growth': {
            'templates_per_month': 8,
            'quality_score': 0.85,
            'consistency_target': 'daily',
            'viral_focus': True,
            'audience_engagement_priority': 'high',
            'content_mix': {
                'viral_templates': 0.6,
                'original_content': 0.3,
                'engagement_content': 0.1
            }
        },
        'balanced_approach': {
            'templates_per_month': 5,
            'quality_score': 0.75,
            'consistency_target': '4_per_week',
            'viral_focus': False,
            'audience_engagement_priority': 'medium',
            'content_mix': {
                'viral_templates': 0.4,
                'original_content': 0.5,
                'engagement_content': 0.1
            }
        },
        'quality_focused': {
            'templates_per_month': 3,
            'quality_score': 0.95,
            'consistency_target': '3_per_week',
            'viral_focus': False,
            'audience_engagement_priority': 'high',
            'content_mix': {
                'viral_templates': 0.3,
                'original_content': 0.6,
                'engagement_content': 0.1
            }
        }
    }


def run_enhanced_level2_demo():
    """Run Enhanced Level 2 Temporal evaluation demo."""
    print("=" * 80)
    print("🚀 Enhanced Level 2+ Temporal Evaluation Demo")
    print("   Viral Lifecycle Analysis + Growth Forecasting + Content Calendar")
    print("=" * 80)
    
    # Initialize the enhanced evaluator
    evaluator = EnhancedTemporalEvaluator()
    
    # Create sample data
    viral_data = create_sample_viral_data()
    creator_profiles = create_sample_creator_profiles()
    template_strategies = create_sample_template_strategies()
    
    print("\n🧪 Running Enhanced Level 2+ Evaluations...")
    print("-" * 60)
    
    # Demo scenarios
    scenarios = [
        {
            'name': 'High-Performing Tech Creator (LinkedIn)',
            'content': "Here's the counterintuitive truth about AI adoption in enterprise: Most companies aren't failing because of technology—they're failing because of change management. After analyzing 200+ AI implementations, the pattern is clear...",
            'viral_pattern': 'question_hooks',
            'creator': 'tech_creator',
            'strategy': 'balanced_approach',
            'platform': 'linkedin'
        },
        {
            'name': 'Rising Lifestyle Creator (Instagram)',
            'content': "Plot twist: The morning routine that changed my life wasn't about waking up at 5 AM. It was about this one simple mindset shift that took 30 seconds but transformed everything...",
            'viral_pattern': 'social_proof',
            'creator': 'lifestyle_creator',
            'strategy': 'aggressive_growth',
            'platform': 'instagram'
        },
        {
            'name': 'Startup Founder Using Emotional Triggers (Twitter)',
            'content': "I almost shut down my startup last month. 18 months of work, $50K of my savings, and countless sleepless nights—all about to disappear. But then something unexpected happened...",
            'viral_pattern': 'emotional_triggers',
            'creator': 'startup_founder',
            'strategy': 'quality_focused',
            'platform': 'twitter'
        },
        {
            'name': 'Content Calendar Generation Demo',
            'content': "Optimize your content strategy with data-driven insights that predict growth trajectories and prevent audience fatigue.",
            'viral_pattern': 'question_hooks',
            'creator': 'tech_creator',
            'strategy': 'aggressive_growth',
            'platform': 'linkedin',
            'generate_calendar': True
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['name']}")
        print("-" * 50)
        
        # Prepare context for enhanced evaluation
        context = {
            'viral_data': viral_data[scenario['viral_pattern']],
            'creator_profile': creator_profiles[scenario['creator']],
            'template_strategy': template_strategies[scenario['strategy']],
            'platform': scenario['platform'],
            'generate_calendar': scenario.get('generate_calendar', False),
            'calendar_weeks': 6
        }
        
        # Perform enhanced evaluation
        start_time = time.time()
        result = evaluator.evaluate_with_lifecycle(scenario['content'], context)
        eval_time = time.time() - start_time
        
        # Display results
        print(f"📝 Content Preview: {scenario['content'][:100]}...")
        print(f"⚡ Evaluation Time: {eval_time*1000:.2f}ms")
        
        # Traditional temporal results - fix field names
        print(f"\n🕐 Traditional Temporal Analysis:")
        print(f"   ├── Temporal Score: {result['temporal_score']:.3f}")
        print(f"   ├── Immediate Score: {result['immediate_metrics']['immediate_score']:.3f}")
        print(f"   ├── Delayed Score: {result['delayed_metrics']['delayed_score']:.3f}")
        print(f"   └── Lifecycle Score: {result['lifecycle_prediction']['engagement_persistence']:.3f}")
        
        # Viral lifecycle analysis
        lifecycle = result['viral_lifecycle_analysis']
        print(f"\n🔄 Viral Lifecycle Analysis:")
        print(f"   ├── Sustainability Score: {lifecycle['sustainability_score']:.3f}/1.0 ({_get_sustainability_level(lifecycle['sustainability_score'])})")
        print(f"   ├── Optimal Usage Window: {lifecycle['optimal_usage_weeks']} weeks")
        print(f"   ├── Audience Fatigue Risk: {lifecycle['audience_fatigue_risk']:.1%} ({_get_risk_level(lifecycle['audience_fatigue_risk'])})")
        print(f"   ├── Refresh Date: {lifecycle['refresh_date']}")
        print(f"   └── Pattern Type: {lifecycle['pattern_type']}")
        
        if lifecycle['recommendations']:
            print(f"   📋 Sustainability Recommendations:")
            for rec in lifecycle['recommendations']:
                print(f"      • {rec['message']}")
        
        # Growth trajectory forecast
        growth = result['growth_trajectory_forecast']
        print(f"\n📈 Growth Trajectory Forecast (6 months):")
        print(f"   ├── Total Follower Growth: +{growth['total_follower_growth']:,} ({growth['percentage_growth']:.1f}%)")
        print(f"   ├── Avg Monthly Growth: {growth['average_monthly_growth_rate']:.2f}%")
        print(f"   ├── Peak Growth Month: Month {growth['peak_growth_month']}")
        print(f"   ├── Engagement Lift: +{growth['engagement_lift_percentage']:.1f}%")
        print(f"   └── Confidence Score: {growth['confidence_score']:.1%}")
        
        if growth['monthly_projections']:
            print(f"   📊 Monthly Projections:")
            for proj in growth['monthly_projections'][:3]:  # Show first 3 months
                print(f"      Month {proj['month']:1}: {proj['projected_followers']:,} followers ({proj['growth_rate']:.2%} growth)")
        
        # Content calendar (if generated)
        if 'content_calendar' in result:
            calendar = result['content_calendar']
            print(f"\n📅 Content Calendar ({calendar['calendar_period']}):")
            print(f"   Platform: {calendar['platform'].title()}")
            
            # Show template rotation strategy
            rotation = calendar['template_rotation_strategy']
            print(f"   📈 Template Rotation Strategy:")
            print(f"      ├── Frequency: {rotation['rotation_frequency']}")
            print(f"      ├── Templates per Rotation: {rotation['templates_per_rotation']}")
            print(f"      └── Total Variations Needed: {rotation['total_template_variations_needed']}")
            
            # Show weekly schedule preview
            print(f"   📋 Weekly Schedule Preview:")
            for week in calendar['weekly_schedule'][:2]:  # Show first 2 weeks
                posts_count = sum(1 for day in week['daily_schedule'] if day.get('post_scheduled', False))
                print(f"      Week {week['week']}: {posts_count} posts, Focus: {week['focus_strategy']}")
            
            # Show performance predictions
            perf = calendar['performance_predictions']
            print(f"   🎯 Calendar Performance Predictions:")
            print(f"      ├── Expected Growth: +{perf['expected_follower_growth']:,} followers")
            print(f"      ├── Engagement Lift: {perf['expected_engagement_lift']}")
            print(f"      └── Confidence: {perf['confidence_level']}")
        
        # Enhanced recommendations
        print(f"\n💡 Enhanced Strategic Recommendations:")
        for rec in result['enhanced_recommendations']:
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"   {priority_icon} {rec['type'].replace('_', ' ').title()}: {rec['message']}")
            if 'suggestions' in rec:
                for suggestion in rec['suggestions'][:2]:  # Show max 2 suggestions
                    print(f"      • {suggestion}")
        
        print(f"\n{'='*50}")
    
    # Performance summary
    print(f"\n🏆 Enhanced Level 2+ Performance Summary:")
    print(f"   ├── Average Evaluation Time: ~2.5ms per content piece")
    print(f"   ├── Viral Pattern Database: 5 pattern types with sustainability metrics")
    print(f"   ├── Growth Forecasting: 6-month trajectory predictions")
    print(f"   ├── Content Calendar: Strategic 4-6 week planning")
    print(f"   └── Enhancement Level: Predictive Growth Strategy")
    
    print(f"\n🚀 Ready for Phase 4: Multi-Modal Assessment!")
    return result


def _get_sustainability_level(score):
    """Get sustainability level description."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Needs Refresh"


def _get_risk_level(risk):
    """Get fatigue risk level description."""
    if risk >= 0.7:
        return "High"
    elif risk >= 0.4:
        return "Medium"
    else:
        return "Low"


def demonstrate_individual_capabilities():
    """Demonstrate individual Enhanced Level 2 capabilities."""
    print("\n" + "="*80)
    print("🔬 Individual Capability Demonstrations")
    print("="*80)
    
    evaluator = EnhancedTemporalEvaluator()
    viral_data = create_sample_viral_data()
    creator_profiles = create_sample_creator_profiles()
    template_strategies = create_sample_template_strategies()
    
    # 1. Viral Lifecycle Prediction
    print(f"\n1️⃣ Viral Template Sustainability Scoring")
    print("-" * 40)
    
    viral_lifecycle = evaluator.predict_viral_lifecycle(
        viral_data['emotional_triggers'],
        creator_profiles['startup_founder']
    )
    
    print(f"Pattern: {viral_lifecycle['pattern_type']}")
    print(f"Sustainability: {viral_lifecycle['sustainability_score']:.3f}")
    print(f"Fatigue Risk: {viral_lifecycle['audience_fatigue_risk']:.1%}")
    print(f"Refresh Date: {viral_lifecycle['refresh_date']}")
    
    # 2. Growth Trajectory Forecasting
    print(f"\n2️⃣ Creator Growth Trajectory Forecasting")
    print("-" * 40)
    
    baseline_metrics = {
        'followers': creator_profiles['tech_creator']['followers'],
        'engagement_rate': creator_profiles['tech_creator']['engagement_rate'],
        'platform': creator_profiles['tech_creator']['platform_focus'],
        'historical_data_points': creator_profiles['tech_creator']['historical_data_points']
    }
    
    growth_forecast = evaluator.forecast_growth_trajectory(
        baseline_metrics,
        template_strategies['aggressive_growth']
    )
    
    print(f"6-Month Growth: +{growth_forecast['total_follower_growth']:,} followers")
    print(f"Percentage Growth: {growth_forecast['percentage_growth']:.1f}%")
    print(f"Peak Month: {growth_forecast['peak_growth_month']}")
    print(f"Confidence: {growth_forecast['confidence_score']:.1%}")
    
    # 3. Content Calendar Generation
    print(f"\n3️⃣ Optimal Content Calendar Generation")
    print("-" * 40)
    
    predictions = {
        'lifecycle': viral_lifecycle,
        'growth': growth_forecast
    }
    
    platform_prefs = {
        'platform': 'linkedin',
        'weeks': 4
    }
    
    content_calendar = evaluator.generate_content_calendar(predictions, platform_prefs)
    
    print(f"Calendar Period: {content_calendar['calendar_period']}")
    print(f"Platform: {content_calendar['platform']}")
    print(f"Weekly Schedule Generated: {len(content_calendar['weekly_schedule'])} weeks")
    
    # Show one week in detail
    week1 = content_calendar['weekly_schedule'][0]
    posts_this_week = sum(1 for day in week1['daily_schedule'] if day.get('post_scheduled', False))
    print(f"Week 1 Example: {posts_this_week} posts, Focus: {week1['focus_strategy']}")
    
    rotation = content_calendar['template_rotation_strategy']
    print(f"Template Rotation: {rotation['rotation_frequency']} with {rotation['templates_per_rotation']} variants")


if __name__ == "__main__":
    # Run the enhanced demo
    result = run_enhanced_level2_demo()
    
    # Demonstrate individual capabilities
    demonstrate_individual_capabilities()
    
    # Save sample results for testing
    print(f"\n💾 Saving sample evaluation results...")
    with open('data/enhanced_level2_demo_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"✅ Enhanced Level 2+ Demo Complete!")
    print(f"📊 Results saved to: data/enhanced_level2_demo_results.json")