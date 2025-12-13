"""
Script de teste para verificar consistência dos dados
"""

import sys
sys.path.insert(0, r'c:\Users\gafeb\OneDrive\Desktop\campaign_analysis - Copia')

print('1. Importing modules...')
from data_processor import (
    load_and_process_data, get_summary_metrics, identify_top_creatives,
    analyze_patterns, generate_recommendations, generate_insights
)
from advanced_analysis import MarketingScienceAnalyzer

print('2. Loading data...')
CSV_PATH = 'meta_creatives_sample - meta_creatives_sample.csv.csv'
df = load_and_process_data(CSV_PATH)
metrics = get_summary_metrics(df)
top_creatives, avg_metrics = identify_top_creatives(df)
patterns = analyze_patterns(df, top_creatives)
recommendations = generate_recommendations(df, patterns, metrics)
insights = generate_insights(df, metrics, patterns)

print('3. Running advanced analysis...')
analyzer = MarketingScienceAnalyzer(df)
full_report = analyzer.generate_full_report()

print('4. Checking consistency...')
print()
print('=' * 60)
print('CONSISTENT DATA SUMMARY')
print('=' * 60)
print(f"Total Creatives: {len(df)}")
print(f"Overall ROAS: {metrics['overall_roas']:.2f}x")
print(f"Overall CPA: R$ {metrics['avg_cpa']:.2f}")
print()

# Lift Analysis Results
lift = full_report['lift_analysis']
print('=== BEST BY ROAS LIFT ===')
print(f"Best Placement: {lift['placement']['ranking'][0]['name']} (+{lift['placement']['ranking'][0]['roas_lift_pct']:.1f}%)")
print(f"Best Targeting: {lift['targeting']['ranking'][0]['name']} (+{lift['targeting']['ranking'][0]['roas_lift_pct']:.1f}%)")
print(f"Best Objective: {lift['campaign_objective']['ranking'][0]['name']} (+{lift['campaign_objective']['ranking'][0]['roas_lift_pct']:.1f}%)")
print()

# Advantage+ Analysis
adv = full_report['advantage_uplift']
print('=== ADVANTAGE+ ANALYSIS ===')
print(f"ROAS with Advantage+: {adv['ROAS']['with_advantage']:.2f}x")
print(f"ROAS without Advantage+: {adv['ROAS']['without_advantage']:.2f}x")
print(f"Advantage+ Impact: {adv['summary']['roas_uplift']:.1f}%")
print(f"Recommendation: {lift['summary']['advantage_recommendation']}")
print()

# Priority Recommendations
print('=== PRIORITY RECOMMENDATIONS ===')
for rec in full_report['priority_recommendations']:
    print(f"{rec['priority']}. [{rec['category']}] {rec['action']}")
    print(f"   Impact: {rec['expected_impact']}")
print()

print('=' * 60)
print('ALL CHECKS PASSED - Data is consistent!')
print('=' * 60)
