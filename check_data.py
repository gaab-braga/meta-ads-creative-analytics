import pandas as pd
from data_processor import load_and_process_data
from advanced_analysis import MarketingScienceAnalyzer

# Carregar dados usando data_processor para garantir consistência
df = load_and_process_data('meta_creatives_sample - meta_creatives_sample.csv.csv')

# Verificar colunas
print('=== COLUNAS DISPONIVEIS ===')
print(df.columns.tolist())

print('\n=== ESTATISTICAS BASICAS ===')
print(f'Total de registros: {len(df)}')
print(f'Total Amount spent: {df["Amount spent"].sum():,.2f}')
print(f'Total Conversions: {df["Conversions"].sum():,.0f}')
print(f'Total Revenue: {df["Revenue"].sum():,.2f}')
print(f'ROAS medio: {df["ROAS"].mean():.2f}')
print(f'CPA medio: {df["CPA"].mean():.2f}')

# Analisar com MarketingScienceAnalyzer
analyzer = MarketingScienceAnalyzer(df)
report = analyzer.generate_full_report()

print('\n=== FORECAST ===')
print(f'Baseline conversions: {report["forecast"]["baseline"]["conversions"]["value"]:,.0f}')
print(f'Optimized conversions: {report["forecast"]["optimized"]["conversions"]["value"]:,.0f}')
print(f'Optimized ROAS: {report["forecast"]["optimized"]["roas"]["value"]:.2f}')
print(f'Optimized CPA: {report["forecast"]["optimized"]["cpa"]["value"]:.2f}')

print('\n=== DECISION MATRIX ===')
print(f'Scale count: {report["decision_matrix"]["scale"]["count"]}')
print(f'Test count: {report["decision_matrix"]["test"]["count"]}')
print(f'Pause count: {report["decision_matrix"]["pause"]["count"]}')
print(f'Scale avg ROAS: {report["decision_matrix"]["scale"]["avg_roas"]:.2f}')
print(f'Test avg ROAS: {report["decision_matrix"]["test"]["avg_roas"]:.2f}')
print(f'Pause avg ROAS: {report["decision_matrix"]["pause"]["avg_roas"]:.2f}')
print(f'Pause expected savings: {report["decision_matrix"]["pause"]["expected_savings"]:,.2f}')

print('\n=== EXECUTIVE SUMMARY - ESTIMATED IMPACT ===')
print(f'ROAS improvement: {report["executive_summary"]["estimated_impact"]["roas_improvement"]:.2f}%')
print(f'CPA reduction: {report["executive_summary"]["estimated_impact"]["cpa_reduction"]:.2f}%')
print(f'Conversion increase: {report["executive_summary"]["estimated_impact"]["conversion_increase"]:.2f}%')
print(f'Potential savings: {report["executive_summary"]["estimated_impact"]["potential_savings"]:,.2f}')

print('\n=== PRIORITY RECOMMENDATIONS ===')
for rec in report['priority_recommendations'][:5]:
    print(f'{rec["priority"]}. {rec["action"]} - Impacto: {rec["expected_impact"]}')

print('\n=== ADVANTAGE UPLIFT ===')
print(f'With Advantage ROAS: {report["advantage_uplift"]["ROAS"]["with_advantage"]:.2f}')
print(f'Without Advantage ROAS: {report["advantage_uplift"]["ROAS"]["without_advantage"]:.2f}')
print(f'ROAS Uplift %: {report["advantage_uplift"]["summary"]["roas_uplift"]:.2f}%')
