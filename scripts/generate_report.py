"""
Relatório Completo de Marketing Science
Meta Ads Creative Performance Analysis
=====================================

Execute este arquivo para gerar o relatório completo
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from campanhas.paths import DADOS, RELATORIOS
from campanhas.data_processor import load_and_process_data
from campanhas.advanced_analysis import MarketingScienceAnalyzer

# ============================================
# CONFIGURAÇÕES
# ============================================
CSV_PATH = DADOS
CHART_TEMPLATE = 'plotly_dark'

COLORS = {
    'primary': '#6366f1',    # Indigo 500
    'secondary': '#10b981',  # Emerald 500
    'accent': '#f43f5e',     # Rose 500
    'danger': '#ef4444',     # Red 500
    'success': '#10b981',    # Emerald 500
    'warning': '#f59e0b',    # Amber 500
    'bg_dark': '#0f172a'     # Slate 900
}

# ============================================
# CARREGAR E ANALISAR DADOS
# ============================================
print("\n" + "="*70)
print("📊 RELATÓRIO DE MARKETING SCIENCE - META ADS")
print("="*70)

print("\n🔄 Carregando dados...")
df = load_and_process_data(CSV_PATH)

print("🔬 Executando análises avançadas...")
analyzer = MarketingScienceAnalyzer(df)
report = analyzer.generate_full_report()

# ============================================
# 1. RESUMO EXECUTIVO
# ============================================
print("\n" + "="*70)
print("📋 1. RESUMO EXECUTIVO")
print("="*70)

summary = report['executive_summary']
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  VISÃO GERAL DO PORTFÓLIO                                           │
├─────────────────────────────────────────────────────────────────────┤
│  Total de Creatives:     {summary['total_creatives']:>10}                              │
│  Investimento Total:     ${summary['total_spend']:>14,.0f}                        │
│  Total de Conversões:    {summary['total_conversions']:>10,.0f}                              │
│  ROAS Geral:             {summary['overall_roas']:>10.2f}x                             │
│  CPA Médio:              ${summary['overall_cpa']:>13.2f}                         │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n📌 PRINCIPAIS DESCOBERTAS:")
for i, finding in enumerate(summary['key_findings'], 1):
    print(f"   {i}. {finding['finding']}")
    print(f"      ➜ Impacto: {finding['impact']}")

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  IMPACTO ESTIMADO COM OTIMIZAÇÕES                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Melhoria no ROAS:       {summary['estimated_impact']['roas_improvement']:>+10.1f}%                            │
│  Redução no CPA:         {summary['estimated_impact']['cpa_reduction']:>+10.1f}%                            │
│  Aumento em Conversões:  {summary['estimated_impact']['conversion_increase']:>+10.1f}%                            │
│  Economia Potencial:     ${summary['estimated_impact']['potential_savings']:>14,.2f}                        │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 2. IMPACTO DAS VARIÁVEIS NO ROAS
# ============================================
print("\n" + "="*70)
print("📈 2. IMPACTO DAS VARIÁVEIS NO ROAS")
print("="*70)

impact = report['roas_impact']
print(f"\n   Modelo com melhor performance: {impact['model_performance']['best_model']}")
print(f"   R² (Ridge): {impact['model_performance']['ridge_r2']:.3f}")
print(f"   R² (Random Forest): {impact['model_performance']['rf_r2']:.3f}")
print(f"   R² (Gradient Boosting): {impact['model_performance']['gb_r2']:.3f}")

print("\n   📊 RANKING DE IMPACTO NO ROAS:")
print("   " + "-"*65)
print(f"   {'#':<3} {'Variável':<25} {'Impacto %':<12} {'Direção':<12} {'Significância'}")
print("   " + "-"*65)

for i, var in enumerate(impact['impact_ranking'], 1):
    sig = "✓ Sig." if var['statistical_significance']['is_significant'] else "○ N/S"
    print(f"   {i:<3} {var['variable']:<25} {var['impact_score']:>8.1f}%    {var['direction']:<12} {sig}")

print("   " + "-"*65)

# Gráfico de impacto
fig_impact = go.Figure()

variables = [v['variable'] for v in impact['impact_ranking']]
scores = [v['impact_score'] for v in impact['impact_ranking']]
colors_impact = [COLORS['success'] if v['direction'] == 'Positivo' else COLORS['danger'] 
                 for v in impact['impact_ranking']]

fig_impact.add_trace(go.Bar(
    y=variables[::-1],
    x=scores[::-1],
    orientation='h',
    marker_color=colors_impact[::-1],
    text=[f"{s:.1f}%" for s in scores[::-1]],
    textposition='outside'
))

fig_impact.update_layout(
    title='Impacto das Variáveis no ROAS',
    template=CHART_TEMPLATE,
    height=400,
    xaxis_title='Importância (%)',
    yaxis_title='',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_impact.write_html(RELATORIOS / 'report_impact_roas.html')
print("\n   📁 Gráfico salvo: report_impact_roas.html")

# ============================================
# 3. ANÁLISE DE CAUSALIDADE
# ============================================
print("\n" + "="*70)
print("🔬 3. ANÁLISE DE CAUSALIDADE (Difference-in-Differences)")
print("="*70)

did = report['diff_in_diff']
print(f"""
   ADVANTAGE+ PLACEMENTS: ANÁLISE COMPARATIVA
   ═══════════════════════════════════════════

   Grupo Tratamento (Advantage+ = Yes): {did['summary']['total_treatment']} creatives
   Grupo Controle (Advantage+ = No):    {did['summary']['total_control']} creatives
""")

print("   " + "-"*70)
print(f"   {'Métrica':<12} {'Com Adv+':<14} {'Sem Adv+':<14} {'Diferença':<12} {'P-Value':<10} {'Efeito'}")
print("   " + "-"*70)

for metric in ['ROAS', 'CPA', 'CTR', 'CPC', 'CPM']:
    data = did[metric]
    sig_mark = "***" if data['p_value'] < 0.01 else "**" if data['p_value'] < 0.05 else "*" if data['p_value'] < 0.1 else ""
    
    if metric in ['ROAS', 'CTR']:
        treat_fmt = f"{data['treatment_mean']:.2f}"
        ctrl_fmt = f"{data['control_mean']:.2f}"
        diff_fmt = f"{data['percentage_difference']:+.1f}%"
    else:
        treat_fmt = f"${data['treatment_mean']:.2f}"
        ctrl_fmt = f"${data['control_mean']:.2f}"
        diff_fmt = f"{data['percentage_difference']:+.1f}%"
    
    print(f"   {metric:<12} {treat_fmt:<14} {ctrl_fmt:<14} {diff_fmt:<12} {data['p_value']:<10.4f} {data['effect_size']}{sig_mark}")

print("   " + "-"*70)
print("   *** p<0.01  ** p<0.05  * p<0.1")

print(f"""
   ┌───────────────────────────────────────────────────────────────────┐
   │  CONCLUSÃO CAUSAL                                                 │
   ├───────────────────────────────────────────────────────────────────┤
   │  Uplift de ROAS:     {did['summary']['roas_uplift']:>+8.1f}%                                  │
   │  Redução de CPA:     {did['summary']['cpa_reduction']:>+8.1f}%                                  │
   │  Confiança:          {did['summary']['statistical_confidence']*100:>8.1f}%                                  │
   └───────────────────────────────────────────────────────────────────┘
""")

# Gráfico Difference-in-Differences
fig_did = make_subplots(rows=1, cols=3, subplot_titles=('ROAS', 'CPA', 'CTR'))

for i, metric in enumerate(['ROAS', 'CPA', 'CTR'], 1):
    data = did[metric]
    fig_did.add_trace(
        go.Bar(
            x=['Com Advantage+', 'Sem Advantage+'],
            y=[data['treatment_mean'], data['control_mean']],
            marker_color=[COLORS['primary'], COLORS['accent']],
            text=[f"{data['treatment_mean']:.2f}", f"{data['control_mean']:.2f}"],
            textposition='outside'
        ),
        row=1, col=i
    )

fig_did.update_layout(
    title='Análise Difference-in-Differences: Advantage+ Placements',
    template=CHART_TEMPLATE,
    height=400,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_did.write_html(RELATORIOS / 'report_diff_in_diff.html')
print("   📁 Gráfico salvo: report_diff_in_diff.html")

# ============================================
# 4. RELAÇÃO FREQUENCY x CPA
# ============================================
print("\n" + "="*70)
print("📉 4. RELAÇÃO MATEMÁTICA: FREQUENCY × CPA")
print("="*70)

freq = report['frequency_analysis']
print(f"""
   ANÁLISE DE CORRELAÇÃO
   ═════════════════════
   
   Correlação de Pearson:  {freq['correlation']['pearson']:>8.4f} (p = {freq['correlation']['pearson_p']:.4f})
   Correlação de Spearman: {freq['correlation']['spearman']:>8.4f} (p = {freq['correlation']['spearman_p']:.4f})
   
   Interpretação: Correlação {freq['correlation']['strength']} {freq['correlation']['interpretation']}
   
   MODELO DE REGRESSÃO
   ═══════════════════
   
   Equação: CPA = {freq['regression']['intercept']:.2f} + {freq['regression']['coefficient']:.2f} × Frequency
   R² = {freq['regression']['r_squared']:.4f}
   
   {freq['regression']['interpretation']}
""")

print("   ANÁLISE POR FAIXA DE FREQUÊNCIA:")
print("   " + "-"*65)
print(f"   {'Faixa':<12} {'Qtd':<8} {'CPA Médio':<12} {'ROAS Médio':<12} {'Conversões'}")
print("   " + "-"*65)

for band in freq['frequency_bands']:
    print(f"   {band['range']:<12} {band['count']:<8} ${band['avg_cpa']:<11.2f} {band['avg_roas']:<12.2f} {band['total_conversions']:.0f}")

print("   " + "-"*65)
print(f"\n   ✓ Frequência Ótima: {freq['optimal_frequency']}")

print(f"""
   ┌───────────────────────────────────────────────────────────────────┐
   │  RECOMENDAÇÃO                                                     │
   ├───────────────────────────────────────────────────────────────────┤
   │  Ação: {freq['recommendation']['action']:<55} │
   │  Razão: {freq['recommendation']['reason']:<54} │
   └───────────────────────────────────────────────────────────────────┘
""")

# Gráfico Frequency x CPA
fig_freq = go.Figure()

fig_freq.add_trace(go.Scatter(
    x=df['Frequency'],
    y=df['CPA'],
    mode='markers',
    marker=dict(
        size=8,
        color=df['ROAS'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='ROAS')
    ),
    text=df['Nome do Creative'],
    hovertemplate='<b>%{text}</b><br>Frequency: %{x:.2f}<br>CPA: $%{y:.2f}<extra></extra>'
))

# Linha de tendência
freq_vals = np.linspace(df['Frequency'].min(), df['Frequency'].max(), 100)
cpa_pred = freq['regression']['intercept'] + freq['regression']['coefficient'] * freq_vals

fig_freq.add_trace(go.Scatter(
    x=freq_vals,
    y=cpa_pred,
    mode='lines',
    line=dict(color=COLORS['danger'], width=3, dash='dash'),
    name='Tendência'
))

fig_freq.update_layout(
    title='Relação Frequency × CPA (com linha de tendência)',
    xaxis_title='Frequency',
    yaxis_title='CPA ($)',
    template=CHART_TEMPLATE,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_freq.write_html(RELATORIOS / 'report_frequency_cpa.html')
print("   📁 Gráfico salvo: report_frequency_cpa.html")

# ============================================
# 5. UPLIFT ADVANTAGE+ PLACEMENTS
# ============================================
print("\n" + "="*70)
print("⚡ 5. UPLIFT MÉDIO: ADVANTAGE+ PLACEMENTS")
print("="*70)

uplift = report['advantage_uplift']
print(f"""
   ┌───────────────────────────────────────────────────────────────────┐
   │  ANÁLISE DE UPLIFT                                                │
   ├───────────────────────────────────────────────────────────────────┤
   │  Creatives com Advantage+:    {uplift['summary']['total_with_advantage']:>10}                          │
   │  Creatives sem Advantage+:    {uplift['summary']['total_without_advantage']:>10}                          │
   │  Nível de Confiança:          {uplift['summary']['confidence_level']:<10}                          │
   └───────────────────────────────────────────────────────────────────┘
""")

print("   COMPARATIVO DE MÉTRICAS:")
print("   " + "-"*70)
print(f"   {'Métrica':<15} {'Com Adv+':<15} {'Sem Adv+':<15} {'Uplift':<15} {'Impacto'}")
print("   " + "-"*70)

for metric in ['ROAS', 'CPA', 'CTR', 'CPC', 'Conversions']:
    data = uplift[metric]
    impact_icon = "✓ Positivo" if data['is_positive_impact'] else "✗ Negativo"
    
    if metric in ['ROAS', 'CTR']:
        with_fmt = f"{data['with_advantage']:.2f}"
        without_fmt = f"{data['without_advantage']:.2f}"
    elif metric == 'Conversions':
        with_fmt = f"{data['with_advantage']:.0f}"
        without_fmt = f"{data['without_advantage']:.0f}"
    else:
        with_fmt = f"${data['with_advantage']:.2f}"
        without_fmt = f"${data['without_advantage']:.2f}"
    
    print(f"   {metric:<15} {with_fmt:<15} {without_fmt:<15} {data['percentage_uplift']:>+.1f}%{'':<9} {impact_icon}")

print("   " + "-"*70)

print(f"""
   ┌───────────────────────────────────────────────────────────────────┐
   │  CONCLUSÃO                                                        │
   ├───────────────────────────────────────────────────────────────────┤
   │  Uplift de ROAS:    {uplift['summary']['roas_uplift']:>+10.1f}%                                │
   │  Variação de CPA:   {uplift['summary']['cpa_change']:>+10.1f}%                                │
   │  Uplift de CTR:     {uplift['summary']['ctr_uplift']:>+10.1f}%                                │
   │                                                                   │
   │  ➜ Recomendação: {uplift['summary']['recommendation']:<45} │
   └───────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 6. PADRÕES DOS TOP PERFORMERS
# ============================================
print("\n" + "="*70)
print("🏆 6. PADRÕES DOS CREATIVES VENCEDORES")
print("="*70)

top = report['top_performers']
patterns = top['patterns']

print(f"""
   PERFIL IDEAL DO CREATIVE VENCEDOR
   ══════════════════════════════════
   
   Placement:             {top['ideal_profile']['placement']}
   Campaign Objective:    {top['ideal_profile']['objective']}
   Targeting:             {top['ideal_profile']['targeting']}
   Advantage+ Placements: {top['ideal_profile']['advantage_placements']}
   Advantage+ Creative:   {top['ideal_profile']['advantage_creative']}
   Frequência Ótima:      {top['ideal_profile']['optimal_frequency']}
""")

print("   MÉTRICAS DOS TOP 20 PERFORMERS:")
print("   " + "-"*50)
print(f"   ROAS Médio:         {patterns['metrics']['avg_roas']:>10.2f}x")
print(f"   CPA Médio:          ${patterns['metrics']['avg_cpa']:>9.2f}")
print(f"   CTR Médio:          {patterns['metrics']['avg_ctr']:>10.2f}%")
print(f"   Frequência Média:   {patterns['metrics']['avg_frequency']:>10.2f}")
print(f"   Total Conversões:   {patterns['metrics']['total_conversions']:>10.0f}")
print(f"   Total Investido:    ${patterns['metrics']['total_spend']:>9.0f}")
print("   " + "-"*50)

print("\n   COMPARAÇÃO COM MÉDIA GERAL:")
print(f"   ROAS: {patterns['vs_average']['roas_diff']:>+.1f}% vs média")
print(f"   CPA:  {patterns['vs_average']['cpa_diff']:>+.1f}% vs média")
print(f"   CTR:  {patterns['vs_average']['ctr_diff']:>+.1f}% vs média")

print("\n   DISTRIBUIÇÃO DE PLACEMENTS (Top Performers):")
for plc, count in patterns['placement']['distribution'].items():
    pct = count / sum(patterns['placement']['distribution'].values()) * 100
    bar = "█" * int(pct / 5)
    print(f"   {plc:<18} {bar} {pct:.1f}%")

print("\n   DISTRIBUIÇÃO DE TARGETING (Top Performers):")
for tgt, count in patterns['targeting']['distribution'].items():
    pct = count / sum(patterns['targeting']['distribution'].values()) * 100
    bar = "█" * int(pct / 5)
    print(f"   {tgt:<22} {bar} {pct:.1f}%")

# ============================================
# 7. MATRIZ DE DECISÃO
# ============================================
print("\n" + "="*70)
print("🎯 7. MATRIZ DE DECISÃO: ESCALAR | TESTAR | PAUSAR")
print("="*70)

matrix = report['decision_matrix']

print(f"""
   ┌──────────────────────────────────────────────────────────────────────┐
   │                      RESUMO DA MATRIZ                                │
   ├──────────────────────────────────────────────────────────────────────┤
   │  🚀 ESCALAR:    {matrix['summary']['total_to_scale']:>5} creatives                                    │
   │  🔬 TESTAR:     {matrix['summary']['total_to_test']:>5} creatives                                    │
   │  ⏸️  PAUSAR:     {matrix['summary']['total_to_pause']:>5} creatives                                    │
   ├──────────────────────────────────────────────────────────────────────┤
   │  💰 Economia Potencial: ${matrix['summary']['potential_savings']:>12,.2f}                         │
   └──────────────────────────────────────────────────────────────────────┘
""")

# ESCALAR
print("\n   🚀 CREATIVES PARA ESCALAR (Top 10):")
print("   " + "-"*75)
print(f"   {'Creative':<20} {'ROAS':<10} {'CPA':<12} {'Conv.':<10} {'Motivo'}")
print("   " + "-"*75)

for c in matrix['scale']['creatives'][:10]:
    print(f"   {c['creative']:<20} {c['roas']:.2f}x     ${c['cpa']:<10.2f} {c['conversions']:<10.0f} {c['reason'][:30]}")

print("   " + "-"*75)
print(f"   ➜ Ação: {matrix['scale']['action']}")
print(f"   ➜ Impacto Esperado: {matrix['scale']['expected_impact']}")

# TESTAR
print("\n   🔬 CREATIVES PARA TESTAR (Amostra):")
print("   " + "-"*75)

for c in matrix['test']['creatives'][:5]:
    print(f"   {c['creative']:<20} ROAS: {c['roas']:.2f}x | CPA: ${c['cpa']:.2f} | {c['reason'][:35]}")

print("   " + "-"*75)
print(f"   ➜ Ação: {matrix['test']['action']}")
print(f"   ➜ Impacto Esperado: {matrix['test']['expected_impact']}")

# PAUSAR
print("\n   ⏸️  CREATIVES PARA PAUSAR (Amostra):")
print("   " + "-"*75)

for c in matrix['pause']['creatives'][:5]:
    print(f"   {c['creative']:<20} ROAS: {c['roas']:.2f}x | CPA: ${c['cpa']:.2f} | {c['reason'][:35]}")

print("   " + "-"*75)
print(f"   ➜ Ação: {matrix['pause']['action']}")
print(f"   ➜ Economia: ${matrix['pause']['expected_savings']:.2f}")

# Gráfico da matriz de decisão
fig_matrix = go.Figure()

categories = ['Escalar', 'Testar', 'Pausar']
counts = [matrix['scale']['count'], matrix['test']['count'], matrix['pause']['count']]
colors_matrix = [COLORS['success'], COLORS['warning'], COLORS['danger']]
roas_vals = [matrix['scale']['avg_roas'], matrix['test']['avg_roas'], matrix['pause']['avg_roas']]

fig_matrix.add_trace(go.Bar(
    x=categories,
    y=counts,
    marker_color=colors_matrix,
    text=[f"{c} creatives<br>ROAS: {r:.2f}x" for c, r in zip(counts, roas_vals)],
    textposition='outside'
))

fig_matrix.update_layout(
    title='Matriz de Decisão: Distribuição de Creatives',
    template=CHART_TEMPLATE,
    height=400,
    yaxis_title='Número de Creatives',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_matrix.write_html(RELATORIOS / 'report_decision_matrix.html')
print("\n   📁 Gráfico salvo: report_decision_matrix.html")

# ============================================
# 8. PREVISÕES 30 DIAS
# ============================================
print("\n" + "="*70)
print("🔮 8. PREVISÕES PARA OS PRÓXIMOS 30 DIAS")
print("="*70)

forecast = report['forecast']

print("""
   ┌──────────────────────────────────────────────────────────────────────┐
   │                   CENÁRIOS DE PREVISÃO                               │
   └──────────────────────────────────────────────────────────────────────┘
""")

# Cenário Baseline
bl = forecast['baseline']
print(f"""   📊 CENÁRIO 1: {bl['scenario'].upper()}
   ─────────────────────────────────────
   ROAS:        {bl['roas']['value']:.2f}x (IC: {bl['roas']['lower']:.2f} - {bl['roas']['upper']:.2f})
   CPA:         ${bl['cpa']['value']:.2f} (IC: ${bl['cpa']['lower']:.2f} - ${bl['cpa']['upper']:.2f})
   Conversões:  {bl['conversions']['value']:,.0f} (IC: {bl['conversions']['lower']:,.0f} - {bl['conversions']['upper']:,.0f})
   Investimento: ${bl['spend']:,.2f}
""")

# Cenário Otimizado
opt = forecast['optimized']
print(f"""   ⚡ CENÁRIO 2: {opt['scenario'].upper()} (RECOMENDADO)
   ─────────────────────────────────────
   ROAS:        {opt['roas']['value']:.2f}x (IC: {opt['roas']['lower']:.2f} - {opt['roas']['upper']:.2f})  │ {opt['improvement_vs_baseline']['roas']:+.1f}% vs baseline
   CPA:         ${opt['cpa']['value']:.2f} (IC: ${opt['cpa']['lower']:.2f} - ${opt['cpa']['upper']:.2f})  │ {opt['improvement_vs_baseline']['cpa']:+.1f}% redução
   Conversões:  {opt['conversions']['value']:,.0f} (IC: {opt['conversions']['lower']:,.0f} - {opt['conversions']['upper']:,.0f})  │ {opt['improvement_vs_baseline']['conversions']:+.1f}% vs baseline
   Investimento: ${opt['spend']:,.2f}
""")

# Cenário Agressivo
agg = forecast['aggressive']
print(f"""   🚀 CENÁRIO 3: {agg['scenario'].upper()}
   ─────────────────────────────────────
   ROAS:        {agg['roas']['value']:.2f}x (IC: {agg['roas']['lower']:.2f} - {agg['roas']['upper']:.2f})  │ {agg['improvement_vs_baseline']['roas']:+.1f}% vs baseline
   CPA:         ${agg['cpa']['value']:.2f} (IC: ${agg['cpa']['lower']:.2f} - ${agg['cpa']['upper']:.2f})
   Conversões:  {agg['conversions']['value']:,.0f} (IC: {agg['conversions']['lower']:,.0f} - {agg['conversions']['upper']:,.0f})  │ {agg['improvement_vs_baseline']['conversions']:+.1f}% vs baseline
   Investimento: ${agg['spend']:,.2f} (+{agg['improvement_vs_baseline']['spend_increase']:.0f}%)
""")

rec = forecast['recommendation']
print(f"""   ┌──────────────────────────────────────────────────────────────────────┐
   │  RECOMENDAÇÃO                                                        │
   ├──────────────────────────────────────────────────────────────────────┤
   │  Cenário Sugerido: {rec['suggested_scenario']:<43}│
   │  Razão: {rec['reason']:<54}│
   │                                                                      │
   │  Previsões:                                                          │
   │  • ROAS Esperado:       {rec['expected_roas']:.2f}x                                      │
   │  • CPA Esperado:        ${rec['expected_cpa']:.2f}                                     │
   │  • Conversões:          {rec['expected_conversions']:,.0f}                                     │
   │  • Confiança:           {rec['confidence']}                                      │
   └──────────────────────────────────────────────────────────────────────┘
""")

# Gráfico de previsões
fig_forecast = make_subplots(rows=1, cols=3, subplot_titles=('ROAS', 'CPA', 'Conversões'))

scenarios = ['Baseline', 'Otimizado', 'Agressivo']
scenario_colors = [COLORS['accent'], COLORS['success'], COLORS['primary']]

# ROAS
roas_vals = [bl['roas']['value'], opt['roas']['value'], agg['roas']['value']]
roas_lower = [bl['roas']['lower'], opt['roas']['lower'], agg['roas']['lower']]
roas_upper = [bl['roas']['upper'], opt['roas']['upper'], agg['roas']['upper']]

fig_forecast.add_trace(
    go.Bar(
        x=scenarios,
        y=roas_vals,
        marker_color=scenario_colors,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[u - v for u, v in zip(roas_upper, roas_vals)],
            arrayminus=[v - l for v, l in zip(roas_vals, roas_lower)]
        )
    ),
    row=1, col=1
)

# CPA
cpa_vals = [bl['cpa']['value'], opt['cpa']['value'], agg['cpa']['value']]
fig_forecast.add_trace(
    go.Bar(x=scenarios, y=cpa_vals, marker_color=scenario_colors),
    row=1, col=2
)

# Conversões
conv_vals = [bl['conversions']['value'], opt['conversions']['value'], agg['conversions']['value']]
fig_forecast.add_trace(
    go.Bar(x=scenarios, y=conv_vals, marker_color=scenario_colors),
    row=1, col=3
)

fig_forecast.update_layout(
    title='Previsões para 30 Dias - Comparativo de Cenários',
    template=CHART_TEMPLATE,
    height=400,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_forecast.write_html(RELATORIOS / 'report_forecast.html')
print("   📁 Gráfico salvo: report_forecast.html")

# ============================================
# 9. RECOMENDAÇÕES PRIORITÁRIAS
# ============================================
print("\n" + "="*70)
print("✅ 9. RECOMENDAÇÕES PRIORITÁRIAS")
print("="*70)

recs = report['priority_recommendations']

print("""
   ┌──────────────────────────────────────────────────────────────────────┐
   │                   PLANO DE AÇÃO PRIORITÁRIO                          │
   └──────────────────────────────────────────────────────────────────────┘
""")

for rec in recs:
    priority_emoji = "🔴" if rec['priority'] <= 2 else "🟡" if rec['priority'] <= 4 else "🟢"
    print(f"""
   {priority_emoji} PRIORIDADE {rec['priority']}: {rec['category'].upper()}
   ───────────────────────────────────────
   Ação:            {rec['action']}
   Impacto:         {rec['expected_impact']}
   Esforço:         {rec['effort']}
   Prazo:           {rec['timeframe']}
""")

# ============================================
# 10. RESUMO FINAL
# ============================================
print("\n" + "="*70)
print("📋 10. RESUMO FINAL PARA APRESENTAÇÃO")
print("="*70)

print(f"""
   ╔══════════════════════════════════════════════════════════════════════╗
   ║              CONCLUSÕES OBJETIVAS                                    ║
   ╠══════════════════════════════════════════════════════════════════════╣
   ║                                                                      ║
   ║  1. IMPACTO NO ROAS                                                  ║
   ║     • Variável mais importante: {impact['impact_ranking'][0]['variable']:<30}║
   ║     • Impacto: {impact['impact_ranking'][0]['impact_score']:.1f}% de influência no resultado             ║
   ║                                                                      ║
   ║  2. CAUSALIDADE (ADVANTAGE+ PLACEMENTS)                              ║
   ║     • Uplift de ROAS: {did['summary']['roas_uplift']:+.1f}%                                     ║
   ║     • Redução de CPA: {did['summary']['cpa_reduction']:+.1f}%                                     ║
   ║     • Confiança estatística: {did['summary']['statistical_confidence']*100:.1f}%                       ║
   ║                                                                      ║
   ║  3. FREQUÊNCIA × CPA                                                 ║
   ║     • Correlação: {freq['correlation']['strength']} ({freq['correlation']['pearson']:.3f})                       ║
   ║     • Frequência ótima: {freq['optimal_frequency']:<35}║
   ║                                                                      ║
   ║  4. MATRIZ DE DECISÃO                                                ║
   ║     • Escalar: {matrix['scale']['count']} creatives (+50-100% budget)                  ║
   ║     • Testar: {matrix['test']['count']} creatives (otimizar)                          ║
   ║     • Pausar: {matrix['pause']['count']} creatives (economia: ${matrix['pause']['expected_savings']:.2f})        ║
   ║                                                                      ║
   ║  5. PREVISÃO 30 DIAS (Cenário Otimizado)                             ║
   ║     • ROAS: {opt['roas']['value']:.2f}x ({opt['improvement_vs_baseline']['roas']:+.1f}% vs atual)                          ║
   ║     • CPA: ${opt['cpa']['value']:.2f} ({opt['improvement_vs_baseline']['cpa']:+.1f}% redução)                        ║
   ║     • Conversões: {opt['conversions']['value']:,.0f} ({opt['improvement_vs_baseline']['conversions']:+.1f}% aumento)                   ║
   ║                                                                      ║
   ╠══════════════════════════════════════════════════════════════════════╣
   ║              IMPACTO ESTIMADO TOTAL                                  ║
   ╠══════════════════════════════════════════════════════════════════════╣
   ║                                                                      ║
   ║  📈 Melhoria no ROAS:        {summary['estimated_impact']['roas_improvement']:>+8.1f}%                          ║
   ║  📉 Redução no CPA:          {summary['estimated_impact']['cpa_reduction']:>+8.1f}%                          ║
   ║  📊 Aumento em Conversões:   {summary['estimated_impact']['conversion_increase']:>+8.1f}%                          ║
   ║  💰 Economia Potencial:      ${summary['estimated_impact']['potential_savings']:>12,.2f}                    ║
   ║                                                                      ║
   ╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "="*70)
print("📁 ARQUIVOS DE GRÁFICOS GERADOS:")
print("="*70)
print("""
   1. report_impact_roas.html      - Impacto das variáveis no ROAS
   2. report_diff_in_diff.html     - Análise Difference-in-Differences
   3. report_frequency_cpa.html    - Relação Frequency × CPA
   4. report_decision_matrix.html  - Matriz de Decisão
   5. report_forecast.html         - Previsões 30 dias
""")

print("\n" + "="*70)
print("✅ RELATÓRIO COMPLETO GERADO COM SUCESSO!")
print("="*70 + "\n")
