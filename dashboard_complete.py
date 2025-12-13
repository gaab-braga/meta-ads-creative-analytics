"""
Meta Ads Creative Performance Dashboard - VERSÃO COMPLETA
Dashboard integrado com todos os relatórios de Marketing Science
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Importar módulos locais
from data_processor import (
    load_and_process_data, get_summary_metrics, identify_top_creatives,
    analyze_patterns, generate_recommendations, generate_insights
)
from advanced_analysis import MarketingScienceAnalyzer

# ============================================
# CONFIGURAÇÃO DO APP
# ============================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Meta Ads Creative Performance Dashboard"

# ============================================
# CARREGAR DADOS
# ============================================
CSV_PATH = "meta_creatives_sample - meta_creatives_sample.csv.csv"

# Carregar e processar dados
df = load_and_process_data(CSV_PATH)
metrics = get_summary_metrics(df)
top_creatives, avg_metrics = identify_top_creatives(df)
patterns = analyze_patterns(df, top_creatives)
recommendations = generate_recommendations(df, patterns, metrics)
insights = generate_insights(df, metrics, patterns)

# Análise avançada de Marketing Science
analyzer = MarketingScienceAnalyzer(df)
full_report = analyzer.generate_full_report()

# ============================================
# CORES E TEMA
# ============================================
COLORS = {
    'primary': '#6366f1',    # Indigo 500
    'secondary': '#10b981',  # Emerald 500
    'accent': '#f43f5e',     # Rose 500
    'danger': '#ef4444',     # Red 500
    'success': '#10b981',    # Emerald 500
    'warning': '#f59e0b',    # Amber 500
    'bg_dark': '#f8fafc',    # Light bg
    'bg_card': '#ffffff',    # White
    'text': '#1f2937',       # Gray 800
    'text_secondary': '#6b7280' # Gray 500
}

CHART_TEMPLATE = 'plotly_white'

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def create_kpi_card(icon, value, label, sublabel=None, color_class='primary'):
    """Cria um card de KPI com sublabel opcional - Layout: Label > Value > Sublabel"""
    sublabel_element = None
    if sublabel:
        sublabel_element = html.Div(sublabel, className="kpi-sublabel")
    
    return html.Div([
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className=f"kpi-value kpi-{color_class}"),
            sublabel_element
        ], className="kpi-content")
    ], className="kpi-card")


def create_performance_bar(name, roas, max_roas, color='blue', cpa=None):
    """Cria uma barra de performance horizontal"""
    bar_width = min((roas / max_roas) * 100, 100) if max_roas > 0 else 0
    
    right_text = f"ROAS: {roas:.2f}x"
    if cpa is not None:
        right_text += f" | CPA: R$ {cpa:.2f}"
    
    return html.Div([
        html.Div([
            html.Span(name, className="perf-bar-name"),
            html.Span(right_text, className="perf-bar-value")
        ], className="perf-bar-header"),
        html.Div([
            html.Div(className=f"perf-bar-fill perf-bar-{color}", style={'width': f'{bar_width}%'})
        ], className="perf-bar-track")
    ], className="perf-bar-item")


def create_advantage_card(title, roas, cpa, count, is_on=True):
    """Cria um card para Advantage+ comparação"""
    color = 'primary' if is_on else 'secondary'
    return html.Div([
        html.Div([
            html.Div(className=f"adv-indicator adv-{color}"),
            html.Div([
                html.Div(title, className="adv-title"),
                html.Div(f"ROAS: {roas:.2f}x | CPA: R$ {cpa:.2f}", className="adv-metrics"),
                html.Div(f"{count} creatives", className="adv-count")
            ], className="adv-content")
        ], className="adv-row")
    ], className="advantage-card")


def format_currency_br(value):
    """Formata número em moeda estilo brasileiro (R$ 1.234,56)."""
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_number_br(value):
    """Formata número com separador de milhar brasileiro."""
    return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_analyst_insights(total_spend, total_conversions, avg_roas, best_placement, best_placement_roas,
                          best_targeting, best_targeting_roas, best_objective, best_objective_roas,
                          avg_ctr, avg_cpc, avg_cpa, avg_cpm):
    """Monta os blocos de narrativa e eficiência responsivos ao filtro.

    A estrutura segue o modelo:
    - Narrativa Geral numerada (1–4), sempre deixando claro "de quem" estamos falando
      (placement, targeting, objective) para o recorte atual dos filtros.
    - Insights Sobre Eficiência em forma de bullet, conectando métricas com ação.
    """
    # Narrativa geral numerada (quem / quanto / onde está melhor)
    narrativa = html.Div([
        html.Div("Narrativa Geral", className="insight-title-lg"),
        html.Ol([
            html.Li([
                "Considerando os filtros atuais (campanhas, conjuntos e segmentações selecionadas), o investimento total analisado é de ",
                html.Strong(format_currency_br(total_spend)),
                ", gerando ", html.Strong(format_number_br(total_conversions)),
                " conversões com um ROAS médio de ", html.Strong(f"{avg_roas:.2f}x"),
                ". Esse é o retorno consolidado desse recorte de mídia."
            ]),
            html.Li([
                "O placement com melhor desempenho em ROAS é ", html.Strong(best_placement),
                ", com ROAS médio de ", html.Strong(f"{best_placement_roas:.2f}x"),
                ". Esse canal tende a gerar conversões mais eficientes e deve ser priorizado em testes e ganho incremental de budget."
            ]),
            html.Li([
                "O tipo de targeting com maior retorno é ", html.Strong(best_targeting),
                ", com ROAS médio de ", html.Strong(f"{best_targeting_roas:.2f}x"),
                ". Isso reforça a importância de concentrar variações dentro desse segmento (novas mensagens, criativos e audiências derivadas como lookalikes)."
            ]),
            html.Li([
                "Entre os Campaign Objectives, o melhor ROAS médio foi observado em ",
                html.Strong(best_objective),
                ", com ROAS médio de ", html.Strong(f"{best_objective_roas:.2f}x"),
                ". Isso indica que esse objetivo está mais alinhado com o resultado de negócio atual e deve ser a referência principal em novos testes de campanha."
            ])
        ], className="insight-list-narrative")
    ], className="insight-section-card")

    eficiencia = html.Div([
        html.Div("Insights Sobre Eficiência", className="insight-title-lg"),
        html.Ul([
            html.Li([
                "A CTR média do recorte atual é de ", html.Strong(f"{avg_ctr:.2f}%"),
                ", indicando o nível de atratividade das peças criativas que estão rodando nesses placements e targetings."
            ]),
            html.Li([
                "KPIs de CPC a ", html.Strong(format_currency_br(avg_cpc)),
                " e CPA médio em ", html.Strong(format_currency_br(avg_cpa)),
                " mostram o custo de levar um usuário à ação. Nesse contexto, combinações de CPC baixo com CPA alto indicam gargalo pós-clique (landing page, oferta ou jornada)."
            ]),
            html.Li([
                "O CPM médio de ", html.Strong(format_currency_br(avg_cpm)),
                " indica o custo de alcance desse público. Placements que combinam CPM mais baixo com ROAS alto, como ", html.Strong(best_placement),
                " quando aplicável, devem ser priorizados no plano de mídia."
            ]),
            html.Li([
                "Se o CPA observado ficar acima da meta interna, priorize redistribuição de orçamento para creatives, placements e targetings com ROAS acima da média e CTR mais alta, pausando gradualmente combinações com CTR baixo, CPM alto e ROAS abaixo do esperado."
            ])
        ], className="insight-list-bullets")
    ], className="insight-section-card")

    return html.Div([
        narrativa,
        eficiencia
    ], className="insight-analyst")


def create_chart_container(title, chart):
    """Cria container para gráficos"""
    return html.Div([
        html.H3(title, className="chart-title"),
        chart
    ], className="chart-container")


# ============================================
# GRÁFICOS DO DASHBOARD PRINCIPAL
# ============================================

def create_placement_distribution():
    placement_data = df.groupby('Placement').agg({
        'Impressions': 'sum', 'Conversions': 'sum', 'Amount spent': 'sum', 'ROAS': 'mean'
    }).reset_index()
    
    fig = px.bar(placement_data, x='Placement', y='Conversions', color='ROAS',
                 color_continuous_scale='Blues', template=CHART_TEMPLATE)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20, r=20, t=30, b=20), height=350)
    return fig


def create_advantage_comparison():
    comparison_data = df.groupby('Advantage+ Placements').agg({
        'ROAS': 'mean', 'CPA': 'mean', 'CTR': 'mean', 'Conversions': 'sum'
    }).reset_index()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROAS Médio", "CPA Médio"))
    colors = [COLORS['primary'] if x == 'Yes' else COLORS['text_secondary'] 
              for x in comparison_data['Advantage+ Placements']]
    
    fig.add_trace(go.Bar(x=comparison_data['Advantage+ Placements'], y=comparison_data['ROAS'],
                         marker_color=colors, name='ROAS'), row=1, col=1)
    fig.add_trace(go.Bar(x=comparison_data['Advantage+ Placements'], y=comparison_data['CPA'],
                         marker_color=colors, name='CPA'), row=1, col=2)
    
    fig.update_layout(template=CHART_TEMPLATE, plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=40, b=20),
                      height=350, showlegend=False)
    return fig


def create_objective_performance():
    objective_data = df.groupby('Campaign Objective').agg({
        'ROAS': 'mean', 'CPA': 'mean', 'Conversions': 'sum', 'Amount spent': 'sum'
    }).reset_index()
    
    fig = px.scatter(objective_data, x='CPA', y='ROAS', size='Conversions',
                     color='Campaign Objective', template=CHART_TEMPLATE, size_max=60)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20, r=20, t=30, b=20), height=350)
    return fig


def create_targeting_performance():
    targeting_data = df.groupby('Targeting').agg({
        'ROAS': 'mean', 'CPA': 'mean', 'CTR': 'mean', 'Conversions': 'sum'
    }).reset_index()
    
    fig = px.bar(targeting_data, x='Targeting', y='ROAS', color='CTR',
                 color_continuous_scale='Viridis', template=CHART_TEMPLATE)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20, r=20, t=30, b=20), height=350)
    return fig


def create_creative_scatter():
    fig = px.scatter(df, x='CPA', y='ROAS', size='Conversions', color='Campaign Objective',
                     hover_name='Nome do Creative', hover_data=['Placement', 'Targeting', 'CTR'],
                     template=CHART_TEMPLATE, size_max=40)
    fig.add_hline(y=df['ROAS'].mean(), line_dash="dash", line_color=COLORS['accent'],
                  annotation_text="ROAS Médio")
    fig.add_vline(x=df['CPA'].mean(), line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="CPA Médio")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20, r=20, t=30, b=20), height=400)
    return fig


def create_top_creatives_chart():
    top_10 = top_creatives.head(10)
    fig = px.bar(top_10, x='ROAS', y='Nome do Creative', orientation='h',
                 color='score', color_continuous_scale='Greens', template=CHART_TEMPLATE,
                 hover_data=['CPA', 'CTR', 'Conversions'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20, r=20, t=30, b=20), height=400,
                      yaxis={'categoryorder': 'total ascending'})
    return fig


# ============================================
# GRÁFICOS DE MARKETING SCIENCE (INTEGRADOS)
# ============================================

def create_impact_roas_chart():
    """Gráfico de impacto das variáveis no ROAS"""
    impact = full_report['roas_impact']
    variables = [v['variable'] for v in impact['impact_ranking']]
    scores = [v['impact_score'] for v in impact['impact_ranking']]
    colors_impact = [COLORS['success'] if v['direction'] == 'Positivo' else COLORS['danger'] 
                     for v in impact['impact_ranking']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=variables[::-1], x=scores[::-1], orientation='h',
        marker_color=colors_impact[::-1],
        text=[f"{s:.1f}%" for s in scores[::-1]], textposition='outside'
    ))
    
    fig.update_layout(
        title='Impacto das Variáveis no ROAS (Importância %)',
        template=CHART_TEMPLATE, height=400,
        xaxis_title='Importância (%)', yaxis_title='',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=80, t=50, b=20)
    )
    return fig


def create_diff_in_diff_chart():
    """Gráfico Difference-in-Differences"""
    did = full_report['diff_in_diff']
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('ROAS', 'CPA ($)', 'CTR (%)'))
    
    for i, metric in enumerate(['ROAS', 'CPA', 'CTR'], 1):
        data = did[metric]
        fig.add_trace(
            go.Bar(
                x=['Com Advantage+', 'Sem Advantage+'],
                y=[data['treatment_mean'], data['control_mean']],
                marker_color=[COLORS['primary'], COLORS['accent']],
                text=[f"{data['treatment_mean']:.2f}", f"{data['control_mean']:.2f}"],
                textposition='outside'
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title='Difference-in-Differences: Advantage+ Placements',
        template=CHART_TEMPLATE, height=400, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_frequency_cpa_chart():
    """Gráfico Frequência x CPA"""
    freq = full_report['frequency_analysis']
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df['Frequency'], y=df['CPA'], mode='markers',
        marker=dict(size=8, color=df['ROAS'], colorscale='Viridis',
                    showscale=True, colorbar=dict(title='ROAS')),
        text=df['Nome do Creative'],
        hovertemplate='<b>%{text}</b><br>Frequency: %{x:.2f}<br>CPA: $%{y:.2f}<extra></extra>',
        name='Creatives'
    ))
    
    # Linha de tendência
    freq_vals = np.linspace(df['Frequency'].min(), df['Frequency'].max(), 100)
    cpa_pred = freq['regression']['intercept'] + freq['regression']['coefficient'] * freq_vals
    
    fig.add_trace(go.Scatter(
        x=freq_vals, y=cpa_pred, mode='lines',
        line=dict(color=COLORS['danger'], width=3, dash='dash'),
        name=f'Tendência (r={freq["correlation"]["pearson"]:.3f})'
    ))
    
    fig.update_layout(
        title=f'Relação Frequency × CPA (Correlação: {freq["correlation"]["strength"]})',
        xaxis_title='Frequency', yaxis_title='CPA ($)',
        template=CHART_TEMPLATE, height=450,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_decision_matrix_chart():
    """Gráfico Matriz de Decisão"""
    matrix = full_report['decision_matrix']
    
    categories = ['🚀 Escalar', '🔬 Testar', '⏸️ Pausar']
    counts = [matrix['scale']['count'], matrix['test']['count'], matrix['pause']['count']]
    colors_matrix = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    roas_vals = [matrix['scale']['avg_roas'], matrix['test']['avg_roas'], matrix['pause']['avg_roas']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=counts, marker_color=colors_matrix,
        text=[f"{c} creatives<br>ROAS: {r:.2f}x" for c, r in zip(counts, roas_vals)],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Matriz de Decisão: Distribuição de Creatives',
        template=CHART_TEMPLATE, height=400,
        yaxis_title='Número de Creatives',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_forecast_chart():
    """Gráfico de Previsões 30 dias"""
    forecast = full_report['forecast']
    bl = forecast['baseline']
    opt = forecast['optimized']
    agg = forecast['aggressive']
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('ROAS', 'CPA ($)', 'Conversões'))
    
    scenarios = ['Baseline', 'Otimizado', 'Agressivo']
    scenario_colors = [COLORS['accent'], COLORS['success'], COLORS['primary']]
    
    # ROAS
    roas_vals = [bl['roas']['value'], opt['roas']['value'], agg['roas']['value']]
    fig.add_trace(go.Bar(x=scenarios, y=roas_vals, marker_color=scenario_colors,
                         text=[f"{v:.1f}x" for v in roas_vals], textposition='outside'), row=1, col=1)
    
    # CPA
    cpa_vals = [bl['cpa']['value'], opt['cpa']['value'], agg['cpa']['value']]
    fig.add_trace(go.Bar(x=scenarios, y=cpa_vals, marker_color=scenario_colors,
                         text=[f"${v:.2f}" for v in cpa_vals], textposition='outside'), row=1, col=2)
    
    # Conversões
    conv_vals = [bl['conversions']['value'], opt['conversions']['value'], agg['conversions']['value']]
    fig.add_trace(go.Bar(x=scenarios, y=conv_vals, marker_color=scenario_colors,
                         text=[f"{v:,.0f}" for v in conv_vals], textposition='outside'), row=1, col=3)
    
    fig.update_layout(
        title='Previsões para 30 Dias - Cenários',
        template=CHART_TEMPLATE, height=400, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_correlation_heatmap():
    """Mapa de correlação entre métricas"""
    corr_cols = ['Impressions', 'Reach', 'Frequency', 'Clicks', 'Conversions', 
                 'Amount spent', 'CTR', 'CPC', 'CPM', 'CPA', 'ROAS']
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
                    template=CHART_TEMPLATE)
    fig.update_layout(
        title='Matriz de Correlação entre Métricas',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20), height=500
    )
    return fig


# ============================================
# GRÁFICOS DE LIFT ANALYSIS
# ============================================

def create_lift_chart_by_dimension(dimension_data, dimension_name, metric='roas_lift_pct'):
    """Cria gráfico de barras horizontais de Lift por dimensão"""
    if 'ranking' not in dimension_data or not dimension_data['ranking']:
        return go.Figure()
    
    ranking = dimension_data['ranking']
    names = [item['name'] for item in ranking]
    lifts = [item[metric] for item in ranking]
    counts = [item['count'] for item in ranking]
    significant = [item.get('is_significant', False) for item in ranking]
    
    # Cores baseadas em lift positivo/negativo
    colors = [COLORS['success'] if l > 0 else COLORS['danger'] for l in lifts]
    
    # Adicionar asterisco para significativos
    labels = [f"{l:+.1f}%{'*' if s else ''}" for l, s in zip(lifts, significant)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names[::-1],  # Reverter para maior em cima
        x=lifts[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=labels[::-1],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Lift: %{x:.1f}%<br><extra></extra>'
    ))
    
    # Linha de referência em 0
    fig.add_vline(x=0, line_dash="solid", line_color="#94a3b8", line_width=2)
    
    fig.update_layout(
        title=f'ROAS Lift por {dimension_name} (vs. Média Global)',
        xaxis_title='Lift (%)',
        yaxis_title='',
        template=CHART_TEMPLATE,
        height=max(300, len(names) * 50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=100, t=50, b=20)
    )
    
    return fig


def create_advantage_lift_comparison():
    """Cria gráfico de comparação Advantage+ ON vs OFF com Lift"""
    lift_data = full_report['lift_analysis']['advantage_placements']
    
    if 'Yes' not in lift_data['categories'] or 'No' not in lift_data['categories']:
        return go.Figure()
    
    yes_data = lift_data['categories']['Yes']
    no_data = lift_data['categories']['No']
    
    metrics = ['ROAS', 'CPA', 'CTR']
    yes_values = [yes_data['avg_roas'], yes_data['avg_cpa'], yes_data['avg_ctr']]
    no_values = [no_data['avg_roas'], no_data['avg_cpa'], no_data['avg_ctr']]
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
    
    for i, (metric, yes_val, no_val) in enumerate(zip(metrics, yes_values, no_values), 1):
        # Calcular lift
        lift = ((yes_val / no_val) - 1) * 100 if no_val != 0 else 0
        
        fig.add_trace(
            go.Bar(
                x=['Advantage+ ON', 'Advantage+ OFF'],
                y=[yes_val, no_val],
                marker_color=[COLORS['success'], COLORS['text_secondary']],
                text=[f"{yes_val:.2f}", f"{no_val:.2f}"],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=i
        )
        
        # Adicionar anotação de lift
        lift_color = COLORS['success'] if (lift > 0 and metric != 'CPA') or (lift < 0 and metric == 'CPA') else COLORS['danger']
        fig.add_annotation(
            x=0.5, y=max(yes_val, no_val) * 1.15,
            text=f"<b>Lift: {lift:+.1f}%</b>",
            showarrow=False,
            font=dict(size=12, color=lift_color),
            xref=f'x{i}', yref=f'y{i}'
        )
    
    fig.update_layout(
        title='Comparação Advantage+ Placements: ON vs OFF',
        template=CHART_TEMPLATE,
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_combined_lift_chart():
    """Cria gráfico combinado mostrando o melhor de cada dimensão"""
    lift = full_report['lift_analysis']
    
    # Pegar o melhor de cada dimensão
    best_items = []
    
    if lift['campaign_objective']['ranking']:
        best = lift['campaign_objective']['ranking'][0]
        best_items.append({
            'dimension': 'Objective',
            'name': best['name'],
            'lift': best['roas_lift_pct'],
            'significant': best.get('is_significant', False)
        })
    
    if lift['targeting']['ranking']:
        best = lift['targeting']['ranking'][0]
        best_items.append({
            'dimension': 'Targeting',
            'name': best['name'],
            'lift': best['roas_lift_pct'],
            'significant': best.get('is_significant', False)
        })
    
    if lift['placement']['ranking']:
        best = lift['placement']['ranking'][0]
        best_items.append({
            'dimension': 'Placement',
            'name': best['name'],
            'lift': best['roas_lift_pct'],
            'significant': best.get('is_significant', False)
        })
    
    if 'uplift' in lift['advantage_placements']:
        best_items.append({
            'dimension': 'Advantage+',
            'name': 'ON vs OFF',
            'lift': lift['advantage_placements']['uplift']['roas_uplift'],
            'significant': True
        })
    
    if not best_items:
        return go.Figure()
    
    # Ordenar por lift
    best_items = sorted(best_items, key=lambda x: x['lift'], reverse=True)
    
    labels = [f"{item['dimension']}: {item['name']}" for item in best_items]
    lifts = [item['lift'] for item in best_items]
    colors = [COLORS['success'] if l > 0 else COLORS['danger'] for l in lifts]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels[::-1],
        x=lifts[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{l:+.1f}%" for l in lifts[::-1]],
        textposition='outside'
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="#94a3b8", line_width=2)
    
    fig.update_layout(
        title='🏆 Maiores Lifts por Dimensão (ROAS vs. Média Global)',
        xaxis_title='ROAS Lift (%)',
        template=CHART_TEMPLATE,
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=80, t=50, b=20)
    )
    
    return fig


# ============================================
# LAYOUT
# ============================================

# Header
header = html.Div([
    html.H1([html.Span("📊 "), "Meta Ads Creative Performance Dashboard"], className="dashboard-title"),
    html.P("Análise avançada de performance de creatives com Marketing Science", className="dashboard-subtitle")
], className="dashboard-header")

# KPIs - Dashboard Principal (4 cards principais)
total_creatives = len(df)
kpi_section = html.Div([
    html.H2("Dashboard Principal", className="section-title"),
    html.Div([
        create_kpi_card(
            "💰", 
            f"R$ {metrics['total_spend']:,.2f}", 
            "INVESTIMENTO TOTAL",
            f"{total_creatives} creatives",
            "purple"
        ),
        create_kpi_card(
            "🎯", 
            f"{metrics['total_conversions']:,.0f}", 
            "CONVERSÕES",
            f"CPA: R$ {metrics['avg_cpa']:.2f}",
            "blue"
        ),
        create_kpi_card(
            "📊", 
            f"{metrics['avg_ctr']:.2f}%", 
            "CTR MÉDIO",
            f"CPC: R$ {metrics['avg_cpc']:.2f}" if 'avg_cpc' in metrics else f"CPC: R$ {(metrics['total_spend']/metrics['total_clicks']) if metrics['total_clicks'] > 0 else 0:.2f}",
            "yellow"
        ),
        create_kpi_card(
            "📈", 
            f"{metrics['overall_roas']:.2f}x", 
            "ROAS MÉDIO",
            f"CPM: R$ {(metrics['total_spend']/metrics['total_impressions']*1000) if metrics['total_impressions'] > 0 else 0:.2f}",
            "green"
        ),
    ], className="kpi-container-main")
], className="dashboard-section")

# Calcular dados para seções de Performance
placement_perf = df.groupby('Placement').agg({'ROAS': 'mean', 'CPA': 'mean'}).reset_index().sort_values('ROAS', ascending=False)
targeting_perf = df.groupby('Targeting').agg({'ROAS': 'mean', 'CPA': 'mean'}).reset_index().sort_values('ROAS', ascending=False)
objective_perf = df.groupby('Campaign Objective').agg({'ROAS': 'mean', 'CPA': 'mean'}).reset_index().sort_values('ROAS', ascending=False)

# Advantage+ dados
advantage_on = df[df['Advantage+ Placements'] == 'Yes']
advantage_off = df[df['Advantage+ Placements'] == 'No']
adv_on_roas = advantage_on['ROAS'].mean() if len(advantage_on) > 0 else 0
adv_on_cpa = advantage_on['CPA'].mean() if len(advantage_on) > 0 else 0
adv_on_count = len(advantage_on)
adv_off_roas = advantage_off['ROAS'].mean() if len(advantage_off) > 0 else 0
adv_off_cpa = advantage_off['CPA'].mean() if len(advantage_off) > 0 else 0
adv_off_count = len(advantage_off)

max_placement_roas = placement_perf['ROAS'].max() if len(placement_perf) > 0 else 1
max_targeting_roas = targeting_perf['ROAS'].max() if len(targeting_perf) > 0 else 1
max_objective_roas = objective_perf['ROAS'].max() if len(objective_perf) > 0 else 1

# Seção Performance por Placement
performance_placement = html.Div([
    html.H3("Performance por Placement", className="perf-section-title"),
    html.Div([
        create_performance_bar(row['Placement'], row['ROAS'], max_placement_roas, 'blue')
        for _, row in placement_perf.iterrows()
    ], className="perf-bar-container")
], className="perf-section")

# Seção Performance por Targeting
performance_targeting = html.Div([
    html.H3("Performance por Targeting", className="perf-section-title"),
    html.Div([
        create_performance_bar(row['Targeting'], row['ROAS'], max_targeting_roas, 'green', row['CPA'])
        for _, row in targeting_perf.iterrows()
    ], className="perf-bar-container")
], className="perf-section")

# Seção Performance por Objetivo
performance_objective = html.Div([
    html.H3("Performance por Objetivo", className="perf-section-title"),
    html.Div([
        create_performance_bar(row['Campaign Objective'], row['ROAS'], max_objective_roas, 'purple')
        for _, row in objective_perf.iterrows()
    ], className="perf-bar-container")
], className="perf-section")

# Seção Advantage+ vs Manual
advantage_section = html.Div([
    html.H3("Advantage+ vs Manual", className="perf-section-title"),
    html.Div([
        create_advantage_card("Advantage+ ON", adv_on_roas, adv_on_cpa, adv_on_count, True),
        create_advantage_card("Advantage+ OFF", adv_off_roas, adv_off_cpa, adv_off_count, False),
    ], className="advantage-container")
], className="perf-section")

# Grid de Performance Sections
performance_grid = html.Div([
    html.Div([
        performance_placement,
        performance_targeting
    ], className="perf-row"),
    html.Div([
        performance_objective,
        advantage_section
    ], className="perf-row")
], className="performance-grid")

# Filtros - Sidebar vertical
filter_sidebar = html.Div([
    html.Div([html.Span("🔍 "), "Filtros"], className="filter-title"),
    html.Div([
        html.Label("Campanha"),
        dcc.Dropdown(id='filter-campaign', options=[{'label': c, 'value': c} for c in df['Campanha'].unique()],
                     multi=True, placeholder="Todas")
    ], className="filter-item"),
    html.Div([
        html.Label("Adset"),
        dcc.Dropdown(id='filter-adset', options=[{'label': a, 'value': a} for a in df['Adset'].unique()],
                     multi=True, placeholder="Todos")
    ], className="filter-item"),
    html.Div([
        html.Label("Placement"),
        dcc.Dropdown(id='filter-placement', options=[{'label': p, 'value': p} for p in df['Placement'].unique()],
                     multi=True, placeholder="Todos")
    ], className="filter-item"),
    html.Div([
        html.Label("Advantage+"),
        dcc.Dropdown(id='filter-advantage', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}],
                     multi=True, placeholder="Todos")
    ], className="filter-item"),
    html.Div([
        html.Label("Objetivo"),
        dcc.Dropdown(id='filter-objective', options=[{'label': o, 'value': o} for o in df['Campaign Objective'].unique()],
                     multi=True, placeholder="Todos")
    ], className="filter-item"),
    html.Div([
        html.Label("Targeting"),
        dcc.Dropdown(id='filter-targeting', options=[{'label': t, 'value': t} for t in df['Targeting'].unique()],
                     multi=True, placeholder="Todos")
    ], className="filter-item"),
], className="filter-sidebar")

# Extrair dados do relatório
did = full_report['diff_in_diff']
freq = full_report['frequency_analysis']
uplift = full_report['advantage_uplift']
top_patterns = full_report['top_performers']
matrix = full_report['decision_matrix']
forecast = full_report['forecast']
impact = full_report['roas_impact']
priority_recs = full_report['priority_recommendations']
exec_summary = full_report['executive_summary']
lift_analysis = full_report['lift_analysis']  # NOVO: Lift Analysis
regression_stats = full_report['regression_stats']  # Estatísticas de Regressão
mmm_analysis = full_report['mmm_analysis']  # Marketing Mix Contribution

# Tabs principais com Dashboard Principal como primeira aba dinâmica
tabs = dcc.Tabs([
    # Tab 1: Dashboard Principal (dinâmico - afetado pelos filtros)
    dcc.Tab(label='📊 Dashboard Principal', children=[
        html.Div([
            # KPIs dinâmicos (Big Numbers)
            html.Div(id='dynamic-kpis'),
            # Tabela de Indicadores por Creative (ROAS, CPA, CTR, CPC, CPM)
            html.Div(id='dynamic-creatives-table'),
            # Gráficos de comparação (Placement, Targeting, Objective, Advantage+)
            html.Div(id='dynamic-performance')
        ], className="tab-content")
    ]),
    
    # Tab 2: Top Creatives
    dcc.Tab(label='🏆 Top Creatives', children=[
        html.Div([
            html.H2("🏆 Top 10 Creatives de Melhor Performance", style={'marginBottom': '20px'}),
            create_chart_container("Ranking dos Top Creatives por ROAS",
                dcc.Graph(figure=create_top_creatives_chart())),
            
            html.Div([
                html.H3("📋 Detalhes dos Top Creatives", className="chart-title"),
                dash_table.DataTable(
                    columns=[
                        {'name': '#', 'id': 'rank'}, {'name': 'Creative', 'id': 'Nome do Creative'},
                        {'name': 'Score', 'id': 'score'}, {'name': 'ROAS', 'id': 'ROAS'},
                        {'name': 'CPA', 'id': 'CPA'}, {'name': 'CTR %', 'id': 'CTR'},
                        {'name': 'Placement', 'id': 'Placement'}, {'name': 'Targeting', 'id': 'Targeting'},
                    ],
                    data=[{**row, 'rank': i+1, 'ROAS': f"{row['ROAS']:.2f}", 'CPA': f"${row['CPA']:.2f}", 'CTR': f"{row['CTR']:.2f}"} 
                          for i, row in enumerate(top_creatives.head(10).to_dict('records'))],
                    style_header={'backgroundColor': '#f8fafc', 'color': '#374151', 'fontWeight': 'bold', 'borderBottom': '2px solid #e5e7eb'},
                    style_cell={'backgroundColor': '#ffffff', 'color': '#1f2937', 'border': '1px solid #e5e7eb', 'padding': '12px'}
                )
            ], className="chart-container"),
            
            html.Div([
                html.H3("🔍 Padrões Identificados", className="chart-title"),
                html.Div([
                    html.Div([html.Div(patterns['top_placement'], className="pattern-value"),
                              html.Div("Placement Predominante", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(patterns['top_targeting'], className="pattern-value"),
                              html.Div("Targeting Predominante", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"{patterns['advantage_placements']:.0f}%", className="pattern-value"),
                              html.Div("Usam Advantage+", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"{patterns['avg_roas']:.2f}x", className="pattern-value"),
                              html.Div("ROAS Médio Top 10", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"${patterns['avg_cpa']:.2f}", className="pattern-value"),
                              html.Div("CPA Médio Top 10", className="pattern-label")], className="pattern-item"),
                ], className="pattern-grid")
            ], className="chart-container")
        ], className="tab-content")
    ]),
    
    # Tab 3: MARKETING SCIENCE (SIMPLIFICADO)
    dcc.Tab(label='🔬 Marketing Science', children=[
        html.Div([
            # Header da seção
            html.Div([
                html.H2([html.Span("✓ ", style={'color': COLORS['success']}), "Marketing Science"], 
                       style={'marginBottom': '10px', 'fontSize': '1.8em', 'fontWeight': '600'}),
                html.P("Análise de impacto causal e recomendações baseadas em dados",
                       style={'color': '#6b7280', 'marginBottom': '30px'})
            ]),
            
            # ========== SEÇÃO 1: CARDS PRINCIPAIS (2 COLUNAS) ==========
            html.Div([
                # Card 1: Uplift de Advantage+ Placements
                html.Div([
                    html.H3("Uplift de Advantage+ Placements", style={'marginBottom': '20px', 'fontWeight': '600', 'fontSize': '1.1em'}),
                    
                    # Advantage+ ON
                    html.Div([
                        html.Div("Advantage+ ON", style={'fontWeight': '600', 'marginBottom': '8px'}),
                        html.Div([
                            html.Span("ROAS médio: ", style={'color': '#6b7280'}),
                            html.Span(f"{uplift['ROAS']['with_advantage']:.2f}x", style={'fontWeight': '600'})
                        ], style={'marginBottom': '4px'}),
                        html.Div([
                            html.Span("CPA médio: ", style={'color': '#6b7280'}),
                            html.Span(f"R$ {uplift['CPA']['with_advantage']:.2f}", style={'fontWeight': '600'})
                        ], style={'marginBottom': '4px'}),
                        html.Div(f"{uplift['summary']['total_with_advantage']} creatives", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px', 'marginBottom': '15px'}),
                    
                    # Advantage+ OFF
                    html.Div([
                        html.Div("Advantage+ OFF", style={'fontWeight': '600', 'marginBottom': '8px'}),
                        html.Div([
                            html.Span("ROAS médio: ", style={'color': '#6b7280'}),
                            html.Span(f"{uplift['ROAS']['without_advantage']:.2f}x", style={'fontWeight': '600'})
                        ], style={'marginBottom': '4px'}),
                        html.Div([
                            html.Span("CPA médio: ", style={'color': '#6b7280'}),
                            html.Span(f"R$ {uplift['CPA']['without_advantage']:.2f}", style={'fontWeight': '600'})
                        ], style={'marginBottom': '4px'}),
                        html.Div(f"{uplift['summary']['total_without_advantage']} creatives", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px', 'marginBottom': '15px'}),
                    
                    # Resultado do Uplift
                    html.Div([
                        html.Span(f"Uplift estimado: {uplift['summary']['roas_uplift']:.2f}% ROAS", 
                                 style={'color': COLORS['primary'], 'fontWeight': '600', 'fontSize': '0.95em'})
                    ], style={'padding': '12px', 'backgroundColor': 'rgba(99, 102, 241, 0.1)', 'borderRadius': '8px', 'borderLeft': f'3px solid {COLORS["primary"]}'})
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'}),
                
                # Card 2: Correlação Frequency x CPA
                html.Div([
                    html.H3("Correlação Frequency x CPA", style={'marginBottom': '10px', 'fontWeight': '600', 'fontSize': '1.1em'}),
                    html.P("Análise da relação entre frequência de exposição e custo por aquisição.",
                           style={'color': '#6b7280', 'marginBottom': '20px', 'fontSize': '0.9em'}),
                    
                    # Métricas
                    html.Div([
                        html.Div([
                            html.Span("Frequency média: ", style={'color': '#6b7280'}),
                            html.Span(f"{df['Frequency'].mean():.2f}", style={'fontWeight': '600'})
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("CPA médio: ", style={'color': '#6b7280'}),
                            html.Span(f"R$ {df['CPA'].mean():.2f}", style={'fontWeight': '600'})
                        ])
                    ], style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px', 'marginBottom': '20px'}),
                    
                    # Recomendação
                    html.Div([
                        html.P([
                            html.Strong("Recomendação: "),
                            f"Manter frequency entre 1.2 e 2.0 para otimizar CPA sem saturar o público."
                        ], style={'color': '#374151', 'fontSize': '0.9em', 'lineHeight': '1.6'})
                    ])
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'})
            ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px'}),
            
            # ========== SEÇÃO 2: IMPACTO DAS VARIÁVEIS ==========
            html.Div([
                html.H3("📊 Impacto das Variáveis no ROAS", style={'marginBottom': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span(f"#{i+1} ", style={'color': COLORS['primary'], 'fontWeight': '700'}),
                            html.Span(f"{var['variable']}", style={'fontWeight': '500'}),
                            html.Span(f" — {var['impact_score']:.1f}%", style={'color': '#6b7280', 'marginLeft': '10px'})
                        ], style={'padding': '12px 15px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px', 'marginBottom': '8px',
                                 'borderLeft': f'3px solid {COLORS["primary"] if i == 0 else COLORS["secondary"] if i == 1 else "#e5e7eb"}'})
                    ]) for i, var in enumerate(impact['impact_ranking'][:5])
                ])
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)', 'marginBottom': '25px'}),
            
            # ========== SEÇÃO 3: TABELA DE ESTATÍSTICAS DE REGRESSÃO ==========
            html.Div([
                html.H3("📈 Análise Estatística de Regressão", style={'marginBottom': '20px', 'fontWeight': '600'}),
                
                # Resumo do modelo completo
                html.Div([
                    html.Div([
                        html.Span("Modelo Completo: ", style={'fontWeight': '600', 'color': '#374151'}),
                        html.Span(f"R² = {regression_stats['full_model']['r_squared']:.4f}", style={'marginRight': '20px'}),
                        html.Span(f"R² Ajustado = {regression_stats['full_model']['r_squared_adj']:.4f}", style={'marginRight': '20px'}),
                        html.Span(f"Erro Padrão = {regression_stats['full_model']['std_error']:.4f}", style={'marginRight': '20px'}),
                        html.Span(f"N = {regression_stats['full_model']['n_observations']}", style={'color': '#6b7280'})
                    ], style={'padding': '15px', 'backgroundColor': 'rgba(59, 130, 246, 0.08)', 'borderRadius': '8px', 'marginBottom': '20px', 'fontSize': '0.9em'})
                ]),
                
                # Tabela de variáveis
                html.Table([
                    # Header
                    html.Thead([
                        html.Tr([
                            html.Th("Variável", style={'textAlign': 'left', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'}),
                            html.Th("R²", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'}),
                            html.Th("R² Ajustado", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'}),
                            html.Th("Erro Padrão", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'}),
                            html.Th("t-Statistic", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'}),
                            html.Th("p-Value", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'}),
                            html.Th("Significância", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151'})
                        ])
                    ]),
                    # Body
                    html.Tbody([
                        html.Tr([
                            html.Td(var['variable'], style={'padding': '12px 15px', 'borderBottom': '1px solid #f3f4f6', 'fontWeight': '500'}),
                            html.Td(f"{var['r_squared']:.4f}", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '1px solid #f3f4f6'}),
                            html.Td(f"{var['r_squared_adj']:.4f}", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '1px solid #f3f4f6'}),
                            html.Td(f"{var['std_error']:.4f}", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '1px solid #f3f4f6'}),
                            html.Td(f"{var['t_statistic']:.3f}", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '1px solid #f3f4f6'}),
                            html.Td(f"{var['p_value']:.4f}", style={'textAlign': 'center', 'padding': '12px 15px', 'borderBottom': '1px solid #f3f4f6'}),
                            html.Td(
                                "✓ Sig." if var['is_significant'] else "✗ N.S.",
                                style={
                                    'textAlign': 'center', 
                                    'padding': '12px 15px', 
                                    'borderBottom': '1px solid #f3f4f6',
                                    'color': COLORS['success'] if var['is_significant'] else '#9ca3af',
                                    'fontWeight': '600'
                                }
                            )
                        ]) for var in regression_stats['variables']
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '0.9em'}),
                
                # Legenda
                html.Div([
                    html.Div("Legenda: ", style={'fontWeight': '600', 'marginRight': '10px', 'display': 'inline'}),
                    html.Span("✓ Sig. = Estatisticamente significativo (p < 0.05)", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginRight': '15px'}),
                    html.Span("✗ N.S. = Não significativo", style={'color': '#6b7280', 'fontSize': '0.85em'})
                ], style={'marginTop': '15px', 'padding': '10px', 'backgroundColor': '#f9fafb', 'borderRadius': '6px'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)', 'marginBottom': '25px'}),
            
            # ========== SEÇÃO 4: MARKETING MIX CONTRIBUTION ==========
            html.Div([
                html.H3("🎯 Marketing Mix Contribution", style={'marginBottom': '20px', 'fontWeight': '600'}),
                
                # Elasticidade e Saturação - Cards resumo
                html.Div([
                    # Card Elasticidade
                    html.Div([
                        html.Div("Elasticidade Geral", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '8px'}),
                        html.Div(f"{mmm_analysis['elasticity_analysis']['overall_elasticity']:.2f}", 
                                style={'fontSize': '2em', 'fontWeight': '700', 'color': '#3b82f6', 'marginBottom': '5px'}),
                        html.Div(mmm_analysis['elasticity_analysis']['interpretation'], 
                                style={'fontSize': '0.8em', 'color': '#6b7280'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(59, 130, 246, 0.08)', 
                             'borderRadius': '10px', 'borderLeft': '4px solid #3b82f6', 'textAlign': 'center'}),
                    
                    # Card Saturação
                    html.Div([
                        html.Div("Nível de Saturação", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '8px'}),
                        html.Div(mmm_analysis['saturation_analysis']['saturation_level'], 
                                style={'fontSize': '2em', 'fontWeight': '700', 
                                      'color': '#ef4444' if mmm_analysis['saturation_analysis']['saturation_level'] == 'Alto' 
                                              else '#f59e0b' if mmm_analysis['saturation_analysis']['saturation_level'] == 'Moderado' 
                                              else '#10b981', 'marginBottom': '5px'}),
                        html.Div(f"Faixa ótima: {mmm_analysis['saturation_analysis']['optimal_spend_range']}", 
                                style={'fontSize': '0.8em', 'color': '#6b7280'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(245, 158, 11, 0.08)', 
                             'borderRadius': '10px', 'borderLeft': '4px solid #f59e0b', 'textAlign': 'center'}),
                    
                    # Card Melhor Canal
                    html.Div([
                        html.Div("Melhor Placement", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '8px'}),
                        html.Div(mmm_analysis['model_summary']['best_placement'], 
                                style={'fontSize': '1.3em', 'fontWeight': '700', 'color': '#10b981', 'marginBottom': '5px'}),
                        html.Div(f"Eficiência: {mmm_analysis['by_placement'][0]['efficiency_index']:.2f}x" if mmm_analysis['by_placement'] else "N/A", 
                                style={'fontSize': '0.8em', 'color': '#6b7280'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(16, 185, 129, 0.08)', 
                             'borderRadius': '10px', 'borderLeft': '4px solid #10b981', 'textAlign': 'center'})
                ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '25px'}),
                
                # Tabela de Contribuição por Placement
                html.Div([
                    html.H4("Contribuição por Placement", style={'marginBottom': '15px', 'fontWeight': '600', 'fontSize': '1em'}),
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Placement", style={'textAlign': 'left', 'padding': '10px 12px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151', 'fontSize': '0.85em'}),
                                html.Th("% Spend", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151', 'fontSize': '0.85em'}),
                                html.Th("% Conv.", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151', 'fontSize': '0.85em'}),
                                html.Th("Eficiência", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151', 'fontSize': '0.85em'}),
                                html.Th("ROAS", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151', 'fontSize': '0.85em'}),
                                html.Th("Ação", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '2px solid #e5e7eb', 'fontWeight': '600', 'color': '#374151', 'fontSize': '0.85em'})
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(p['segment'], style={'padding': '10px 12px', 'borderBottom': '1px solid #f3f4f6', 'fontWeight': '500', 'fontSize': '0.9em'}),
                                html.Td(f"{p['spend_share']:.1f}%", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '1px solid #f3f4f6', 'fontSize': '0.9em'}),
                                html.Td(f"{p['conv_share']:.1f}%", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '1px solid #f3f4f6', 'fontSize': '0.9em'}),
                                html.Td(f"{p['efficiency_index']:.2f}x", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '1px solid #f3f4f6', 'fontSize': '0.9em',
                                        'color': '#10b981' if p['efficiency_index'] > 1.1 else '#ef4444' if p['efficiency_index'] < 0.9 else '#6b7280',
                                        'fontWeight': '600'}),
                                html.Td(f"{p['roas']:.2f}x", style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '1px solid #f3f4f6', 'fontSize': '0.9em'}),
                                html.Td(
                                    html.Span(p['recommendation'], style={
                                        'padding': '4px 10px', 'borderRadius': '12px', 'fontSize': '0.8em', 'fontWeight': '600',
                                        'backgroundColor': 'rgba(16, 185, 129, 0.15)' if p['recommendation'] == 'Aumentar' 
                                                          else 'rgba(239, 68, 68, 0.15)' if p['recommendation'] == 'Reduzir' 
                                                          else 'rgba(107, 114, 128, 0.15)',
                                        'color': '#10b981' if p['recommendation'] == 'Aumentar' 
                                                else '#ef4444' if p['recommendation'] == 'Reduzir' 
                                                else '#6b7280'
                                    }),
                                    style={'textAlign': 'center', 'padding': '10px 12px', 'borderBottom': '1px solid #f3f4f6'}
                                )
                            ]) for p in mmm_analysis['by_placement'][:5]
                        ])
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '20px'})
                ]),
                
                # Insight de elasticidade
                html.Div([
                    html.Div("💡 ", style={'display': 'inline'}),
                    html.Span("Insight: ", style={'fontWeight': '600'}),
                    html.Span(mmm_analysis['elasticity_analysis']['recommendation'], style={'color': '#6b7280'})
                ], style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px', 'fontSize': '0.9em'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)', 'marginBottom': '25px'}),
            
            # ========== SEÇÃO 5: PREVISÕES DE 30 DIAS ==========
            html.Div([
                html.H3("Previsões de 30 dias", style={'marginBottom': '25px', 'fontWeight': '600', 'fontSize': '1.1em'}),
                
                # Cards de previsão com cores distintas
                html.Div([
                    # Card 1: Conversões esperadas (azul)
                    html.Div([
                        html.Div("Conversões esperadas", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '10px'}),
                        html.Div(f"{forecast['optimized']['conversions']['value']:,.0f}".replace(',', '.'), 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#3b82f6'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(59, 130, 246, 0.08)', 'borderRadius': '10px', 'borderBottom': '3px solid #3b82f6'}),
                    
                    # Card 2: Investimento projetado (roxo)
                    html.Div([
                        html.Div("Investimento projetado", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '10px'}),
                        html.Div(f"R$ {metrics['total_spend'] * 1.1:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'), 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#8b5cf6'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(139, 92, 246, 0.08)', 'borderRadius': '10px', 'borderBottom': '3px solid #8b5cf6'}),
                    
                    # Card 3: CPA esperado (verde)
                    html.Div([
                        html.Div("CPA esperado", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '10px'}),
                        html.Div(f"R$ {forecast['optimized']['cpa']['value']:.2f}".replace('.', ','), 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#10b981'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(16, 185, 129, 0.08)', 'borderRadius': '10px', 'borderBottom': '3px solid #10b981'}),
                    
                    # Card 4: ROAS esperado (laranja)
                    html.Div([
                        html.Div("ROAS esperado", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '10px'}),
                        html.Div(f"{forecast['optimized']['roas']['value']:.2f}x".replace('.', ','), 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#f59e0b'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(245, 158, 11, 0.08)', 'borderRadius': '10px', 'borderBottom': '3px solid #f59e0b'})
                ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '25px'}),
                
                # Recomendações baseadas nas previsões
                html.Div([
                    html.Div("Recomendações baseadas nas previsões:", style={'fontWeight': '600', 'marginBottom': '12px', 'color': '#374151'}),
                    html.Div([
                        html.Div(f"• Incrementar orçamento gradualmente mantendo a mesma estrutura", style={'marginBottom': '6px', 'color': '#6b7280', 'fontSize': '0.9em'}),
                        html.Div(f"• Priorizar creatives com ROAS acima de {metrics['overall_roas']:.2f}x", style={'marginBottom': '6px', 'color': '#6b7280', 'fontSize': '0.9em'}),
                        html.Div(f"• Monitorar CPA semanalmente para ajustes táticos", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ])
                ], style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'})
            
        ], className="tab-content", style={'backgroundColor': '#f1f5f9', 'padding': '30px'})
    ]),
    
    # Tab 4: Ações Recomendadas (Novo Layout inspirado na imagem)
    dcc.Tab(label='✅ Ações Recomendadas', children=[
        html.Div([
            html.H2([html.Span("⚡ ", style={'color': '#f59e0b'}), "Ações Recomendadas"], 
                   style={'marginBottom': '30px', 'fontWeight': '600'}),
            
            # ========== PRIORIDADE 1 - ALTO IMPACTO / BAIXO ESFORÇO ==========
            html.Div([
                html.Div([
                    html.Div("1", style={'backgroundColor': '#ef4444', 'color': 'white', 'width': '28px', 'height': '28px',
                                        'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center', 
                                        'justifyContent': 'center', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
                    html.Span("Prioridade 1 – Alto Impacto / Baixo Esforço", 
                             style={'marginLeft': '12px', 'fontWeight': '600', 'fontSize': '1.1em'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
                
                html.Ul([
                    html.Li([
                        html.Strong("Escalar orçamento ", style={'color': '#374151'}),
                        html.Span(f"nos creatives com ROAS acima de {metrics['overall_roas']:.1f}x e CPA abaixo de R$ {metrics['avg_cpa']:.2f}.", style={'color': '#6b7280'})
                    ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
                    html.Li([
                        html.Strong("Pausar ou reduzir ", style={'color': '#374151'}),
                        html.Span(f"investimento em {matrix['pause']['count']} creatives com ROAS consistentemente abaixo de 10x.", style={'color': '#6b7280'})
                    ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
                    html.Li([
                        html.Strong("Replicar o formato ", style={'color': '#374151'}),
                        html.Span(f"(vídeo/imagem, copy, CTA) dos Top Creatives em novos testes.", style={'color': '#6b7280'})
                    ], style={'lineHeight': '1.6'}),
                ], style={'listStyleType': 'disc', 'paddingLeft': '20px', 'margin': '0'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 
                     'boxShadow': '0 1px 3px rgba(0,0,0,0.08)', 'marginBottom': '20px',
                     'borderLeft': '4px solid #ef4444'}),
            
            # ========== PRIORIDADE 2 - OTIMIZAÇÃO ESTRUTURAL ==========
            html.Div([
                html.Div([
                    html.Div("2", style={'backgroundColor': '#f59e0b', 'color': 'white', 'width': '28px', 'height': '28px',
                                        'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center', 
                                        'justifyContent': 'center', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
                    html.Span("Prioridade 2 – Otimização Estrutural", 
                             style={'marginLeft': '12px', 'fontWeight': '600', 'fontSize': '1.1em'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
                
                html.Ul([
                    html.Li([
                        html.Span("Reavaliar objetivos de campanha com pior ROAS e realocar verba para objetivos mais eficientes.", style={'color': '#6b7280'})
                    ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
                    html.Li([
                        html.Span("Consolidar campanhas semelhantes para concentrar o aprendizado do algoritmo.", style={'color': '#6b7280'})
                    ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
                    html.Li([
                        html.Span("Ajustar o mix de ", style={'color': '#6b7280'}),
                        html.Strong("Targeting", style={'color': '#374151'}),
                        html.Span(f" (ex.: {'aumentar' if uplift['summary']['roas_uplift'] > 0 else 'testar'} Advantage+ quando apresenta melhor ROAS).", style={'color': '#6b7280'})
                    ], style={'lineHeight': '1.6'}),
                ], style={'listStyleType': 'disc', 'paddingLeft': '20px', 'margin': '0'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 
                     'boxShadow': '0 1px 3px rgba(0,0,0,0.08)', 'marginBottom': '20px',
                     'borderLeft': '4px solid #f59e0b'}),
            
            # ========== PRIORIDADE 3 - TESTES DE MARKETING SCIENCE ==========
            html.Div([
                html.Div([
                    html.Div("3", style={'backgroundColor': '#10b981', 'color': 'white', 'width': '28px', 'height': '28px',
                                        'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center', 
                                        'justifyContent': 'center', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
                    html.Span("Prioridade 3 – Testes de Marketing Science", 
                             style={'marginLeft': '12px', 'fontWeight': '600', 'fontSize': '1.1em'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
                
                html.Ul([
                    html.Li([
                        html.Span("Rodar ", style={'color': '#6b7280'}),
                        html.Strong("testes A/B ", style={'color': '#374151'}),
                        html.Span("isolando o impacto de Advantage+ Placements vs configurações manuais.", style={'color': '#6b7280'})
                    ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
                    html.Li([
                        html.Span(f"Experimentar diferentes níveis de frequency alvo e monitorar o impacto em CPA e ROAS.", style={'color': '#6b7280'})
                    ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
                    html.Li([
                        html.Span(f"Testar novos placements (ex.: {patterns['top_placement']}) com base nos padrões dos top performers.", style={'color': '#6b7280'})
                    ], style={'lineHeight': '1.6'}),
                ], style={'listStyleType': 'disc', 'paddingLeft': '20px', 'margin': '0'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 
                     'boxShadow': '0 1px 3px rgba(0,0,0,0.08)', 'marginBottom': '25px',
                     'borderLeft': '4px solid #10b981'}),
            
            # ========== RESUMO DE IMPACTO ESTIMADO ==========
            html.Div([
                html.H3("📊 Impacto Estimado das Ações", style={'marginBottom': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.Div(f"+{exec_summary['estimated_impact']['roas_improvement']:.1f}%", 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#10b981'}),
                        html.Div("Melhoria no ROAS", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(16, 185, 129, 0.08)', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div(f"-{abs(exec_summary['estimated_impact']['cpa_reduction']):.1f}%", 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#3b82f6'}),
                        html.Div("Redução no CPA", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(59, 130, 246, 0.08)', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div(f"+{exec_summary['estimated_impact']['conversion_increase']:.1f}%", 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#8b5cf6'}),
                        html.Div("Aumento em Conversões", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(139, 92, 246, 0.08)', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div(f"R$ {exec_summary['estimated_impact']['potential_savings']:,.0f}".replace(',', '.'), 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#f59e0b'}),
                        html.Div("Economia Potencial", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(245, 158, 11, 0.08)', 'borderRadius': '10px'})
                ], style={'display': 'flex', 'gap': '15px'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'})
            
        ], className="tab-content", style={'backgroundColor': '#f1f5f9', 'padding': '30px'})
    ]),
    
    # Tab 5: Padrões Vencedores (Novo Layout)
    dcc.Tab(label='🎯 Padrões Vencedores', children=[
        html.Div([
            html.H2([html.Span("🎯 ", style={'color': '#ef4444'}), "Padrões dos Creatives Vencedores"], 
                   style={'marginBottom': '30px', 'fontWeight': '600'}),
            
            # ========== DUAS COLUNAS PRINCIPAIS ==========
            html.Div([
                # COLUNA 1: Características em Comum
                html.Div([
                    html.H3("Características em Comum", style={'marginBottom': '25px', 'fontWeight': '600', 'fontSize': '1.1em'}),
                    
                    # Placements líderes
                    html.Div([
                        html.Div("Placements líderes", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '6px'}),
                        html.Div(", ".join([f"{plc}" for plc, _ in sorted(top_patterns['patterns']['placement']['distribution'].items(), key=lambda x: x[1], reverse=True)[:3]]),
                                style={'fontWeight': '600', 'fontSize': '1em', 'color': '#1f2937'})
                    ], style={'marginBottom': '20px', 'paddingBottom': '20px', 'borderBottom': '1px solid #e5e7eb'}),
                    
                    # Tipos de Targeting
                    html.Div([
                        html.Div("Tipos de Targeting mais comuns", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '6px'}),
                        html.Div(", ".join([f"{tgt}" for tgt, _ in sorted(top_patterns['patterns']['targeting']['distribution'].items(), key=lambda x: x[1], reverse=True)[:3]]),
                                style={'fontWeight': '600', 'fontSize': '1em', 'color': '#1f2937'})
                    ], style={'marginBottom': '20px', 'paddingBottom': '20px', 'borderBottom': '1px solid #e5e7eb'}),
                    
                    # Objectives mais frequentes
                    html.Div([
                        html.Div("Objectives mais frequentes", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '6px'}),
                        html.Div(", ".join([f"{obj}" for obj, _ in sorted(top_patterns['patterns']['objective']['distribution'].items(), key=lambda x: x[1], reverse=True)[:3]]),
                                style={'fontWeight': '600', 'fontSize': '1em', 'color': '#1f2937'})
                    ], style={'marginBottom': '20px', 'paddingBottom': '20px', 'borderBottom': '1px solid #e5e7eb'}),
                    
                    # Creatives vencedores identificados
                    html.Div([
                        html.Div("Creatives vencedores identificados", style={'color': '#6b7280', 'fontSize': '0.85em', 'marginBottom': '6px'}),
                        html.Div(f"{len(top_patterns['top_creatives'])} creatives",
                                style={'fontWeight': '600', 'fontSize': '1em', 'color': '#1f2937'})
                    ])
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 
                         'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'}),
                
                # COLUNA 2: Como Transformar em Briefing Criativo
                html.Div([
                    html.H3("Como Transformar em Briefing Criativo", style={'marginBottom': '25px', 'fontWeight': '600', 'fontSize': '1.1em'}),
                    
                    html.Div([
                        # Recomendação 1
                        html.Div([
                            html.Span("✓ ", style={'color': '#ef4444', 'fontWeight': 'bold', 'marginRight': '8px'}),
                            html.Span("Priorize a produção de novos creatives ", style={'color': '#6b7280'}),
                            html.Strong("seguindo o mesmo Placement e Objective ", style={'color': '#1f2937'}),
                            html.Span("que aparecem nos padrões vencedores.", style={'color': '#6b7280'})
                        ], style={'marginBottom': '18px', 'lineHeight': '1.6'}),
                        
                        # Recomendação 2
                        html.Div([
                            html.Span("✓ ", style={'color': '#ef4444', 'fontWeight': 'bold', 'marginRight': '8px'}),
                            html.Span("Use o tipo de ", style={'color': '#6b7280'}),
                            html.Strong("Targeting vencedor ", style={'color': '#1f2937'}),
                            html.Span("como base e apenas teste variações incrementais.", style={'color': '#6b7280'})
                        ], style={'marginBottom': '18px', 'lineHeight': '1.6'}),
                        
                        # Recomendação 3
                        html.Div([
                            html.Span("✓ ", style={'color': '#ef4444', 'fontWeight': 'bold', 'marginRight': '8px'}),
                            html.Span("Mantenha frequency em faixa próxima à média observada para evitar saturação e queda de CTR.", style={'color': '#6b7280'})
                        ], style={'marginBottom': '18px', 'lineHeight': '1.6'}),
                        
                        # Recomendação 4
                        html.Div([
                            html.Span("✓ ", style={'color': '#ef4444', 'fontWeight': 'bold', 'marginRight': '8px'}),
                            html.Span("Use os vencedores como ", style={'color': '#6b7280'}),
                            html.Strong("benchmark ", style={'color': '#1f2937'}),
                            html.Span("em todo novo teste criativo (sempre rodar contra um champion).", style={'color': '#6b7280'})
                        ], style={'lineHeight': '1.6'})
                    ])
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 
                         'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'})
            ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px'}),
            
            # ========== MÉTRICAS DOS VENCEDORES ==========
            html.Div([
                html.H3("📊 Métricas dos Top Performers", style={'marginBottom': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.Div(f"{top_patterns['patterns']['metrics']['avg_roas']:.2f}x", 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#10b981'}),
                        html.Div("ROAS Médio", style={'color': '#6b7280', 'fontSize': '0.9em'}),
                        html.Div(f"+{top_patterns['patterns']['vs_average']['roas_diff']:.1f}% vs média", 
                                style={'color': '#10b981', 'fontSize': '0.8em', 'marginTop': '4px'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(16, 185, 129, 0.08)', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div(f"R$ {top_patterns['patterns']['metrics']['avg_cpa']:.2f}", 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#3b82f6'}),
                        html.Div("CPA Médio", style={'color': '#6b7280', 'fontSize': '0.9em'}),
                        html.Div(f"{top_patterns['patterns']['vs_average']['cpa_diff']:.1f}% vs média", 
                                style={'color': '#10b981' if top_patterns['patterns']['vs_average']['cpa_diff'] < 0 else '#ef4444', 'fontSize': '0.8em', 'marginTop': '4px'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(59, 130, 246, 0.08)', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div(f"{top_patterns['patterns']['metrics']['avg_ctr']:.2f}%", 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#8b5cf6'}),
                        html.Div("CTR Médio", style={'color': '#6b7280', 'fontSize': '0.9em'}),
                        html.Div(f"+{top_patterns['patterns']['vs_average']['ctr_diff']:.1f}% vs média", 
                                style={'color': '#10b981', 'fontSize': '0.8em', 'marginTop': '4px'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(139, 92, 246, 0.08)', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div(f"{top_patterns['patterns']['metrics']['total_conversions']:,.0f}".replace(',', '.'), 
                                style={'fontSize': '1.8em', 'fontWeight': '700', 'color': '#f59e0b'}),
                        html.Div("Total Conversões", style={'color': '#6b7280', 'fontSize': '0.9em'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'rgba(245, 158, 11, 0.08)', 'borderRadius': '10px'})
                ], style={'display': 'flex', 'gap': '15px'})
            ], style={'padding': '25px', 'backgroundColor': '#ffffff', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'})
            
        ], className="tab-content", style={'backgroundColor': '#f1f5f9', 'padding': '30px'})
    ]),
    
    # Tab 6: Insights
    dcc.Tab(label='💡 Insights', children=[
        html.Div([
            html.H2("🧠 Insight do Analista", style={'marginBottom': '20px'}),
            html.Div(id='insights-analyst-container', style={'marginBottom': '30px'}),
            html.H2("💡 Insights Automáticos", style={'marginBottom': '20px'}),
            html.Div(id='insights-list')
        ], className="tab-content")
    ]),
    
    # Tab 7: RELATÓRIO COMPLETO PARA APRESENTAÇÃO
    dcc.Tab(label='📑 Relatório Completo', children=[
        html.Div([
            # Header do Relatório
            html.Div([
                html.H1("📊 Relatório Completo de Marketing Science", 
                       style={'textAlign': 'center', 'background': 'linear-gradient(135deg, #1877F2, #42b72a)',
                              'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
                              'fontSize': '2.2em', 'fontWeight': '800', 'marginBottom': '10px'}),
                html.P("Meta Ads Creative Performance Analysis", 
                       style={'textAlign': 'center', 'color': '#a0aec0', 'fontSize': '1.2em'}),
            ], style={'padding': '30px', 'background': 'linear-gradient(135deg, rgba(24, 119, 242, 0.15), rgba(66, 183, 42, 0.08))',
                     'borderRadius': '15px', 'marginBottom': '30px', 'border': '1px solid rgba(255,255,255,0.1)'}),
            
            # ========== RESUMO EXECUTIVO ==========
            html.Div([
                html.H2("📋 Resumo Executivo", style={'marginBottom': '20px', 'borderBottom': '2px solid #1877F2', 'paddingBottom': '10px'}),
                html.Div([
                    html.P([
                        f"Este relatório analisa o desempenho de ", html.Strong(f"{exec_summary['total_creatives']:,}"), " criativos de Meta Ads, ",
                        f"com investimento total de ", html.Strong(f"${exec_summary['total_spend']:,.2f}"), " e ",
                        html.Strong(f"{exec_summary['total_conversions']:,.0f}"), " conversões geradas."
                    ], style={'fontSize': '1.1em', 'lineHeight': '1.8', 'marginBottom': '15px'}),
                    html.P([
                        f"O ROAS médio é de ", html.Strong(f"{exec_summary['overall_roas']:.2f}x"), 
                        f", com CPA médio de ", html.Strong(f"${exec_summary['overall_cpa']:.2f}"), ". ",
                        f"A análise identificou ", html.Strong(f"{len(top_creatives)}"), " criativos de alta performance (top 10%)."
                    ], style={'fontSize': '1.1em', 'lineHeight': '1.8'}),
                ], style={'padding': '20px', 'background': 'linear-gradient(135deg, rgba(24,119,242,0.1), rgba(24,119,242,0.02))',
                         'borderRadius': '10px', 'borderLeft': '4px solid #1877F2'}),
                
                html.Div([
                    html.Div([html.Div(f"{exec_summary['total_creatives']:,}", className="pattern-value"), 
                              html.Div("Total Criativos", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"${exec_summary['total_spend']:,.0f}", className="pattern-value"),
                              html.Div("Investimento", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"{exec_summary['total_conversions']:,.0f}", className="pattern-value"),
                              html.Div("Conversões", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"{exec_summary['overall_roas']:.1f}x", className="pattern-value"),
                              html.Div("ROAS", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"${exec_summary['overall_cpa']:.2f}", className="pattern-value"),
                              html.Div("CPA", className="pattern-label")], className="pattern-item"),
                    html.Div([html.Div(f"{metrics['avg_ctr']:.2f}%", className="pattern-value"),
                              html.Div("CTR", className="pattern-label")], className="pattern-item"),
                ], className="pattern-grid", style={'marginTop': '25px'})
            ], className="chart-container"),
            
            # ========== 10 DESCOBERTAS PRINCIPAIS ==========
            html.Div([
                html.H2("🔍 10 Descobertas Principais", style={'marginBottom': '25px', 'borderBottom': '2px solid #1877F2', 'paddingBottom': '10px'}),
                html.Div([
                    # Descoberta 1
                    html.Div([
                        html.Div("1", style={'background': '#42b72a', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center', 
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Placement com Melhor ROAS: {patterns['top_placement']}", style={'marginBottom': '5px'}),
                            html.P(f"O placement \"{patterns['top_placement']}\" apresenta ROAS significativamente acima da média da conta.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(66,183,42,0.1)', 
                             'borderRadius': '10px', 'borderLeft': '4px solid #42b72a', 'marginBottom': '12px'}),
                    
                    # Descoberta 2
                    html.Div([
                        html.Div("2", style={'background': '#42b72a', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Objetivo de Campanha Mais Eficiente: {top_patterns['ideal_profile']['objective']}", style={'marginBottom': '5px'}),
                            html.P("Este objetivo demonstra melhor combinação de ROAS e volume de conversões.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(66,183,42,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #42b72a', 'marginBottom': '12px'}),
                    
                    # Descoberta 3
                    html.Div([
                        html.Div("3", style={'background': '#1877F2', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Targeting Vencedor: {patterns['top_targeting']}", style={'marginBottom': '5px'}),
                            html.P(f"O targeting \"{patterns['top_targeting']}\" supera outras estratégias consistentemente.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(24,119,242,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #1877F2', 'marginBottom': '12px'}),
                    
                    # Descoberta 4
                    html.Div([
                        html.Div("4", style={'background': '#ffc107', 'color': '#000', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("Impacto do Advantage+ Placements", style={'marginBottom': '5px'}),
                            html.P(f"ROAS com Advantage+: {did['ROAS']['treatment_mean']:.2f}x vs {did['ROAS']['control_mean']:.2f}x sem. "
                                   f"Diferença {'estatisticamente significativa' if did['summary']['statistical_confidence'] > 0.95 else 'não significativa'}.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(255,193,7,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #ffc107', 'marginBottom': '12px'}),
                    
                    # Descoberta 5
                    html.Div([
                        html.Div("5", style={'background': '#1877F2', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Frequência Ótima: {freq['optimal_frequency']}", style={'marginBottom': '5px'}),
                            html.P("Esta faixa apresenta melhor ROAS, indicando ponto ótimo antes de saturação de audiência.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(24,119,242,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #1877F2', 'marginBottom': '12px'}),
                    
                    # Descoberta 6
                    html.Div([
                        html.Div("6", style={'background': '#42b72a', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Top 10% dos Criativos = {len(top_creatives)} High Performers", style={'marginBottom': '5px'}),
                            html.P(f"Estes criativos geram ROAS acima de {df['ROAS'].quantile(0.90):.2f}x - candidatos prioritários para escala.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(66,183,42,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #42b72a', 'marginBottom': '12px'}),
                    
                    # Descoberta 7
                    html.Div([
                        html.Div("7", style={'background': '#fa3e3e', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Criativos para Pausar: {matrix['pause']['count']}", style={'marginBottom': '5px'}),
                            html.P(f"Performance abaixo do aceitável. Economia potencial: ${matrix['pause']['expected_savings']:,.2f}",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(250,62,62,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #fa3e3e', 'marginBottom': '12px'}),
                    
                    # Descoberta 8
                    html.Div([
                        html.Div("8", style={'background': '#1877F2', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4(f"Variável de Maior Impacto: {impact['impact_ranking'][0]['variable']}", style={'marginBottom': '5px'}),
                            html.P(f"Machine Learning identificou {impact['impact_ranking'][0]['impact_score']:.1f}% de importância para predição de ROAS.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(24,119,242,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #1877F2', 'marginBottom': '12px'}),
                    
                    # Descoberta 9
                    html.Div([
                        html.Div("9", style={'background': '#1877F2', 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("Correlação CTR × Conversões", style={'marginBottom': '5px'}),
                            html.P(f"Correlação de {df['CTR'].corr(df['Conversions']):.2f} indica relação {'forte' if abs(df['CTR'].corr(df['Conversions'])) > 0.5 else 'moderada'} entre engajamento e resultado.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(24,119,242,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #1877F2', 'marginBottom': '12px'}),
                    
                    # Descoberta 10
                    html.Div([
                        html.Div("10", style={'background': '#42b72a', 'color': 'white', 'width': '30px', 'height': '30px',
                                             'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                             'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0', 'fontSize': '12px'}),
                        html.Div([
                            html.H4("Modelo Preditivo Validado", style={'marginBottom': '5px'}),
                            html.P(f"Modelo {impact['model_performance']['best_model']} com R² = {max(impact['model_performance']['ridge_r2'], impact['model_performance']['rf_r2'], impact['model_performance']['gb_r2']):.3f} para predição de ROAS.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '15px', 'background': 'rgba(66,183,42,0.1)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #42b72a', 'marginBottom': '12px'}),
                ])
            ], className="chart-container"),
            
            # ========== 5 OPORTUNIDADES DE OTIMIZAÇÃO ==========
            html.Div([
                html.H2("🎯 5 Oportunidades de Otimização", style={'marginBottom': '25px', 'borderBottom': '2px solid #1877F2', 'paddingBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.H4("💰 1. Realocar Budget para Top Performers", style={'color': COLORS['success']}),
                        html.P(f"Transferir investimento dos {matrix['pause']['count']} criativos de baixa performance para os {matrix['scale']['count']} de alta performance."),
                        html.Div(f"Economia estimada: ${matrix['pause']['expected_savings']:,.2f}", 
                                style={'color': COLORS['success'], 'fontWeight': 'bold', 'marginTop': '10px'})
                    ], style={'flex': '1', 'padding': '20px', 'background': 'linear-gradient(135deg, rgba(66,183,42,0.15), rgba(66,183,42,0.05))',
                             'borderRadius': '12px', 'border': '1px solid rgba(66,183,42,0.3)'}),
                    
                    html.Div([
                        html.H4("📍 2. Priorizar Placement Vencedor", style={'color': COLORS['success']}),
                        html.P(f"Concentrar novos criativos no placement \"{patterns['top_placement']}\" que apresenta ROAS superior."),
                        html.Div(f"Potencial: Aumento significativo no ROAS", 
                                style={'color': COLORS['success'], 'fontWeight': 'bold', 'marginTop': '10px'})
                    ], style={'flex': '1', 'padding': '20px', 'background': 'linear-gradient(135deg, rgba(66,183,42,0.15), rgba(66,183,42,0.05))',
                             'borderRadius': '12px', 'border': '1px solid rgba(66,183,42,0.3)'}),
                ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}),
                
                html.Div([
                    html.Div([
                        html.H4("🎯 3. Otimizar Targeting", style={'color': COLORS['primary']}),
                        html.P(f"Expandir uso de targeting \"{patterns['top_targeting']}\" e testar lookalikes dos top performers."),
                        html.Div(f"ROAS alvo: {top_patterns['patterns']['metrics']['avg_roas']:.2f}x", 
                                style={'color': COLORS['primary'], 'fontWeight': 'bold', 'marginTop': '10px'})
                    ], style={'flex': '1', 'padding': '20px', 'background': 'linear-gradient(135deg, rgba(24,119,242,0.15), rgba(24,119,242,0.05))',
                             'borderRadius': '12px', 'border': '1px solid rgba(24,119,242,0.3)'}),
                    
                    html.Div([
                        html.H4("📊 4. Controle de Frequência", style={'color': COLORS['primary']}),
                        html.P(f"Implementar caps de frequência na faixa ótima \"{freq['optimal_frequency']}\" para evitar saturação."),
                        html.Div("Redução de waste estimada: 15-20%", 
                                style={'color': COLORS['primary'], 'fontWeight': 'bold', 'marginTop': '10px'})
                    ], style={'flex': '1', 'padding': '20px', 'background': 'linear-gradient(135deg, rgba(24,119,242,0.15), rgba(24,119,242,0.05))',
                             'borderRadius': '12px', 'border': '1px solid rgba(24,119,242,0.3)'}),
                    
                    html.Div([
                        html.H4("🧪 5. Programa de Testes A/B", style={'color': COLORS['warning']}),
                        html.P(f"Estruturar testes para os {matrix['test']['count']} criativos em zona de incerteza."),
                        html.Div(f"{matrix['test']['count']} criativos para otimizar", 
                                style={'color': COLORS['warning'], 'fontWeight': 'bold', 'marginTop': '10px'})
                    ], style={'flex': '1', 'padding': '20px', 'background': 'linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.05))',
                             'borderRadius': '12px', 'border': '1px solid rgba(255,193,7,0.3)'}),
                ], style={'display': 'flex', 'gap': '15px'})
            ], className="chart-container"),
            
            # ========== 3 RECOMENDAÇÕES DE MÍDIA ==========
            html.Div([
                html.H2("📺 3 Recomendações de Mídia", style={'marginBottom': '25px', 'borderBottom': '2px solid #1877F2', 'paddingBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.H4("🚀 URGENTE: Escalar Vencedores", style={'color': COLORS['danger']}),
                        html.P([html.Strong("Ação: "), f"Aumentar budget em 50% para os {matrix['scale']['count']} criativos classificados para escalar."]),
                        html.P([html.Strong("Prazo: "), "Próximos 7 dias"]),
                        html.P([html.Strong("Critério: "), f"ROAS > {df['ROAS'].quantile(0.75):.2f}x e baixo CPA"]),
                        html.Div(f"Impacto esperado: +{exec_summary['estimated_impact']['conversion_increase']:.0f}% conversões",
                                style={'color': COLORS['success'], 'fontWeight': 'bold', 'marginTop': '15px', 'padding': '10px',
                                      'background': 'rgba(66,183,42,0.2)', 'borderRadius': '8px'})
                    ], style={'flex': '1', 'padding': '25px', 'background': 'linear-gradient(135deg, rgba(250,62,62,0.15), rgba(250,62,62,0.05))',
                             'borderRadius': '12px', 'border': '2px solid rgba(250,62,62,0.5)'}),
                    
                    html.Div([
                        html.H4("⏸️ MODERADA: Pausar Ineficientes", style={'color': COLORS['warning']}),
                        html.P([html.Strong("Ação: "), f"Pausar os {matrix['pause']['count']} criativos de baixa performance."]),
                        html.P([html.Strong("Prazo: "), "Próximas 48 horas"]),
                        html.P([html.Strong("Critério: "), "ROAS abaixo da mediana consistentemente"]),
                        html.Div(f"Economia: ${matrix['pause']['expected_savings']:,.2f}",
                                style={'color': COLORS['accent'], 'fontWeight': 'bold', 'marginTop': '15px', 'padding': '10px',
                                      'background': 'rgba(245,166,35,0.2)', 'borderRadius': '8px'})
                    ], style={'flex': '1', 'padding': '25px', 'background': 'linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.05))',
                             'borderRadius': '12px', 'border': '2px solid rgba(255,193,7,0.5)'}),
                    
                    html.Div([
                        html.H4("🔬 ESTRATÉGICA: Expandir Testes", style={'color': COLORS['primary']}),
                        html.P([html.Strong("Ação: "), "Criar variações dos top 10 criativos para teste."]),
                        html.P([html.Strong("Prazo: "), "Próximos 14 dias"]),
                        html.P([html.Strong("Critério: "), "3 variações por criativo vencedor"]),
                        html.Div("Pipeline de 30+ novos criativos para teste",
                                style={'color': COLORS['primary'], 'fontWeight': 'bold', 'marginTop': '15px', 'padding': '10px',
                                      'background': 'rgba(24,119,242,0.2)', 'borderRadius': '8px'})
                    ], style={'flex': '1', 'padding': '25px', 'background': 'linear-gradient(135deg, rgba(24,119,242,0.15), rgba(24,119,242,0.05))',
                             'borderRadius': '12px', 'border': '2px solid rgba(24,119,242,0.5)'})
                ], style={'display': 'flex', 'gap': '20px'})
            ], className="chart-container"),
            
            # ========== PADRÕES DOS CREATIVES VENCEDORES ==========
            html.Div([
                html.H2("🏆 Padrões dos Creatives Vencedores", style={'marginBottom': '25px', 'borderBottom': '2px solid #FFD700', 'paddingBottom': '10px'}),
                html.Div([
                    html.H3("🥇 DNA do Creative Vencedor", style={'color': '#FFD700', 'marginBottom': '20px'}),
                    html.P(f"Baseado na análise dos {len(top_creatives)} criativos top performers (top 10% por ROAS)", 
                          style={'color': '#a0aec0', 'marginBottom': '20px'}),
                    html.Div([
                        html.Div([html.Div("Placement", style={'color': '#718096', 'fontSize': '0.85em'}),
                                  html.Div(top_patterns['ideal_profile']['placement'], style={'color': '#FFD700', 'fontWeight': '700', 'fontSize': '1.1em'})], 
                                style={'background': 'rgba(0,0,0,0.2)', 'padding': '15px', 'borderRadius': '10px'}),
                        html.Div([html.Div("Objective", style={'color': '#718096', 'fontSize': '0.85em'}),
                                  html.Div(top_patterns['ideal_profile']['objective'], style={'color': '#FFD700', 'fontWeight': '700', 'fontSize': '1.1em'})],
                                style={'background': 'rgba(0,0,0,0.2)', 'padding': '15px', 'borderRadius': '10px'}),
                        html.Div([html.Div("Targeting", style={'color': '#718096', 'fontSize': '0.85em'}),
                                  html.Div(top_patterns['ideal_profile']['targeting'], style={'color': '#FFD700', 'fontWeight': '700', 'fontSize': '1.1em'})],
                                style={'background': 'rgba(0,0,0,0.2)', 'padding': '15px', 'borderRadius': '10px'}),
                        html.Div([html.Div("Advantage+", style={'color': '#718096', 'fontSize': '0.85em'}),
                                  html.Div(top_patterns['ideal_profile']['advantage_placements'], style={'color': '#FFD700', 'fontWeight': '700', 'fontSize': '1.1em'})],
                                style={'background': 'rgba(0,0,0,0.2)', 'padding': '15px', 'borderRadius': '10px'}),
                        html.Div([html.Div("Frequência", style={'color': '#718096', 'fontSize': '0.85em'}),
                                  html.Div(top_patterns['ideal_profile']['optimal_frequency'], style={'color': '#FFD700', 'fontWeight': '700', 'fontSize': '1.1em'})],
                                style={'background': 'rgba(0,0,0,0.2)', 'padding': '15px', 'borderRadius': '10px'}),
                        html.Div([html.Div("ROAS Médio", style={'color': '#718096', 'fontSize': '0.85em'}),
                                  html.Div(f"{top_patterns['patterns']['metrics']['avg_roas']:.2f}x", style={'color': '#FFD700', 'fontWeight': '700', 'fontSize': '1.1em'})],
                                style={'background': 'rgba(0,0,0,0.2)', 'padding': '15px', 'borderRadius': '10px'}),
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '15px'})
                ], style={'background': 'linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,215,0,0.02))',
                         'borderRadius': '15px', 'padding': '25px', 'border': '2px solid rgba(255,215,0,0.3)'})
            ], className="chart-container"),
            
            # ========== MATRIZ DE DECISÃO ==========
            html.Div([
                html.H2("🎲 Matriz de Decisão", style={'marginBottom': '25px', 'borderBottom': '2px solid #1877F2', 'paddingBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Div(f"{matrix['scale']['count']}", style={'fontSize': '3em', 'fontWeight': '800', 'color': COLORS['success']}),
                        html.Div("ESCALAR", style={'fontWeight': '600', 'fontSize': '1.1em'}),
                        html.Div("Alto ROAS + Baixo CPA", style={'color': '#a0aec0', 'marginTop': '10px'}),
                        html.Div("Investir +50% budget", style={'marginTop': '10px'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '25px', 
                             'background': 'linear-gradient(135deg, rgba(66,183,42,0.2), rgba(66,183,42,0.05))',
                             'borderRadius': '15px', 'border': '2px solid #42b72a'}),
                    
                    html.Div([
                        html.Div(f"{matrix['test']['count']}", style={'fontSize': '3em', 'fontWeight': '800', 'color': COLORS['warning']}),
                        html.Div("TESTAR", style={'fontWeight': '600', 'fontSize': '1.1em'}),
                        html.Div("Performance mediana", style={'color': '#a0aec0', 'marginTop': '10px'}),
                        html.Div("Otimizar e validar", style={'marginTop': '10px'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '25px',
                             'background': 'linear-gradient(135deg, rgba(255,193,7,0.2), rgba(255,193,7,0.05))',
                             'borderRadius': '15px', 'border': '2px solid #ffc107'}),
                    
                    html.Div([
                        html.Div(f"{matrix['pause']['count']}", style={'fontSize': '3em', 'fontWeight': '800', 'color': COLORS['danger']}),
                        html.Div("PAUSAR", style={'fontWeight': '600', 'fontSize': '1.1em'}),
                        html.Div("Baixa performance", style={'color': '#a0aec0', 'marginTop': '10px'}),
                        html.Div(f"Economia: ${matrix['pause']['expected_savings']:,.0f}", style={'marginTop': '10px'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '25px',
                             'background': 'linear-gradient(135deg, rgba(250,62,62,0.2), rgba(250,62,62,0.05))',
                             'borderRadius': '15px', 'border': '2px solid #fa3e3e'})
                ], style={'display': 'flex', 'gap': '20px'})
            ], className="chart-container"),
            
            # ========== PREVISÃO 30 DIAS ==========
            html.Div([
                html.H2("🔮 Previsão para os Próximos 30 Dias", style={'marginBottom': '25px', 'borderBottom': '2px solid #9370DB', 'paddingBottom': '10px'}),
                html.Div([
                    html.H3("Projeção de Conversões (30 dias)", style={'textAlign': 'center'}),
                    html.Div(f"{forecast['optimized']['conversions']['value']:,.0f}", 
                            style={'fontSize': '3.5em', 'fontWeight': '800', 'color': '#9370DB', 'textAlign': 'center', 'margin': '20px 0'}),
                    html.P("conversões estimadas com estratégia otimizada", style={'color': '#a0aec0', 'textAlign': 'center'}),
                    
                    html.Div([
                        html.Div([
                            html.Div(f"{forecast['baseline']['conversions']['value']*0.85:,.0f}", style={'color': COLORS['danger'], 'fontSize': '1.5em', 'fontWeight': '700'}),
                            html.Div("Cenário Pessimista", style={'color': '#718096'})
                        ], style={'flex': '1', 'textAlign': 'center'}),
                        html.Div([
                            html.Div(f"{forecast['baseline']['conversions']['value']:,.0f}", style={'color': COLORS['warning'], 'fontSize': '1.5em', 'fontWeight': '700'}),
                            html.Div("Cenário Base", style={'color': '#718096'})
                        ], style={'flex': '1', 'textAlign': 'center'}),
                        html.Div([
                            html.Div(f"{forecast['aggressive']['conversions']['value']:,.0f}", style={'color': COLORS['success'], 'fontSize': '1.5em', 'fontWeight': '700'}),
                            html.Div("Cenário Otimista", style={'color': '#718096'})
                        ], style={'flex': '1', 'textAlign': 'center'}),
                    ], style={'display': 'flex', 'marginTop': '30px'})
                ], style={'background': 'linear-gradient(135deg, rgba(138,43,226,0.15), rgba(138,43,226,0.05))',
                         'borderRadius': '15px', 'padding': '30px', 'border': '1px solid rgba(138,43,226,0.3)', 'textAlign': 'center'})
            ], className="chart-container"),
            
            # ========== AÇÕES IMEDIATAS ==========
            html.Div([
                html.H2("⚡ Ações Imediatas Recomendadas", style={'marginBottom': '25px', 'borderBottom': '2px solid #1877F2', 'paddingBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Div("1", style={'background': COLORS['danger'], 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("🔴 HOJE: Pausar Criativos de Baixa Performance", style={'marginBottom': '5px'}),
                            html.P(f"Pausar imediatamente os {matrix['pause']['count']} criativos classificados como 'PAUSAR'. Economia: ${matrix['pause']['expected_savings']:,.2f}",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '20px', 'background': 'rgba(250,62,62,0.15)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #fa3e3e', 'marginBottom': '12px'}),
                    
                    html.Div([
                        html.Div("2", style={'background': COLORS['warning'], 'color': '#000', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("🟡 ESTA SEMANA: Aumentar Budget dos Vencedores", style={'marginBottom': '5px'}),
                            html.P(f"Incrementar em 30-50% o investimento nos {matrix['scale']['count']} criativos de alta performance.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '20px', 'background': 'rgba(255,193,7,0.15)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #ffc107', 'marginBottom': '12px'}),
                    
                    html.Div([
                        html.Div("3", style={'background': COLORS['success'], 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("🟢 PRÓXIMAS 2 SEMANAS: Criar Variações dos Top 10", style={'marginBottom': '5px'}),
                            html.P("Desenvolver 3 variações de cada um dos 10 melhores criativos, testando elementos isolados.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '20px', 'background': 'rgba(66,183,42,0.15)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #42b72a', 'marginBottom': '12px'}),
                    
                    html.Div([
                        html.Div("4", style={'background': COLORS['primary'], 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("📊 CONTÍNUO: Implementar Frequência Cap", style={'marginBottom': '5px'}),
                            html.P(f"Configurar frequency cap na faixa ótima \"{freq['optimal_frequency']}\" para evitar saturação.",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '20px', 'background': 'rgba(24,119,242,0.15)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #1877F2', 'marginBottom': '12px'}),
                    
                    html.Div([
                        html.Div("5", style={'background': COLORS['primary'], 'color': 'white', 'width': '30px', 'height': '30px',
                                            'borderRadius': '50%', 'display': 'flex', 'alignItems': 'center',
                                            'justifyContent': 'center', 'fontWeight': 'bold', 'flexShrink': '0'}),
                        html.Div([
                            html.H4("🎯 PRÓXIMO MÊS: Expandir Targeting Vencedor", style={'marginBottom': '5px'}),
                            html.P(f"Criar lookalike audiences baseadas nas conversões com targeting \"{patterns['top_targeting']}\".",
                                  style={'color': '#a0aec0', 'margin': '0'})
                        ], style={'marginLeft': '15px'})
                    ], style={'display': 'flex', 'padding': '20px', 'background': 'rgba(24,119,242,0.15)',
                             'borderRadius': '10px', 'borderLeft': '4px solid #1877F2', 'marginBottom': '12px'}),
                ])
            ], className="chart-container"),
            
            # Footer
            html.Div([
                html.P("📊 Relatório gerado automaticamente por Marketing Science Analysis", style={'textAlign': 'center', 'color': '#718096'}),
                html.P("Meta Ads Creative Performance Dashboard", style={'textAlign': 'center', 'color': '#718096'})
            ], style={'marginTop': '40px', 'padding': '20px', 'borderTop': '1px solid rgba(255,255,255,0.1)'})
            
        ], className="tab-content")
    ]),
], className="custom-tabs", id="main-tabs")

# Layout final com sidebar à esquerda e conteúdo principal à direita
app.layout = html.Div([
    header,
    html.Div([
        # Sidebar com filtros (esquerda)
        filter_sidebar,
        # Conteúdo principal (direita) - apenas as tabs
        html.Div([
            tabs
        ], className="main-content")
    ], className="app-layout")
], className="main-container")


# ============================================
# CALLBACKS
# ============================================

@callback(
    [Output('dynamic-kpis', 'children'),
     Output('dynamic-creatives-table', 'children'),
     Output('dynamic-performance', 'children'),
     Output('insights-analyst-container', 'children'),
     Output('insights-list', 'children')],
    [Input('filter-campaign', 'value'), Input('filter-adset', 'value'),
     Input('filter-placement', 'value'), Input('filter-advantage', 'value'),
     Input('filter-objective', 'value'), Input('filter-targeting', 'value')]
)
def update_dashboard(campaigns, adsets, placements, advantage, objectives, targeting):
    filtered_df = df.copy()
    
    if campaigns:
        filtered_df = filtered_df[filtered_df['Campanha'].isin(campaigns)]
    if adsets:
        filtered_df = filtered_df[filtered_df['Adset'].isin(adsets)]
    if placements:
        filtered_df = filtered_df[filtered_df['Placement'].isin(placements)]
    if advantage:
        filtered_df = filtered_df[filtered_df['Advantage+ Placements'].isin(advantage)]
    if objectives:
        filtered_df = filtered_df[filtered_df['Campaign Objective'].isin(objectives)]
    if targeting:
        filtered_df = filtered_df[filtered_df['Targeting'].isin(targeting)]
    
    # Calcular métricas filtradas
    total_spend = filtered_df['Amount spent'].sum()
    total_conversions = filtered_df['Conversions'].sum()
    total_clicks = filtered_df['Clicks'].sum()
    total_impressions = filtered_df['Impressions'].sum()
    avg_ctr = filtered_df['CTR'].mean() if len(filtered_df) > 0 else 0
    avg_cpa = filtered_df['CPA'].mean() if len(filtered_df) > 0 else 0
    overall_roas = filtered_df['ROAS'].mean() if len(filtered_df) > 0 else 0
    num_creatives = len(filtered_df)
    avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
    avg_cpm = (total_spend / total_impressions * 1000) if total_impressions > 0 else 0
    avg_roas = overall_roas
    
    # KPIs dinâmicos
    kpis = html.Div([
        html.H2("Dashboard Principal", className="section-title"),
        html.Div([
            create_kpi_card("💰", f"R$ {total_spend:,.0f}", "INVESTIMENTO TOTAL", f"{num_creatives} creatives", "purple"),
            create_kpi_card("🎯", f"{total_conversions:,.0f}", "CONVERSÕES", f"CPA: R$ {avg_cpa:.2f}", "blue"),
            create_kpi_card("📊", f"{avg_ctr:.2f}%", "CTR MÉDIO", f"CPC: R$ {avg_cpc:.2f}", "yellow"),
            create_kpi_card("📈", f"{overall_roas:.2f}x", "ROAS MÉDIO", f"CPM: R$ {avg_cpm:.2f}", "green"),
        ], className="kpi-container-main")
    ], className="dashboard-section")
    
    # Performance sections dinâmicas
    placement_perf = filtered_df.groupby('Placement').agg({'ROAS': 'mean', 'CPA': 'mean'}).reset_index().sort_values('ROAS', ascending=False)
    targeting_perf = filtered_df.groupby('Targeting').agg({'ROAS': 'mean', 'CPA': 'mean'}).reset_index().sort_values('ROAS', ascending=False)
    objective_perf = filtered_df.groupby('Campaign Objective').agg({'ROAS': 'mean', 'CPA': 'mean'}).reset_index().sort_values('ROAS', ascending=False)
    
    max_placement_roas = placement_perf['ROAS'].max() if len(placement_perf) > 0 else 1
    max_targeting_roas = targeting_perf['ROAS'].max() if len(targeting_perf) > 0 else 1
    max_objective_roas = objective_perf['ROAS'].max() if len(objective_perf) > 0 else 1
    
    # Advantage+ dados
    adv_on = filtered_df[filtered_df['Advantage+ Placements'] == 'Yes']
    adv_off = filtered_df[filtered_df['Advantage+ Placements'] == 'No']
    adv_on_roas = adv_on['ROAS'].mean() if len(adv_on) > 0 else 0
    adv_on_cpa = adv_on['CPA'].mean() if len(adv_on) > 0 else 0
    adv_on_count = len(adv_on)
    adv_off_roas = adv_off['ROAS'].mean() if len(adv_off) > 0 else 0
    adv_off_cpa = adv_off['CPA'].mean() if len(adv_off) > 0 else 0
    adv_off_count = len(adv_off)
    
    performance = html.Div([
        html.Div([
            html.Div([
                html.H3("Performance por Placement", className="perf-section-title"),
                html.Div([create_performance_bar(row['Placement'], row['ROAS'], max_placement_roas, 'blue') for _, row in placement_perf.iterrows()], className="perf-bar-container")
            ], className="perf-section"),
            html.Div([
                html.H3("Performance por Targeting", className="perf-section-title"),
                html.Div([create_performance_bar(row['Targeting'], row['ROAS'], max_targeting_roas, 'green', row['CPA']) for _, row in targeting_perf.iterrows()], className="perf-bar-container")
            ], className="perf-section"),
        ], className="perf-row"),
        html.Div([
            html.Div([
                html.H3("Performance por Objetivo", className="perf-section-title"),
                html.Div([create_performance_bar(row['Campaign Objective'], row['ROAS'], max_objective_roas, 'purple') for _, row in objective_perf.iterrows()], className="perf-bar-container")
            ], className="perf-section"),
            html.Div([
                html.H3("Advantage+ vs Manual", className="perf-section-title"),
                html.Div([
                    create_advantage_card("Advantage+ ON", adv_on_roas, adv_on_cpa, adv_on_count, True),
                    create_advantage_card("Advantage+ OFF", adv_off_roas, adv_off_cpa, adv_off_count, False),
                ], className="advantage-container")
            ], className="perf-section"),
        ], className="perf-row")
    ], className="performance-grid")

    # Insights do analista (narrativa + eficiência)
    best_placement = placement_perf.iloc[0]['Placement'] if len(placement_perf) > 0 else 'N/A'
    best_placement_roas = placement_perf.iloc[0]['ROAS'] if len(placement_perf) > 0 else 0
    best_targeting = targeting_perf.iloc[0]['Targeting'] if len(targeting_perf) > 0 else 'N/A'
    best_targeting_roas = targeting_perf.iloc[0]['ROAS'] if len(targeting_perf) > 0 else 0
    best_objective = objective_perf.iloc[0]['Campaign Objective'] if len(objective_perf) > 0 else 'N/A'
    best_objective_roas = objective_perf.iloc[0]['ROAS'] if len(objective_perf) > 0 else 0

    analyst_insights = build_analyst_insights(
        total_spend, total_conversions, avg_roas,
        best_placement, best_placement_roas,
        best_targeting, best_targeting_roas,
        best_objective, best_objective_roas,
        avg_ctr, avg_cpc, avg_cpa, avg_cpm
    ) if len(filtered_df) > 0 else html.Div("Nenhum dado para os filtros selecionados.", className="insight-empty")

    # Insights automáticos da aba "Insights" responsivos ao filtro
    filtered_metrics = get_summary_metrics(filtered_df) if len(filtered_df) > 0 else get_summary_metrics(df)
    filtered_top_creatives, _ = identify_top_creatives(filtered_df if len(filtered_df) > 0 else df)
    filtered_patterns = analyze_patterns(filtered_df if len(filtered_df) > 0 else df, filtered_top_creatives)
    filtered_insights = generate_insights(filtered_df if len(filtered_df) > 0 else df, filtered_metrics, filtered_patterns)
    insights_cards = [
        html.Div([
            html.Div(ins['icon'], className="insight-icon"),
            html.Div(ins['category'], className="insight-category"),
            html.Div(ins['insight'], className="insight-text")
        ], className="insight-card")
        for ins in filtered_insights
    ]
    
    # ============================================
    # TABELA DE INDICADORES POR CREATIVE
    # ============================================
    # Preparar dados para a tabela na ordem: Creative, Placement, Impressions, Clicks, Conv, Spent, CTR, CPA, ROAS
    table_data = []
    for _, row in filtered_df.iterrows():
        table_data.append({
            'Creative': row['Nome do Creative'],
            'Placement': row['Placement'],
            'Impressions': f"{row['Impressions']:,.0f}",
            'Clicks': f"{row['Clicks']:,.0f}",
            'Conv': f"{row['Conversions']:,.0f}",
            'Spent': f"R$ {row['Amount spent']:,.2f}",
            'CTR': f"{row['CTR']:.2f}%",
            'CPA': f"R$ {row['CPA']:.2f}",
            'ROAS': f"{row['ROAS']:.2f}x"
        })
    
    # Definir colunas da tabela na ordem solicitada
    table_columns = [
        {'name': 'Creative', 'id': 'Creative'},
        {'name': 'Placement', 'id': 'Placement'},
        {'name': 'Impressions', 'id': 'Impressions'},
        {'name': 'Clicks', 'id': 'Clicks'},
        {'name': 'Conv', 'id': 'Conv'},
        {'name': 'Spent', 'id': 'Spent'},
        {'name': 'CTR', 'id': 'CTR'},
        {'name': 'CPA', 'id': 'CPA'},
        {'name': 'ROAS', 'id': 'ROAS'},
    ]
    
    # Criar tabela de creatives
    creatives_table = html.Div([
        html.H3("📋 Indicadores por Creative", className="chart-title"),
        html.P(f"Exibindo {len(filtered_df)} creatives", style={'color': '#6b7280', 'marginBottom': '16px'}),
        dash_table.DataTable(
            id='creatives-datatable',
            columns=table_columns,
            data=table_data,
            page_size=15,
            page_action='native',
            sort_action='native',
            sort_mode='multi',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#f8fafc',
                'color': '#374151',
                'fontWeight': 'bold',
                'borderBottom': '2px solid #e5e7eb',
                'padding': '12px 10px',
                'textAlign': 'center',
                'fontSize': '13px'
            },
            style_cell={
                'backgroundColor': '#ffffff',
                'color': '#1f2937',
                'border': '1px solid #e5e7eb',
                'padding': '12px 10px',
                'fontSize': '13px',
                'textAlign': 'center',
                'minWidth': '90px'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Creative'}, 'textAlign': 'left', 'minWidth': '160px', 'fontWeight': '500'},
                {'if': {'column_id': 'Placement'}, 'textAlign': 'left', 'minWidth': '120px'},
                {'if': {'column_id': 'ROAS'}, 'minWidth': '100px'}
            ],
            style_header_conditional=[
                {'if': {'column_id': 'ROAS'}, 'backgroundColor': '#10b981', 'color': 'white', 'fontWeight': 'bold'}
            ],
            style_data_conditional=[
                {'if': {'state': 'active'}, 'backgroundColor': 'rgba(99, 102, 241, 0.1)', 'border': '1px solid #6366f1'},
                {'if': {'column_id': 'ROAS'}, 'backgroundColor': 'rgba(16, 185, 129, 0.1)', 'color': '#10b981', 'fontWeight': 'bold', 'fontSize': '14px'},
                {'if': {'column_id': 'CPA'}, 'color': '#6366f1', 'fontWeight': '500'},
                {'if': {'column_id': 'CTR'}, 'color': '#f59e0b', 'fontWeight': '500'},
            ],
            style_filter={'backgroundColor': '#f8fafc', 'padding': '8px'},
            export_format='xlsx',
            export_headers='display',
        )
    ], className="chart-container", style={'marginTop': '24px'})
    
    return kpis, creatives_table, performance, analyst_insights, insights_cards


# ============================================
# INICIAR APP
# ============================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Meta Ads Creative Performance Dashboard - VERSÃO COMPLETA")
    print("="*70)
    print("\n📊 Dashboard iniciado!")
    print("🌐 Acesse: http://127.0.0.1:8050")
    print("\n📋 Abas disponíveis:")
    print("   1. Dashboard Principal")
    print("   2. Top Creatives")
    print("   3. Relatório Marketing Science (INTEGRADO)")
    print("   4. Ações Recomendadas")
    print("   5. Padrões Vencedores")
    print("   6. Insights")
    print("   7. 📑 RELATÓRIO COMPLETO (NOVO!)")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, port=8050)
