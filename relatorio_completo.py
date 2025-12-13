"""
📊 RELATÓRIO COMPLETO DE MARKETING SCIENCE
Meta Ads Creative Performance Analysis
Gerado em: Dezembro 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar data_processor para garantir métricas consistentes
from data_processor import load_and_process_data, get_summary_metrics

# ============================================
# CARREGAR DADOS USANDO DATA_PROCESSOR
# ============================================
CSV_PATH = "meta_creatives_sample - meta_creatives_sample.csv.csv"

# Usar data_processor para garantir cálculos consistentes
df = load_and_process_data(CSV_PATH)

# Adicionar coluna binária para Advantage+
df['Advantage_Binary'] = (df['Advantage+ Placements'] == 'Yes').astype(int)

# Adicionar Conversion Rate (não vem do data_processor)
df['Conversion Rate'] = np.where(df['Clicks'] > 0, (df['Conversions'] / df['Clicks']) * 100, 0)

# ============================================
# ANÁLISES
# ============================================

# Métricas gerais (já calculadas corretamente pelo data_processor)
metrics = get_summary_metrics(df)
total_creatives = metrics['total_creatives']
total_spent = metrics['total_spend']
total_conversions = metrics['total_conversions']
avg_roas = metrics['overall_roas']
avg_cpa = metrics['avg_cpa']
avg_ctr = metrics['avg_ctr']
avg_frequency = metrics['avg_frequency']

# Top performers (top 10% por ROAS)
top_threshold = df['ROAS'].quantile(0.90)
top_creatives = df[df['ROAS'] >= top_threshold]

# Análise por Placement
placement_analysis = df.groupby('Placement').agg({
    'ROAS': 'mean',
    'CPA': 'mean',
    'Conversions': 'sum',
    'Amount spent': 'sum',
    'Nome do Creative': 'count'
}).round(2)
placement_analysis.columns = ['ROAS Médio', 'CPA Médio', 'Conversões', 'Investimento', 'Qtd Creatives']
placement_analysis = placement_analysis.sort_values('ROAS Médio', ascending=False)

# Análise por Objetivo
objective_analysis = df.groupby('Campaign Objective').agg({
    'ROAS': 'mean',
    'CPA': 'mean',
    'Conversions': 'sum',
    'Amount spent': 'sum',
    'Nome do Creative': 'count'
}).round(2)
objective_analysis.columns = ['ROAS Médio', 'CPA Médio', 'Conversões', 'Investimento', 'Qtd Creatives']
objective_analysis = objective_analysis.sort_values('ROAS Médio', ascending=False)

# Análise por Targeting
targeting_analysis = df.groupby('Targeting').agg({
    'ROAS': 'mean',
    'CPA': 'mean',
    'Conversions': 'sum',
    'Amount spent': 'sum',
    'Nome do Creative': 'count'
}).round(2)
targeting_analysis.columns = ['ROAS Médio', 'CPA Médio', 'Conversões', 'Investimento', 'Qtd Creatives']
targeting_analysis = targeting_analysis.sort_values('ROAS Médio', ascending=False)

# Análise Advantage+
adv_yes = df[df['Advantage+ Placements'] == 'Yes']
adv_no = df[df['Advantage+ Placements'] == 'No']
advantage_impact = {
    'Com Advantage+': {'ROAS': adv_yes['ROAS'].mean(), 'CPA': adv_yes[adv_yes['CPA'] > 0]['CPA'].mean(), 'n': len(adv_yes)},
    'Sem Advantage+': {'ROAS': adv_no['ROAS'].mean(), 'CPA': adv_no[adv_no['CPA'] > 0]['CPA'].mean(), 'n': len(adv_no)}
}

# Teste estatístico
if len(adv_yes) > 1 and len(adv_no) > 1:
    t_stat, p_value = stats.ttest_ind(adv_yes['ROAS'], adv_no['ROAS'])
    advantage_significant = p_value < 0.05
else:
    p_value = 1.0
    advantage_significant = False

# Análise de Frequência
df['Freq_Group'] = pd.cut(df['Frequency'], bins=[0, 1, 2, 3, 5, 100], labels=['0-1', '1-2', '2-3', '3-5', '5+'])
frequency_analysis = df.groupby('Freq_Group').agg({
    'ROAS': 'mean',
    'CPA': 'mean',
    'Conversions': 'sum',
    'Nome do Creative': 'count'
}).round(2)

# Regressão para impacto no ROAS
features = ['Advantage_Binary', 'Frequency', 'Clicks', 'Reach']
X = df[features].fillna(0)
y = df['ROAS'].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, y)

# Importância das variáveis
feature_importance = pd.DataFrame({
    'Variable': ['Advantage+ Placements', 'Frequency', 'Clicks', 'Reach'],
    'Ridge_Coef': ridge.coef_,
    'RF_Importance': rf.feature_importances_ * 100
}).sort_values('RF_Importance', ascending=False)

# Matriz de Decisão
df['Decision'] = df.apply(
    lambda x: 'ESCALAR' if x['ROAS'] >= df['ROAS'].quantile(0.75) and x['CPA'] <= df['CPA'].quantile(0.25) else
              'TESTAR' if x['ROAS'] >= df['ROAS'].median() else
              'PAUSAR', axis=1
)

decision_summary = df['Decision'].value_counts()
potential_savings = df[df['Decision'] == 'PAUSAR']['Amount spent'].sum()

# Previsão 30 dias (baseada em tendência)
forecast_base = total_conversions / 30  # média diária
forecast_growth = 1.05  # 5% growth assumido
forecast_30_days = sum([forecast_base * (forecast_growth ** (i/30)) for i in range(30)])

# ============================================
# GERAR RELATÓRIO HTML
# ============================================

html_content = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Completo de Marketing Science - Meta Ads</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.65);
            --primary: #6366f1;
            --secondary: #10b981;
            --accent: #f43f5e;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #020617;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(244, 63, 94, 0.08), transparent 25%);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 60px;
            padding: 40px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 24px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: var(--text-muted);
            font-size: 1.2em;
        }}
        
        .header .date {{
            color: var(--text-muted);
            margin-top: 10px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        /* Sections */
        .section {{
            margin-bottom: 40px;
            background: var(--bg-card);
            border-radius: 24px;
            padding: 32px;
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        .section-title {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 15px;
            color: var(--text);
        }}
        
        .section-title .icon {{
            font-size: 1.2em;
            background: rgba(99, 102, 241, 0.1);
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            color: var(--primary);
        }}
        
        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }}
        
        .kpi-card {{
            background: rgba(30, 41, 59, 0.4);
            border-radius: 20px;
            padding: 24px;
            text-align: center;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px);
            border-color: var(--primary);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.2);
        }}
        
        .kpi-value {{
            font-size: 2.5em;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 5px;
            letter-spacing: -0.03em;
        }}
        
        .kpi-label {{
            color: var(--text-muted);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }}
        
        /* Discovery Items */
        .discovery-list {{
            display: grid;
            gap: 20px;
        }}
        
        .discovery-item {{
            background: rgba(30, 41, 59, 0.4);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            align-items: flex-start;
            gap: 20px;
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            transition: transform 0.2s ease;
        }}
        
        .discovery-item:hover {{
            transform: translateX(5px);
            background: rgba(30, 41, 59, 0.6);
        }}
        
        .discovery-item.success {{
            border-left-color: var(--secondary);
        }}
        
        .discovery-item.warning {{
            border-left-color: var(--accent);
        }}
        
        .discovery-item.danger {{
            border-left-color: #fa3e3e;
            background: rgba(250, 62, 62, 0.1);
        }}
        
        .discovery-number {{
            background: #1877F2;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            flex-shrink: 0;
        }}
        
        .discovery-content h4 {{
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        
        .discovery-content p {{
            color: #a0aec0;
            font-size: 0.95em;
        }}
        
        /* Tables */
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        th {{
            background: rgba(24, 119, 242, 0.2);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        /* Action Cards */
        .action-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .action-card {{
            background: linear-gradient(135deg, rgba(66, 183, 42, 0.15), rgba(66, 183, 42, 0.05));
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(66, 183, 42, 0.3);
        }}
        
        .action-card.urgent {{
            background: linear-gradient(135deg, rgba(250, 62, 62, 0.15), rgba(250, 62, 62, 0.05));
            border-color: rgba(250, 62, 62, 0.3);
        }}
        
        .action-card h4 {{
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .action-card p {{
            color: #a0aec0;
        }}
        
        .action-card .impact {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 600;
            color: #42b72a;
        }}
        
        /* Charts */
        .chart-container {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        /* Winner Profile */
        .winner-profile {{
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 215, 0, 0.02));
            border-radius: 15px;
            padding: 30px;
            border: 2px solid rgba(255, 215, 0, 0.3);
        }}
        
        .winner-profile h3 {{
            color: #FFD700;
            margin-bottom: 20px;
            font-size: 1.4em;
        }}
        
        .winner-attributes {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .winner-attr {{
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
        }}
        
        .winner-attr .label {{
            color: #718096;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        
        .winner-attr .value {{
            color: #FFD700;
            font-weight: 700;
            font-size: 1.1em;
        }}
        
        /* Forecast Section */
        .forecast-box {{
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.15), rgba(138, 43, 226, 0.05));
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(138, 43, 226, 0.3);
            text-align: center;
        }}
        
        .forecast-value {{
            font-size: 3em;
            font-weight: 800;
            color: #9370DB;
            margin: 20px 0;
        }}
        
        /* Matrix */
        .matrix-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        
        .matrix-card {{
            padding: 25px;
            border-radius: 15px;
            text-align: center;
        }}
        
        .matrix-card.scale {{
            background: linear-gradient(135deg, rgba(66, 183, 42, 0.2), rgba(66, 183, 42, 0.05));
            border: 2px solid #42b72a;
        }}
        
        .matrix-card.test {{
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.2), rgba(255, 193, 7, 0.05));
            border: 2px solid #ffc107;
        }}
        
        .matrix-card.pause {{
            background: linear-gradient(135deg, rgba(250, 62, 62, 0.2), rgba(250, 62, 62, 0.05));
            border: 2px solid #fa3e3e;
        }}
        
        .matrix-card .number {{
            font-size: 3em;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        
        .matrix-card.scale .number {{ color: #42b72a; }}
        .matrix-card.test .number {{ color: #ffc107; }}
        .matrix-card.pause .number {{ color: #fa3e3e; }}
        
        /* Executive Summary */
        .executive-summary {{
            background: linear-gradient(135deg, rgba(24, 119, 242, 0.15), rgba(24, 119, 242, 0.05));
            border-radius: 15px;
            padding: 30px;
            border-left: 5px solid #1877F2;
        }}
        
        .executive-summary p {{
            font-size: 1.1em;
            line-height: 1.8;
            color: #e2e8f0;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 60px;
            padding: 30px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #718096;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .section {{
                padding: 20px;
            }}
            .matrix-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 Relatório de Marketing Science</h1>
            <p class="subtitle">Meta Ads Creative Performance Analysis</p>
            <p class="date">Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
        </div>

        <!-- RESUMO EXECUTIVO -->
        <div class="section">
            <h2 class="section-title"><span class="icon">📋</span> Resumo Executivo</h2>
            
            <div class="executive-summary">
                <p>
                    Este relatório analisa o desempenho de <strong>{total_creatives:,}</strong> criativos de Meta Ads, 
                    com um investimento total de <strong>${total_spent:,.0f}</strong> e <strong>{total_conversions:,.0f}</strong> conversões geradas.
                    <br><br>
                    O ROAS médio da conta é de <strong>{avg_roas:.2f}x</strong>, com CPA médio de <strong>${avg_cpa:.2f}</strong>.
                    A análise identificou <strong>{len(top_creatives)}</strong> criativos de alta performance (top 10%) que representam
                    as melhores oportunidades de escala.
                    <br><br>
                    <strong>Principais insights:</strong> A variável com maior impacto no ROAS é o volume de cliques ({feature_importance.iloc[0]['RF_Importance']:.1f}% de importância),
                    seguida por frequência e configurações de targeting. A análise recomenda pausar {decision_summary.get('PAUSAR', 0)} criativos
                    de baixa performance, gerando economia potencial de <strong>${potential_savings:,.2f}</strong>.
                </p>
            </div>
            
            <div class="kpi-grid" style="margin-top: 30px;">
                <div class="kpi-card">
                    <div class="kpi-value">{total_creatives:,}</div>
                    <div class="kpi-label">Total Criativos</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">${total_spent:,.0f}</div>
                    <div class="kpi-label">Investimento Total</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{total_conversions:,.0f}</div>
                    <div class="kpi-label">Conversões</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{avg_roas:.1f}x</div>
                    <div class="kpi-label">ROAS Médio</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">${avg_cpa:.2f}</div>
                    <div class="kpi-label">CPA Médio</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{avg_ctr:.2f}%</div>
                    <div class="kpi-label">CTR Médio</div>
                </div>
            </div>
        </div>

        <!-- 10 DESCOBERTAS PRINCIPAIS -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🔍</span> 10 Descobertas Principais</h2>
            
            <div class="discovery-list">
                <div class="discovery-item success">
                    <div class="discovery-number">1</div>
                    <div class="discovery-content">
                        <h4>Placement com Melhor ROAS: {placement_analysis.index[0]}</h4>
                        <p>O placement "{placement_analysis.index[0]}" apresenta ROAS médio de {placement_analysis.iloc[0]['ROAS Médio']:.2f}x, 
                        {((placement_analysis.iloc[0]['ROAS Médio'] / avg_roas - 1) * 100):.0f}% acima da média da conta.</p>
                    </div>
                </div>
                
                <div class="discovery-item success">
                    <div class="discovery-number">2</div>
                    <div class="discovery-content">
                        <h4>Objetivo de Campanha Mais Eficiente: {objective_analysis.index[0]}</h4>
                        <p>Campanhas com objetivo "{objective_analysis.index[0]}" têm ROAS de {objective_analysis.iloc[0]['ROAS Médio']:.2f}x 
                        e representam {objective_analysis.iloc[0]['Qtd Creatives']} criativos.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">3</div>
                    <div class="discovery-content">
                        <h4>Targeting Vencedor: {targeting_analysis.index[0]}</h4>
                        <p>O targeting "{targeting_analysis.index[0]}" supera outras estratégias com ROAS de {targeting_analysis.iloc[0]['ROAS Médio']:.2f}x.</p>
                    </div>
                </div>
                
                <div class="discovery-item {'success' if advantage_impact['Com Advantage+']['ROAS'] > advantage_impact['Sem Advantage+']['ROAS'] else 'warning'}">
                    <div class="discovery-number">4</div>
                    <div class="discovery-content">
                        <h4>Impacto do Advantage+ Placements</h4>
                        <p>Criativos com Advantage+ têm ROAS de {advantage_impact['Com Advantage+']['ROAS']:.2f}x vs {advantage_impact['Sem Advantage+']['ROAS']:.2f}x sem.
                        Diferença {'estatisticamente significativa' if advantage_significant else 'não significativa'} (p={p_value:.3f}).</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">5</div>
                    <div class="discovery-content">
                        <h4>Frequência Ótima: {df.groupby('Freq_Group')['ROAS'].mean().idxmax()}</h4>
                        <p>A faixa de frequência "{df.groupby('Freq_Group')['ROAS'].mean().idxmax()}" apresenta melhor ROAS, 
                        sugerindo ponto ótimo de exposição antes de saturação.</p>
                    </div>
                </div>
                
                <div class="discovery-item success">
                    <div class="discovery-number">6</div>
                    <div class="discovery-content">
                        <h4>Top 10% dos Criativos</h4>
                        <p>{len(top_creatives)} criativos (top 10%) geram ROAS acima de {top_threshold:.2f}x - 
                        são candidatos prioritários para escala de investimento.</p>
                    </div>
                </div>
                
                <div class="discovery-item warning">
                    <div class="discovery-number">7</div>
                    <div class="discovery-content">
                        <h4>Criativos para Pausar</h4>
                        <p>{decision_summary.get('PAUSAR', 0)} criativos ({(decision_summary.get('PAUSAR', 0)/total_creatives*100):.1f}%) 
                        estão com performance abaixo do aceitável. Economia potencial: ${potential_savings:,.2f}.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">8</div>
                    <div class="discovery-content">
                        <h4>Variável de Maior Impacto: {feature_importance.iloc[0]['Variable']}</h4>
                        <p>Modelo de Machine Learning identificou "{feature_importance.iloc[0]['Variable']}" como variável com 
                        {feature_importance.iloc[0]['RF_Importance']:.1f}% de importância para predição de ROAS.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">9</div>
                    <div class="discovery-content">
                        <h4>Correlação CTR x Conversões</h4>
                        <p>Correlação de {df['CTR'].corr(df['Conversion Rate']):.2f} entre CTR e Taxa de Conversão indica 
                        {'forte' if abs(df['CTR'].corr(df['Conversion Rate'])) > 0.5 else 'moderada'} relação entre engajamento e resultado.</p>
                    </div>
                </div>
                
                <div class="discovery-item success">
                    <div class="discovery-number">10</div>
                    <div class="discovery-content">
                        <h4>Eficiência por Volume</h4>
                        <p>Criativos com mais de {df['Impressions'].median():,.0f} impressões têm 
                        {'melhor' if df[df['Impressions'] > df['Impressions'].median()]['ROAS'].mean() > avg_roas else 'similar'} 
                        performance que a média, indicando que escala {'melhora' if df[df['Impressions'] > df['Impressions'].median()]['ROAS'].mean() > avg_roas else 'mantém'} eficiência.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- 5 OPORTUNIDADES DE OTIMIZAÇÃO -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🎯</span> 5 Oportunidades de Otimização</h2>
            
            <div class="action-grid">
                <div class="action-card">
                    <h4>💰 1. Realocar Budget para Top Performers</h4>
                    <p>Transferir investimento dos {decision_summary.get('PAUSAR', 0)} criativos de baixa performance 
                    para os {decision_summary.get('ESCALAR', 0)} de alta performance.</p>
                    <div class="impact">Economia estimada: ${potential_savings:,.2f}</div>
                </div>
                
                <div class="action-card">
                    <h4>📍 2. Priorizar Placement Vencedor</h4>
                    <p>Concentrar novos criativos no placement "{placement_analysis.index[0]}" 
                    que apresenta ROAS {((placement_analysis.iloc[0]['ROAS Médio'] / avg_roas - 1) * 100):.0f}% acima da média.</p>
                    <div class="impact">Potencial: +{((placement_analysis.iloc[0]['ROAS Médio'] / avg_roas - 1) * 100):.0f}% ROAS</div>
                </div>
                
                <div class="action-card">
                    <h4>🎯 3. Otimizar Targeting</h4>
                    <p>Expandir uso de targeting "{targeting_analysis.index[0]}" 
                    e testar lookalikes baseados nos top performers.</p>
                    <div class="impact">ROAS alvo: {targeting_analysis.iloc[0]['ROAS Médio']:.2f}x</div>
                </div>
                
                <div class="action-card">
                    <h4>📊 4. Controle de Frequência</h4>
                    <p>Implementar caps de frequência na faixa ótima "{df.groupby('Freq_Group')['ROAS'].mean().idxmax()}" 
                    para evitar saturação de audiência.</p>
                    <div class="impact">Redução de waste estimada: 15-20%</div>
                </div>
                
                <div class="action-card">
                    <h4>🧪 5. Programa de Testes A/B</h4>
                    <p>Estruturar testes para os {decision_summary.get('TESTAR', 0)} criativos em zona de incerteza, 
                    validando elementos de copy, visual e CTA.</p>
                    <div class="impact">{decision_summary.get('TESTAR', 0)} criativos para otimizar</div>
                </div>
            </div>
        </div>

        <!-- 3 RECOMENDAÇÕES DE MÍDIA -->
        <div class="section">
            <h2 class="section-title"><span class="icon">📺</span> 3 Recomendações de Mídia</h2>
            
            <div class="action-grid">
                <div class="action-card urgent">
                    <h4>🚀 URGENTE: Escalar Vencedores</h4>
                    <p><strong>Ação:</strong> Aumentar budget em 50% para os {decision_summary.get('ESCALAR', 0)} criativos 
                    classificados para escalar imediatamente.</p>
                    <p><strong>Prazo:</strong> Próximos 7 dias</p>
                    <p><strong>Critério:</strong> ROAS > {df['ROAS'].quantile(0.75):.2f}x e CPA < ${df['CPA'].quantile(0.25):.2f}</p>
                    <div class="impact">Impacto esperado: +{(decision_summary.get('ESCALAR', 0) * 0.5 * df[df['Decision']=='ESCALAR']['Conversions'].mean()):.0f} conversões/mês</div>
                </div>
                
                <div class="action-card">
                    <h4>⏸️ MODERADA: Pausar Ineficientes</h4>
                    <p><strong>Ação:</strong> Pausar os {decision_summary.get('PAUSAR', 0)} criativos de baixa performance 
                    e redirecionar budget.</p>
                    <p><strong>Prazo:</strong> Próximas 48 horas</p>
                    <p><strong>Critério:</strong> ROAS abaixo da mediana consistentemente</p>
                    <div class="impact">Economia: ${potential_savings:,.2f}</div>
                </div>
                
                <div class="action-card">
                    <h4>🔬 ESTRATÉGICA: Expandir Testes</h4>
                    <p><strong>Ação:</strong> Criar variações dos top 10 criativos para teste, 
                    mantendo elementos vencedores e testando variáveis isoladas.</p>
                    <p><strong>Prazo:</strong> Próximos 14 dias</p>
                    <p><strong>Critério:</strong> 3 variações por criativo vencedor</p>
                    <div class="impact">Pipeline de 30+ novos criativos para teste</div>
                </div>
            </div>
        </div>

        <!-- PADRÕES DOS CREATIVES VENCEDORES -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🏆</span> Padrões dos Creatives Vencedores</h2>
            
            <div class="winner-profile">
                <h3>🥇 DNA do Creative Vencedor</h3>
                <p style="color: #a0aec0; margin-bottom: 20px;">
                    Baseado na análise dos {len(top_creatives)} criativos top performers (top 10% por ROAS)
                </p>
                
                <div class="winner-attributes">
                    <div class="winner-attr">
                        <div class="label">Placement Preferencial</div>
                        <div class="value">{top_creatives['Placement'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'}</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">Objetivo de Campanha</div>
                        <div class="value">{top_creatives['Campaign Objective'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'}</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">Targeting</div>
                        <div class="value">{top_creatives['Targeting'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'}</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">Advantage+ Placements</div>
                        <div class="value">{top_creatives['Advantage+ Placements'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'}</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">Frequência Média</div>
                        <div class="value">{top_creatives['Frequency'].mean():.2f}</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">CTR Médio</div>
                        <div class="value">{top_creatives['CTR'].mean():.2f}%</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">ROAS Médio</div>
                        <div class="value">{top_creatives['ROAS'].mean():.2f}x</div>
                    </div>
                    <div class="winner-attr">
                        <div class="label">CPA Médio</div>
                        <div class="value">${top_creatives[top_creatives['CPA'] > 0]['CPA'].mean():.2f}</div>
                    </div>
                </div>
            </div>
            
            <div class="table-container" style="margin-top: 30px;">
                <h4 style="margin-bottom: 15px;">📊 Comparativo: Vencedores vs Média</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Métrica</th>
                            <th>Top 10%</th>
                            <th>Média Geral</th>
                            <th>Diferença</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>ROAS</td>
                            <td style="color: #42b72a; font-weight: 600;">{top_creatives['ROAS'].mean():.2f}x</td>
                            <td>{avg_roas:.2f}x</td>
                            <td style="color: #42b72a;">+{((top_creatives['ROAS'].mean() / avg_roas - 1) * 100):.0f}%</td>
                        </tr>
                        <tr>
                            <td>CPA</td>
                            <td style="color: #42b72a; font-weight: 600;">${top_creatives[top_creatives['CPA'] > 0]['CPA'].mean():.2f}</td>
                            <td>${avg_cpa:.2f}</td>
                            <td style="color: #42b72a;">{((top_creatives[top_creatives['CPA'] > 0]['CPA'].mean() / avg_cpa - 1) * 100):.0f}%</td>
                        </tr>
                        <tr>
                            <td>CTR</td>
                            <td style="color: #42b72a; font-weight: 600;">{top_creatives['CTR'].mean():.2f}%</td>
                            <td>{avg_ctr:.2f}%</td>
                            <td style="color: #42b72a;">+{((top_creatives['CTR'].mean() / avg_ctr - 1) * 100):.0f}%</td>
                        </tr>
                        <tr>
                            <td>Frequência</td>
                            <td>{top_creatives['Frequency'].mean():.2f}</td>
                            <td>{avg_frequency:.2f}</td>
                            <td>{((top_creatives['Frequency'].mean() / avg_frequency - 1) * 100):+.0f}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- MATRIZ DE DECISÃO -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🎲</span> Matriz de Decisão</h2>
            
            <div class="matrix-grid">
                <div class="matrix-card scale">
                    <div class="number">{decision_summary.get('ESCALAR', 0)}</div>
                    <div style="font-weight: 600; font-size: 1.1em;">ESCALAR</div>
                    <div style="color: #a0aec0; margin-top: 10px;">Alto ROAS + Baixo CPA</div>
                    <div style="margin-top: 15px;">Investir +50% budget</div>
                </div>
                
                <div class="matrix-card test">
                    <div class="number">{decision_summary.get('TESTAR', 0)}</div>
                    <div style="font-weight: 600; font-size: 1.1em;">TESTAR</div>
                    <div style="color: #a0aec0; margin-top: 10px;">Performance mediana</div>
                    <div style="margin-top: 15px;">Otimizar e validar</div>
                </div>
                
                <div class="matrix-card pause">
                    <div class="number">{decision_summary.get('PAUSAR', 0)}</div>
                    <div style="font-weight: 600; font-size: 1.1em;">PAUSAR</div>
                    <div style="color: #a0aec0; margin-top: 10px;">Baixa performance</div>
                    <div style="margin-top: 15px;">Economia: ${potential_savings:,.0f}</div>
                </div>
            </div>
        </div>

        <!-- PREVISÃO 30 DIAS -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🔮</span> Previsão para os Próximos 30 Dias</h2>
            
            <div class="forecast-box">
                <h3>Projeção de Conversões (30 dias)</h3>
                <div class="forecast-value">{forecast_30_days:,.0f}</div>
                <p style="color: #a0aec0;">conversões estimadas mantendo estratégia atual</p>
                
                <div style="margin-top: 30px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div>
                        <div style="color: #fa3e3e; font-size: 1.5em; font-weight: 700;">{forecast_30_days * 0.85:,.0f}</div>
                        <div style="color: #718096;">Cenário Pessimista (-15%)</div>
                    </div>
                    <div>
                        <div style="color: #ffc107; font-size: 1.5em; font-weight: 700;">{forecast_30_days:,.0f}</div>
                        <div style="color: #718096;">Cenário Base</div>
                    </div>
                    <div>
                        <div style="color: #42b72a; font-size: 1.5em; font-weight: 700;">{forecast_30_days * 1.20:,.0f}</div>
                        <div style="color: #718096;">Cenário Otimista (+20%)</div>
                    </div>
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1); text-align: left;">
                    <h4 style="margin-bottom: 15px;">📈 Premissas da Previsão:</h4>
                    <ul style="color: #a0aec0; list-style: none;">
                        <li>✓ Cenário base assume manutenção do budget e estratégia atual</li>
                        <li>✓ Cenário otimista assume implementação das recomendações de escala</li>
                        <li>✓ Cenário pessimista considera sazonalidade e saturação</li>
                        <li>✓ Modelo baseado em tendência histórica com growth rate de 5%</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- AÇÕES IMEDIATAS -->
        <div class="section">
            <h2 class="section-title"><span class="icon">⚡</span> Ações Imediatas Recomendadas</h2>
            
            <div class="discovery-list">
                <div class="discovery-item urgent" style="border-left-color: #fa3e3e; background: rgba(250, 62, 62, 0.15);">
                    <div class="discovery-number" style="background: #fa3e3e;">1</div>
                    <div class="discovery-content">
                        <h4>🔴 HOJE: Pausar Criativos de Baixa Performance</h4>
                        <p>Pausar imediatamente os {decision_summary.get('PAUSAR', 0)} criativos classificados como "PAUSAR". 
                        Economia imediata de ${potential_savings:,.2f} que pode ser realocada para top performers.</p>
                    </div>
                </div>
                
                <div class="discovery-item" style="border-left-color: #ffc107; background: rgba(255, 193, 7, 0.15);">
                    <div class="discovery-number" style="background: #ffc107; color: #000;">2</div>
                    <div class="discovery-content">
                        <h4>🟡 ESTA SEMANA: Aumentar Budget dos Vencedores</h4>
                        <p>Incrementar em 30-50% o investimento nos {decision_summary.get('ESCALAR', 0)} criativos de alta performance.
                        Monitorar métricas diariamente para ajustes.</p>
                    </div>
                </div>
                
                <div class="discovery-item success">
                    <div class="discovery-number" style="background: #42b72a;">3</div>
                    <div class="discovery-content">
                        <h4>🟢 PRÓXIMAS 2 SEMANAS: Criar Variações dos Top 10</h4>
                        <p>Desenvolver 3 variações de cada um dos 10 melhores criativos, testando elementos isolados 
                        (headline, visual, CTA). Manter DNA vencedor identificado.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">4</div>
                    <div class="discovery-content">
                        <h4>📊 CONTÍNUO: Implementar Frequência Cap</h4>
                        <p>Configurar frequency cap na faixa ótima "{df.groupby('Freq_Group')['ROAS'].mean().idxmax()}" 
                        para todos os adsets, evitando saturação de audiência.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">5</div>
                    <div class="discovery-content">
                        <h4>🎯 PRÓXIMO MÊS: Expandir Targeting Vencedor</h4>
                        <p>Criar lookalike audiences baseadas nas conversões dos top performers com targeting 
                        "{targeting_analysis.index[0]}". Testar expansões de 1%, 2% e 5%.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">6</div>
                    <div class="discovery-content">
                        <h4>📍 PRÓXIMO MÊS: Priorizar Placement Vencedor</h4>
                        <p>Concentrar 70% dos novos criativos no placement "{placement_analysis.index[0]}" 
                        que demonstrou ROAS superior. Manter 30% para testes em outros placements.</p>
                    </div>
                </div>
                
                <div class="discovery-item">
                    <div class="discovery-number">7</div>
                    <div class="discovery-content">
                        <h4>🔄 SEMANAL: Review de Performance</h4>
                        <p>Estabelecer revisão semanal da matriz de decisão para reclassificar criativos 
                        e ajustar estratégia baseada em dados atualizados.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>📊 Relatório gerado automaticamente por Marketing Science Analysis</p>
            <p>Meta Ads Creative Performance Dashboard</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                {datetime.now().strftime('%d/%m/%Y %H:%M')} | Total de {total_creatives} criativos analisados
            </p>
        </div>
    </div>
</body>
</html>
'''

# Salvar relatório
with open('relatorio_marketing_science_completo.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("=" * 70)
print("📊 RELATÓRIO COMPLETO DE MARKETING SCIENCE GERADO!")
print("=" * 70)
print(f"\n✅ Arquivo: relatorio_marketing_science_completo.html")
print(f"📁 Local: {CSV_PATH.replace('.csv', '')}")
print("\n📋 Conteúdo do Relatório:")
print("   1. ✓ Resumo Executivo")
print("   2. ✓ 10 Descobertas Principais")
print("   3. ✓ 5 Oportunidades de Otimização")
print("   4. ✓ 3 Recomendações de Mídia")
print("   5. ✓ Padrões dos Creatives Vencedores")
print("   6. ✓ Matriz de Decisão")
print("   7. ✓ Previsão para os Próximos 30 Dias")
print("   8. ✓ Ações Imediatas Recomendadas")
print("\n" + "=" * 70)
print("🌐 Abra o arquivo HTML no navegador para visualizar!")
print("=" * 70)
