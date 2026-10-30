"""Carga do CSV, métricas derivadas (CTR, CPC, CPM, CPA, receita simulada e ROAS) e resumos do painel."""

import pandas as pd
import numpy as np


def load_and_process_data(filepath):
    """
    Carrega o CSV e calcula todas as métricas derivadas
    """
    # Carregar dados
    df = pd.read_csv(filepath)
    
    # Limpar coluna Amount spent (remover $ e converter para float)
    df['Amount spent'] = df['Amount spent'].replace(r'[\$,]', '', regex=True).astype(float)
    
    # Garantir tipos numéricos
    numeric_cols = ['Impressions', 'Reach', 'Frequency', 'Clicks', 'Results', 'Conversions', 'Amount spent']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calcular métricas derivadas
    # CTR (Click-Through Rate) = (Clicks / Impressions) * 100
    df['CTR'] = np.where(df['Impressions'] > 0, (df['Clicks'] / df['Impressions']) * 100, 0)
    
    # CPC (Cost Per Click) = Amount spent / Clicks
    df['CPC'] = np.where(df['Clicks'] > 0, df['Amount spent'] / df['Clicks'], 0)
    
    # CPM (Cost Per Mille) = (Amount spent / Impressions) * 1000
    df['CPM'] = np.where(df['Impressions'] > 0, (df['Amount spent'] / df['Impressions']) * 1000, 0)
    
    # CPA (Cost Per Acquisition/Conversion) = Amount spent / Conversions
    df['CPA'] = np.where(df['Conversions'] > 0, df['Amount spent'] / df['Conversions'], 0)
    
    # ROAS (Return on Ad Spend) - Simulando receita = Conversions * valor médio estimado
    # Assumindo valor médio de conversão de $50 para simulação
    avg_conversion_value = 50
    df['Revenue'] = df['Conversions'] * avg_conversion_value
    df['ROAS'] = np.where(df['Amount spent'] > 0, df['Revenue'] / df['Amount spent'], 0)
    
    return df


def get_summary_metrics(df):
    """
    Retorna métricas agregadas para o dashboard
    """
    total_spend = df['Amount spent'].sum()
    total_impressions = df['Impressions'].sum()
    total_clicks = df['Clicks'].sum()
    total_conversions = df['Conversions'].sum()
    total_reach = df['Reach'].sum()
    total_revenue = df['Revenue'].sum()
    
    return {
        'total_spend': total_spend,
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'total_conversions': total_conversions,
        'total_reach': total_reach,
        'total_revenue': total_revenue,
        'avg_ctr': (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
        'avg_cpc': total_spend / total_clicks if total_clicks > 0 else 0,
        'avg_cpm': (total_spend / total_impressions * 1000) if total_impressions > 0 else 0,
        'avg_cpa': total_spend / total_conversions if total_conversions > 0 else 0,
        'overall_roas': total_revenue / total_spend if total_spend > 0 else 0,
        'avg_frequency': df['Frequency'].mean(),
        'active_creatives': len(df[df['Delivery status'] == 'Active']),
        'total_creatives': len(df)
    }


def identify_top_creatives(df, n=10):
    """
    Identifica os top creatives baseado em critérios de performance
    """
    # Calcular médias gerais
    avg_roas = df['ROAS'].mean()
    avg_cpa = df['CPA'].mean()
    avg_ctr = df['CTR'].mean()
    
    # Criar score de performance
    df_scored = df.copy()
    
    # Score baseado em múltiplos critérios
    df_scored['score'] = 0
    
    # ROAS acima da média (+2 pontos)
    df_scored.loc[df_scored['ROAS'] > avg_roas, 'score'] += 2
    
    # CPA abaixo da média (+2 pontos)
    df_scored.loc[df_scored['CPA'] < avg_cpa, 'score'] += 2
    
    # CTR acima da média (+1 ponto)
    df_scored.loc[df_scored['CTR'] > avg_ctr, 'score'] += 1
    
    # Advantage+ Placements = Yes (+1 ponto)
    df_scored.loc[df_scored['Advantage+ Placements'] == 'Yes', 'score'] += 1
    
    # Campaign Objective = Sales (+1 ponto)
    df_scored.loc[df_scored['Campaign Objective'] == 'Sales', 'score'] += 1
    
    # Targeting = Advantage+ (+1 ponto)
    df_scored.loc[df_scored['Targeting'] == 'Advantage+', 'score'] += 1
    
    # Ordenar por score e ROAS
    top_creatives = df_scored.nlargest(n, ['score', 'ROAS'])
    
    return top_creatives, {
        'avg_roas': avg_roas,
        'avg_cpa': avg_cpa,
        'avg_ctr': avg_ctr
    }


def analyze_patterns(df, top_creatives):
    """
    Analisa padrões dos creatives vencedores
    """
    patterns = {}
    
    # Placement mais comum
    patterns['top_placement'] = top_creatives['Placement'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'
    patterns['placement_dist'] = top_creatives['Placement'].value_counts().to_dict()
    
    # Advantage+ Placements
    patterns['advantage_placements'] = (top_creatives['Advantage+ Placements'] == 'Yes').mean() * 100
    
    # Advantage+ Creative
    patterns['advantage_creative'] = (top_creatives['Advantage+ Creative'] == 'Yes').mean() * 100
    
    # Campaign Objective mais comum
    patterns['top_objective'] = top_creatives['Campaign Objective'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'
    patterns['objective_dist'] = top_creatives['Campaign Objective'].value_counts().to_dict()
    
    # Targeting mais comum
    patterns['top_targeting'] = top_creatives['Targeting'].mode().iloc[0] if len(top_creatives) > 0 else 'N/A'
    patterns['targeting_dist'] = top_creatives['Targeting'].value_counts().to_dict()
    
    # Métricas médias dos top creatives
    patterns['avg_roas'] = top_creatives['ROAS'].mean()
    patterns['avg_cpa'] = top_creatives['CPA'].mean()
    patterns['avg_ctr'] = top_creatives['CTR'].mean()
    patterns['avg_frequency'] = top_creatives['Frequency'].mean()
    
    return patterns


def generate_recommendations(df, patterns, metrics):
    """
    Gera recomendações automáticas baseadas na análise
    """
    recommendations = []
    
    # Recomendação baseada em Advantage+ Placements
    adv_yes = df[df['Advantage+ Placements'] == 'Yes']
    adv_no = df[df['Advantage+ Placements'] == 'No']
    
    if len(adv_yes) > 0 and len(adv_no) > 0:
        roas_yes = adv_yes['ROAS'].mean()
        roas_no = adv_no['ROAS'].mean()
        if roas_yes > roas_no:
            uplift = ((roas_yes - roas_no) / roas_no) * 100 if roas_no > 0 else 0
            recommendations.append({
                'priority': 'Alta',
                'action': 'Ativar Advantage+ Placements',
                'reason': f'Creatives com Advantage+ Placements têm ROAS {uplift:.1f}% maior',
                'impact': 'Alto'
            })
    
    # Recomendação baseada em Campaign Objective
    sales_creatives = df[df['Campaign Objective'] == 'Sales']
    if len(sales_creatives) > 0:
        sales_roas = sales_creatives['ROAS'].mean()
        overall_roas = df['ROAS'].mean()
        if sales_roas > overall_roas:
            recommendations.append({
                'priority': 'Alta',
                'action': 'Priorizar objetivo de Sales',
                'reason': f'Campanhas de Sales têm ROAS médio de {sales_roas:.2f}x',
                'impact': 'Alto'
            })
    
    # Recomendação baseada em Targeting
    targeting_performance = df.groupby('Targeting')['ROAS'].mean().sort_values(ascending=False)
    best_targeting = targeting_performance.index[0]
    recommendations.append({
        'priority': 'Média',
        'action': f'Expandir uso de targeting "{best_targeting}"',
        'reason': f'Melhor ROAS médio entre tipos de targeting ({targeting_performance.iloc[0]:.2f}x)',
        'impact': 'Médio'
    })
    
    # Recomendação baseada em Placement
    placement_performance = df.groupby('Placement')['ROAS'].mean().sort_values(ascending=False)
    best_placement = placement_performance.index[0]
    recommendations.append({
        'priority': 'Média',
        'action': f'Aumentar investimento em "{best_placement}"',
        'reason': f'Melhor performance entre placements (ROAS: {placement_performance.iloc[0]:.2f}x)',
        'impact': 'Médio'
    })
    
    # Recomendação baseada em Frequency
    high_freq = df[df['Frequency'] > 2]
    low_freq = df[df['Frequency'] <= 2]
    if len(high_freq) > 0 and len(low_freq) > 0:
        if high_freq['CPA'].mean() > low_freq['CPA'].mean():
            recommendations.append({
                'priority': 'Média',
                'action': 'Controlar frequência de exibição',
                'reason': 'Frequência alta está associada a CPA maior',
                'impact': 'Médio'
            })
    
    # Recomendação para creatives com baixa performance
    low_performers = df[df['ROAS'] < df['ROAS'].quantile(0.25)]
    if len(low_performers) > 0:
        recommendations.append({
            'priority': 'Alta',
            'action': f'Revisar ou pausar {len(low_performers)} creatives de baixa performance',
            'reason': 'Creatives com ROAS abaixo do percentil 25%',
            'impact': 'Alto'
        })
    
    return recommendations


def generate_insights(df, metrics, patterns):
    """
    Gera insights em linguagem natural
    """
    insights = []
    
    # Insight geral
    insights.append({
        'category': 'Visão Geral',
        'insight': f"O portfólio possui {metrics['total_creatives']} creatives, com {metrics['active_creatives']} ativos. "
                   f"O investimento total foi de ${metrics['total_spend']:,.0f}, gerando {metrics['total_conversions']:,.0f} conversões.",
        'icon': ''
    })
    
    # Insight de ROAS
    insights.append({
        'category': 'Retorno sobre Investimento',
        'insight': f"O ROAS geral é de {metrics['overall_roas']:.2f}x, significando que para cada $1 investido, "
                   f"há um retorno de ${metrics['overall_roas']:.2f}. Os top performers alcançam ROAS de {patterns['avg_roas']:.2f}x.",
        'icon': ''
    })
    
    # Insight de CTR
    avg_ctr = metrics['avg_ctr']
    ctr_benchmark = 1.0  # Benchmark típico de Meta Ads
    ctr_status = "acima" if avg_ctr > ctr_benchmark else "abaixo"
    insights.append({
        'category': 'Engajamento',
        'insight': f"O CTR médio é de {avg_ctr:.2f}%, {ctr_status} do benchmark típico de {ctr_benchmark}%. "
                   f"Os top creatives têm CTR de {patterns['avg_ctr']:.2f}%.",
        'icon': ''
    })
    
    # Insight de Placements
    insights.append({
        'category': 'Placements',
        'insight': f"O placement mais eficiente é '{patterns['top_placement']}'. "
                   f"{patterns['advantage_placements']:.0f}% dos top creatives usam Advantage+ Placements.",
        'icon': ''
    })
    
    # Insight de Targeting
    insights.append({
        'category': 'Targeting',
        'insight': f"O targeting '{patterns['top_targeting']}' é predominante entre os melhores creatives. "
                   f"A estratégia de targeting deve priorizar esta abordagem.",
        'icon': ''
    })
    
    # Insight de Frequency
    insights.append({
        'category': 'Frequência',
        'insight': f"A frequência média é de {metrics['avg_frequency']:.2f}. "
                   f"Os top creatives mantêm frequência de {patterns['avg_frequency']:.2f}, indicando "
                   f"{'controle adequado' if patterns['avg_frequency'] < 2 else 'possível saturação'}.",
        'icon': ''
    })
    
    # Insight de Custo
    insights.append({
        'category': 'Eficiência de Custo',
        'insight': f"O CPA médio é de ${metrics['avg_cpa']:.2f}, enquanto os top performers alcançam "
                   f"${patterns['avg_cpa']:.2f}. Há oportunidade de redução de {((metrics['avg_cpa'] - patterns['avg_cpa']) / metrics['avg_cpa'] * 100):.1f}% no CPA.",
        'icon': ''
    })
    
    return insights
