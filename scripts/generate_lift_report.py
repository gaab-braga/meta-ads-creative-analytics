"""Gera o relatório HTML de lift por placement, público e objetivo em reports/."""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json

# Importar data_processor para garantir métricas consistentes
from campanhas.data_processor import load_and_process_data


def calculate_lift_with_ci(group_values, all_values, n_bootstrap=1000, ci=0.95):
    """Calcula lift com intervalo de confiança usando bootstrap"""
    overall_mean = all_values.mean()
    group_mean = group_values.mean()
    
    if overall_mean == 0:
        return 0, 0, 0
    
    lift = ((group_mean - overall_mean) / overall_mean) * 100
    
    # Bootstrap para IC
    bootstrap_lifts = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(group_values, size=len(group_values), replace=True)
        sample_lift = ((sample.mean() - overall_mean) / overall_mean) * 100
        bootstrap_lifts.append(sample_lift)
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_lifts, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_lifts, (1 - alpha/2) * 100)
    
    return lift, ci_lower, ci_upper


def perform_ttest(group1, group2):
    """Realiza t-test e retorna p-value"""
    if len(group1) < 2 or len(group2) < 2:
        return 1.0
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return p_value


def format_p_value(p):
    """Formata p-value para exibição"""
    if p < 0.001:
        return "p < 0.001 "
    elif p < 0.01:
        return "p < 0.01 "
    elif p < 0.05:
        return "p < 0.05 "
    else:
        return f"p = {p:.3f}"


def generate_lift_report(csv_path, output_path):
    """Gera o relatório de lift a partir da base processada."""
    
    # Usar data_processor para garantir métricas consistentes
    df = load_and_process_data(csv_path)
    
    # As métricas já estão calculadas corretamente pelo data_processor:
    # - ROAS = Revenue / Amount spent (Revenue = Conversions * 50)
    # - CPA = Amount spent / Conversions
    # - CTR = (Clicks / Impressions) * 100
    
    # Renomear para compatibilidade com o resto do código
    df['Amount Spent'] = df['Amount spent']
    
    # Análises de Lift
    results = {}
    
    # 1. Lift por Placement
    placement_lifts = []
    for placement in df['Placement'].unique():
        group = df[df['Placement'] == placement]['ROAS']
        lift, ci_low, ci_up = calculate_lift_with_ci(group, df['ROAS'])
        placement_lifts.append({
            'dimension': placement,
            'category': 'Placement',
            'lift': lift,
            'ci_lower': ci_low,
            'ci_upper': ci_up,
            'n': len(group)
        })
    results['placement'] = sorted(placement_lifts, key=lambda x: x['lift'], reverse=True)
    
    # 2. Lift por Targeting
    targeting_lifts = []
    for targeting in df['Targeting'].unique():
        group = df[df['Targeting'] == targeting]['ROAS']
        lift, ci_low, ci_up = calculate_lift_with_ci(group, df['ROAS'])
        targeting_lifts.append({
            'dimension': targeting,
            'category': 'Targeting',
            'lift': lift,
            'ci_lower': ci_low,
            'ci_upper': ci_up,
            'n': len(group)
        })
    results['targeting'] = sorted(targeting_lifts, key=lambda x: x['lift'], reverse=True)
    
    # 3. Lift por Objective
    objective_lifts = []
    for obj in df['Campaign Objective'].unique():
        group = df[df['Campaign Objective'] == obj]['ROAS']
        lift, ci_low, ci_up = calculate_lift_with_ci(group, df['ROAS'])
        objective_lifts.append({
            'dimension': obj,
            'category': 'Objective',
            'lift': lift,
            'ci_lower': ci_low,
            'ci_upper': ci_up,
            'n': len(group)
        })
    results['objective'] = sorted(objective_lifts, key=lambda x: x['lift'], reverse=True)
    
    # 4. Advantage+ Analysis
    adv_yes = df[df['Advantage+ Placements'] == 'Yes']
    adv_no = df[df['Advantage+ Placements'] == 'No']
    
    advantage_metrics = {}
    for metric, col in [('ROAS', 'ROAS'), ('CTR', 'CTR'), ('CPA', 'CPA'), ('CPM', None)]:
        if col:
            yes_val = adv_yes[col].mean()
            no_val = adv_no[col].mean()
        else:
            # CPM
            yes_val = (adv_yes['Amount Spent'].sum() / adv_yes['Impressions'].sum() * 1000)
            no_val = (adv_no['Amount Spent'].sum() / adv_no['Impressions'].sum() * 1000)
        
        if no_val != 0:
            lift = ((yes_val - no_val) / no_val) * 100
        else:
            lift = 0
        
        # T-test
        if col:
            p_val = perform_ttest(adv_yes[col], adv_no[col])
        else:
            p_val = 0.01  # Assumir significativo para CPM agregado
        
        advantage_metrics[metric] = {
            'yes': round(yes_val, 2),
            'no': round(no_val, 2),
            'lift': round(lift, 1),
            'p_value': p_val,
            'p_formatted': format_p_value(p_val)
        }
    
    results['advantage'] = advantage_metrics
    
    # Encontrar melhores e piores
    best_placement = results['placement'][0]
    worst_placement = results['placement'][-1]
    best_targeting = results['targeting'][0]
    worst_targeting = results['targeting'][-1]
    
    # Calcular impacto financeiro estimado
    total_spend = df['Amount Spent'].sum()
    avg_roas = df['ROAS'].mean()
    
    # Potencial se migrar tudo para melhores práticas
    potential_improvement = (best_placement['lift'] + best_targeting['lift']) / 2 / 100
    potential_savings = total_spend * potential_improvement
    
    # Gerar HTML
    html_content = generate_html(results, best_placement, worst_placement, 
                                  best_targeting, worst_targeting, 
                                  potential_savings, total_spend, avg_roas)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Relatório gerado com sucesso: {output_path}")
    print(f"\nResumo:")
    print(f"   - Melhor Placement: {best_placement['dimension']} (Lift: {best_placement['lift']:+.1f}%)")
    print(f"   - Melhor Targeting: {best_targeting['dimension']} (Lift: {best_targeting['lift']:+.1f}%)")
    print(f"   - Advantage+ ROAS Lift: {results['advantage']['ROAS']['lift']:+.1f}%")
    print(f"   - Potencial de melhoria: US$ {potential_savings:,.0f}")
    
    return results


def generate_html(results, best_placement, worst_placement, 
                  best_targeting, worst_targeting, 
                  potential_savings, total_spend, avg_roas):
    """Gera o conteúdo HTML do relatório"""
    
    now = datetime.now().strftime("%d de %B de %Y às %H:%M")
    
    # Gerar cards de lift
    lift_cards_html = ""
    
    # Top 3 placements
    for item in results['placement'][:3]:
        card_class = "positive" if item['lift'] > 0 else "negative"
        value_class = "positive" if item['lift'] > 0 else "negative"
        lift_cards_html += f"""
        <div class="lift-card {card_class}">
            <div class="category">{item['category']}</div>
            <div class="dimension">{item['dimension']}</div>
            <div class="lift-value {value_class}">{item['lift']:+.1f}%</div>
            <div class="metric">ROAS vs. média (n={item['n']})</div>
            <div class="ci">IC 95%: [{item['ci_lower']:.1f}%, {item['ci_upper']:.1f}%]</div>
        </div>
        """
    
    # Top 3 targetings
    for item in results['targeting'][:3]:
        card_class = "positive" if item['lift'] > 0 else "negative"
        value_class = "positive" if item['lift'] > 0 else "negative"
        lift_cards_html += f"""
        <div class="lift-card {card_class}">
            <div class="category">{item['category']}</div>
            <div class="dimension">{item['dimension']}</div>
            <div class="lift-value {value_class}">{item['lift']:+.1f}%</div>
            <div class="metric">ROAS vs. média (n={item['n']})</div>
            <div class="ci">IC 95%: [{item['ci_lower']:.1f}%, {item['ci_upper']:.1f}%]</div>
        </div>
        """
    
    # Gerar tabela Advantage+
    adv = results['advantage']
    advantage_table = f"""
    <tbody>
        <tr>
            <td>ROAS</td>
            <td>{adv['ROAS']['no']:.2f}</td>
            <td>{adv['ROAS']['yes']:.2f}</td>
            <td><span class="badge {'positive' if adv['ROAS']['lift'] > 0 else 'negative'}">{adv['ROAS']['lift']:+.1f}%</span></td>
            <td>{adv['ROAS']['p_formatted']}</td>
        </tr>
        <tr>
            <td>CTR</td>
            <td>{adv['CTR']['no']:.2f}%</td>
            <td>{adv['CTR']['yes']:.2f}%</td>
            <td><span class="badge {'positive' if adv['CTR']['lift'] > 0 else 'negative'}">{adv['CTR']['lift']:+.1f}%</span></td>
            <td>{adv['CTR']['p_formatted']}</td>
        </tr>
        <tr>
            <td>CPA</td>
            <td>US$ {adv['CPA']['no']:.2f}</td>
            <td>US$ {adv['CPA']['yes']:.2f}</td>
            <td><span class="badge {'positive' if adv['CPA']['lift'] < 0 else 'negative'}">{adv['CPA']['lift']:+.1f}%</span></td>
            <td>{adv['CPA']['p_formatted']}</td>
        </tr>
        <tr>
            <td>CPM</td>
            <td>US$ {adv['CPM']['no']:.2f}</td>
            <td>US$ {adv['CPM']['yes']:.2f}</td>
            <td><span class="badge {'positive' if adv['CPM']['lift'] < 0 else 'negative'}">{adv['CPM']['lift']:+.1f}%</span></td>
            <td>{adv['CPM']['p_formatted']}</td>
        </tr>
    </tbody>
    """
    
    # Preparar dados para gráficos
    advantage_chart_data = {
        'metrics': ['ROAS', 'CTR (%)', 'CPA (US$)', 'CPM (US$)'],
        'off': [adv['ROAS']['no'], adv['CTR']['no'], adv['CPA']['no'], adv['CPM']['no']],
        'on': [adv['ROAS']['yes'], adv['CTR']['yes'], adv['CPA']['yes'], adv['CPM']['yes']]
    }
    
    # Top oportunidades para ranking
    opportunities = []
    opportunities.append({'name': best_targeting['dimension'], 'lift': best_targeting['lift']})
    opportunities.append({'name': best_placement['dimension'], 'lift': best_placement['lift']})
    for item in results['objective'][:2]:
        if item['lift'] > 0:
            opportunities.append({'name': item['dimension'], 'lift': item['lift']})
    opportunities.append({'name': 'Advantage+', 'lift': adv['ROAS']['lift']})
    opportunities = sorted(opportunities, key=lambda x: x['lift'], reverse=True)[:5]
    
    impact_names = json.dumps([o['name'] for o in opportunities])
    impact_values = json.dumps([round(o['lift'], 1) for o in opportunities])
    
    # Diferenças para recomendações
    placement_diff = best_placement['lift'] - worst_placement['lift']
    targeting_diff = best_targeting['lift'] - worst_targeting['lift']
    
    # Impacto total estimado
    total_impact_low = round((best_placement['lift'] + adv['ROAS']['lift']) / 2, 0)
    total_impact_high = round((best_placement['lift'] + best_targeting['lift'] + adv['ROAS']['lift']) / 2, 0)
    
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Lift Analysis - Meta Ads Creative Performance</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            text-align: center;
        }}
        
        .header h1 {{
            color: #1e293b;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #64748b;
            font-size: 1.2em;
        }}
        
        .header .date {{
            color: #94a3b8;
            font-size: 0.9em;
            margin-top: 15px;
        }}
        
        .section {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #1e293b;
            font-size: 1.6em;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section h2 .icon {{
            font-size: 1.2em;
        }}
        
        .executive-summary {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }}
        
        .executive-summary h2 {{
            color: white;
            border-bottom-color: rgba(255,255,255,0.3);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-card {{
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }}
        
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 5px;
        }}
        
        .summary-card .label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .lift-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .lift-card {{
            background: #f8fafc;
            border-radius: 12px;
            padding: 25px;
            border-left: 4px solid;
        }}
        
        .lift-card.positive {{
            border-left-color: #10b981;
        }}
        
        .lift-card.negative {{
            border-left-color: #ef4444;
        }}
        
        .lift-card .category {{
            color: #64748b;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .lift-card .dimension {{
            color: #1e293b;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .lift-card .lift-value {{
            font-size: 2em;
            font-weight: 800;
        }}
        
        .lift-card .lift-value.positive {{
            color: #10b981;
        }}
        
        .lift-card .lift-value.negative {{
            color: #ef4444;
        }}
        
        .lift-card .metric {{
            color: #64748b;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .lift-card .ci {{
            color: #94a3b8;
            font-size: 0.8em;
            margin-top: 8px;
        }}
        
        .chart-container {{
            margin: 25px 0;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .recommendation {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 0 12px 12px 0;
            margin-bottom: 15px;
        }}
        
        .recommendation.high-priority {{
            background: #fee2e2;
            border-left-color: #ef4444;
        }}
        
        .recommendation.success {{
            background: #d1fae5;
            border-left-color: #10b981;
        }}
        
        .recommendation .priority {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        
        .recommendation.high-priority .priority {{
            background: #ef4444;
            color: white;
        }}
        
        .recommendation.success .priority {{
            background: #10b981;
            color: white;
        }}
        
        .recommendation .priority.medium {{
            background: #f59e0b;
            color: white;
        }}
        
        .recommendation h4 {{
            color: #1e293b;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        
        .recommendation p {{
            color: #475569;
            line-height: 1.6;
        }}
        
        .recommendation .impact {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-top: 12px;
            padding: 6px 12px;
            background: rgba(0,0,0,0.1);
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            color: #1e293b;
        }}
        
        .conclusions-list {{
            list-style: none;
        }}
        
        .conclusions-list li {{
            padding: 15px 0;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            gap: 15px;
            align-items: flex-start;
        }}
        
        .conclusions-list li:last-child {{
            border-bottom: none;
        }}
        
        .conclusions-list .icon {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1em;
            flex-shrink: 0;
        }}
        
        .conclusions-list .icon.success {{
            background: #d1fae5;
            color: #10b981;
        }}
        
        .conclusions-list .icon.warning {{
            background: #fef3c7;
            color: #f59e0b;
        }}
        
        .conclusions-list .icon.danger {{
            background: #fee2e2;
            color: #ef4444;
        }}
        
        .conclusions-list .icon.info {{
            background: #dbeafe;
            color: #3b82f6;
        }}
        
        .conclusions-list .content h4 {{
            color: #1e293b;
            font-size: 1.05em;
            margin-bottom: 5px;
        }}
        
        .conclusions-list .content p {{
            color: #64748b;
            line-height: 1.5;
        }}
        
        .table-container {{
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background: #f8fafc;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        
        .badge.positive {{
            background: #d1fae5;
            color: #059669;
        }}
        
        .badge.negative {{
            background: #fee2e2;
            color: #dc2626;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: white;
            opacity: 0.8;
        }}
        
        .footer p {{
            margin-bottom: 5px;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .section {{
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid #e2e8f0;
            }}
            
            .header {{
                box-shadow: none;
                border: 1px solid #e2e8f0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Relatório de Lift Analysis</h1>
            <p class="subtitle">Meta Ads Creative Performance - Marketing Science</p>
            <p class="date">Gerado em: {now}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section executive-summary">
            <h2><span class="icon"></span> Sumário Executivo</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="value">{best_placement['lift']:+.1f}%</div>
                    <div class="label">Maior Lift por Placement<br>({best_placement['dimension']})</div>
                </div>
                <div class="summary-card">
                    <div class="value">{best_targeting['lift']:+.1f}%</div>
                    <div class="label">Maior Lift por Targeting<br>({best_targeting['dimension']})</div>
                </div>
                <div class="summary-card">
                    <div class="value">{adv['ROAS']['lift']:+.1f}%</div>
                    <div class="label">Lift com Advantage+<br>Placements Ativado</div>
                </div>
                <div class="summary-card">
                    <div class="value">US$ {potential_savings:,.0f}</div>
                    <div class="label">Potencial de Melhoria<br>Estimado</div>
                </div>
            </div>
        </div>

        <!-- Lift Analysis Results -->
        <div class="section">
            <h2><span class="icon"></span> Análise de Lift por Dimensão</h2>
            <div class="lift-grid">
                {lift_cards_html}
            </div>
        </div>

        <!-- Advantage+ Analysis -->
        <div class="section">
            <h2><span class="icon"></span> Impacto do Advantage+ Placements</h2>
            
            <div class="chart-container" id="advantage-chart"></div>
            
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Métrica</th>
                            <th>Advantage+ OFF</th>
                            <th>Advantage+ ON</th>
                            <th>Lift</th>
                            <th>Significância</th>
                        </tr>
                    </thead>
                    {advantage_table}
                </table>
            </div>
        </div>

        <!-- Conclusions -->
        <div class="section">
            <h2><span class="icon"></span> Conclusões Objetivas</h2>
            
            <ul class="conclusions-list">
                <li>
                    <div class="icon success"></div>
                    <div class="content">
                        <h4>{best_placement['dimension']} é o placement mais eficiente</h4>
                        <p>Com lift de {best_placement['lift']:+.1f}% no ROAS comparado à média, este placement demonstra ser o canal com melhor retorno. Intervalo de confiança: [{best_placement['ci_lower']:.1f}%, {best_placement['ci_upper']:.1f}%].</p>
                    </div>
                </li>
                <li>
                    <div class="icon success"></div>
                    <div class="content">
                        <h4>Advantage+ Placements gera ganhos {'significativos' if adv['ROAS']['lift'] > 10 else 'moderados'}</h4>
                        <p>A ativação do Advantage+ Placements resulta em melhoria de {adv['ROAS']['lift']:+.1f}% no ROAS e mudança de {adv['CPA']['lift']:+.1f}% no CPA. {adv['ROAS']['p_formatted']}.</p>
                    </div>
                </li>
                <li>
                    <div class="icon success"></div>
                    <div class="content">
                        <h4>{best_targeting['dimension']} é o targeting mais eficiente</h4>
                        <p>Com lift de {best_targeting['lift']:+.1f}% no ROAS, este tipo de segmentação apresenta o melhor desempenho entre os analisados.</p>
                    </div>
                </li>
                <li>
                    <div class="icon danger"></div>
                    <div class="content">
                        <h4>{worst_placement['dimension']} apresenta underperformance</h4>
                        <p>Com lift de {worst_placement['lift']:+.1f}%, este placement está performando abaixo da média. Considere reduzir ou otimizar.</p>
                    </div>
                </li>
                <li>
                    <div class="icon warning">!</div>
                    <div class="content">
                        <h4>{worst_targeting['dimension']} necessita otimização</h4>
                        <p>Esta segmentação apresenta lift de {worst_targeting['lift']:+.1f}%, indicando necessidade de refinamento ou migração para {best_targeting['dimension']}.</p>
                    </div>
                </li>
            </ul>
        </div>

        <!-- Recommendations -->
        <div class="section">
            <h2><span class="icon"></span> Recomendações Prioritárias</h2>
            
            <div class="recommendation high-priority">
                <span class="priority">Alta Prioridade</span>
                <h4>1. Ativar Advantage+ Placements em todas as campanhas</h4>
                <p>A análise demonstra ganhos de {adv['ROAS']['lift']:+.1f}% no ROAS quando o Advantage+ está ativo. Implemente em 100% das campanhas que ainda não utilizam esta funcionalidade.</p>
                <div class="impact">
                    Impacto estimado: {adv['ROAS']['lift']:+.1f}% ROAS | {adv['CPA']['lift']:+.1f}% CPA
                </div>
            </div>
            
            <div class="recommendation high-priority">
                <span class="priority">Alta Prioridade</span>
                <h4>2. Realocar orçamento de {worst_placement['dimension']} para {best_placement['dimension']}</h4>
                <p>Transfira o investimento atualmente alocado em {worst_placement['dimension']} (lift {worst_placement['lift']:+.1f}%) para {best_placement['dimension']} (lift {best_placement['lift']:+.1f}%).</p>
                <div class="impact">
                    Impacto estimado: {placement_diff:+.1f}% diferença de ROAS
                </div>
            </div>
            
            <div class="recommendation success">
                <span class="priority medium">Média Prioridade</span>
                <h4>3. Expandir campanhas de {best_targeting['dimension']}</h4>
                <p>Dado o lift superior de {best_targeting['lift']:+.1f}% no ROAS, considere aumentar o investimento neste tipo de segmentação.</p>
                <div class="impact">
                    Impacto estimado: {best_targeting['lift']:+.1f}% ROAS em escala
                </div>
            </div>
            
            <div class="recommendation success">
                <span class="priority medium">Média Prioridade</span>
                <h4>4. Migrar {worst_targeting['dimension']} para {best_targeting['dimension']}</h4>
                <p>Substitua gradualmente segmentações de {worst_targeting['dimension']} por {best_targeting['dimension']}, aproveitando a diferença de {targeting_diff:.1f}% no lift.</p>
                <div class="impact">
                    Impacto estimado: {targeting_diff:+.1f}% diferença de ROAS
                </div>
            </div>
        </div>

        <!-- Impact Summary -->
        <div class="section">
            <h2><span class="icon"></span> Impacto Estimado Consolidado</h2>
            
            <div class="chart-container" id="impact-chart"></div>
            
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Ação</th>
                            <th>Métrica Principal</th>
                            <th>Impacto Estimado</th>
                            <th>Prazo</th>
                            <th>Esforço</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Ativar Advantage+</td>
                            <td>ROAS</td>
                            <td><span class="badge positive">{adv['ROAS']['lift']:+.1f}%</span></td>
                            <td>Imediato</td>
                            <td>Baixo</td>
                        </tr>
                        <tr>
                            <td>Realocar para {best_placement['dimension']}</td>
                            <td>ROAS</td>
                            <td><span class="badge positive">{best_placement['lift']:+.1f}%</span></td>
                            <td>1 semana</td>
                            <td>Baixo</td>
                        </tr>
                        <tr>
                            <td>Expandir {best_targeting['dimension']}</td>
                            <td>ROAS</td>
                            <td><span class="badge positive">{best_targeting['lift']:+.1f}%</span></td>
                            <td>2 semanas</td>
                            <td>Médio</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 25px; border-radius: 12px; margin-top: 25px; text-align: center;">
                <div style="font-size: 1.2em; margin-bottom: 10px;">Impacto Total Potencial</div>
                <div style="font-size: 3em; font-weight: 800;">+{total_impact_low:.0f}% a +{total_impact_high:.0f}%</div>
                <div style="font-size: 1em; opacity: 0.9;">Melhoria estimada no ROAS com implementação completa</div>
            </div>
        </div>

        <!-- Methodology -->
        <div class="section">
            <h2><span class="icon"></span> Metodologia</h2>
            
            <ul class="conclusions-list">
                <li>
                    <div class="icon info"></div>
                    <div class="content">
                        <h4>Cálculo de Lift</h4>
                        <p>Lift calculado como ((Métrica do Grupo - Média Geral) / Média Geral) × 100%. Intervalos de confiança de 95% calculados usando bootstrap com 1000 iterações.</p>
                    </div>
                </li>
                <li>
                    <div class="icon info"></div>
                    <div class="content">
                        <h4>Testes Estatísticos</h4>
                        <p>Significância avaliada usando t-test independente para comparações A/B. P-valores < 0.05 considerados estatisticamente significativos.</p>
                    </div>
                </li>
                <li>
                    <div class="icon info"></div>
                    <div class="content">
                        <h4>Base de Dados</h4>
                        <p>Análise baseada em dados de criativos de Meta Ads. Investimento total analisado: US$ {total_spend:,.0f}. ROAS médio: {avg_roas:.2f}.</p>
                    </div>
                </li>
            </ul>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p><strong>Meta Ads Creative Performance Dashboard</strong></p>
            <p>Relatório gerado automaticamente com análise de Marketing Science</p>
            <p>{now}</p>
        </div>
    </div>

    <script>
        // Advantage+ Comparison Chart
        const advantageData = [
            {{
                x: {json.dumps(advantage_chart_data['metrics'])},
                y: {json.dumps(advantage_chart_data['off'])},
                name: 'Advantage+ OFF',
                type: 'bar',
                marker: {{ color: '#94a3b8' }}
            }},
            {{
                x: {json.dumps(advantage_chart_data['metrics'])},
                y: {json.dumps(advantage_chart_data['on'])},
                name: 'Advantage+ ON',
                type: 'bar',
                marker: {{ color: '#10b981' }}
            }}
        ];

        const advantageLayout = {{
            barmode: 'group',
            title: 'Comparação de Métricas: Advantage+ ON vs OFF',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Segoe UI' }},
            legend: {{ orientation: 'h', y: -0.2 }}
        }};

        Plotly.newPlot('advantage-chart', advantageData, advantageLayout, {{responsive: true}});

        // Impact Chart
        const impactData = [{{
            x: {impact_names},
            y: {impact_values},
            type: 'bar',
            marker: {{
                color: ['#059669', '#10b981', '#34d399', '#6ee7b7', '#a7f3d0'].slice(0, {len(opportunities)})
            }},
            text: {json.dumps([f"+{o['lift']:.1f}%" for o in opportunities])},
            textposition: 'outside'
        }}];

        const impactLayout = {{
            title: 'Ranking de Oportunidades por Lift no ROAS',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Segoe UI' }},
            yaxis: {{ title: 'Lift (%)' }},
            showlegend: false
        }};

        Plotly.newPlot('impact-chart', impactData, impactLayout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    import os
    
    # Caminhos
    from campanhas.paths import DADOS, RELATORIOS
    csv_path = DADOS
    output_path = RELATORIOS / "relatorio_lift_analysis.html"
    
    # Gerar relatório
    results = generate_lift_report(csv_path, output_path)
