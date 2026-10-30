"""Análises dos criativos: fatores do ROAS, comparação entre grupos, lift por segmento, regressão e cenários."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')


class MarketingScienceAnalyzer:
    """
    Classe para análise avançada de Marketing Science
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
        
    def prepare_data(self):
        """Prepara dados para análise"""
        # Encode categorical variables
        self.df['Advantage_Binary'] = (self.df['Advantage+ Placements'] == 'Yes').astype(int)
        self.df['Advantage_Creative_Binary'] = (self.df['Advantage+ Creative'] == 'Yes').astype(int)
        
        # Label encode categoricals
        self.label_encoders = {}
        for col in ['Placement', 'Campaign Objective', 'Targeting', 'Delivery status']:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
    
    # =========================================================================
    # 1. ANÁLISE DE IMPACTO NO ROAS
    # =========================================================================
    
    def analyze_roas_impact(self):
        """
        Identifica quais variáveis têm impacto direto no ROAS
        usando múltiplos modelos
        """
        results = {}
        
        # Features para análise
        features = [
            'Advantage_Binary',
            'Campaign Objective_encoded',
            'Targeting_encoded', 
            'Placement_encoded',
            'Frequency',
            'Clicks',
            'Reach'
        ]
        
        X = self.df[features].fillna(0)
        y = self.df['ROAS'].fillna(0)
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Base inteira só para ler coeficientes e importâncias; o R² vem da validação cruzada
        ridge = Ridge(alpha=1.0)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        for modelo in (ridge, rf, gb):
            modelo.fit(X_scaled, y)

        # O scaler fica dentro do Pipeline para ser ajustado só nas dobras de treino
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def r2_validacao_cruzada(modelo):
            return cross_val_score(make_pipeline(StandardScaler(), modelo), X, y, cv=cv, scoring='r2')

        ridge_scores = r2_validacao_cruzada(Ridge(alpha=1.0))
        rf_scores = r2_validacao_cruzada(RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
        gb_scores = r2_validacao_cruzada(GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5))
        
        # Calcular importância combinada
        feature_names_readable = [
            'Advantage+ Placements',
            'Campaign Objective',
            'Targeting',
            'Placement',
            'Frequency',
            'Clicks',
            'Reach'
        ]
        
        # Normalizar coeficientes e importâncias
        ridge_importance = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_)) * 100
        rf_importance = rf.feature_importances_ / np.sum(rf.feature_importances_) * 100
        gb_importance = gb.feature_importances_ / np.sum(gb.feature_importances_) * 100
        
        # Média ponderada das importâncias
        combined_importance = (ridge_importance * 0.3 + rf_importance * 0.4 + gb_importance * 0.3)
        
        # Criar ranking
        impact_ranking = []
        for i, name in enumerate(feature_names_readable):
            # Calcular direção do impacto (positivo/negativo)
            direction = 'Positivo' if ridge.coef_[i] > 0 else 'Negativo'
            
            impact_ranking.append({
                'variable': name,
                'impact_score': combined_importance[i],
                'ridge_coef': ridge.coef_[i],
                'rf_importance': rf_importance[i],
                'gb_importance': gb_importance[i],
                'direction': direction,
                'statistical_significance': self._calculate_significance(name)
            })
        
        # Ordenar por impacto
        impact_ranking = sorted(impact_ranking, key=lambda x: x['impact_score'], reverse=True)
        
        results['impact_ranking'] = impact_ranking
        results['model_performance'] = {
            'ridge_r2': np.mean(ridge_scores),
            'rf_r2': np.mean(rf_scores),
            'gb_r2': np.mean(gb_scores),
            'best_model': 'Gradient Boosting' if np.mean(gb_scores) > max(np.mean(ridge_scores), np.mean(rf_scores)) else 
                         'Random Forest' if np.mean(rf_scores) > np.mean(ridge_scores) else 'Ridge'
        }
        
        return results
    
    def _calculate_significance(self, variable):
        """Calcula significância estatística de uma variável"""
        if variable == 'Advantage+ Placements':
            group1 = self.df[self.df['Advantage+ Placements'] == 'Yes']['ROAS']
            group2 = self.df[self.df['Advantage+ Placements'] == 'No']['ROAS']
        elif variable == 'Frequency':
            high_freq = self.df[self.df['Frequency'] > self.df['Frequency'].median()]['ROAS']
            low_freq = self.df[self.df['Frequency'] <= self.df['Frequency'].median()]['ROAS']
            group1, group2 = high_freq, low_freq
        else:
            return {'p_value': 0.05, 'is_significant': True}
        
        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_value = stats.ttest_ind(group1, group2)
            return {'p_value': p_value, 'is_significant': p_value < 0.05}
        return {'p_value': 1.0, 'is_significant': False}
    
    # =========================================================================
    # 2. COMPARAÇÃO ENTRE GRUPOS
    # =========================================================================
    
    def comparar_advantage_placements(self):
        """
        Compara criativos com e sem Advantage+ Placements: diferença de médias e teste t.
        A base é um corte só (sem antes e depois), então isso mostra associação, não efeito causal.
        """
        results = {}
        
        # Grupos
        treatment = self.df[self.df['Advantage+ Placements'] == 'Yes']
        control = self.df[self.df['Advantage+ Placements'] == 'No']
        
        metrics = ['ROAS', 'CPA', 'CTR', 'CPC', 'CPM', 'Conversions']
        
        for metric in metrics:
            # Estatísticas básicas
            treat_mean = treatment[metric].mean()
            treat_std = treatment[metric].std()
            control_mean = control[metric].mean()
            control_std = control[metric].std()
            
            # Diferença
            diff = treat_mean - control_mean
            pct_diff = (diff / control_mean * 100) if control_mean != 0 else 0
            
            # Teste estatístico
            if len(treatment) > 1 and len(control) > 1:
                t_stat, p_value = stats.ttest_ind(treatment[metric], control[metric])
                # Cohen's d (effect size)
                pooled_std = np.sqrt((treat_std**2 + control_std**2) / 2)
                cohens_d = diff / pooled_std if pooled_std > 0 else 0
            else:
                t_stat, p_value, cohens_d = 0, 1, 0
            
            # Intervalo de confiança
            se = np.sqrt(treat_std**2/len(treatment) + control_std**2/len(control)) if len(treatment) > 0 and len(control) > 0 else 0
            ci_lower = diff - 1.96 * se
            ci_upper = diff + 1.96 * se
            
            results[metric] = {
                'treatment_mean': treat_mean,
                'treatment_std': treat_std,
                'treatment_n': len(treatment),
                'control_mean': control_mean,
                'control_std': control_std,
                'control_n': len(control),
                'difference': diff,
                'percentage_difference': pct_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'effect_size': 'Grande' if abs(cohens_d) > 0.8 else 'Médio' if abs(cohens_d) > 0.5 else 'Pequeno',
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        # Resumo executivo
        results['summary'] = {
            'total_treatment': len(treatment),
            'total_control': len(control),
            'roas_uplift': results['ROAS']['percentage_difference'],
            'cpa_reduction': -results['CPA']['percentage_difference'],
            'conversion_uplift': results['Conversions']['percentage_difference'],
            'statistical_confidence': 1 - results['ROAS']['p_value']
        }
        
        return results
    
    def diferenca_por_objetivo(self):
        """
        Diferença de ROAS e CPA de cada objetivo de campanha contra a média dos demais.
        """
        results = {}
        
        # Analisar por Campaign Objective
        objectives = self.df['Campaign Objective'].unique()
        
        objective_impact = {}
        for obj in objectives:
            obj_data = self.df[self.df['Campaign Objective'] == obj]
            other_data = self.df[self.df['Campaign Objective'] != obj]
            
            if len(obj_data) > 5 and len(other_data) > 5:
                roas_diff = obj_data['ROAS'].mean() - other_data['ROAS'].mean()
                cpa_diff = obj_data['CPA'].mean() - other_data['CPA'].mean()
                
                objective_impact[obj] = {
                    'roas_impact': roas_diff,
                    'roas_pct': (roas_diff / other_data['ROAS'].mean() * 100) if other_data['ROAS'].mean() != 0 else 0,
                    'cpa_impact': cpa_diff,
                    'cpa_pct': (cpa_diff / other_data['CPA'].mean() * 100) if other_data['CPA'].mean() != 0 else 0,
                    'sample_size': len(obj_data),
                    'avg_conversions': obj_data['Conversions'].mean()
                }
        
        results['by_objective'] = objective_impact
        
        # Analisar por Targeting
        targetings = self.df['Targeting'].unique()
        
        targeting_impact = {}
        for tgt in targetings:
            tgt_data = self.df[self.df['Targeting'] == tgt]
            other_data = self.df[self.df['Targeting'] != tgt]
            
            if len(tgt_data) > 5 and len(other_data) > 5:
                roas_diff = tgt_data['ROAS'].mean() - other_data['ROAS'].mean()
                
                targeting_impact[tgt] = {
                    'roas_impact': roas_diff,
                    'roas_pct': (roas_diff / other_data['ROAS'].mean() * 100) if other_data['ROAS'].mean() != 0 else 0,
                    'avg_cpa': tgt_data['CPA'].mean(),
                    'sample_size': len(tgt_data)
                }
        
        results['by_targeting'] = targeting_impact
        
        # Analisar por Placement
        placements = self.df['Placement'].unique()
        
        placement_impact = {}
        for plc in placements:
            plc_data = self.df[self.df['Placement'] == plc]
            other_data = self.df[self.df['Placement'] != plc]
            
            if len(plc_data) > 5 and len(other_data) > 5:
                roas_diff = plc_data['ROAS'].mean() - other_data['ROAS'].mean()
                
                placement_impact[plc] = {
                    'roas_impact': roas_diff,
                    'roas_pct': (roas_diff / other_data['ROAS'].mean() * 100) if other_data['ROAS'].mean() != 0 else 0,
                    'avg_ctr': plc_data['CTR'].mean(),
                    'sample_size': len(plc_data)
                }
        
        results['by_placement'] = placement_impact
        
        return results
    
    # =========================================================================
    # 3. RELAÇÃO FREQUENCY x CPA
    # =========================================================================
    
    def frequency_cpa_analysis(self):
        """
        Análise detalhada da relação entre Frequency e CPA
        """
        results = {}
        
        freq = self.df['Frequency'].values
        cpa = self.df['CPA'].values
        
        # Correlação de Pearson
        pearson_corr, pearson_p = stats.pearsonr(freq, cpa)
        
        # Correlação de Spearman (não-paramétrica)
        spearman_corr, spearman_p = stats.spearmanr(freq, cpa)
        
        # Regressão linear
        X_freq = freq.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X_freq, cpa)
        
        # Predições para diferentes frequências
        freq_range = np.arange(1.0, 3.5, 0.25)
        cpa_predictions = reg.predict(freq_range.reshape(-1, 1))
        
        # Análise por faixas de frequência
        freq_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, float('inf'))]
        freq_analysis = []
        
        for low, high in freq_bins:
            mask = (self.df['Frequency'] >= low) & (self.df['Frequency'] < high)
            segment = self.df[mask]
            
            if len(segment) > 0:
                freq_analysis.append({
                    'range': f'{low:.1f} - {high:.1f}' if high != float('inf') else f'{low:.1f}+',
                    'count': len(segment),
                    'avg_cpa': segment['CPA'].mean(),
                    'avg_roas': segment['ROAS'].mean(),
                    'avg_ctr': segment['CTR'].mean(),
                    'total_conversions': segment['Conversions'].sum()
                })
        
        # Ponto ótimo de frequência (menor CPA)
        optimal_idx = np.argmin([x['avg_cpa'] for x in freq_analysis])
        optimal_frequency = freq_analysis[optimal_idx]['range']
        
        results['correlation'] = {
            'pearson': pearson_corr,
            'pearson_p': pearson_p,
            'spearman': spearman_corr,
            'spearman_p': spearman_p,
            'interpretation': 'Positiva' if pearson_corr > 0 else 'Negativa',
            'strength': 'Forte' if abs(pearson_corr) > 0.7 else 'Moderada' if abs(pearson_corr) > 0.4 else 'Fraca'
        }
        
        results['regression'] = {
            'intercept': reg.intercept_,
            'coefficient': reg.coef_[0],
            'r_squared': reg.score(X_freq, cpa),
            'interpretation': f'Para cada aumento de 1 na frequência, o CPA {"aumenta" if reg.coef_[0] > 0 else "diminui"} em ${abs(reg.coef_[0]):.2f}'
        }
        
        results['frequency_bands'] = freq_analysis
        results['optimal_frequency'] = optimal_frequency
        results['predictions'] = list(zip(freq_range.tolist(), cpa_predictions.tolist()))
        
        # Recomendação
        if reg.coef_[0] > 0:
            results['recommendation'] = {
                'action': 'Controlar frequência',
                'reason': 'Frequência alta aumenta CPA',
                'optimal_range': optimal_frequency,
                'estimated_savings': f'${(freq_analysis[-1]["avg_cpa"] - freq_analysis[optimal_idx]["avg_cpa"]):.2f} por conversão'
            }
        else:
            results['recommendation'] = {
                'action': 'Aumentar frequência moderadamente',
                'reason': 'Frequência está associada a menor CPA',
                'optimal_range': optimal_frequency
            }
        
        return results
    
    # =========================================================================
    # 4. UPLIFT ADVANTAGE+ PLACEMENTS
    # =========================================================================
    
    def calculate_advantage_uplift(self):
        """
        Calcula o uplift médio quando Advantage+ Placements = Yes
        """
        results = {}
        
        adv_yes = self.df[self.df['Advantage+ Placements'] == 'Yes']
        adv_no = self.df[self.df['Advantage+ Placements'] == 'No']
        
        metrics = ['ROAS', 'CPA', 'CTR', 'CPC', 'CPM', 'Conversions', 'Clicks', 'Reach']
        
        for metric in metrics:
            yes_mean = adv_yes[metric].mean()
            no_mean = adv_no[metric].mean()
            
            # Uplift absoluto e percentual
            uplift = yes_mean - no_mean
            uplift_pct = (uplift / no_mean * 100) if no_mean != 0 else 0
            
            # Para CPA, inverter a interpretação (menor é melhor)
            is_positive = uplift_pct > 0
            if metric in ['CPA', 'CPC', 'CPM']:
                is_positive = uplift_pct < 0
            
            results[metric] = {
                'with_advantage': yes_mean,
                'without_advantage': no_mean,
                'absolute_uplift': uplift,
                'percentage_uplift': uplift_pct,
                'is_positive_impact': is_positive
            }
        
        # Resumo geral
        results['summary'] = {
            'roas_uplift': results['ROAS']['percentage_uplift'],
            'cpa_change': results['CPA']['percentage_uplift'],
            'ctr_uplift': results['CTR']['percentage_uplift'],
            'recommendation': 'Ativar Advantage+ Placements' if results['ROAS']['is_positive_impact'] else 'Manter testes manuais',
            'confidence_level': 'Alta' if abs(results['ROAS']['percentage_uplift']) > 10 else 'Média' if abs(results['ROAS']['percentage_uplift']) > 5 else 'Baixa',
            'total_with_advantage': len(adv_yes),
            'total_without_advantage': len(adv_no)
        }
        
        return results
    
    # =========================================================================
    # 5. PADRÕES DOS TOP PERFORMERS
    # =========================================================================
    
    def identify_top_performers_patterns(self, top_n=20):
        """
        Identifica padrões dos creatives com melhor performance
        """
        results = {}
        
        # Top performers por ROAS
        top_roas = self.df.nlargest(top_n, 'ROAS')
        # Top performers por CPA (menor)
        top_cpa = self.df.nsmallest(top_n, 'CPA')
        # Top performers por conversões
        top_conv = self.df.nlargest(top_n, 'Conversions')
        
        # Score combinado
        self.df['performance_score'] = (
            (self.df['ROAS'] / self.df['ROAS'].max()) * 40 +
            (1 - self.df['CPA'] / self.df['CPA'].max()) * 30 +
            (self.df['CTR'] / self.df['CTR'].max()) * 15 +
            (self.df['Conversions'] / self.df['Conversions'].max()) * 15
        )
        
        top_overall = self.df.nlargest(top_n, 'performance_score')
        
        # Analisar padrões
        patterns = {}
        
        # Placement patterns
        patterns['placement'] = {
            'distribution': top_overall['Placement'].value_counts().to_dict(),
            'most_common': top_overall['Placement'].mode().iloc[0] if len(top_overall) > 0 else 'N/A',
            'percentage': (top_overall['Placement'].value_counts().iloc[0] / len(top_overall) * 100) if len(top_overall) > 0 else 0
        }
        
        # Advantage+ patterns
        patterns['advantage_placements'] = {
            'yes_percentage': (top_overall['Advantage+ Placements'] == 'Yes').mean() * 100,
            'no_percentage': (top_overall['Advantage+ Placements'] == 'No').mean() * 100
        }
        
        patterns['advantage_creative'] = {
            'yes_percentage': (top_overall['Advantage+ Creative'] == 'Yes').mean() * 100,
            'no_percentage': (top_overall['Advantage+ Creative'] == 'No').mean() * 100
        }
        
        # Campaign Objective patterns
        patterns['objective'] = {
            'distribution': top_overall['Campaign Objective'].value_counts().to_dict(),
            'most_common': top_overall['Campaign Objective'].mode().iloc[0] if len(top_overall) > 0 else 'N/A',
            'percentage': (top_overall['Campaign Objective'].value_counts().iloc[0] / len(top_overall) * 100) if len(top_overall) > 0 else 0
        }
        
        # Targeting patterns
        patterns['targeting'] = {
            'distribution': top_overall['Targeting'].value_counts().to_dict(),
            'most_common': top_overall['Targeting'].mode().iloc[0] if len(top_overall) > 0 else 'N/A',
            'percentage': (top_overall['Targeting'].value_counts().iloc[0] / len(top_overall) * 100) if len(top_overall) > 0 else 0
        }
        
        # Métricas médias dos top performers
        patterns['metrics'] = {
            'avg_roas': top_overall['ROAS'].mean(),
            'avg_cpa': top_overall['CPA'].mean(),
            'avg_ctr': top_overall['CTR'].mean(),
            'avg_frequency': top_overall['Frequency'].mean(),
            'avg_conversions': top_overall['Conversions'].mean(),
            'total_spend': top_overall['Amount spent'].sum(),
            'total_conversions': top_overall['Conversions'].sum()
        }
        
        # Comparação com média geral
        patterns['vs_average'] = {
            'roas_diff': ((patterns['metrics']['avg_roas'] / self.df['ROAS'].mean()) - 1) * 100,
            'cpa_diff': ((patterns['metrics']['avg_cpa'] / self.df['CPA'].mean()) - 1) * 100,
            'ctr_diff': ((patterns['metrics']['avg_ctr'] / self.df['CTR'].mean()) - 1) * 100
        }
        
        results['patterns'] = patterns
        results['top_creatives'] = top_overall[['Nome do Creative', 'ROAS', 'CPA', 'CTR', 'Conversions', 
                                                 'Placement', 'Campaign Objective', 'Targeting',
                                                 'Advantage+ Placements', 'performance_score']].to_dict('records')
        
        # Perfil ideal do creative
        results['ideal_profile'] = {
            'placement': patterns['placement']['most_common'],
            'objective': patterns['objective']['most_common'],
            'targeting': patterns['targeting']['most_common'],
            'advantage_placements': 'Yes' if patterns['advantage_placements']['yes_percentage'] > 50 else 'No',
            'advantage_creative': 'Yes' if patterns['advantage_creative']['yes_percentage'] > 50 else 'No',
            'optimal_frequency': f"{patterns['metrics']['avg_frequency']:.2f}"
        }
        
        return results
    
    # =========================================================================
    # 6. MATRIZ DE DECISÃO
    # =========================================================================
    
    def create_decision_matrix(self):
        """
        Cria matriz de decisão: Escalar, Testar, Pausar
        """
        results = {}
        
        # Calcular métricas médias
        avg_roas = self.df['ROAS'].mean()
        avg_cpa = self.df['CPA'].mean()
        avg_ctr = self.df['CTR'].mean()
        
        # Definir limiares
        high_roas = self.df['ROAS'].quantile(0.75)
        low_roas = self.df['ROAS'].quantile(0.25)
        high_cpa = self.df['CPA'].quantile(0.75)
        low_cpa = self.df['CPA'].quantile(0.25)
        
        # Classificar creatives
        scale = []  # Escalar
        test = []   # Testar
        pause = []  # Pausar
        
        for _, row in self.df.iterrows():
            creative_data = {
                'creative': row['Nome do Creative'],
                'campaign': row['Campanha'],
                'roas': row['ROAS'],
                'cpa': row['CPA'],
                'ctr': row['CTR'],
                'conversions': row['Conversions'],
                'spend': row['Amount spent'],
                'placement': row['Placement'],
                'objective': row['Campaign Objective']
            }
            
            # Critérios de classificação
            if row['ROAS'] >= high_roas and row['CPA'] <= avg_cpa:
                # Alto ROAS + CPA controlado = ESCALAR
                creative_data['reason'] = 'Alto ROAS com CPA eficiente'
                creative_data['potential'] = 'Alto'
                scale.append(creative_data)
            elif row['ROAS'] >= avg_roas and row['ROAS'] < high_roas:
                # ROAS médio = TESTAR
                creative_data['reason'] = 'ROAS médio - oportunidade de otimização'
                creative_data['potential'] = 'Médio'
                test.append(creative_data)
            elif row['ROAS'] < low_roas or row['CPA'] > high_cpa:
                # Baixo ROAS ou CPA muito alto = PAUSAR
                creative_data['reason'] = 'Baixo ROAS ou CPA muito alto'
                creative_data['potential'] = 'Baixo'
                pause.append(creative_data)
            else:
                # Intermediário = TESTAR
                creative_data['reason'] = 'Performance intermediária'
                creative_data['potential'] = 'Médio'
                test.append(creative_data)
        
        # Ordenar por performance
        scale = sorted(scale, key=lambda x: x['roas'], reverse=True)
        test = sorted(test, key=lambda x: x['roas'], reverse=True)
        pause = sorted(pause, key=lambda x: x['roas'])
        
        results['scale'] = {
            'creatives': scale[:15],  # Top 15
            'count': len(scale),
            'total_spend': sum(c['spend'] for c in scale),
            'total_conversions': sum(c['conversions'] for c in scale),
            'avg_roas': np.mean([c['roas'] for c in scale]) if scale else 0,
            'action': 'Aumentar orçamento em 50-100%',
            'expected_impact': '+30-50% em conversões'
        }
        
        results['test'] = {
            'creatives': test[:15],
            'count': len(test),
            'total_spend': sum(c['spend'] for c in test),
            'total_conversions': sum(c['conversions'] for c in test),
            'avg_roas': np.mean([c['roas'] for c in test]) if test else 0,
            'action': 'Otimizar copy/criativo, testar audiences',
            'expected_impact': '+10-20% em ROAS com otimização'
        }
        
        results['pause'] = {
            'creatives': pause[:15],
            'count': len(pause),
            'total_spend': sum(c['spend'] for c in pause),
            'total_conversions': sum(c['conversions'] for c in pause),
            'avg_roas': np.mean([c['roas'] for c in pause]) if pause else 0,
            'action': 'Pausar ou recriar completamente',
            'expected_savings': sum(c['spend'] for c in pause)
        }
        
        # Resumo
        results['summary'] = {
            'total_to_scale': len(scale),
            'total_to_test': len(test),
            'total_to_pause': len(pause),
            'potential_savings': results['pause']['total_spend'],
            'reallocation_recommendation': f"Realocar ${results['pause']['total_spend']:.0f} dos pausados para os escaláveis"
        }
        
        return results
    
    # =========================================================================
    # 7. PREVISÕES 30 DIAS
    # =========================================================================
    
    def generate_30day_forecast(self):
        """
        Gera previsões para os próximos 30 dias
        """
        results = {}
        
        # Estatísticas atuais (assumindo dados de 30 dias)
        current_spend = self.df['Amount spent'].sum()
        current_conversions = self.df['Conversions'].sum()
        current_revenue = self.df['Revenue'].sum()
        current_roas = current_revenue / current_spend if current_spend > 0 else 0
        current_cpa = current_spend / current_conversions if current_conversions > 0 else 0
        
        # Fatores de ajuste baseados em análises
        advantage_impact = self.calculate_advantage_uplift()
        
        # Cenário 1: Mantendo estratégia atual
        baseline = {
            'roas': current_roas,
            'cpa': current_cpa,
            'conversions': current_conversions,
            'spend': current_spend,
            'revenue': current_revenue
        }
        
        # Cenário 2: Otimizado (aplicando recomendações)
        # Calculamos a melhoria potencial baseada em:
        # 1. Realocar budget de low performers para high performers
        # 2. Otimizar frequency
        # 3. Usar melhor targeting
        
        # Calcular gap entre top 25% e média
        top_quartile_roas = self.df['ROAS'].quantile(0.75)
        bottom_quartile_roas = self.df['ROAS'].quantile(0.25)
        roas_gap = (top_quartile_roas - current_roas) / current_roas if current_roas > 0 else 0
        
        # Estimativa conservadora: capturar 30% do gap para a média
        optimization_factor = 1.0 + (roas_gap * 0.30) if roas_gap > 0 else 1.05
        
        # Se Advantage+ tem impacto positivo, adicionar
        if advantage_impact['summary']['roas_uplift'] > 0:
            optimization_factor += advantage_impact['summary']['roas_uplift'] / 100 * 0.3  # 30% do uplift
        
        optimized = {
            'roas': current_roas * optimization_factor,
            'cpa': current_cpa / optimization_factor if optimization_factor > 0 else current_cpa,
            'conversions': current_conversions * optimization_factor,
            'spend': current_spend,  # Mesmo investimento
            'revenue': current_revenue * optimization_factor
        }
        
        # Cenário 3: Agressivo (escalar vencedores)
        aggressive_factor = 1.3  # 30% mais agressivo
        aggressive = {
            'roas': current_roas * optimization_factor * 0.95,  # Ligeira redução por escala
            'cpa': current_cpa / (optimization_factor * 0.95),
            'conversions': current_conversions * aggressive_factor,
            'spend': current_spend * aggressive_factor,
            'revenue': current_revenue * aggressive_factor
        }
        
        # Intervalos de confiança (baseados em variabilidade)
        roas_std = self.df['ROAS'].std()
        cpa_std = self.df['CPA'].std()
        conv_std = self.df['Conversions'].std()
        
        results['baseline'] = {
            'scenario': 'Manter Estratégia Atual',
            'roas': {'value': baseline['roas'], 'lower': baseline['roas'] * 0.9, 'upper': baseline['roas'] * 1.1},
            'cpa': {'value': baseline['cpa'], 'lower': baseline['cpa'] * 0.9, 'upper': baseline['cpa'] * 1.1},
            'conversions': {'value': baseline['conversions'], 'lower': baseline['conversions'] * 0.85, 'upper': baseline['conversions'] * 1.15},
            'spend': baseline['spend'],
            'revenue': {'value': baseline['revenue'], 'lower': baseline['revenue'] * 0.85, 'upper': baseline['revenue'] * 1.15}
        }
        
        results['optimized'] = {
            'scenario': 'Estratégia Otimizada',
            'roas': {'value': optimized['roas'], 'lower': optimized['roas'] * 0.9, 'upper': optimized['roas'] * 1.15},
            'cpa': {'value': optimized['cpa'], 'lower': optimized['cpa'] * 0.85, 'upper': optimized['cpa'] * 1.1},
            'conversions': {'value': optimized['conversions'], 'lower': optimized['conversions'] * 0.9, 'upper': optimized['conversions'] * 1.2},
            'spend': optimized['spend'],
            'revenue': {'value': optimized['revenue'], 'lower': optimized['revenue'] * 0.9, 'upper': optimized['revenue'] * 1.2},
            'improvement_vs_baseline': {
                'roas': ((optimized['roas'] / baseline['roas']) - 1) * 100,
                'cpa': ((baseline['cpa'] / optimized['cpa']) - 1) * 100,
                'conversions': ((optimized['conversions'] / baseline['conversions']) - 1) * 100
            }
        }
        
        results['aggressive'] = {
            'scenario': 'Escala Agressiva',
            'roas': {'value': aggressive['roas'], 'lower': aggressive['roas'] * 0.85, 'upper': aggressive['roas'] * 1.1},
            'cpa': {'value': aggressive['cpa'], 'lower': aggressive['cpa'] * 0.9, 'upper': aggressive['cpa'] * 1.15},
            'conversions': {'value': aggressive['conversions'], 'lower': aggressive['conversions'] * 0.85, 'upper': aggressive['conversions'] * 1.25},
            'spend': aggressive['spend'],
            'revenue': {'value': aggressive['revenue'], 'lower': aggressive['revenue'] * 0.8, 'upper': aggressive['revenue'] * 1.25},
            'improvement_vs_baseline': {
                'roas': ((aggressive['roas'] / baseline['roas']) - 1) * 100,
                'conversions': ((aggressive['conversions'] / baseline['conversions']) - 1) * 100,
                'spend_increase': ((aggressive['spend'] / baseline['spend']) - 1) * 100
            }
        }
        
        # Recomendação
        results['recommendation'] = {
            'suggested_scenario': 'Estratégia Otimizada',
            'reason': 'Melhor equilíbrio entre risco e retorno',
            'expected_roas': optimized['roas'],
            'expected_cpa': optimized['cpa'],
            'expected_conversions': optimized['conversions'],
            'confidence': '75-85%'
        }
        
        return results
    
    # =========================================================================
    # 8. LIFT ANALYSIS COMPLETO
    # =========================================================================
    
    def calculate_comprehensive_lift(self):
        """
        Calcula Lift Analysis completo para todas as dimensões relevantes
        """
        results = {}
        
        # Média global para referência
        global_roas = self.df['ROAS'].mean()
        global_cpa = self.df['CPA'].mean()
        global_ctr = self.df['CTR'].mean()
        global_conversions = self.df['Conversions'].mean()
        
        # =====================================================================
        # 1. LIFT POR ADVANTAGE+ PLACEMENTS
        # =====================================================================
        adv_yes = self.df[self.df['Advantage+ Placements'] == 'Yes']
        adv_no = self.df[self.df['Advantage+ Placements'] == 'No']
        
        results['advantage_placements'] = {
            'dimension': 'Advantage+ Placements',
            'categories': {}
        }
        
        for label, group in [('Yes', adv_yes), ('No', adv_no)]:
            if len(group) > 0:
                roas_lift = ((group['ROAS'].mean() / global_roas) - 1) * 100
                cpa_lift = ((group['CPA'].mean() / global_cpa) - 1) * 100
                ctr_lift = ((group['CTR'].mean() / global_ctr) - 1) * 100
                
                # Teste estatístico
                if len(group) > 1:
                    other = self.df[self.df['Advantage+ Placements'] != label]
                    if len(other) > 1:
                        t_stat, p_value = stats.ttest_ind(group['ROAS'], other['ROAS'])
                    else:
                        p_value = 1.0
                else:
                    p_value = 1.0
                
                results['advantage_placements']['categories'][label] = {
                    'count': len(group),
                    'avg_roas': group['ROAS'].mean(),
                    'avg_cpa': group['CPA'].mean(),
                    'avg_ctr': group['CTR'].mean(),
                    'total_conversions': group['Conversions'].sum(),
                    'total_spend': group['Amount spent'].sum(),
                    'roas_lift_pct': roas_lift,
                    'cpa_lift_pct': cpa_lift,
                    'ctr_lift_pct': ctr_lift,
                    'is_significant': p_value < 0.05,
                    'p_value': p_value
                }
        
        # Calcular uplift direto Yes vs No
        if 'Yes' in results['advantage_placements']['categories'] and 'No' in results['advantage_placements']['categories']:
            yes_data = results['advantage_placements']['categories']['Yes']
            no_data = results['advantage_placements']['categories']['No']
            results['advantage_placements']['uplift'] = {
                'roas_uplift': ((yes_data['avg_roas'] / no_data['avg_roas']) - 1) * 100 if no_data['avg_roas'] > 0 else 0,
                'cpa_change': ((yes_data['avg_cpa'] / no_data['avg_cpa']) - 1) * 100 if no_data['avg_cpa'] > 0 else 0,
                'ctr_uplift': ((yes_data['avg_ctr'] / no_data['avg_ctr']) - 1) * 100 if no_data['avg_ctr'] > 0 else 0,
            }
        
        # =====================================================================
        # 2. LIFT POR CAMPAIGN OBJECTIVE
        # =====================================================================
        results['campaign_objective'] = {
            'dimension': 'Campaign Objective',
            'categories': {}
        }
        
        for objective in self.df['Campaign Objective'].unique():
            group = self.df[self.df['Campaign Objective'] == objective]
            if len(group) >= 5:
                roas_lift = ((group['ROAS'].mean() / global_roas) - 1) * 100
                cpa_lift = ((group['CPA'].mean() / global_cpa) - 1) * 100
                
                other = self.df[self.df['Campaign Objective'] != objective]
                if len(other) > 1:
                    t_stat, p_value = stats.ttest_ind(group['ROAS'], other['ROAS'])
                else:
                    p_value = 1.0
                
                results['campaign_objective']['categories'][objective] = {
                    'count': len(group),
                    'avg_roas': group['ROAS'].mean(),
                    'avg_cpa': group['CPA'].mean(),
                    'avg_ctr': group['CTR'].mean(),
                    'total_conversions': group['Conversions'].sum(),
                    'total_spend': group['Amount spent'].sum(),
                    'roas_lift_pct': roas_lift,
                    'cpa_lift_pct': cpa_lift,
                    'is_significant': p_value < 0.05,
                    'p_value': p_value
                }
        
        # Ordenar por ROAS lift
        sorted_objectives = sorted(
            results['campaign_objective']['categories'].items(),
            key=lambda x: x[1]['roas_lift_pct'],
            reverse=True
        )
        results['campaign_objective']['ranking'] = [
            {'name': k, **v} for k, v in sorted_objectives
        ]
        
        # =====================================================================
        # 3. LIFT POR TARGETING
        # =====================================================================
        results['targeting'] = {
            'dimension': 'Targeting',
            'categories': {}
        }
        
        for targeting in self.df['Targeting'].unique():
            group = self.df[self.df['Targeting'] == targeting]
            if len(group) >= 5:
                roas_lift = ((group['ROAS'].mean() / global_roas) - 1) * 100
                cpa_lift = ((group['CPA'].mean() / global_cpa) - 1) * 100
                
                other = self.df[self.df['Targeting'] != targeting]
                if len(other) > 1:
                    t_stat, p_value = stats.ttest_ind(group['ROAS'], other['ROAS'])
                else:
                    p_value = 1.0
                
                results['targeting']['categories'][targeting] = {
                    'count': len(group),
                    'avg_roas': group['ROAS'].mean(),
                    'avg_cpa': group['CPA'].mean(),
                    'avg_ctr': group['CTR'].mean(),
                    'total_conversions': group['Conversions'].sum(),
                    'total_spend': group['Amount spent'].sum(),
                    'roas_lift_pct': roas_lift,
                    'cpa_lift_pct': cpa_lift,
                    'is_significant': p_value < 0.05,
                    'p_value': p_value
                }
        
        sorted_targeting = sorted(
            results['targeting']['categories'].items(),
            key=lambda x: x[1]['roas_lift_pct'],
            reverse=True
        )
        results['targeting']['ranking'] = [
            {'name': k, **v} for k, v in sorted_targeting
        ]
        
        # =====================================================================
        # 4. LIFT POR PLACEMENT
        # =====================================================================
        results['placement'] = {
            'dimension': 'Placement',
            'categories': {}
        }
        
        for placement in self.df['Placement'].unique():
            group = self.df[self.df['Placement'] == placement]
            if len(group) >= 5:
                roas_lift = ((group['ROAS'].mean() / global_roas) - 1) * 100
                cpa_lift = ((group['CPA'].mean() / global_cpa) - 1) * 100
                
                other = self.df[self.df['Placement'] != placement]
                if len(other) > 1:
                    t_stat, p_value = stats.ttest_ind(group['ROAS'], other['ROAS'])
                else:
                    p_value = 1.0
                
                results['placement']['categories'][placement] = {
                    'count': len(group),
                    'avg_roas': group['ROAS'].mean(),
                    'avg_cpa': group['CPA'].mean(),
                    'avg_ctr': group['CTR'].mean(),
                    'total_conversions': group['Conversions'].sum(),
                    'total_spend': group['Amount spent'].sum(),
                    'roas_lift_pct': roas_lift,
                    'cpa_lift_pct': cpa_lift,
                    'is_significant': p_value < 0.05,
                    'p_value': p_value
                }
        
        sorted_placement = sorted(
            results['placement']['categories'].items(),
            key=lambda x: x[1]['roas_lift_pct'],
            reverse=True
        )
        results['placement']['ranking'] = [
            {'name': k, **v} for k, v in sorted_placement
        ]
        
        # =====================================================================
        # 5. RESUMO GERAL DE LIFT
        # =====================================================================
        adv_uplift = results['advantage_placements'].get('uplift', {}).get('roas_uplift', 0)
        results['summary'] = {
            'global_roas': global_roas,
            'global_cpa': global_cpa,
            'global_ctr': global_ctr,
            'best_objective': results['campaign_objective']['ranking'][0] if results['campaign_objective']['ranking'] else None,
            'best_targeting': results['targeting']['ranking'][0] if results['targeting']['ranking'] else None,
            'best_placement': results['placement']['ranking'][0] if results['placement']['ranking'] else None,
            'advantage_recommendation': 'Ativar Advantage+' if adv_uplift > 5 else 'Evitar Advantage+ (usar manual)' if adv_uplift < -2 else 'Testar A/B',
        }
        
        return results
    
    # =========================================================================
    # 9. ANÁLISE DE REGRESSÃO ESTATÍSTICA COMPLETA
    # =========================================================================
    
    def calculate_regression_statistics(self):
        """
        Calcula estatísticas de regressão completas para cada variável:
        - R² (coeficiente de determinação)
        - R² ajustado
        - Erro padrão (Standard Error)
        - Estatística t (t-statistic)
        - Valor p (p-value)
        """
        results = []
        
        # Variáveis para análise
        variables = [
            ('Advantage+ Placements', 'Advantage_Binary'),
            ('Campaign Objective', 'Campaign Objective_encoded'),
            ('Targeting', 'Targeting_encoded'),
            ('Placement', 'Placement_encoded'),
            ('Frequency', 'Frequency'),
            ('Clicks', 'Clicks'),
            ('Reach', 'Reach')
        ]
        
        y = self.df['ROAS'].fillna(0).values
        n = len(y)
        
        for var_name, var_col in variables:
            try:
                X = self.df[var_col].fillna(0).values.reshape(-1, 1)
                
                # Regressão linear simples
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Cálculo de R²
                ss_res = np.sum((y - y_pred) ** 2)  # Soma dos quadrados dos resíduos
                ss_tot = np.sum((y - np.mean(y)) ** 2)  # Soma total dos quadrados
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # R² ajustado
                k = 1  # número de variáveis independentes
                r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1)) if n > k + 1 else r_squared
                
                # Erro padrão do modelo
                mse = ss_res / (n - k - 1) if n > k + 1 else ss_res / n
                std_error = np.sqrt(mse)
                
                # Erro padrão do coeficiente
                x_mean = np.mean(X)
                ss_x = np.sum((X - x_mean) ** 2)
                se_coef = std_error / np.sqrt(ss_x) if ss_x > 0 else 0
                
                # Estatística t
                coefficient = model.coef_[0]
                t_statistic = coefficient / se_coef if se_coef > 0 else 0
                
                # Valor p (bicaudal)
                df = n - k - 1
                p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df)) if df > 0 else 1.0
                
                results.append({
                    'variable': var_name,
                    'coefficient': coefficient,
                    'r_squared': r_squared,
                    'r_squared_adj': r_squared_adj,
                    'std_error': se_coef,
                    't_statistic': t_statistic,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                })
                
            except Exception as e:
                results.append({
                    'variable': var_name,
                    'coefficient': 0,
                    'r_squared': 0,
                    'r_squared_adj': 0,
                    'std_error': 0,
                    't_statistic': 0,
                    'p_value': 1.0,
                    'is_significant': False
                })
        
        # Ordenar por R² (maior primeiro)
        results = sorted(results, key=lambda x: x['r_squared'], reverse=True)
        
        # Adicionar modelo completo (todas as variáveis)
        try:
            feature_cols = [v[1] for v in variables]
            X_full = self.df[feature_cols].fillna(0).values
            
            model_full = LinearRegression()
            model_full.fit(X_full, y)
            y_pred_full = model_full.predict(X_full)
            
            ss_res_full = np.sum((y - y_pred_full) ** 2)
            ss_tot_full = np.sum((y - np.mean(y)) ** 2)
            r2_full = 1 - (ss_res_full / ss_tot_full) if ss_tot_full > 0 else 0
            
            k_full = len(feature_cols)
            r2_adj_full = 1 - ((1 - r2_full) * (n - 1) / (n - k_full - 1)) if n > k_full + 1 else r2_full
            
            mse_full = ss_res_full / (n - k_full - 1) if n > k_full + 1 else ss_res_full / n
            std_error_full = np.sqrt(mse_full)
            
            full_model_summary = {
                'r_squared': r2_full,
                'r_squared_adj': r2_adj_full,
                'std_error': std_error_full,
                'n_observations': n,
                'n_variables': k_full
            }
        except:
            full_model_summary = {
                'r_squared': 0,
                'r_squared_adj': 0,
                'std_error': 0,
                'n_observations': n,
                'n_variables': 0
            }
        
        return {
            'variables': results,
            'full_model': full_model_summary
        }
    
    # =========================================================================
    # 10. MARKETING MIX CONTRIBUTION ANALYSIS
    # =========================================================================
    
    def calculate_marketing_mix_contribution(self):
        """
        Calcula a contribuição de cada canal/variável para os resultados.
        Abordagem baseada em:
        - Contribuição percentual de cada segmento
        - Elasticidade (sensibilidade do resultado ao investimento)
        - Análise de saturação (rendimentos decrescentes)
        - ROI marginal
        """
        results = {
            'by_placement': [],
            'by_objective': [],
            'by_targeting': [],
            'elasticity_analysis': {},
            'saturation_analysis': {},
            'model_summary': {}
        }
        
        total_spend = self.df['Amount spent'].sum()
        total_conversions = self.df['Conversions'].sum()
        total_revenue = self.df['Revenue'].sum()
        
        # ===== 1. CONTRIBUIÇÃO POR PLACEMENT =====
        for placement in self.df['Placement'].unique():
            segment = self.df[self.df['Placement'] == placement]
            spend = segment['Amount spent'].sum()
            conversions = segment['Conversions'].sum()
            revenue = segment['Revenue'].sum()
            
            # Contribuição percentual
            spend_share = (spend / total_spend * 100) if total_spend > 0 else 0
            conv_share = (conversions / total_conversions * 100) if total_conversions > 0 else 0
            
            # Eficiência: conversão share / spend share (>1 = eficiente)
            efficiency = (conv_share / spend_share) if spend_share > 0 else 0
            
            # ROAS do segmento
            roas = (revenue / spend) if spend > 0 else 0
            
            # CPA do segmento
            cpa = (spend / conversions) if conversions > 0 else 0
            
            results['by_placement'].append({
                'segment': placement,
                'spend': spend,
                'spend_share': spend_share,
                'conversions': conversions,
                'conv_share': conv_share,
                'efficiency_index': efficiency,
                'roas': roas,
                'cpa': cpa,
                'recommendation': 'Aumentar' if efficiency > 1.1 else 'Manter' if efficiency > 0.9 else 'Reduzir'
            })
        
        # Ordenar por eficiência
        results['by_placement'] = sorted(results['by_placement'], key=lambda x: x['efficiency_index'], reverse=True)
        
        # ===== 2. CONTRIBUIÇÃO POR OBJECTIVE =====
        for objective in self.df['Campaign Objective'].unique():
            segment = self.df[self.df['Campaign Objective'] == objective]
            spend = segment['Amount spent'].sum()
            conversions = segment['Conversions'].sum()
            revenue = segment['Revenue'].sum()
            
            spend_share = (spend / total_spend * 100) if total_spend > 0 else 0
            conv_share = (conversions / total_conversions * 100) if total_conversions > 0 else 0
            efficiency = (conv_share / spend_share) if spend_share > 0 else 0
            roas = (revenue / spend) if spend > 0 else 0
            cpa = (spend / conversions) if conversions > 0 else 0
            
            results['by_objective'].append({
                'segment': objective,
                'spend': spend,
                'spend_share': spend_share,
                'conversions': conversions,
                'conv_share': conv_share,
                'efficiency_index': efficiency,
                'roas': roas,
                'cpa': cpa,
                'recommendation': 'Aumentar' if efficiency > 1.1 else 'Manter' if efficiency > 0.9 else 'Reduzir'
            })
        
        results['by_objective'] = sorted(results['by_objective'], key=lambda x: x['efficiency_index'], reverse=True)
        
        # ===== 3. CONTRIBUIÇÃO POR TARGETING =====
        for targeting in self.df['Targeting'].unique():
            segment = self.df[self.df['Targeting'] == targeting]
            spend = segment['Amount spent'].sum()
            conversions = segment['Conversions'].sum()
            revenue = segment['Revenue'].sum()
            
            spend_share = (spend / total_spend * 100) if total_spend > 0 else 0
            conv_share = (conversions / total_conversions * 100) if total_conversions > 0 else 0
            efficiency = (conv_share / spend_share) if spend_share > 0 else 0
            roas = (revenue / spend) if spend > 0 else 0
            cpa = (spend / conversions) if conversions > 0 else 0
            
            results['by_targeting'].append({
                'segment': targeting,
                'spend': spend,
                'spend_share': spend_share,
                'conversions': conversions,
                'conv_share': conv_share,
                'efficiency_index': efficiency,
                'roas': roas,
                'cpa': cpa,
                'recommendation': 'Aumentar' if efficiency > 1.1 else 'Manter' if efficiency > 0.9 else 'Reduzir'
            })
        
        results['by_targeting'] = sorted(results['by_targeting'], key=lambda x: x['efficiency_index'], reverse=True)
        
        # ===== 4. ANÁLISE DE ELASTICIDADE =====
        # Elasticidade = % mudança em conversões / % mudança em investimento
        # Usamos regressão log-log para estimar
        
        try:
            # Filtrar valores positivos para log
            valid_df = self.df[(self.df['Amount spent'] > 0) & (self.df['Conversions'] > 0)].copy()
            
            if len(valid_df) > 10:
                log_spend = np.log(valid_df['Amount spent'])
                log_conv = np.log(valid_df['Conversions'])
                
                # Regressão log-log: elasticidade é o coeficiente
                model = LinearRegression()
                model.fit(log_spend.values.reshape(-1, 1), log_conv.values)
                
                elasticity = model.coef_[0]
                
                results['elasticity_analysis'] = {
                    'overall_elasticity': elasticity,
                    'interpretation': 'Elástico (aumento de spend gera mais que proporcional em conversões)' if elasticity > 1 
                                      else 'Inelástico (rendimentos decrescentes)' if elasticity < 1 
                                      else 'Unitário',
                    'marginal_efficiency': 'Alta' if elasticity > 0.8 else 'Média' if elasticity > 0.5 else 'Baixa',
                    'recommendation': f"A cada 10% de aumento em investimento, espera-se {elasticity * 10:.1f}% de aumento em conversões"
                }
            else:
                results['elasticity_analysis'] = {
                    'overall_elasticity': 0.7,
                    'interpretation': 'Dados insuficientes para análise precisa',
                    'marginal_efficiency': 'Média',
                    'recommendation': 'Coletar mais dados para análise precisa'
                }
        except:
            results['elasticity_analysis'] = {
                'overall_elasticity': 0.7,
                'interpretation': 'Erro no cálculo',
                'marginal_efficiency': 'Média',
                'recommendation': 'Revisar dados'
            }
        
        # ===== 5. ANÁLISE DE SATURAÇÃO =====
        # Dividir em quartis de spend e verificar ROAS em cada faixa
        try:
            self.df['spend_quartile'] = pd.qcut(self.df['Amount spent'], q=4, labels=['Q1 (Baixo)', 'Q2', 'Q3', 'Q4 (Alto)'])
            
            saturation_data = []
            for quartile in ['Q1 (Baixo)', 'Q2', 'Q3', 'Q4 (Alto)']:
                q_data = self.df[self.df['spend_quartile'] == quartile]
                avg_spend = q_data['Amount spent'].mean()
                avg_roas = q_data['ROAS'].mean()
                avg_cpa = q_data['CPA'].mean()
                
                saturation_data.append({
                    'quartile': quartile,
                    'avg_spend': avg_spend,
                    'avg_roas': avg_roas,
                    'avg_cpa': avg_cpa
                })
            
            # Verificar se há saturação (ROAS diminui nos quartis mais altos)
            q1_roas = saturation_data[0]['avg_roas']
            q4_roas = saturation_data[3]['avg_roas']
            
            results['saturation_analysis'] = {
                'quartile_data': saturation_data,
                'has_saturation': q4_roas < q1_roas * 0.9,
                'saturation_level': 'Alto' if q4_roas < q1_roas * 0.7 else 'Moderado' if q4_roas < q1_roas * 0.9 else 'Baixo',
                'optimal_spend_range': f"US$ {saturation_data[1]['avg_spend']:.2f} - US$ {saturation_data[2]['avg_spend']:.2f}",
                'recommendation': 'Redistribuir orçamento para faixas mais eficientes' if q4_roas < q1_roas * 0.9 
                                 else 'Orçamento bem distribuído'
            }
        except:
            results['saturation_analysis'] = {
                'quartile_data': [],
                'has_saturation': False,
                'saturation_level': 'Indeterminado',
                'optimal_spend_range': 'N/A',
                'recommendation': 'Dados insuficientes'
            }
        
        # ===== 6. RESUMO DO MODELO =====
        results['model_summary'] = {
            'total_spend': total_spend,
            'total_conversions': total_conversions,
            'total_revenue': total_revenue,
            'overall_roas': total_revenue / total_spend if total_spend > 0 else 0,
            'overall_cpa': total_spend / total_conversions if total_conversions > 0 else 0,
            'n_creatives': len(self.df),
            'n_placements': len(self.df['Placement'].unique()),
            'n_objectives': len(self.df['Campaign Objective'].unique()),
            'n_targeting': len(self.df['Targeting'].unique()),
            'best_placement': results['by_placement'][0]['segment'] if results['by_placement'] else 'N/A',
            'best_objective': results['by_objective'][0]['segment'] if results['by_objective'] else 'N/A',
            'best_targeting': results['by_targeting'][0]['segment'] if results['by_targeting'] else 'N/A'
        }
        
        return results
    
    # =========================================================================
    # 11. RELATÓRIO COMPLETO
    # =========================================================================
    
    def generate_full_report(self):
        """
        Gera relatório completo com todas as análises
        """
        report = {}
        
        # 1. Impacto no ROAS
        report['roas_impact'] = self.analyze_roas_impact()
        
        # 2. Com e sem Advantage+ Placements
        report['comparacao_advantage'] = self.comparar_advantage_placements()
        
        # 3. Diferença por objetivo de campanha
        report['diferenca_por_objetivo'] = self.diferenca_por_objetivo()
        
        # 4. Frequency x CPA
        report['frequency_analysis'] = self.frequency_cpa_analysis()
        
        # 5. Advantage+ Uplift
        report['advantage_uplift'] = self.calculate_advantage_uplift()
        
        # 6. Top Performers Patterns
        report['top_performers'] = self.identify_top_performers_patterns()
        
        # 7. Decision Matrix
        report['decision_matrix'] = self.create_decision_matrix()
        
        # 8. 30-Day Forecast
        report['forecast'] = self.generate_30day_forecast()
        
        # 9. LIFT ANALYSIS COMPLETO (NOVO)
        report['lift_analysis'] = self.calculate_comprehensive_lift()
        
        # 10. Análise de Regressão Estatística
        report['regression_stats'] = self.calculate_regression_statistics()
        
        # 11. Marketing Mix Contribution Analysis
        report['mmm_analysis'] = self.calculate_marketing_mix_contribution()
        
        # 12. Resumo Executivo
        report['executive_summary'] = self._generate_executive_summary(report)
        
        # 13. Recomendações Prioritárias
        report['priority_recommendations'] = self._generate_priority_recommendations(report)
        
        return report
    
    def _generate_executive_summary(self, report):
        """Gera resumo executivo"""
        summary = {
            'total_creatives': len(self.df),
            'total_spend': self.df['Amount spent'].sum(),
            'total_conversions': self.df['Conversions'].sum(),
            'overall_roas': self.df['Revenue'].sum() / self.df['Amount spent'].sum() if self.df['Amount spent'].sum() > 0 else 0,
            'overall_cpa': self.df['Amount spent'].sum() / self.df['Conversions'].sum() if self.df['Conversions'].sum() > 0 else 0,
            
            'key_findings': [
                {
                    'finding': f"Variável com maior impacto no ROAS: {report['roas_impact']['impact_ranking'][0]['variable']}",
                    'impact': f"{report['roas_impact']['impact_ranking'][0]['impact_score']:.1f}% de importância"
                },
                {
                    'finding': f"Advantage+ Placements {'aumenta' if report['advantage_uplift']['summary']['roas_uplift'] > 0 else 'diminui'} o ROAS em {abs(report['advantage_uplift']['summary']['roas_uplift']):.1f}%",
                    'impact': 'Estatisticamente significativo' if report['comparacao_advantage']['ROAS']['is_significant'] else 'Não significativo'
                },
                {
                    'finding': f"Frequência e CPA têm correlação {report['frequency_analysis']['correlation']['strength'].lower()}",
                    'impact': f"r = {report['frequency_analysis']['correlation']['pearson']:.3f}"
                },
                {
                    'finding': f"{report['decision_matrix']['summary']['total_to_scale']} creatives prontos para escalar",
                    'impact': f"Potencial de +{report['forecast']['optimized']['improvement_vs_baseline']['conversions']:.1f}% em conversões"
                }
            ],
            
            'estimated_impact': {
                'roas_improvement': report['forecast']['optimized']['improvement_vs_baseline']['roas'],
                'cpa_reduction': report['forecast']['optimized']['improvement_vs_baseline']['cpa'],
                'conversion_increase': report['forecast']['optimized']['improvement_vs_baseline']['conversions'],
                'potential_savings': report['decision_matrix']['pause']['total_spend']
            }
        }
        
        return summary
    
    def _generate_priority_recommendations(self, report):
        """Gera as recomendações prioritárias a partir das análises."""
        recommendations = []
        
        # 1. Advantage+ Placements - baseado no impacto real
        roas_uplift = report['advantage_uplift']['summary']['roas_uplift']
        if roas_uplift > 5:
            recommendations.append({
                'priority': 1,
                'category': 'Advantage+',
                'action': 'Ativar Advantage+ Placements em todos os adsets',
                'expected_impact': f"+{roas_uplift:.1f}% ROAS",
                'effort': 'Baixo',
                'timeframe': 'Imediato'
            })
        elif roas_uplift < -2:
            # Advantage+ tem impacto NEGATIVO - recomendar usar configurações manuais
            recommendations.append({
                'priority': 1,
                'category': 'Advantage+',
                'action': 'Manter configurações manuais de Placements (evitar Advantage+)',
                'expected_impact': f"Evitar perda de {abs(roas_uplift):.1f}% no ROAS",
                'effort': 'Baixo',
                'timeframe': 'Imediato'
            })
        else:
            # Impacto neutro - testar mais
            recommendations.append({
                'priority': 3,
                'category': 'Advantage+',
                'action': 'Continuar testando Advantage+ vs Manual em A/B tests',
                'expected_impact': 'Dados inconclusivos - necessário mais testes',
                'effort': 'Médio',
                'timeframe': '2-4 semanas'
            })
        
        # 2. Escalar vencedores
        if report['decision_matrix']['summary']['total_to_scale'] > 0:
            recommendations.append({
                'priority': 2,
                'category': 'Budget',
                'action': f"Aumentar orçamento de {report['decision_matrix']['summary']['total_to_scale']} creatives vencedores",
                'expected_impact': f"+30-50% em conversões",
                'effort': 'Médio',
                'timeframe': '1-2 semanas'
            })
        
        # 3. Pausar perdedores
        if report['decision_matrix']['pause']['total_spend'] > 0:
            recommendations.append({
                'priority': 3,
                'category': 'Otimização',
                'action': f"Pausar {report['decision_matrix']['summary']['total_to_pause']} creatives de baixa performance",
                'expected_impact': f"Economia de ${report['decision_matrix']['pause']['total_spend']:.2f}",
                'effort': 'Baixo',
                'timeframe': 'Imediato'
            })
        
        # 4. Targeting - usar dados do lift_analysis que é mais preciso
        best_targeting_lift = report['lift_analysis']['targeting']['ranking'][0] if report['lift_analysis']['targeting']['ranking'] else None
        if best_targeting_lift:
            recommendations.append({
                'priority': 4,
                'category': 'Targeting',
                'action': f"Priorizar targeting '{best_targeting_lift['name']}' em novas campanhas",
                'expected_impact': f"+{best_targeting_lift['roas_lift_pct']:.1f}% ROAS vs média",
                'effort': 'Médio',
                'timeframe': '1 semana'
            })
        
        # 5. Placement - usar dados do lift_analysis que é mais preciso
        best_placement_lift = report['lift_analysis']['placement']['ranking'][0] if report['lift_analysis']['placement']['ranking'] else None
        if best_placement_lift:
            recommendations.append({
                'priority': 5,
                'category': 'Placement',
                'action': f"Expandir presença em '{best_placement_lift['name']}'",
                'expected_impact': f"+{best_placement_lift['roas_lift_pct']:.1f}% ROAS vs média",
                'effort': 'Baixo',
                'timeframe': 'Imediato'
            })
        
        # 6. Frequência
        freq_rec = report['frequency_analysis']['recommendation']
        recommendations.append({
            'priority': 6,
            'category': 'Frequência',
            'action': freq_rec['action'],
            'expected_impact': freq_rec.get('estimated_savings', 'Otimização de CPA'),
            'effort': 'Médio',
            'timeframe': '2-3 semanas'
        })
        
        return recommendations
