"""Os números do painel e os da análise avançada saem da mesma base e batem entre si."""
import pytest

from campanhas.advanced_analysis import MarketingScienceAnalyzer
from campanhas.data_processor import get_summary_metrics, load_and_process_data
from campanhas.paths import DADOS


@pytest.fixture(scope="module")
def df():
    return load_and_process_data(DADOS)


@pytest.fixture(scope="module")
def relatorio(df):
    return MarketingScienceAnalyzer(df).generate_full_report()


def test_base_carrega(df):
    assert len(df) == 1000
    assert {"ROAS", "CPA", "Revenue"} <= set(df.columns)


def test_roas_do_painel_bate_com_a_base(df):
    metricas = get_summary_metrics(df)
    assert metricas["overall_roas"] == pytest.approx(df["Revenue"].sum() / df["Amount spent"].sum())


def test_relatorio_completo_tem_as_secoes(relatorio):
    for secao in ("lift_analysis", "advantage_uplift", "priority_recommendations"):
        assert secao in relatorio
