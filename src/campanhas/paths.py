"""Caminhos do projeto, resolvidos a partir da raiz (funciona de qualquer pasta)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data" / "raw" / "meta_creatives_sample.csv"
RELATORIOS = RAIZ / "reports"
