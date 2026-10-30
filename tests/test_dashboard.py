"""O app monta o layout sem erro."""
from campanhas.dashboard.app import app


def test_layout_monta():
    assert app.layout is not None
