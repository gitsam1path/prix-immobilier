import json
from pathlib import Path

import numpy as np
import pytest

from immobilier.data import TARGET, TEST_YEAR, feature_types, load, load_raw, split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load()


def test_nettoyage(df):
    raw = load_raw()
    assert len(raw) - len(df) == 5               # les 5 maisons de plus de 4 000 pieds² sont écartées
    assert df["Gr Liv Area"].max() <= 4000
    assert (df.loc[df["Pool Area"] == 0, "Pool QC"] == 0).all()  # « pas de piscine » -> note 0, pas une valeur manquante
    assert df["Garage Yr Blt"].max() <= 2010     # l'année 2207 (erreur de saisie) est corrigée


def test_decoupage_temporel(df):
    train, test = split(df)
    assert train["Yr Sold"].max() < TEST_YEAR and (test["Yr Sold"] == TEST_YEAR).all()
    assert len(train) + len(test) == len(df)


def test_pas_de_fuite_de_la_vente(df):
    """Les informations connues seulement au moment de la vente ne servent pas à estimer le prix."""
    numeric, categorical = feature_types(df)
    used = set(numeric + categorical)
    for col in (TARGET, "Sale Type", "Sale Condition", "Yr Sold", "Mo Sold", "PID", "Order"):
        assert col not in used


def test_variables_metier(df):
    row = df.iloc[0]
    assert row["surface_totale"] == row["Gr Liv Area"] + row["Total Bsmt SF"]
    assert (df["age_a_la_vente"] >= -1).all()    # une maison peut être vendue l'année avant sa fin de construction


def test_rapport():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("lancer d'abord : cd src && python -m immobilier.train")
    r = json.loads(path.read_text(encoding="utf-8"))
    res = {m["modele"]: m for m in r["resultats"]}
    best = res[r["modele_retenu"]]
    assert best["test_mediane_ecart_pct"] < res["Référence (prix au m² du quartier)"]["test_mediane_ecart_pct"]
    assert 80 <= r["intervalle_90"]["couverture_test_pct"] <= 100
