"""
Ventes immobilières d'Ames (Iowa), 2006-2010 : 2 930 maisons, 80 caractéristiques (De Cock, 2011).

Nettoyage :
- les valeurs manquantes qui signifient « absent » (pas de piscine, pas de garage…) deviennent « Aucun » ;
- les 5 maisons de plus de 4 000 pieds² sont écartées, comme le recommande l'auteur des données ;
- des variables métier sont ajoutées (surface totale, âge, nombre de salles de bains…).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ames_ventes.csv"

M2_PAR_PIED2 = 0.092903
TEST_YEAR = 2010  # on entraîne sur 2006-2009 et on prévoit les ventes de 2010

# variables dont l'absence de valeur signifie « l'équipement n'existe pas »
ABSENCE = ["Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu", "Garage Type", "Garage Finish",
           "Garage Qual", "Garage Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
           "BsmtFin Type 2", "Mas Vnr Type"]
ZERO_IF_ABSENT = ["Mas Vnr Area", "Bsmt Full Bath", "Bsmt Half Bath", "BsmtFin SF 1", "BsmtFin SF 2",
                  "Bsmt Unf SF", "Total Bsmt SF", "Garage Cars", "Garage Area"]

# échelles de qualité : texte -> note
QUALITY = {"Aucun": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = ["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual",
                "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC"]

TARGET = "SalePrice"
DROP = ["Order", "PID", TARGET, "Yr Sold", "Mo Sold", "Sale Type", "Sale Condition"]
# Le type et les conditions de vente ne sont connus qu'au moment de la vente : on ne les utilise pas pour estimer un bien.


def load_raw(path: Path = RAW) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Gr Liv Area"] <= 4000]  # recommandation de De Cock : 5 maisons atypiques
    for c in ABSENCE:
        df[c] = df[c].fillna("Aucun")
    for c in ZERO_IF_ABSENT:
        df[c] = df[c].fillna(0)
    df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])
    df.loc[df["Garage Yr Blt"] > 2010, "Garage Yr Blt"] = df["Year Built"]  # une erreur de saisie connue (2207)
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    df["MS SubClass"] = df["MS SubClass"].astype(str)  # code de type de logement, pas une quantité
    for c in QUALITY_COLS:
        df[c] = df[c].map(QUALITY).astype(float)

    # variables métier
    df["surface_totale"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["salles_de_bain"] = df["Full Bath"] + 0.5 * df["Half Bath"] + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    df["age_a_la_vente"] = df["Yr Sold"] - df["Year Built"]
    df["annees_depuis_renovation"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["renove"] = (df["Year Remod/Add"] > df["Year Built"]).astype(int)
    df["surface_exterieure"] = df[["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]].sum(axis=1)
    df["qualite_x_surface"] = df["Overall Qual"] * df["Gr Liv Area"]
    return df.reset_index(drop=True)


def load() -> pd.DataFrame:
    return clean(load_raw())


def split(df: pd.DataFrame):
    """Découpage temporel : on apprend sur le passé (2006-2009) et on prévoit le futur (2010)."""
    train, test = df[df["Yr Sold"] < TEST_YEAR], df[df["Yr Sold"] == TEST_YEAR]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c not in DROP]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in features if c not in categorical]
    return numeric, categorical
