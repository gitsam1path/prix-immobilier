"""
Estimer le prix de vente d'une maison.

- 6 modèles comparés par validation croisée sur les ventes 2006-2009, puis testés sur les ventes de 2010 ;
- la cible est le LOGARITHME du prix : une erreur de 10 % pèse autant sur une maison à 100 k$ qu'à 500 k$ ;
- intervalles de prix à 90 % par prédiction conforme ;
- interprétation : ce que vaut une note de qualité, un m², un garage… (coefficients du Lasso) et importance des variables.

Lancer depuis src/ :  python -m immobilier.train
"""

from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import M2_PAR_PIED2, ROOT, TARGET, feature_types, load, split

warnings.filterwarnings("ignore")
REPORTS, FIG, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
SEED = 42
ALPHA = 0.10  # intervalles à 90 %

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1a73e8", "#e8710a", "#188038", "#d93025", "#9aa0a6", "#8e6bbf"


class NeighborhoodMedian:
    """Référence : le prix médian au m² du quartier × la surface habitable."""

    def fit(self, X, y):
        price_m2 = np.exp(y) / X["Gr Liv Area"]
        self.by_hood = price_m2.groupby(X["Neighborhood"]).median()
        self.default = price_m2.median()
        return self

    def predict(self, X):
        p = X["Neighborhood"].map(self.by_hood).fillna(self.default).to_numpy()
        return np.log(p * X["Gr Liv Area"].to_numpy())

    def get_params(self, deep=False):
        return {}

    def set_params(self, **p):
        return self


def linear_prep(numeric, categorical):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical),
    ])


def tree_prep(numeric, categorical):
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), categorical),
    ])


def models(numeric, categorical) -> dict:
    simple = ["Gr Liv Area", "Overall Qual", "Year Built", "Total Bsmt SF", "Garage Cars"]
    return {
        "Référence (prix au m² du quartier)": NeighborhoodMedian(),
        "Régression linéaire (5 variables)": ColumnTransformer([("num", "passthrough", simple)]),
        "Ridge (toutes les variables)": make_pipeline(linear_prep(numeric, categorical), RidgeCV(alphas=np.logspace(-1, 3, 30))),
        "Lasso (toutes les variables)": make_pipeline(linear_prep(numeric, categorical),
                                                      LassoCV(alphas=np.logspace(-4, -1, 30), max_iter=20000, random_state=SEED)),
        "Forêt aléatoire": make_pipeline(tree_prep(numeric, categorical),
                                         RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.33,
                                                               n_jobs=-1, random_state=SEED)),
        "LightGBM": make_pipeline(tree_prep(numeric, categorical),
                                  lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.01, num_leaves=15, min_child_samples=10,
                                                    subsample=0.7, subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0,
                                                    random_state=SEED, verbose=-1)),
    }


def _model(name, obj):
    if name.startswith("Régression linéaire"):
        return make_pipeline(obj, LinearRegression())
    return obj


def metrics(y_log, pred_log) -> dict:
    price, pred = np.exp(y_log), np.exp(pred_log)
    return {"rmse_log": float(np.sqrt(np.mean((y_log - pred_log) ** 2))),
            "mae_dollars": float(np.mean(np.abs(price - pred))),
            "mape": float(np.mean(np.abs(price - pred) / price) * 100),
            "mediane_ecart_pct": float(np.median(np.abs(price - pred) / price) * 100)}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for d in (FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    df = load()
    train, test = split(df)
    numeric, categorical = feature_types(df)
    X, y = train[numeric + categorical], np.log(train[TARGET])
    Xt, yt = test[numeric + categorical], np.log(test[TARGET])
    print(f"{len(df)} ventes ({len(train)} en 2006-2009, {len(test)} en 2010) · {len(numeric)} variables numériques, "
          f"{len(categorical)} catégorielles")

    cv = KFold(5, shuffle=True, random_state=SEED)
    rows, oof, test_preds, fitted = [], {}, {}, {}
    for name, obj in models(numeric, categorical).items():
        m = _model(name, obj)
        if isinstance(m, NeighborhoodMedian):
            p = np.zeros(len(X))
            for tr, va in cv.split(X):
                p[va] = NeighborhoodMedian().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
            final = NeighborhoodMedian().fit(X, y)
        else:
            p = cross_val_predict(clone(m), X, y, cv=cv)
            final = clone(m).fit(X, y)
        oof[name], fitted[name] = p, final
        test_preds[name] = final.predict(Xt)
        rows.append({"modele": name, **{f"cv_{k}": v for k, v in metrics(y, p).items()},
                     **{f"test_{k}": v for k, v in metrics(yt, test_preds[name]).items()}})
        print(f"  {name:36s} CV erreur médiane {rows[-1]['cv_mediane_ecart_pct']:5.1f} % · "
              f"test : RMSE(log) {rows[-1]['test_rmse_log']:.3f}, erreur médiane {rows[-1]['test_mediane_ecart_pct']:5.1f} %")

    # Combinaison : moyenne d'un modèle linéaire et d'un modèle d'arbres (erreurs différentes, elles se compensent)
    duo = ["Lasso (toutes les variables)", "LightGBM"]
    name = "Combinaison Lasso + LightGBM"
    oof[name] = np.mean([oof[m] for m in duo], axis=0)
    test_preds[name] = np.mean([test_preds[m] for m in duo], axis=0)
    rows.append({"modele": name, **{f"cv_{k}": v for k, v in metrics(y, oof[name]).items()},
                 **{f"test_{k}": v for k, v in metrics(yt, test_preds[name]).items()}})
    table = pd.DataFrame(rows).sort_values("cv_rmse_log").reset_index(drop=True)
    best = table.iloc[0]["modele"]
    print(f"Modèle retenu (meilleure validation croisée) : {best}")

    # ─── Intervalles de prix par prédiction conforme ───
    # résidus hors échantillon de la validation croisée -> quantile à 90 % -> intervalle autour de chaque prévision
    resid = np.abs(y - oof[best])
    n = len(resid)
    q = float(np.quantile(resid, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    low, high = np.exp(test_preds[best] - q), np.exp(test_preds[best] + q)
    price_t = np.exp(yt)
    coverage = float(np.mean((price_t >= low) & (price_t <= high)) * 100)
    print(f"Intervalle à 90 % : prix prévu ×/÷ {np.exp(q):.2f} ; couverture sur 2010 : {coverage:.1f} %")

    # ─── Interprétation ───
    lasso = fitted["Lasso (toutes les variables)"]
    names = lasso[0].get_feature_names_out()
    coefs = pd.Series(lasso[-1].coef_, index=names)
    n_kept = int((coefs != 0).sum())
    # Effet de chaque caractéristique, par simulation « et si… ? » : on modifie une caractéristique de chaque maison
    # de 2010 (en mettant à jour les variables qui en dépendent) et on mesure la variation du prix estimé.
    def predict_best(data):
        if best == name:
            return np.mean([fitted[m].predict(data) for m in duo], axis=0)
        return fitted[best].predict(data)

    pied2 = 10 / M2_PAR_PIED2  # 10 m² en pieds²

    def plus_surface(d):
        d["Gr Liv Area"] += pied2
        d["surface_totale"] += pied2
        d["qualite_x_surface"] += d["Overall Qual"] * pied2

    def plus_sous_sol(d):
        d["Total Bsmt SF"] += pied2
        d["Bsmt Unf SF"] += pied2
        d["surface_totale"] += pied2

    def plus_qualite(d):
        d["Overall Qual"] = (d["Overall Qual"] + 1).clip(upper=10)
        d["qualite_x_surface"] = d["Overall Qual"] * d["Gr Liv Area"]

    def plus_garage(d):
        d["Garage Cars"] += 1
        d["Garage Area"] += 250

    def plus_age(d):
        d["Year Built"] -= 10
        d["age_a_la_vente"] += 10

    def plus_cuisine(d):
        d["Kitchen Qual"] = (d["Kitchen Qual"] + 1).clip(upper=5)

    def plus_cheminee(d):
        d["Fireplaces"] += 1

    base_pred = predict_best(Xt)
    effects = {}
    for label, change in [("+1 point de qualité générale (sur 10)", plus_qualite), ("+10 m² habitables", plus_surface),
                          ("+10 m² de sous-sol", plus_sous_sol), ("+1 place de garage", plus_garage),
                          ("+10 ans d'âge", plus_age), ("+1 cran de qualité de cuisine", plus_cuisine),
                          ("+1 cheminée", plus_cheminee)]:
        d = Xt.copy()
        change(d)
        effects[label] = float(np.median(np.exp(predict_best(d) - base_pred) - 1) * 100)

    lgbm = fitted["LightGBM"]
    sample = Xt
    imp = permutation_importance(lgbm, sample, yt, n_repeats=8, random_state=SEED, n_jobs=-1,
                                 scoring="neg_root_mean_squared_error")
    importance = pd.Series(imp.importances_mean, index=sample.columns).sort_values(ascending=False)

    # ─── Analyse des erreurs ───
    err = pd.DataFrame({"prix": price_t, "prevu": np.exp(test_preds[best]), "quartier": test["Neighborhood"],
                        "condition": test["Sale Condition"]})
    err["ecart_pct"] = (err["prevu"] - err["prix"]) / err["prix"] * 100
    err["tranche"] = pd.qcut(err["prix"], 4, labels=["1er quart (moins chères)", "2e quart", "3e quart", "4e quart (plus chères)"])
    by_band = err.groupby("tranche", observed=True)["ecart_pct"].agg(lambda s: float(np.median(np.abs(s)))).round(1)
    by_cond = err.groupby("condition")["ecart_pct"].agg(["count", lambda s: float(np.median(np.abs(s)))]).round(1)
    by_cond.columns = ["ventes", "erreur_mediane_pct"]

    # ─── Figures ───
    fig_prix(df)
    fig_comparaison(table)
    fig_reel_prevu(price_t, np.exp(test_preds[best]), low, high, best)
    fig_effets(effects)
    fig_importance(importance.head(12))
    fig_erreurs(err)

    joblib.dump({"modeles": {m: fitted[m] for m in duo} if best == name else {best: fitted[best]},
                 "combinaison": best == name, "marge_log_90": q, "variables": numeric + categorical},
                MODELS / "model.joblib")
    report = {
        "donnees": {"ventes": len(df), "entrainement_2006_2009": len(train), "test_2010": len(test),
                    "variables_numeriques": len(numeric), "variables_categorielles": len(categorical),
                    "prix_median": float(df[TARGET].median())},
        "modele_retenu": best,
        "resultats": json.loads(table.round(4).to_json(orient="records", force_ascii=False)),
        "intervalle_90": {"facteur": round(float(np.exp(q)), 3), "couverture_test_pct": round(coverage, 1),
                          "largeur_mediane_dollars": round(float(np.median(high - low)), 0)},
        "lasso": {"variables_conservees": n_kept, "variables_totales": len(coefs)},
        "effets_simulation_pct": {k: round(v, 1) for k, v in effects.items()},
        "importance_lightgbm": {k: round(float(v), 4) for k, v in importance.head(12).items()},
        "erreur_par_tranche_de_prix": by_band.to_dict(),
        "erreur_par_condition_de_vente": by_cond.to_dict(orient="index"),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table[["modele", "cv_rmse_log", "test_rmse_log", "test_mae_dollars", "test_mediane_ecart_pct"]].round(3).to_string(index=False))
    print("Effets simulés :", {k: round(v, 1) for k, v in effects.items()})
    print("Erreur par condition de vente :", by_cond.to_dict(orient="index"))


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────
def fig_prix(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(df[TARGET] / 1000, bins=50, color=BLUE)
    axes[0].set_title("Prix de vente : une distribution asymétrique")
    axes[0].set_xlabel("Prix (k$)")
    axes[1].hist(np.log(df[TARGET]), bins=50, color=GREEN)
    axes[1].set_title("…presque symétrique en logarithme")
    axes[1].set_xlabel("log(prix)")
    _save(fig, "distribution_prix")


def fig_comparaison(table):
    t = table.sort_values("test_mediane_ecart_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [RED if m.startswith("Combinaison") else GREY if m.startswith(("Référence", "Régression")) else BLUE for m in t["modele"]]
    bars = ax.barh(t["modele"], t["test_mediane_ecart_pct"], color=colors)
    for b, (_, r) in zip(bars, t.iterrows()):
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
                f"{r['test_mediane_ecart_pct']:.1f} %  (erreur moyenne {r['test_mae_dollars'] / 1000:.1f} k$)", va="center", fontsize=8.5)
    ax.set_xlim(0, t["test_mediane_ecart_pct"].max() * 1.5)
    ax.set_xlabel("Erreur médiane sur les ventes de 2010 (%)")
    ax.set_title("Estimer le prix d'une maison : comparaison des modèles")
    ax.grid(axis="y", visible=False)
    _save(fig, "comparaison_modeles")


def fig_reel_prevu(price, pred, low, high, name):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    order = np.argsort(pred)
    ax.vlines(pred[order] / 1000, low[order] / 1000, high[order] / 1000, color=BLUE, alpha=0.12, lw=2)
    inside = (price >= low) & (price <= high)
    ax.scatter(pred[inside] / 1000, price[inside] / 1000, s=10, color=BLUE, label="Prix réel dans l'intervalle")
    ax.scatter(pred[~inside] / 1000, price[~inside] / 1000, s=14, color=RED, label="Prix réel hors intervalle")
    lim = max(price.max(), pred.max()) / 1000 * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=1, ls="--")
    ax.set_xlabel("Prix estimé (k$)")
    ax.set_ylabel("Prix de vente réel (k$)")
    ax.set_title(f"Ventes de 2010 : estimation et intervalle à 90 %\n({name})", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    _save(fig, "reel_vs_estime")


def fig_effets(effects):
    s = pd.Series(effects).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(s.index, s.values, color=[GREEN if v > 0 else RED for v in s.values])
    for i, v in enumerate(s.values):
        ax.text(v + 0.15 if v >= 0 else 0.15, i, f"{v:+.1f} %", va="center", ha="left", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Variation médiane du prix estimé (simulation sur les maisons vendues en 2010)")
    ax.set_title("Ce que vaut chaque caractéristique")
    ax.set_xlim(min(s.min() * 1.6, -2), s.max() * 1.35)
    ax.grid(axis="y", visible=False)
    _save(fig, "effets_caracteristiques")


def fig_importance(imp):
    s = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(s.index, s.values, color=BLUE)
    ax.set_xlabel("Hausse de l'erreur si l'on brouille la variable (LightGBM)")
    ax.set_title("Les variables qui comptent le plus")
    ax.grid(axis="y", visible=False)
    _save(fig, "importance_variables")


def fig_erreurs(err):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    conds = err.groupby("condition")["ecart_pct"].median().sort_values().index
    data = [err.loc[err["condition"] == c, "ecart_pct"] for c in conds]
    labels = {"Normal": "Normale", "Partial": "Neuve (partielle)", "Abnorml": "Anormale (saisie, vente forcée)",
              "Family": "Entre membres de la famille", "Alloca": "Allocation", "AdjLand": "Terrain adjacent"}
    ax.boxplot(data, vert=False, tick_labels=[f"{labels.get(c, c)} ({len(d)})" for c, d in zip(conds, data)],
               showfliers=False, patch_artist=True, boxprops=dict(facecolor="#e8f0fe"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Écart estimation − prix réel (%) ; à droite : maison surestimée")
    ax.set_title("Où le modèle se trompe : selon les conditions de la vente")
    ax.grid(axis="y", visible=False)
    _save(fig, "erreurs_conditions_vente")


if __name__ == "__main__":
    main()
