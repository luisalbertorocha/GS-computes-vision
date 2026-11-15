"""Score de severidade explicável.

    score = W_AREA * area_normalizada + W_CONF * confianca + W_CLASSE * peso_da_classe

Tudo fica em [0, 1]. Os pesos e as faixas abaixo são os documentados no relatório;
se mudar aqui, mude lá. Casos de severidade "alta" devem ser validados por uma pessoa.
"""
from __future__ import annotations

W_AREA = 0.5
W_CONF = 0.2
W_CLASSE = 0.3

# Área (cm²) a partir da qual area_normalizada satura em 1.0 (1 m² de pavimento danificado).
AREA_REF_CM2 = 10000.0

CLASS_WEIGHTS = {
    "buraco": 1.0,
    "calcada_quebrada": 0.7,
    "trinca": 0.4,
}
DEFAULT_CLASS_WEIGHT = 0.5

# score < 0.40 -> baixa; < 0.65 -> média; senão alta
THRESHOLD_MEDIA = 0.40
THRESHOLD_ALTA = 0.65


def severity_score(area_cm2: float, confianca: float, classe: str) -> float:
    area_norm = min(max(area_cm2, 0.0) / AREA_REF_CM2, 1.0)
    conf = min(max(confianca, 0.0), 1.0)
    weight = CLASS_WEIGHTS.get(classe, DEFAULT_CLASS_WEIGHT)
    return round(W_AREA * area_norm + W_CONF * conf + W_CLASSE * weight, 4)


def severity_label(score: float) -> str:
    if score >= THRESHOLD_ALTA:
        return "alta"
    if score >= THRESHOLD_MEDIA:
        return "média"
    return "baixa"
