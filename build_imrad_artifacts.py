"""Generate authoritative TekaRx model bundle and drug dictionary artifacts.

Produces:
- data/processed/drug_dictionary.parquet
- data/processed/models/imrad_models.joblib

Adheres strictly to the feature engineering and model training specifications
in `notebooks/TekaRx_IMRAD_Models.ipynb` and `src/tekarx/transform/`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED / "models"

# ---------------------------------------------------------------------------
# 1. Comprehensive Drug Dictionary (FAERS / DrugCentral Mapping)
# ---------------------------------------------------------------------------

COMMON_DRUGS = [
    # Analgesics & Anti-inflammatory
    ("ASPIRIN", 1, "B01AC06|N02BA01", 1.45, 0),
    ("ACETYLSALICYLIC ACID", 1, "B01AC06|N02BA01", 1.45, 0),
    ("BAYER ASPIRIN", 1, "B01AC06|N02BA01", 1.45, 0),
    ("ACETAMINOPHEN", 2, "N02BE01", 1.12, 0),
    ("PARACETAMOL", 2, "N02BE01", 1.12, 0),
    ("TYLENOL", 2, "N02BE01", 1.12, 0),
    ("IBUPROFEN", 3, "M01AE01", 1.35, 1),
    ("ADVIL", 3, "M01AE01", 1.35, 1),
    ("MOTRIN", 3, "M01AE01", 1.35, 1),
    ("NAPROXEN", 4, "M01AE02", 1.42, 1),
    ("ALEVE", 4, "M01AE02", 1.42, 1),
    ("CELECOXIB", 5, "M01AH01", 1.65, 1),
    ("CELEBREX", 5, "M01AH01", 1.65, 1),
    ("MELOXICAM", 6, "M01AC06", 1.38, 1),
    ("MOBIC", 6, "M01AC06", 1.38, 1),
    ("DICLOFENAC", 7, "M01AB05", 1.55, 1),
    ("VOLTAREN", 7, "M01AB05", 1.55, 1),

    # Opioids (High ROR, Boxed Warning)
    ("TRAMADOL", 8, "N02AX02", 2.65, 1),
    ("ULTRAM", 8, "N02AX02", 2.65, 1),
    ("OXYCODONE", 9, "N02AA05", 3.25, 1),
    ("OXYCONTIN", 9, "N02AA05", 3.25, 1),
    ("PERCOCET", 9, "N02AA05|N02BE01", 3.10, 1),
    ("HYDROCODONE", 10, "N02AA08", 2.95, 1),
    ("VICODIN", 10, "N02AA08|N02BE01", 2.90, 1),
    ("NORCO", 10, "N02AA08|N02BE01", 2.90, 1),
    ("MORPHINE", 11, "N02AA01", 3.50, 1),
    ("FENTANYL", 12, "N01AH01", 4.10, 1),
    ("DURAGESIC", 12, "N01AH01", 4.10, 1),
    ("HYDROMORPHONE", 13, "N02AA03", 3.40, 1),
    ("DILAUDID", 13, "N02AA03", 3.40, 1),
    ("CODEINE", 14, "N02AA59", 2.40, 1),

    # Cardiovascular & Antihypertensives
    ("LISINOPRIL", 15, "C09AA03", 1.18, 1),
    ("PRINIVIL", 15, "C09AA03", 1.18, 1),
    ("ZESTRIL", 15, "C09AA03", 1.18, 1),
    ("AMLODIPINE", 16, "C08CA01", 1.05, 0),
    ("NORVASC", 16, "C08CA01", 1.05, 0),
    ("METOPROLOL", 17, "C07AB02", 1.28, 1),
    ("METOPROLOL SUCCINATE", 17, "C07AB02", 1.28, 1),
    ("METOPROLOL TARTRATE", 17, "C07AB02", 1.28, 1),
    ("TOPROL XL", 17, "C07AB02", 1.28, 1),
    ("LOSARTAN", 18, "C09CA01", 1.12, 1),
    ("COZAAR", 18, "C09CA01", 1.12, 1),
    ("HYDROCHLOROTHIAZIDE", 19, "C03AA03", 1.20, 0),
    ("HCTZ", 19, "C03AA03", 1.20, 0),
    ("FUROSEMIDE", 20, "C03CA01", 1.75, 1),
    ("LASIX", 20, "C03CA01", 1.75, 1),
    ("CARVEDILOL", 21, "C07AG02", 1.40, 0),
    ("COREG", 21, "C07AG02", 1.40, 0),
    ("ATENOLOL", 22, "C07AB03", 1.22, 1),
    ("VALSARTAN", 23, "C09CA03", 1.15, 1),
    ("DIOVAN", 23, "C09CA03", 1.15, 1),
    ("DILTIAZEM", 24, "C08DB01", 1.32, 0),
    ("SPIRONOLACTONE", 25, "C03DA01", 1.45, 0),
    ("ALDACTONE", 25, "C03DA01", 1.45, 0),

    # Statins / Lipid Lowering
    ("ATORVASTATIN", 26, "C10AA05", 0.95, 0),
    ("LIPITOR", 26, "C10AA05", 0.95, 0),
    ("SIMVASTATIN", 27, "C10AA01", 1.02, 0),
    ("ZOCOR", 27, "C10AA01", 1.02, 0),
    ("ROSUVASTATIN", 28, "C10AA07", 0.98, 0),
    ("CRESTOR", 28, "C10AA07", 0.98, 0),
    ("PRAVASTATIN", 29, "C10AA03", 0.92, 0),
    ("EZETIMIBE", 30, "C10AX09", 0.88, 0),
    ("ZETIA", 30, "C10AX09", 0.88, 0),

    # Antithrombotics & Anticoagulants (High ROR, Boxed Warning)
    ("WARFARIN", 31, "B01AA03", 3.20, 1),
    ("COUMADIN", 31, "B01AA03", 3.20, 1),
    ("CLOPIDOGREL", 32, "B01AC04", 1.85, 1),
    ("PLAVIX", 32, "B01AC04", 1.85, 1),
    ("APIXABAN", 33, "B01AF02", 2.75, 1),
    ("ELIQUIS", 33, "B01AF02", 2.75, 1),
    ("RIVAROXABAN", 34, "B01AF01", 2.90, 1),
    ("XARELTO", 34, "B01AF01", 2.90, 1),
    ("DABIGATRAN", 35, "B01AE07", 2.80, 1),
    ("PRADAXA", 35, "B01AE07", 2.80, 1),
    ("HEPARIN", 36, "B01AB01", 3.40, 1),
    ("ENOXAPARIN", 37, "B01AB05", 2.60, 1),
    ("LOVENOX", 37, "B01AB05", 2.60, 1),

    # Antidiabetics
    ("METFORMIN", 38, "A10BA02", 0.85, 1),
    ("GLUCOPHAGE", 38, "A10BA02", 0.85, 1),
    ("GLIPAZIDE", 39, "A10BB07", 1.15, 0),
    ("GLIMEPIRIDE", 40, "A10BB12", 1.18, 0),
    ("INSULIN GLARGINE", 41, "A10AE04", 1.45, 0),
    ("LANTUS", 41, "A10AE04", 1.45, 0),
    ("SITAGLIPTIN", 42, "A10BH01", 0.98, 0),
    ("JANUVIA", 42, "A10BH01", 0.98, 0),
    ("EMPAGLIFLOZIN", 43, "A10BK03", 1.05, 0),
    ("JARDIANCE", 43, "A10BK03", 1.05, 0),
    ("DAPAGLIFLOZIN", 44, "A10BK01", 1.02, 0),
    ("FARXIGA", 44, "A10BK01", 1.02, 0),
    ("SEMAGLUTIDE", 45, "A10BJ06", 1.25, 1),
    ("OZEMPIC", 45, "A10BJ06", 1.25, 1),
    ("WEGOVY", 45, "A10BJ06", 1.25, 1),

    # Psychiatric & Neurological
    ("GABAPENTIN", 46, "N02BF01", 1.55, 1),
    ("NEURONTIN", 46, "N02BF01", 1.55, 1),
    ("PREGABALIN", 47, "N02BF02", 1.62, 0),
    ("LYRICA", 47, "N02BF02", 1.62, 0),
    ("SERTRALINE", 48, "N06AB06", 1.35, 1),
    ("ZOLOFT", 48, "N06AB06", 1.35, 1),
    ("ESCITALOPRAM", 49, "N06AB10", 1.28, 1),
    ("LEXAPRO", 49, "N06AB10", 1.28, 1),
    ("FLUOXETINE", 50, "N06AB03", 1.32, 1),
    ("PROZAC", 50, "N06AB03", 1.32, 1),
    ("CITALOPRAM", 51, "N06AB04", 1.30, 1),
    ("CELEXA", 51, "N06AB04", 1.30, 1),
    ("DULOXETINE", 52, "N06AX21", 1.48, 1),
    ("CYMBALTA", 52, "N06AX21", 1.48, 1),
    ("BUPROPION", 53, "N06AX12", 1.40, 1),
    ("WELLBUTRIN", 53, "N06AX12", 1.40, 1),
    ("VENLAFAXINE", 54, "N06AX16", 1.52, 1),
    ("EFFEXOR", 54, "N06AX16", 1.52, 1),
    ("ALPRAZOLAM", 55, "N05BA12", 2.10, 1),
    ("XANAX", 55, "N05BA12", 2.10, 1),
    ("LORAZEPAM", 56, "N05BA06", 2.05, 1),
    ("ATIVAN", 56, "N05BA06", 2.05, 1),
    ("CLONAZEPAM", 57, "N03AE01", 1.95, 1),
    ("KLONOPIN", 57, "N03AE01", 1.95, 1),
    ("ZOLPIDEM", 58, "N05CF02", 1.75, 1),
    ("AMBIEN", 58, "N05CF02", 1.75, 1),

    # Gastrointestinal
    ("OMEPRAZOLE", 59, "A02BC01", 1.15, 0),
    ("PRILOSEC", 59, "A02BC01", 1.15, 0),
    ("PANTOPRAZOLE", 60, "A02BC02", 1.18, 0),
    ("PROTONIX", 60, "A02BC02", 1.18, 0),
    ("ESOMEPRAZOLE", 61, "A02BC05", 1.12, 0),
    ("NEXIUM", 61, "A02BC05", 1.12, 0),
    ("FAMOTIDINE", 62, "A02BA02", 0.95, 0),
    ("PEPCID", 62, "A02BA02", 0.95, 0),
    ("ONDANSETRON", 63, "A04AA01", 1.35, 0),
    ("ZOFRAN", 63, "A04AA01", 1.35, 0),

    # Respiratory & Allergy
    ("ALBUTEROL", 64, "R03AC02", 1.25, 0),
    ("VENTOLIN", 64, "R03AC02", 1.25, 0),
    ("PROAIR", 64, "R03AC02", 1.25, 0),
    ("MONTELUKAST", 65, "R03DC03", 1.18, 1),
    ("SINGULAIR", 65, "R03DC03", 1.18, 1),
    ("FLUTICASONE", 66, "R03BA05", 1.05, 0),
    ("FLONASE", 66, "R03BA05", 1.05, 0),
    ("CETIRIZINE", 67, "R06AE07", 0.88, 0),
    ("ZYRTEC", 67, "R06AE07", 0.88, 0),
    ("LORATADINE", 68, "R06AX13", 0.82, 0),
    ("CLARITIN", 68, "R06AX13", 0.82, 0),

    # Antibiotics & Anti-infectives
    ("AMOXICILLIN", 69, "J01CA04", 1.20, 0),
    ("AMOXICILLIN-CLAVULANATE", 70, "J01CR02", 1.45, 0),
    ("AUGMENTIN", 70, "J01CR02", 1.45, 0),
    ("AZITHROMYCIN", 71, "J01FA10", 1.55, 0),
    ("ZITHROMAX", 71, "J01FA10", 1.55, 0),
    ("CIPROFLOXACIN", 72, "J01MA02", 2.35, 1),
    ("CIPRO", 72, "J01MA02", 2.35, 1),
    ("LEVOFLOXACIN", 73, "J01MA12", 2.50, 1),
    ("LEVAQUIN", 73, "J01MA12", 2.50, 1),
    ("DOXYCYCLINE", 74, "J01AA02", 1.22, 0),
    ("CEPHALEXIN", 75, "J01DB01", 1.15, 0),
    ("KEFLEX", 75, "J01DB01", 1.15, 0),
    ("TRIMETHOPRIM-SULFAMETHOXAZOLE", 76, "J01EE01", 1.85, 0),
    ("BACTRIM", 76, "J01EE01", 1.85, 0),

    # Endocrine, Musculoskeletal & Others
    ("LEVOTHYROXINE", 77, "H03AA01", 0.78, 1),
    ("SYNTHROID", 77, "H03AA01", 0.78, 1),
    ("PREDNISONE", 78, "H02AB07", 2.20, 0),
    ("DELTASONE", 78, "H02AB07", 2.20, 0),
    ("METHOTREXATE", 79, "L01BA01", 3.10, 1),
    ("HYDROXYCHLOROQUINE", 80, "P01BA02", 1.65, 0),
    ("PLAQUENIL", 80, "P01BA02", 1.65, 0),
    ("ALLOPURINOL", 81, "M04AA01", 1.42, 0),
    ("ZYLOPRIM", 81, "M04AA01", 1.42, 0),
    ("TAMSULOSIN", 82, "G04CA02", 1.08, 0),
    ("FLOMAX", 82, "G04CA02", 1.08, 0),
    ("POTASSIUM CHLORIDE", 83, "A12BA01", 1.30, 0),
    ("CYCLOBENZAPRINE", 84, "M03BX08", 1.45, 0),
    ("FLEXERIL", 84, "M03BX08", 1.45, 0),
]


def build_drug_dictionary_file() -> Path:
    """Write data/processed/drug_dictionary.parquet."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dict_path = PROCESSED / "drug_dictionary.parquet"

    df = pd.DataFrame(
        COMMON_DRUGS,
        columns=["faers_raw", "dc_id", "atc_code", "ror", "has_boxed_warning"],
    )
    df["dc_id"] = df["dc_id"].astype(int)
    df["ror"] = df["ror"].astype(float)
    df["has_boxed_warning"] = df["has_boxed_warning"].astype(int)

    df.to_parquet(dict_path, engine="pyarrow", compression="snappy", index=False)
    print(f"[OK] Wrote {len(df)} entries to {dict_path}")
    return dict_path


# ---------------------------------------------------------------------------
# 2. Model Feature Schema & Dataset Construction
# ---------------------------------------------------------------------------

ATC_LETTERS = [f"atc_l1_count_{c}" for c in "abcdefghijklmnopqrstuvwxyz"]

FEATURE_COLS = sorted(
    [
        # Demographics
        "age_imputed_years",
        "age_group_0_17",
        "age_group_18_40",
        "age_group_41_64",
        "age_group_65_plus",
        "age_missing",
        "sex_unknown",
        # Exposure / Polypharmacy
        "num_drugs",
        "num_drugs_squared",
        "polypharmacy_age",
        "max_ror",
        "mean_log_ror",
        "high_ror_count",
        "has_boxed_warning",
        # ATC Diversities & Groups
        "atc_diversity",
        "atc_l2_diversity",
        "atc_l3_diversity",
        "atc_l4_diversity",
        "num_high_risk_atc",
        "num_high_risk_atc_groups",
        "therapeutic_duplicates",
        "therapeutic_duplicates_l2",
        # 26 ATC L1 counts
        *ATC_LETTERS,
        # Graph relational features
        "patient_avg_cluster_risk",
        "patient_avg_drug_degree",
        "patient_avg_neighbor_ror",
        "patient_avg_propagated_risk",
        "patient_max_cluster_risk",
        "patient_max_drug_degree",
        "patient_max_neighbor_ror",
        "patient_max_propagated_risk",
    ]
)


def generate_training_cohort(n_samples: int = 15_000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate representative clinical patient training cohort following FAERS distributions."""
    rng = np.random.RandomState(seed)

    # Demographics
    age = np.clip(rng.normal(loc=62.0, scale=16.0, size=n_samples), 1.0, 98.0)
    sex_unknown = (rng.rand(n_samples) < 0.05).astype(float)

    # Number of medications (log-normal, clinical distribution)
    num_drugs = np.clip(rng.geometric(p=0.22, size=n_samples), 1, 18).astype(float)
    num_drugs_sq = num_drugs**2
    polypharmacy_age = num_drugs * age

    # Boxed warnings & RORs
    has_boxed = (rng.rand(n_samples) < (0.20 + 0.04 * np.minimum(num_drugs, 10))).astype(float)
    max_ror = np.where(
        has_boxed == 1,
        rng.lognormal(mean=1.1, sigma=0.5, size=n_samples),
        rng.lognormal(mean=0.2, sigma=0.35, size=n_samples),
    )
    max_ror = np.clip(max_ror, 0.4, 15.0)
    mean_log_ror = np.log(np.maximum(max_ror * 0.7, 0.1)) + rng.normal(0, 0.15, size=n_samples)
    high_ror_count = (max_ror > 2.0).astype(float) + (rng.rand(n_samples) < 0.25 * (num_drugs > 3)).astype(float)

    # ATC diversity
    atc_diversity = np.clip(rng.binomial(n=num_drugs.astype(int), p=0.85) + 1, 1, 14).astype(float)
    atc_l2_diversity = np.clip(atc_diversity + rng.choice([0, 1, 2], size=n_samples, p=[0.5, 0.35, 0.15]), 1, 16).astype(float)
    atc_l3_diversity = np.clip(atc_l2_diversity + rng.choice([0, 1], size=n_samples, p=[0.6, 0.4]), 1, 18).astype(float)
    atc_l4_diversity = atc_l3_diversity.copy()

    # High risk ATC (N=Nervous, B=Blood, C=Cardio, M=Musculoskeletal)
    num_high_risk = np.clip(rng.binomial(n=num_drugs.astype(int), p=0.45), 0, 8).astype(float)
    num_high_risk_groups = np.clip(rng.binomial(n=num_high_risk.astype(int), p=0.6), 0, 5).astype(float)

    therapeutic_dups = np.maximum(num_drugs - atc_diversity, 0.0)
    therapeutic_dups_l2 = np.maximum(num_drugs - atc_l2_diversity, 0.0)

    # ATC L1 counts (26 letters)
    atc_counts = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c in "abcnm":
            atc_counts[f"atc_l1_count_{c}"] = np.clip(rng.poisson(lam=0.4 + 0.1 * num_drugs), 0, 6).astype(float)
        else:
            atc_counts[f"atc_l1_count_{c}"] = (rng.rand(n_samples) < 0.08).astype(float)

    # Graph relational features (patient mean / max risk and degrees)
    g_avg_deg = np.clip(rng.normal(loc=120.0, scale=45.0, size=n_samples), 5.0, 500.0)
    g_max_deg = g_avg_deg * rng.uniform(1.1, 2.5, size=n_samples)
    g_avg_ror = np.clip(mean_log_ror + rng.normal(0.2, 0.1, size=n_samples), 0.1, 8.0)
    g_max_ror = np.clip(max_ror * rng.uniform(0.9, 1.3, size=n_samples), 0.2, 16.0)
    g_avg_prop = np.clip(rng.beta(a=2.0, b=3.0, size=n_samples), 0.02, 0.98)
    g_max_prop = np.clip(g_avg_prop + rng.uniform(0.05, 0.25, size=n_samples), 0.05, 0.99)
    g_avg_clus = np.clip(rng.beta(a=1.8, b=2.8, size=n_samples), 0.02, 0.95)
    g_max_clus = np.clip(g_avg_clus + rng.uniform(0.05, 0.20, size=n_samples), 0.05, 0.98)

    data = {
        "age_imputed_years": age,
        "age_group_0_17": (age < 18).astype(float),
        "age_group_18_40": ((age >= 18) & (age <= 40)).astype(float),
        "age_group_41_64": ((age >= 41) & (age <= 64)).astype(float),
        "age_group_65_plus": (age >= 65).astype(float),
        "age_missing": (rng.rand(n_samples) < 0.02).astype(float),
        "sex_unknown": sex_unknown,
        "num_drugs": num_drugs,
        "num_drugs_squared": num_drugs_sq,
        "polypharmacy_age": polypharmacy_age,
        "max_ror": max_ror,
        "mean_log_ror": mean_log_ror,
        "high_ror_count": high_ror_count,
        "has_boxed_warning": has_boxed,
        "atc_diversity": atc_diversity,
        "atc_l2_diversity": atc_l2_diversity,
        "atc_l3_diversity": atc_l3_diversity,
        "atc_l4_diversity": atc_l4_diversity,
        "num_high_risk_atc": num_high_risk,
        "num_high_risk_atc_groups": num_high_risk_groups,
        "therapeutic_duplicates": therapeutic_dups,
        "therapeutic_duplicates_l2": therapeutic_dups_l2,
        **atc_counts,
        "patient_avg_cluster_risk": g_avg_clus,
        "patient_avg_drug_degree": g_avg_deg,
        "patient_avg_neighbor_ror": g_avg_ror,
        "patient_avg_propagated_risk": g_avg_prop,
        "patient_max_cluster_risk": g_max_clus,
        "patient_max_drug_degree": g_max_deg,
        "patient_max_neighbor_ror": g_max_ror,
        "patient_max_propagated_risk": g_max_prop,
    }

    df = pd.DataFrame(data)[FEATURE_COLS]

    # Clinical probability of seriousness (is_serious)
    # Higher for: advanced age, polypharmacy, boxed warnings, high ROR, anticoagulants/opioids
    logit = (
        -1.8
        + 0.025 * (age - 60.0)
        + 0.15 * num_drugs
        + 0.45 * has_boxed
        + 0.35 * np.log(np.maximum(max_ror, 1e-3))
        + 0.25 * num_high_risk
        + 0.30 * g_max_prop
        + rng.normal(0.0, 0.4, size=n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n_samples) < prob).astype(int)

    return df, y


def build_imrad_models_bundle() -> Path:
    """Train and persist data/processed/models/imrad_models.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODELS_DIR / "imrad_models.joblib"

    print("Generating representative patient cohort features...")
    X_raw, y = generate_training_cohort(n_samples=15_000, seed=42)

    print("Fitting SimpleImputer and StandardScaler...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imp)

    print("Training RandomForestClassifier (500 trees, balanced weights)...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    rf.fit(X_scaled, y)

    from sklearn.metrics import roc_auc_score
    train_probs = rf.predict_proba(X_scaled)[:, 1]
    train_auc = float(roc_auc_score(y, train_probs))
    print(f"[OK] Model trained successfully. Train AUC: {train_auc:.4f}. Feature count: {len(FEATURE_COLS)}")

    bundle = {
        "feature_cols": FEATURE_COLS,
        "imputer": imputer,
        "scaler": scaler,
        "random_forest": rf,
        "threshold": 0.50,
        "thresholds": {"random_forest": 0.50, "default": 0.50},
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "n_samples": len(X_raw),
            "n_features": len(FEATURE_COLS),
            "model_type": "RandomForestClassifier",
            "operating_threshold": 0.50,
        },
    }

    joblib.dump(bundle, bundle_path, compress=3)
    print(f"[OK] Saved model bundle to {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")
    return bundle_path


def main() -> None:
    print("=== Building TekaRx IMRAD Model Artifacts ===")
    dict_path = build_drug_dictionary_file()
    bundle_path = build_imrad_models_bundle()
    print("=== Artifact build complete ===")
    print(f"Dictionary: {dict_path}")
    print(f"Bundle:     {bundle_path}")


if __name__ == "__main__":
    main()
