"""
preprocessor.py
===============
Limpia y prepara el DataFrame de flujos extraídos para el pipeline
de análisis forense. Sin ML, sin escalado, sin generación de features
para modelos. Solo limpieza, validación y persistencia.

Responsabilidades:
    - Eliminar filas duplicadas y completamente vacías
    - Reemplazar valores infinitos y NaN por ceros o medianas
    - Validar rangos de columnas críticas (no negativos, no absurdos)
    - Registrar un reporte de calidad del dataset
    - Guardar raw_flows.csv limpio para el resto del pipeline
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================
#  CONFIGURACIÓN
# =============================================================

# Columnas que nunca deben ser negativas
NON_NEGATIVE_COLS = [
    "Flow Duration", "Total Packets", "Total Bytes",
    "Average Packet Size", "Packets/s", "Bytes/s",
    "Total Fwd Packets", "Total Bwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Bytes",
    "Fwd Packet Length Min", "Fwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Max",
    "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
]

# Columnas de texto que identifican el flujo (no se imputan)
ID_COLS = ["Src IP", "Dst IP", "Src Port", "Destination Port", "Protocol"]

# Columnas que deben existir para que el pipeline funcione
REQUIRED_COLS = ["Src IP", "Dst IP"]

# Límites de sanidad para valores extremos
SANITY_LIMITS = {
    "Flow Duration"      : 86_400_000,   # max 24 horas en ms
    "Total Packets"      : 10_000_000,   # max 10M paquetes por flujo
    "Total Bytes"        : 10_000_000_000,  # max 10 GB por flujo
    "Packets/s"          : 1_000_000,    # max 1M pps
    "Bytes/s"            : 10_000_000_000,  # max 10 Gbps
    "Average Packet Size": 65_535,       # max MTU estándar
}


# =============================================================
#  HELPERS
# =============================================================

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{value / total * 100:.2f}%"


# =============================================================
#  ETAPAS DE LIMPIEZA
# =============================================================

def _check_required_columns(df: pd.DataFrame) -> None:
    """
    Valida que las columnas mínimas necesarias existan.
    Lanza ValueError si falta alguna.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[preprocessor] Columnas requeridas faltantes: {missing}\n"
            f"  Columnas disponibles: {list(df.columns)}"
        )


def _drop_empty_rows(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Elimina filas completamente vacías (todos NaN)."""
    before = len(df)
    df = df.dropna(how="all")
    stats["dropped_all_nan"] = before - len(df)
    return df


def _drop_duplicates(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Elimina filas duplicadas exactas.
    Mantiene la primera ocurrencia.
    """
    before = len(df)
    df = df.drop_duplicates()
    stats["dropped_duplicates"] = before - len(df)
    return df


def _drop_invalid_ips(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Elimina filas con IPs nulas, vacías o con formato inválido.
    """
    before = len(df)

    for col in ["Src IP", "Dst IP"]:
        if col in df.columns:
            df = df[df[col].notna()]
            df = df[df[col].astype(str).str.strip() != ""]
            # Patrón básico IPv4 / IPv6
            valid_ipv4 = df[col].str.match(
                r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
            )
            valid_ipv6 = df[col].str.contains(":", na=False)
            df = df[valid_ipv4 | valid_ipv6]

    stats["dropped_invalid_ips"] = before - len(df)
    return df.reset_index(drop=True)


def _replace_inf(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Reemplaza valores infinitos (positivos y negativos) por NaN
    para luego ser imputados en la siguiente etapa.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    stats["replaced_inf"] = int(inf_count)
    return df


def _impute_nan(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Imputa NaN en columnas numéricas:
    - Columnas de flags y conteos → 0
    - Columnas de tamaño/duración → mediana de la columna
    - Throughput (Packets/s, Bytes/s) → 0
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_nan_before = df[numeric_cols].isna().sum().sum()

    zero_fill_keywords = [
        "Flag", "Count", "Win", "act_data",
        "Packets/s", "Bytes/s",
    ]

    for col in numeric_cols:
        if df[col].isna().sum() == 0:
            continue

        should_zero = any(kw in col for kw in zero_fill_keywords)

        if should_zero:
            df[col] = df[col].fillna(0)
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    stats["imputed_nan"] = int(total_nan_before)
    return df


def _clip_non_negative(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Aplica clip(lower=0) a columnas que no pueden ser negativas.
    Registra cuántos valores fueron corregidos.
    """
    corrected = 0
    for col in NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        neg_mask = df[col] < 0
        corrected += neg_mask.sum()
        df[col] = df[col].clip(lower=0)

    stats["clipped_negative"] = int(corrected)
    return df


def _apply_sanity_limits(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Aplica límites superiores de sanidad a columnas con valores absurdos.
    Los valores que superan el límite se reemplazan por la mediana.
    """
    corrected = 0
    for col, limit in SANITY_LIMITS.items():
        if col not in df.columns:
            continue
        over_limit = df[col] > limit
        if over_limit.any():
            median_val = df.loc[~over_limit, col].median()
            df.loc[over_limit, col] = median_val if not np.isnan(median_val) else limit
            corrected += int(over_limit.sum())

    stats["sanity_corrected"] = corrected
    return df


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura tipos de datos correctos:
    - IPs y protocolo como string/int
    - Puertos como int
    - Flags como int
    - Métricas flotantes
    """
    int_cols = [
        "Total Packets", "Total Bytes", "Total Fwd Packets", "Total Bwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Bytes",
        "Subflow Fwd Packets", "Subflow Bwd Packets",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "ECE Flag Count", "CWR Flag Count",
        "Src Port", "Destination Port", "Protocol",
        "act_data_pkt_fwd",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    str_cols = ["Src IP", "Dst IP"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# =============================================================
#  REPORTE DE CALIDAD
# =============================================================

def _build_quality_report(
    df_original: pd.DataFrame,
    df_clean: pd.DataFrame,
    stats: dict,
) -> dict:
    """
    Construye un reporte detallado de la calidad del dataset
    antes y después de la limpieza.
    """
    original_rows = stats.get("original_rows", len(df_original))
    final_rows    = len(df_clean)

    report = {
        "original_rows"        : original_rows,
        "final_rows"           : final_rows,
        "rows_removed"         : original_rows - final_rows,
        "rows_removed_pct"     : _pct(original_rows - final_rows, original_rows),
        "columns"              : df_clean.shape[1],
        "dropped_all_nan"      : stats.get("dropped_all_nan", 0),
        "dropped_duplicates"   : stats.get("dropped_duplicates", 0),
        "dropped_invalid_ips"  : stats.get("dropped_invalid_ips", 0),
        "replaced_inf"         : stats.get("replaced_inf", 0),
        "imputed_nan"          : stats.get("imputed_nan", 0),
        "clipped_negative"     : stats.get("clipped_negative", 0),
        "sanity_corrected"     : stats.get("sanity_corrected", 0),
        "remaining_nan"        : int(df_clean.select_dtypes(include=[np.number]).isna().sum().sum()),
        "unique_src_ips"       : int(df_clean["Src IP"].nunique()) if "Src IP" in df_clean.columns else 0,
        "unique_dst_ips"       : int(df_clean["Dst IP"].nunique()) if "Dst IP" in df_clean.columns else 0,
        "protocol_counts"      : {},
        "column_null_summary"  : {},
    }

    # Distribución de protocolos
    if "Protocol" in df_clean.columns:
        proto_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
        proto_counts = df_clean["Protocol"].value_counts().to_dict()
        report["protocol_counts"] = {
            proto_map.get(int(k), f"PROTO_{k}"): int(v)
            for k, v in proto_counts.items()
        }

    # Resumen de nulos por columna (solo columnas con nulos)
    null_counts = df_clean.isnull().sum()
    report["column_null_summary"] = {
        col: int(cnt)
        for col, cnt in null_counts.items()
        if cnt > 0
    }

    return report


# =============================================================
#  FUNCIÓN PRINCIPAL
# =============================================================

def preprocess(
    df: pd.DataFrame,
    output_dir: str = "",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el pipeline completo de limpieza y validación del DataFrame.

    Parámetros
    ----------
    df         : DataFrame de flujos (salida de flow_extractor.py).
    output_dir : Directorio donde guardar raw_flows.csv. Si es vacío,
                 no guarda el archivo.
    verbose    : Imprime resumen del proceso si True.

    Retorna
    -------
    tuple (df_clean, quality_report)
        df_clean       — DataFrame limpio listo para flow_grouper.py
        quality_report — dict con métricas de calidad del proceso
    """
    if df is None or df.empty:
        raise ValueError("[preprocessor] El DataFrame de entrada está vacío.")

    stats = {"original_rows": len(df)}

    if verbose:
        print(f"[preprocessor] Filas de entrada  : {len(df):,}")
        print(f"[preprocessor] Columnas          : {df.shape[1]}")

    # --- Pipeline de limpieza ---
    _check_required_columns(df)

    if verbose:
        print("[preprocessor] Eliminando filas vacias...")
    df = _drop_empty_rows(df, stats)

    if verbose:
        print("[preprocessor] Eliminando duplicados...")
    df = _drop_duplicates(df, stats)

    if verbose:
        print("[preprocessor] Validando IPs...")
    df = _drop_invalid_ips(df, stats)

    if verbose:
        print("[preprocessor] Reemplazando infinitos...")
    df = _replace_inf(df, stats)

    if verbose:
        print("[preprocessor] Imputando NaN...")
    df = _impute_nan(df, stats)

    if verbose:
        print("[preprocessor] Corrigiendo valores negativos...")
    df = _clip_non_negative(df, stats)

    if verbose:
        print("[preprocessor] Aplicando limites de sanidad...")
    df = _apply_sanity_limits(df, stats)

    if verbose:
        print("[preprocessor] Ajustando tipos de datos...")
    df = _enforce_dtypes(df)

    df = df.reset_index(drop=True)

    # --- Reporte de calidad ---
    quality_report = _build_quality_report(pd.DataFrame(), df, stats)

    if verbose:
        print(f"[preprocessor] Filas finales     : {len(df):,}")
        print(f"[preprocessor] Filas eliminadas  : {stats['original_rows'] - len(df):,} "
              f"({_pct(stats['original_rows'] - len(df), stats['original_rows'])})")
        print(f"[preprocessor] NaN imputados     : {stats['imputed_nan']:,}")
        print(f"[preprocessor] Inf reemplazados  : {stats['replaced_inf']:,}")
        print(f"[preprocessor] Negativos corregidos: {stats['clipped_negative']:,}")

    # --- Guardar CSV ---
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "raw_flows.csv"
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"[preprocessor] raw_flows.csv guardado en: {csv_path}")

    return df, quality_report