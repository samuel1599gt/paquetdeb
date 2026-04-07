"""
flow_grouper.py
===============
Agrupa flujos de red por par (Src IP, Dst IP) y calcula
métricas agregadas extendidas para el análisis forense.

Mejoras sobre la versión anterior:
    - Métricas de asimetría (skewness) y curtosis de paquetes
    - Ratio de protocolos por par IP
    - Detección de puertos únicos por par (base para port scan)
    - Estadísticas de flags TCP agregadas
    - Métricas de ventana TCP (init_win)
    - Cálculo de entropía de puertos destino por par
    - Timestamps de inicio y fin del par
    - Columna de duración total real (no suma de duración de flujos)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")


# =============================================================
#  HELPERS
# =============================================================

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    """Verifica si una columna existe y tiene datos."""
    return col in df.columns and df[col].notna().any()


def _entropy(series: pd.Series) -> float:
    """
    Calcula la entropía de Shannon de una serie categórica.
    Alta entropía en puertos → muchos puertos distintos → posible scan.
    """
    counts = series.value_counts(normalize=True)
    return float(-np.sum(counts * np.log2(counts + 1e-9)))


def _top_n(series: pd.Series, n: int = 5) -> dict:
    """Retorna los N valores más frecuentes como dict {valor: conteo}."""
    return {str(k): int(v) for k, v in series.value_counts().head(n).to_dict().items()}


# =============================================================
#  AGRUPACIONES ESPECÍFICAS
# =============================================================

def _aggregate_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas base de volumen y temporales por par Src IP → Dst IP.
    """
    agg = {}

    # --- Conteo de flujos ---
    agg["flow_count"] = ("Src IP", "count")

    # --- Temporales ---
    if _safe_col(df, "Flow Start"):
        agg["first_seen"]      = ("Flow Start", "min")
        agg["last_seen"]       = ("Flow Start", "max")
    if _safe_col(df, "Flow Duration"):
        agg["total_duration"]  = ("Flow Duration", "sum")
        agg["avg_duration"]    = ("Flow Duration", "mean")
        agg["max_duration"]    = ("Flow Duration", "max")
        agg["min_duration"]    = ("Flow Duration", "min")

    # --- Paquetes y bytes ---
    if _safe_col(df, "Total Packets"):
        agg["total_packets"]   = ("Total Packets", "sum")
    if _safe_col(df, "Total Bytes"):
        agg["total_bytes"]     = ("Total Bytes", "sum")
    if _safe_col(df, "Average Packet Size"):
        agg["avg_packet_size"] = ("Average Packet Size", "mean")
        agg["max_packet_size"] = ("Average Packet Size", "max")

    # --- Throughput ---
    if _safe_col(df, "Packets/s"):
        agg["avg_packets_per_sec"] = ("Packets/s", "mean")
        agg["max_packets_per_sec"] = ("Packets/s", "max")
    if _safe_col(df, "Bytes/s"):
        agg["avg_bytes_per_sec"]   = ("Bytes/s", "mean")
        agg["max_bytes_per_sec"]   = ("Bytes/s", "max")

    return df.groupby(["Src IP", "Dst IP"]).agg(**agg).reset_index()


def _aggregate_forward(df: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas de tráfico forward (cliente → servidor)."""
    if not _safe_col(df, "Subflow Fwd Bytes"):
        return grouped

    agg = {}
    if _safe_col(df, "Total Fwd Packets"):
        agg["total_fwd_packets"]  = ("Total Fwd Packets", "sum")
    if _safe_col(df, "Subflow Fwd Bytes"):
        agg["total_fwd_bytes"]    = ("Subflow Fwd Bytes", "sum")
    if _safe_col(df, "Fwd Packet Length Mean"):
        agg["avg_fwd_pkt_size"]   = ("Fwd Packet Length Mean", "mean")
    if _safe_col(df, "Fwd Packet Length Max"):
        agg["max_fwd_pkt_size"]   = ("Fwd Packet Length Max", "max")
    if _safe_col(df, "Fwd Packet Length Min"):
        agg["min_fwd_pkt_size"]   = ("Fwd Packet Length Min", "min")
    if _safe_col(df, "Fwd Packet Length Std"):
        agg["std_fwd_pkt_size"]   = ("Fwd Packet Length Std", "mean")
    if _safe_col(df, "Fwd IAT Mean"):
        agg["avg_fwd_iat"]        = ("Fwd IAT Mean", "mean")
    if _safe_col(df, "Fwd IAT Max"):
        agg["max_fwd_iat"]        = ("Fwd IAT Max", "max")
    if _safe_col(df, "Fwd IAT Std"):
        agg["std_fwd_iat"]        = ("Fwd IAT Std", "mean")
    if _safe_col(df, "Init_Win_bytes_forward"):
        agg["avg_init_win_fwd"]   = ("Init_Win_bytes_forward", "mean")
    if _safe_col(df, "act_data_pkt_fwd"):
        agg["total_act_data_fwd"] = ("act_data_pkt_fwd", "sum")

    if not agg:
        return grouped

    fwd_agg = df.groupby(["Src IP", "Dst IP"]).agg(**agg).reset_index()
    return grouped.merge(fwd_agg, on=["Src IP", "Dst IP"], how="left")


def _aggregate_backward(df: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas de tráfico backward (servidor → cliente)."""
    if not _safe_col(df, "Subflow Bwd Bytes"):
        return grouped

    agg = {}
    if _safe_col(df, "Total Bwd Packets"):
        agg["total_bwd_packets"] = ("Total Bwd Packets", "sum")
    if _safe_col(df, "Subflow Bwd Bytes"):
        agg["total_bwd_bytes"]   = ("Subflow Bwd Bytes", "sum")
    if _safe_col(df, "Bwd Packet Length Mean"):
        agg["avg_bwd_pkt_size"]  = ("Bwd Packet Length Mean", "mean")
    if _safe_col(df, "Bwd Packet Length Max"):
        agg["max_bwd_pkt_size"]  = ("Bwd Packet Length Max", "max")
    if _safe_col(df, "Bwd Packet Length Min"):
        agg["min_bwd_pkt_size"]  = ("Bwd Packet Length Min", "min")
    if _safe_col(df, "Bwd Packet Length Std"):
        agg["std_bwd_pkt_size"]  = ("Bwd Packet Length Std", "mean")
    if _safe_col(df, "Bwd IAT Mean"):
        agg["avg_bwd_iat"]       = ("Bwd IAT Mean", "mean")
    if _safe_col(df, "Bwd IAT Max"):
        agg["max_bwd_iat"]       = ("Bwd IAT Max", "max")
    if _safe_col(df, "Bwd IAT Std"):
        agg["std_bwd_iat"]       = ("Bwd IAT Std", "mean")
    if _safe_col(df, "Init_Win_bytes_backward"):
        agg["avg_init_win_bwd"]  = ("Init_Win_bytes_backward", "mean")

    if not agg:
        return grouped

    bwd_agg = df.groupby(["Src IP", "Dst IP"]).agg(**agg).reset_index()
    return grouped.merge(bwd_agg, on=["Src IP", "Dst IP"], how="left")


def _aggregate_flags(df: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """Agrega conteos de flags TCP por par IP."""
    flag_cols = {
        "FIN Flag Count": "total_fin",
        "SYN Flag Count": "total_syn",
        "RST Flag Count": "total_rst",
        "PSH Flag Count": "total_psh",
        "ACK Flag Count": "total_ack",
        "URG Flag Count": "total_urg",
        "ECE Flag Count": "total_ece",
        "CWR Flag Count": "total_cwr",
    }

    agg = {
        new_name: (col, "sum")
        for col, new_name in flag_cols.items()
        if _safe_col(df, col)
    }

    if not agg:
        return grouped

    flag_agg = df.groupby(["Src IP", "Dst IP"]).agg(**agg).reset_index()
    return grouped.merge(flag_agg, on=["Src IP", "Dst IP"], how="left")


def _aggregate_ports(df: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas de puertos destino por par IP.
    Incluye número de puertos únicos y entropía (clave para detectar port scan).
    """
    if not _safe_col(df, "Destination Port"):
        return grouped

    port_metrics = (
        df.groupby(["Src IP", "Dst IP"])["Destination Port"]
        .agg(
            unique_dst_ports="nunique",
            entropy_dst_ports=_entropy,
        )
        .reset_index()
    )

    return grouped.merge(port_metrics, on=["Src IP", "Dst IP"], how="left")


def _aggregate_protocols(df: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega distribución de protocolos por par IP.
    Agrega columnas: proto_tcp_ratio, proto_udp_ratio, proto_icmp_ratio.
    """
    if not _safe_col(df, "Protocol"):
        return grouped

    PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp"}

    def _proto_ratios(group):
        total = len(group)
        counts = group["Protocol"].value_counts()
        return pd.Series({
            f"proto_{name}_ratio": round(counts.get(num, 0) / total, 4)
            for num, name in PROTO_MAP.items()
        })

    proto_agg = (
        df.groupby(["Src IP", "Dst IP"])
        .apply(_proto_ratios)
        .reset_index()
    )

    return grouped.merge(proto_agg, on=["Src IP", "Dst IP"], how="left")


def _compute_derived_metrics(grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas derivadas a partir de las columnas ya agregadas.
    Ratios, asimetrías y métricas compuestas.
    """
    df = grouped.copy()

    # --- Ratios fwd / bwd ---
    if "total_fwd_bytes" in df.columns and "total_bwd_bytes" in df.columns:
        df["bytes_ratio_fwd_bwd"] = (
            (df["total_fwd_bytes"] + 1) / (df["total_bwd_bytes"] + 1)
        ).round(4)

    if "total_fwd_packets" in df.columns and "total_bwd_packets" in df.columns:
        df["packets_ratio_fwd_bwd"] = (
            (df["total_fwd_packets"] + 1) / (df["total_bwd_packets"] + 1)
        ).round(4)

    # --- Bytes por paquete promedio ---
    if "total_bytes" in df.columns and "total_packets" in df.columns:
        df["bytes_per_packet"] = (
            df["total_bytes"] / (df["total_packets"] + 1)
        ).round(3)

    # --- Duración real del par (last_seen - first_seen) ---
    if "first_seen" in df.columns and "last_seen" in df.columns:
        df["pair_duration_s"] = (
            (df["last_seen"] - df["first_seen"]).clip(lower=0)
        ).round(3)

    # --- Ratio SYN / total flags (indicador de flood) ---
    flag_total_cols = [c for c in ["total_syn", "total_fin", "total_rst", "total_ack"]
                       if c in df.columns]
    if "total_syn" in df.columns and flag_total_cols:
        total_flags = df[flag_total_cols].sum(axis=1)
        df["syn_ratio"] = (df["total_syn"] / (total_flags + 1)).round(4)

    # --- Ratio RST / total flags ---
    if "total_rst" in df.columns and flag_total_cols:
        total_flags = df[flag_total_cols].sum(axis=1)
        df["rst_ratio"] = (df["total_rst"] / (total_flags + 1)).round(4)

    # --- Tamaño máximo de paquete global ---
    max_cols = [c for c in ["max_fwd_pkt_size", "max_bwd_pkt_size"] if c in df.columns]
    if max_cols:
        df["max_packet_size_global"] = df[max_cols].max(axis=1)

    # --- Tamaño mínimo de paquete global ---
    min_cols = [c for c in ["min_fwd_pkt_size", "min_bwd_pkt_size"] if c in df.columns]
    if min_cols:
        df["min_packet_size_global"] = df[min_cols].min(axis=1)

    return df


def _compute_skewness_kurtosis(df: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula asimetría (skewness) y curtosis de tamaños de paquete
    por par IP. Alta asimetría puede indicar tráfico anómalo.
    """
    results = []

    for (src, dst), group in df.groupby(["Src IP", "Dst IP"]):
        row = {"Src IP": src, "Dst IP": dst}

        fwd_col = "Fwd Packet Length Mean"
        bwd_col = "Bwd Packet Length Mean"

        if _safe_col(group, fwd_col) and len(group) >= 3:
            vals = group[fwd_col].dropna().values
            row["skewness_fwd_pkt"] = round(float(skew(vals)), 4) if len(vals) >= 3 else 0.0
            row["kurtosis_fwd_pkt"] = round(float(kurtosis(vals)), 4) if len(vals) >= 3 else 0.0

        if _safe_col(group, bwd_col) and len(group) >= 3:
            vals = group[bwd_col].dropna().values
            row["skewness_bwd_pkt"] = round(float(skew(vals)), 4) if len(vals) >= 3 else 0.0
            row["kurtosis_bwd_pkt"] = round(float(kurtosis(vals)), 4) if len(vals) >= 3 else 0.0

        results.append(row)

    if not results:
        return grouped

    sk_df = pd.DataFrame(results)
    return grouped.merge(sk_df, on=["Src IP", "Dst IP"], how="left")


# =============================================================
#  FUNCIÓN PRINCIPAL
# =============================================================

def group_flows(
    df_or_path,
    output_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Agrupa flujos por par (Src IP, Dst IP) y calcula métricas extendidas.

    Parámetros
    ----------
    df_or_path  : pd.DataFrame ya cargado O ruta string a un CSV.
    output_path : Si se especifica, guarda el resultado en CSV.
    verbose     : Imprime progreso si True.

    Retorna
    -------
    pd.DataFrame con una fila por par (Src IP, Dst IP) y todas
    las métricas agregadas calculadas.
    """
    # --- Cargar datos ---
    if isinstance(df_or_path, str):
        if verbose:
            print(f"[flow_grouper] Cargando CSV: {df_or_path}")
        df = pd.read_csv(df_or_path)
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        raise TypeError("df_or_path debe ser un DataFrame o una ruta a CSV.")

    if df.empty:
        print("[flow_grouper] ADVERTENCIA: DataFrame de entrada vacío.")
        return pd.DataFrame()

    if verbose:
        print(f"[flow_grouper] Flujos de entrada : {len(df):,}")
        print(f"[flow_grouper] Pares únicos Src→Dst: {df.groupby(['Src IP','Dst IP']).ngroups:,}")

    # --- Pipeline de agregación ---
    if verbose:
        print("[flow_grouper] Agregando metricas base...")
    grouped = _aggregate_base(df)

    if verbose:
        print("[flow_grouper] Agregando metricas forward...")
    grouped = _aggregate_forward(df, grouped)

    if verbose:
        print("[flow_grouper] Agregando metricas backward...")
    grouped = _aggregate_backward(df, grouped)

    if verbose:
        print("[flow_grouper] Agregando flags TCP...")
    grouped = _aggregate_flags(df, grouped)

    if verbose:
        print("[flow_grouper] Agregando metricas de puertos...")
    grouped = _aggregate_ports(df, grouped)

    if verbose:
        print("[flow_grouper] Agregando distribucion de protocolos...")
    grouped = _aggregate_protocols(df, grouped)

    if verbose:
        print("[flow_grouper] Calculando metricas derivadas...")
    grouped = _compute_derived_metrics(grouped)

    if verbose:
        print("[flow_grouper] Calculando asimetria y curtosis...")
    grouped = _compute_skewness_kurtosis(df, grouped)

    # --- Redondear floats ---
    float_cols = grouped.select_dtypes(include="float").columns
    grouped[float_cols] = grouped[float_cols].round(4)

    # --- Ordenar por volumen de tráfico descendente ---
    if "total_bytes" in grouped.columns:
        grouped = grouped.sort_values("total_bytes", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"[flow_grouper] Pares agrupados    : {len(grouped):,}")
        print(f"[flow_grouper] Columnas generadas : {grouped.shape[1]}")

    # --- Exportar ---
    if output_path:
        grouped.to_csv(output_path, index=False)
        if verbose:
            print(f"[flow_grouper] CSV guardado en: {output_path}")

    return grouped