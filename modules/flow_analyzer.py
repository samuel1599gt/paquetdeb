"""
flow_analyzer.py
================
Análisis forense determinístico de flujos de red.
No usa ML. Detecta comportamientos anómalos por reglas
basadas en umbrales estadísticos y patrones conocidos.

Detecciones:
    - Port scanning (vertical y horizontal)
    - SYN flood
    - UDP flood
    - ICMP flood / ping sweep
    - Exfiltración de datos
    - Flags TCP anómalos (RST, FIN scan, NULL)
    - Flows de larga duración sospechosos
    - Beaconing / C2
    - Puertos inusuales con actividad
"""

import json
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")


# =============================================================
#  UMBRALES GLOBALES (ajustables)
# =============================================================

THRESHOLDS = {
    # Port scanning
    "port_scan_unique_dst_ports"    : 20,
    "horizontal_scan_unique_dst_ips": 15,
    "port_scan_min_flows"           : 10,

    # Floods
    "syn_flood_ratio"               : 0.85,
    "syn_flood_min_count"           : 200,
    "udp_flood_packets_per_sec"     : 800,
    "icmp_sweep_unique_dst_ips"     : 20,

    # Exfiltración
    "exfil_bytes_ratio"             : 5.0,
    "exfil_min_fwd_bytes"           : 500_000,

    # Flags TCP anómalos
    "rst_ratio_threshold"           : 0.60,
    "rst_min_count"                 : 50,
    "fin_without_syn_min"           : 50,

    # Duración (en milisegundos)
    "long_flow_duration_ms"         : 600_000,   # 10 minutos

    # Beaconing
    "beaconing_min_flows"           : 10,
    "beaconing_cv_threshold"        : 0.20,
    "beaconing_min_interval_ms"     : 5_000,

    # Puertos inusuales
    "known_safe_ports": {
        20, 21, 22, 23, 25, 53, 67, 68, 80, 110,
        123, 143, 443, 445, 465, 587, 993, 995,
        3306, 3389, 5432, 5900, 8080, 8443,
        27017, 1433, 5353,
    },
    "unusual_port_min_count": 5,
}

PROTO_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}


# =============================================================
#  HELPERS
# =============================================================

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _protocol_label(proto_num) -> str:
    try:
        return PROTO_NAMES.get(int(proto_num), f"PROTO_{int(proto_num)}")
    except Exception:
        return str(proto_num)


def _severity(value, low, medium, high) -> str:
    if value >= high:
        return "CRITICA"
    elif value >= medium:
        return "ALTA"
    elif value >= low:
        return "MEDIA"
    return "BAJA"


# =============================================================
#  DETECCIONES
# =============================================================

def _detect_port_scan(df: pd.DataFrame) -> list:
    """
    Port scan vertical: una src contacta muchos puertos en una misma dst.
    Port scan horizontal: una src contacta muchas dst en pocos puertos.
    """
    alerts = []

    if not _safe_col(df, "Destination Port"):
        return alerts

    # Vertical
    vertical = (
        df.groupby(["Src IP", "Dst IP"])
        .agg(
            unique_ports=("Destination Port", "nunique"),
            total_flows =("Destination Port", "count"),
            top_ports   =("Destination Port", lambda x: x.value_counts().head(5).to_dict()),
        )
        .reset_index()
    )
    hits = vertical[
        (vertical["unique_ports"] >= THRESHOLDS["port_scan_unique_dst_ports"]) &
        (vertical["total_flows"]  >= THRESHOLDS["port_scan_min_flows"])
    ]
    for _, row in hits.iterrows():
        alerts.append({
            "tipo"        : "Port Scan Vertical",
            "severidad"   : _severity(row["unique_ports"], 20, 50, 100),
            "src_ip"      : row["Src IP"],
            "dst_ip"      : row["Dst IP"],
            "detalle"     : (
                f"{int(row['unique_ports'])} puertos unicos escaneados "
                f"en {row['Dst IP']} desde {row['Src IP']}"
            ),
            "unique_ports": int(row["unique_ports"]),
            "total_flows" : int(row["total_flows"]),
            "top_ports"   : {str(k): int(v) for k, v in row["top_ports"].items()},
        })

    # Horizontal
    horizontal = (
        df.groupby("Src IP")
        .agg(
            unique_dst_ips=("Dst IP", "nunique"),
            unique_ports  =("Destination Port", "nunique"),
            total_flows   =("Dst IP", "count"),
        )
        .reset_index()
    )
    hits_h = horizontal[
        (horizontal["unique_dst_ips"] >= THRESHOLDS["horizontal_scan_unique_dst_ips"]) &
        (horizontal["unique_ports"]   <= 5)
    ]
    for _, row in hits_h.iterrows():
        alerts.append({
            "tipo"          : "Port Scan Horizontal",
            "severidad"     : _severity(row["unique_dst_ips"], 15, 40, 80),
            "src_ip"        : row["Src IP"],
            "dst_ip"        : "multiples",
            "detalle"       : (
                f"{int(row['unique_dst_ips'])} hosts escaneados desde "
                f"{row['Src IP']} en {int(row['unique_ports'])} puerto(s)"
            ),
            "unique_dst_ips": int(row["unique_dst_ips"]),
            "unique_ports"  : int(row["unique_ports"]),
            "total_flows"   : int(row["total_flows"]),
        })

    return alerts


def _detect_syn_flood(df: pd.DataFrame) -> list:
    """
    SYN flood: ratio alto de SYN vs total de flags por IP origen.
    """
    alerts = []
    required = ["SYN Flag Count", "FIN Flag Count", "RST Flag Count"]
    if not all(_safe_col(df, c) for c in required):
        return alerts

    grouped = (
        df.groupby("Src IP")
        .agg(
            total_syn =("SYN Flag Count", "sum"),
            total_fin =("FIN Flag Count", "sum"),
            total_rst =("RST Flag Count", "sum"),
            total_flows=("Src IP", "count"),
            unique_dst =("Dst IP", "nunique"),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        total_flags = row["total_syn"] + row["total_fin"] + row["total_rst"]
        if total_flags == 0:
            continue
        syn_ratio = row["total_syn"] / total_flags

        if (syn_ratio >= THRESHOLDS["syn_flood_ratio"] and
                row["total_syn"] >= THRESHOLDS["syn_flood_min_count"]):
            alerts.append({
                "tipo"       : "SYN Flood",
                "severidad"  : _severity(row["total_syn"], 200, 1000, 5000),
                "src_ip"     : row["Src IP"],
                "dst_ip"     : "multiples" if row["unique_dst"] > 1 else "unica",
                "detalle"    : (
                    f"{int(row['total_syn'])} SYNs ({syn_ratio*100:.1f}% del total "
                    f"de flags) desde {row['Src IP']}"
                ),
                "total_syn"  : int(row["total_syn"]),
                "syn_ratio"  : round(float(syn_ratio), 3),
                "unique_dst" : int(row["unique_dst"]),
            })

    return alerts


def _detect_udp_flood(df: pd.DataFrame) -> list:
    """
    UDP flood: alto volumen de paquetes UDP/s desde una IP.
    """
    alerts = []
    if not (_safe_col(df, "Protocol") and _safe_col(df, "Packets/s")):
        return alerts

    udp = df[df["Protocol"].apply(lambda x: _protocol_label(x) == "UDP")].copy()
    if udp.empty:
        return alerts

    grouped = (
        udp.groupby("Src IP")
        .agg(
            avg_pps    =("Packets/s", "mean"),
            max_pps    =("Packets/s", "max"),
            total_flows=("Src IP", "count"),
            unique_dst =("Dst IP", "nunique"),
        )
        .reset_index()
    )

    for _, row in grouped[grouped["max_pps"] >= THRESHOLDS["udp_flood_packets_per_sec"]].iterrows():
        alerts.append({
            "tipo"       : "UDP Flood",
            "severidad"  : _severity(row["max_pps"], 800, 5000, 20000),
            "src_ip"     : row["Src IP"],
            "dst_ip"     : "multiples" if row["unique_dst"] > 1 else "unica",
            "detalle"    : (
                f"Pico de {row['max_pps']:.0f} paquetes/s UDP desde {row['Src IP']}"
            ),
            "max_pps"    : round(float(row["max_pps"]), 2),
            "avg_pps"    : round(float(row["avg_pps"]), 2),
            "total_flows": int(row["total_flows"]),
        })

    return alerts


def _detect_icmp_sweep(df: pd.DataFrame) -> list:
    """
    ICMP ping sweep: una IP sondea muchos hosts con ICMP.
    """
    alerts = []
    if not _safe_col(df, "Protocol"):
        return alerts

    icmp = df[df["Protocol"].apply(lambda x: _protocol_label(x) == "ICMP")].copy()
    if icmp.empty:
        return alerts

    grouped = (
        icmp.groupby("Src IP")
        .agg(
            unique_dst =("Dst IP", "nunique"),
            total_flows=("Dst IP", "count"),
        )
        .reset_index()
    )

    for _, row in grouped[grouped["unique_dst"] >= THRESHOLDS["icmp_sweep_unique_dst_ips"]].iterrows():
        alerts.append({
            "tipo"       : "ICMP Ping Sweep",
            "severidad"  : _severity(row["unique_dst"], 20, 50, 100),
            "src_ip"     : row["Src IP"],
            "dst_ip"     : "multiples",
            "detalle"    : (
                f"{int(row['unique_dst'])} hosts distintos sondeados "
                f"con ICMP desde {row['Src IP']}"
            ),
            "unique_dst" : int(row["unique_dst"]),
            "total_flows": int(row["total_flows"]),
        })

    return alerts


def _detect_exfiltration(df: pd.DataFrame) -> list:
    """
    Exfiltración: ratio fwd_bytes / bwd_bytes muy alto y volumen
    considerable — indica más datos saliendo que entrando.
    """
    alerts = []
    fwd_col, bwd_col = "Subflow Fwd Bytes", "Subflow Bwd Bytes"
    if not (_safe_col(df, fwd_col) and _safe_col(df, bwd_col)):
        return alerts

    agg_dict = {
        "total_fwd_bytes": (fwd_col, "sum"),
        "total_bwd_bytes": (bwd_col, "sum"),
        "total_flows"    : ("Src IP", "count"),
    }
    if _safe_col(df, "Flow Duration"):
        agg_dict["avg_duration"] = ("Flow Duration", "mean")

    grouped = df.groupby(["Src IP", "Dst IP"]).agg(**agg_dict).reset_index()
    grouped["bytes_ratio"] = (
        (grouped["total_fwd_bytes"] + 1) / (grouped["total_bwd_bytes"] + 1)
    )

    hits = grouped[
        (grouped["bytes_ratio"]     >= THRESHOLDS["exfil_bytes_ratio"]) &
        (grouped["total_fwd_bytes"] >= THRESHOLDS["exfil_min_fwd_bytes"])
    ]

    for _, row in hits.iterrows():
        alerts.append({
            "tipo"            : "Posible Exfiltracion",
            "severidad"       : _severity(row["bytes_ratio"], 5, 20, 100),
            "src_ip"          : row["Src IP"],
            "dst_ip"          : row["Dst IP"],
            "detalle"         : (
                f"{row['Src IP']} envio {row['total_fwd_bytes']/1e6:.2f} MB "
                f"(ratio fwd/bwd: {row['bytes_ratio']:.1f}x) hacia {row['Dst IP']}"
            ),
            "total_fwd_bytes" : int(row["total_fwd_bytes"]),
            "total_bwd_bytes" : int(row["total_bwd_bytes"]),
            "bytes_ratio"     : round(float(row["bytes_ratio"]), 2),
            "total_flows"     : int(row["total_flows"]),
        })

    return alerts


def _detect_anomalous_flags(df: pd.DataFrame) -> list:
    """
    Flags TCP anómalos:
    - Alto ratio de RST (RST scanning)
    - FIN sin SYN previo (FIN scanning / evasion)
    """
    alerts = []
    flag_cols = [
        "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
        "PSH Flag Count", "URG Flag Count",
    ]
    available = [c for c in flag_cols if _safe_col(df, c)]
    if len(available) < 2:
        return alerts

    grouped = df.groupby("Src IP")[available].sum().reset_index()
    grouped["total_flags"] = grouped[available].sum(axis=1)

    for _, row in grouped.iterrows():
        if row["total_flags"] == 0:
            continue

        # RST anómalo
        if "RST Flag Count" in available:
            rst_ratio = row["RST Flag Count"] / row["total_flags"]
            if (rst_ratio >= THRESHOLDS["rst_ratio_threshold"] and
                    row["RST Flag Count"] >= THRESHOLDS["rst_min_count"]):
                alerts.append({
                    "tipo"      : "RST Anomalo",
                    "severidad" : _severity(rst_ratio, 0.60, 0.80, 0.95),
                    "src_ip"    : row["Src IP"],
                    "dst_ip"    : "multiples",
                    "detalle"   : (
                        f"{int(row['RST Flag Count'])} RSTs "
                        f"({rst_ratio*100:.1f}%) desde {row['Src IP']} — "
                        f"posible RST scanning o evasion de firewall"
                    ),
                    "rst_count" : int(row["RST Flag Count"]),
                    "rst_ratio" : round(float(rst_ratio), 3),
                })

        # FIN scan: FIN sin SYN
        if "FIN Flag Count" in available and "SYN Flag Count" in available:
            fin = row["FIN Flag Count"]
            syn = row["SYN Flag Count"]
            if fin >= THRESHOLDS["fin_without_syn_min"] and syn == 0:
                alerts.append({
                    "tipo"     : "FIN Scan",
                    "severidad": "ALTA",
                    "src_ip"   : row["Src IP"],
                    "dst_ip"   : "multiples",
                    "detalle"  : (
                        f"{int(fin)} paquetes FIN sin ningun SYN desde "
                        f"{row['Src IP']} — posible FIN scanning"
                    ),
                    "fin_count": int(fin),
                    "syn_count": int(syn),
                })

    return alerts


def _detect_long_flows(df: pd.DataFrame) -> list:
    """
    Flujos de larga duración: posibles sesiones persistentes,
    tunneling o conexiones C2 mantenidas.
    """
    alerts = []
    if not _safe_col(df, "Flow Duration"):
        return alerts

    threshold_ms = THRESHOLDS["long_flow_duration_ms"]
    long = df[df["Flow Duration"] >= threshold_ms].copy()
    if long.empty:
        return alerts

    agg_dict = {
        "max_duration": ("Flow Duration", "max"),
        "avg_duration": ("Flow Duration", "mean"),
        "total_flows" : ("Flow Duration", "count"),
    }
    if _safe_col(df, "Total Bytes"):
        agg_dict["total_bytes"] = ("Total Bytes", "sum")

    grouped = long.groupby(["Src IP", "Dst IP"]).agg(**agg_dict).reset_index()

    for _, row in grouped.iterrows():
        duration_min = row["max_duration"] / 60_000
        alerts.append({
            "tipo"             : "Flujo de Larga Duracion",
            "severidad"        : _severity(
                row["max_duration"],
                threshold_ms,
                threshold_ms * 3,
                threshold_ms * 10,
            ),
            "src_ip"           : row["Src IP"],
            "dst_ip"           : row["Dst IP"],
            "detalle"          : (
                f"Conexion persistente de {duration_min:.1f} min "
                f"entre {row['Src IP']} -> {row['Dst IP']}"
            ),
            "max_duration_ms"  : int(row["max_duration"]),
            "max_duration_min" : round(float(duration_min), 2),
            "total_flows"      : int(row["total_flows"]),
        })

    return alerts


def _detect_beaconing(df: pd.DataFrame) -> list:
    """
    Beaconing: comunicación periódica y regular entre dos IPs.
    Coeficiente de variación (CV) bajo en intervalos → automatizado.
    Típico en malware contactando C2.
    """
    alerts = []
    if not _safe_col(df, "Flow Duration"):
        return alerts

    min_flows = THRESHOLDS["beaconing_min_flows"]
    cv_thresh  = THRESHOLDS["beaconing_cv_threshold"]
    min_intv   = THRESHOLDS["beaconing_min_interval_ms"]

    for (src, dst), group in df.groupby(["Src IP", "Dst IP"]):
        if len(group) < min_flows:
            continue
        durations = group["Flow Duration"].dropna().sort_values().values
        if len(durations) < 2:
            continue
        intervals = np.diff(durations)
        mean_intv = intervals.mean()
        if mean_intv < min_intv:
            continue
        cv = intervals.std() / (mean_intv + 1e-9)
        if cv <= cv_thresh:
            alerts.append({
                "tipo"            : "Beaconing / C2 Sospechoso",
                "severidad"       : _severity(1 - cv, 0.80, 0.90, 0.98),
                "src_ip"          : src,
                "dst_ip"          : dst,
                "detalle"         : (
                    f"Comunicacion periodica: {len(group)} flows con "
                    f"intervalo promedio de {mean_intv/1000:.1f}s y CV={cv:.3f}"
                ),
                "flow_count"      : len(group),
                "avg_interval_ms" : round(float(mean_intv), 2),
                "coef_variacion"  : round(float(cv), 4),
            })

    return alerts


def _detect_unusual_ports(df: pd.DataFrame) -> list:
    """
    Puertos destino no estándar con actividad significativa.
    Puede indicar backdoors, C2, o servicios ocultos.
    """
    alerts = []
    if not _safe_col(df, "Destination Port"):
        return alerts

    safe_ports = THRESHOLDS["known_safe_ports"]
    min_count  = THRESHOLDS["unusual_port_min_count"]

    port_counts = (
        df.groupby("Destination Port")
        .agg(
            flow_count=("Src IP", "count"),
            unique_src=("Src IP", "nunique"),
            unique_dst=("Dst IP", "nunique"),
        )
        .reset_index()
    )

    unusual = port_counts[
        (~port_counts["Destination Port"].isin(safe_ports)) &
        (port_counts["Destination Port"] > 1024) &
        (port_counts["flow_count"] >= min_count)
    ].sort_values("flow_count", ascending=False).head(10)

    for _, row in unusual.iterrows():
        alerts.append({
            "tipo"      : "Puerto Inusual con Actividad",
            "severidad" : _severity(row["flow_count"], 5, 50, 200),
            "src_ip"    : "multiples",
            "dst_ip"    : "multiples",
            "detalle"   : (
                f"Puerto {int(row['Destination Port'])} recibe {int(row['flow_count'])} flows "
                f"desde {int(row['unique_src'])} origenes hacia {int(row['unique_dst'])} destinos"
            ),
            "port"       : int(row["Destination Port"]),
            "flow_count" : int(row["flow_count"]),
            "unique_src" : int(row["unique_src"]),
            "unique_dst" : int(row["unique_dst"]),
        })

    return alerts


# =============================================================
#  MÉTRICAS GLOBALES
# =============================================================

def _compute_global_metrics(df: pd.DataFrame) -> dict:
    metrics = {
        "total_flows"   : len(df),
        "unique_src_ips": int(df["Src IP"].nunique()) if _safe_col(df, "Src IP") else None,
        "unique_dst_ips": int(df["Dst IP"].nunique()) if _safe_col(df, "Dst IP") else None,
        "total_bytes"   : int(df["Total Bytes"].sum()) if _safe_col(df, "Total Bytes") else None,
        "total_packets" : int(df["Total Packets"].sum()) if _safe_col(df, "Total Packets") else None,
    }

    if _safe_col(df, "Protocol"):
        proto_dist = df["Protocol"].apply(_protocol_label).value_counts().to_dict()
        metrics["protocol_distribution"] = {k: int(v) for k, v in proto_dist.items()}
    else:
        metrics["protocol_distribution"] = {}

    if _safe_col(df, "Src IP"):
        metrics["top_src_ips"] = {
            k: int(v) for k, v in
            df["Src IP"].value_counts().head(10).to_dict().items()
        }
    if _safe_col(df, "Dst IP"):
        metrics["top_dst_ips"] = {
            k: int(v) for k, v in
            df["Dst IP"].value_counts().head(10).to_dict().items()
        }
    if _safe_col(df, "Destination Port"):
        metrics["top_dst_ports"] = {
            str(k): int(v) for k, v in
            df["Destination Port"].value_counts().head(10).to_dict().items()
        }

    flags_map = {
        "SYN Flag Count": "syn",
        "FIN Flag Count": "fin",
        "RST Flag Count": "rst",
        "PSH Flag Count": "psh",
        "URG Flag Count": "urg",
    }
    metrics["flags_summary"] = {
        label: int(df[col].sum())
        for col, label in flags_map.items()
        if _safe_col(df, col)
    }

    if _safe_col(df, "Flow Duration"):
        metrics["flow_duration"] = {
            "min_ms": round(float(df["Flow Duration"].min()), 2),
            "max_ms": round(float(df["Flow Duration"].max()), 2),
            "avg_ms": round(float(df["Flow Duration"].mean()), 2),
            "std_ms": round(float(df["Flow Duration"].std()), 2),
        }
    if _safe_col(df, "Average Packet Size"):
        metrics["packet_size"] = {
            "avg": round(float(df["Average Packet Size"].mean()), 2),
            "max": round(float(df["Average Packet Size"].max()), 2),
            "min": round(float(df["Average Packet Size"].min()), 2),
        }
    if _safe_col(df, "Bytes/s"):
        metrics["avg_bytes_per_sec"]   = round(float(df["Bytes/s"].mean()), 2)
    if _safe_col(df, "Packets/s"):
        metrics["avg_packets_per_sec"] = round(float(df["Packets/s"].mean()), 2)

    return metrics


# =============================================================
#  RESUMEN DE ALERTAS
# =============================================================

def _build_alert_summary(alerts: list) -> dict:
    summary = {
        "total_alerts": len(alerts),
        "by_severity" : defaultdict(int),
        "by_type"     : defaultdict(int),
        "critical_ips": defaultdict(int),
    }
    for alert in alerts:
        summary["by_severity"][alert.get("severidad", "DESCONOCIDA")] += 1
        summary["by_type"][alert.get("tipo", "DESCONOCIDA")] += 1
        src = alert.get("src_ip", "")
        if src and src not in ("multiples", "unica"):
            summary["critical_ips"][src] += 1

    return {
        "total_alerts": summary["total_alerts"],
        "by_severity" : dict(summary["by_severity"]),
        "by_type"     : dict(summary["by_type"]),
        "critical_ips": dict(
            sorted(summary["critical_ips"].items(), key=lambda x: x[1], reverse=True)
        ),
    }


# =============================================================
#  FUNCIÓN PRINCIPAL
# =============================================================

def analyze_flows(
    df: pd.DataFrame,
    export_json: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Ejecuta el análisis forense completo sobre el DataFrame de flujos.

    Parámetros
    ----------
    df          : DataFrame de flujos (salida de flow_grouper.py).
    export_json : Ruta JSON de salida opcional.
    verbose     : Imprime progreso en consola si True.

    Retorna
    -------
    dict con claves:
        global_metrics  — métricas generales de la captura
        alerts          — lista de alertas detectadas
        alert_summary   — resumen agrupado por tipo y severidad
    """
    if verbose:
        print("[flow_analyzer] Iniciando analisis forense...")

    global_metrics = _compute_global_metrics(df)

    detectors = [
        ("Port Scan"         , _detect_port_scan),
        ("SYN Flood"         , _detect_syn_flood),
        ("UDP Flood"         , _detect_udp_flood),
        ("ICMP Sweep"        , _detect_icmp_sweep),
        ("Exfiltracion"      , _detect_exfiltration),
        ("Flags Anomalos"    , _detect_anomalous_flags),
        ("Flows Largos"      , _detect_long_flows),
        ("Beaconing"         , _detect_beaconing),
        ("Puertos Inusuales" , _detect_unusual_ports),
    ]

    all_alerts = []
    for name, detector in detectors:
        if verbose:
            print(f"  [+] Detectando: {name}...")
        try:
            found = detector(df)
            all_alerts.extend(found)
            if verbose and found:
                print(f"      -> {len(found)} alerta(s)")
        except Exception as e:
            print(f"  [!] Error en detector '{name}': {e}")

    alert_summary = _build_alert_summary(all_alerts)

    if verbose:
        print(f"[flow_analyzer] Total alertas: {alert_summary['total_alerts']}")

    result = {
        "global_metrics": global_metrics,
        "alerts"        : all_alerts,
        "alert_summary" : alert_summary,
    }

    if export_json:
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        if verbose:
            print(f"[flow_analyzer] Exportado a: {export_json}")

    return result