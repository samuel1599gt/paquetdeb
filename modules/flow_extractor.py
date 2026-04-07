"""
flow_extractor.py
=================
Extrae flujos de red desde archivos .pcap / .pcapng usando Scapy.

Mejoras sobre la versión anterior:
    - Procesamiento por chunks para archivos pesados (memoria controlada)
    - Soporte ICMP además de TCP y UDP
    - Cálculo de IAT (Inter-Arrival Time) fwd y bwd
    - Detección de flags TCP completa (incluye ACK, ECE, CWR)
    - Manejo robusto de errores por paquete
    - Barra de progreso con tqdm
    - Estadísticas de extracción al finalizar
"""

import gc
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP, ICMP
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================
#  CONSTANTES
# =============================================================

# Tamaño del chunk en número de paquetes (ajustable según RAM)
DEFAULT_CHUNK_SIZE = 50_000

# Número máximo de paquetes por flujo antes de cerrar y abrir nuevo
MAX_PKTS_PER_FLOW = 100_000

# Protocolo → número
PROTO_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1}


# =============================================================
#  HELPERS
# =============================================================

def _parse_tcp_flags(flags_int: int) -> dict:
    """
    Descompone el entero de flags TCP en sus bits individuales.
    Retorna dict con conteo (0 o 1) por flag.
    """
    return {
        "FIN": int(bool(flags_int & 0x01)),
        "SYN": int(bool(flags_int & 0x02)),
        "RST": int(bool(flags_int & 0x04)),
        "PSH": int(bool(flags_int & 0x08)),
        "ACK": int(bool(flags_int & 0x10)),
        "URG": int(bool(flags_int & 0x20)),
        "ECE": int(bool(flags_int & 0x40)),
        "CWR": int(bool(flags_int & 0x80)),
    }


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto) -> tuple:
    """
    Genera la clave de flujo.
    Se normaliza para que A→B y B→A sean el mismo flujo
    únicamente si el protocolo no es orientado a conexión.
    Para TCP se mantiene la dirección original.
    """
    if proto == PROTO_MAP["TCP"]:
        return (src_ip, dst_ip, src_port, dst_port, proto)
    # UDP / ICMP: normalizar para agrupar ambas direcciones
    if (src_ip, src_port) > (dst_ip, dst_port):
        return (dst_ip, src_ip, dst_port, src_port, proto)
    return (src_ip, dst_ip, src_port, dst_port, proto)


def _safe_stat(values: list, func, default=0.0):
    """Aplica una función estadística de forma segura."""
    if not values:
        return default
    try:
        return float(func(values))
    except Exception:
        return default


# =============================================================
#  PROCESAMIENTO DE PAQUETES
# =============================================================

def _process_packet(pkt, flows: dict, stats: dict) -> None:
    """
    Procesa un único paquete y lo agrega al flujo correspondiente.
    Modifica `flows` in-place.
    """
    try:
        if IP not in pkt:
            stats["skipped_no_ip"] += 1
            return

        ip_layer = pkt[IP]
        src_ip   = ip_layer.src
        dst_ip   = ip_layer.dst
        proto    = ip_layer.proto
        length   = len(pkt)
        ts       = float(pkt.time)

        # Puertos y flags según protocolo
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            flags    = int(pkt[TCP].flags)
            win_size = int(pkt[TCP].window)
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            flags    = 0
            win_size = 0
        elif ICMP in pkt:
            src_port = int(pkt[ICMP].type)
            dst_port = int(pkt[ICMP].code)
            flags    = 0
            win_size = 0
        else:
            # Otro protocolo IP sin capa de transporte conocida
            src_port = 0
            dst_port = 0
            flags    = 0
            win_size = 0

        key = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)

        # Determinar dirección dentro del flujo
        fwd = (src_ip == key[0])

        flow = flows[key]

        # Inicializar flujo si es nuevo
        if not flow:
            flow["src_ip"]    = key[0]
            flow["dst_ip"]    = key[1]
            flow["src_port"]  = key[2]
            flow["dst_port"]  = key[3]
            flow["proto"]     = proto
            flow["fwd_pkts"]  = []
            flow["bwd_pkts"]  = []
            flow["fwd_times"] = []
            flow["bwd_times"] = []
            flow["all_times"] = []
            flow["flags"]     = []
            flow["win_sizes"] = []
            flow["pkt_count"] = 0

        # Truncar flujos demasiado largos (evita OOM)
        if flow["pkt_count"] >= MAX_PKTS_PER_FLOW:
            stats["truncated_flows"] += 1
            return

        # Agregar datos al flujo
        if fwd:
            flow["fwd_pkts"].append(length)
            flow["fwd_times"].append(ts)
        else:
            flow["bwd_pkts"].append(length)
            flow["bwd_times"].append(ts)

        flow["all_times"].append(ts)
        flow["flags"].append(flags)
        flow["win_sizes"].append(win_size)
        flow["pkt_count"] += 1

        stats["processed"] += 1

    except Exception as e:
        stats["errors"] += 1


# =============================================================
#  CÁLCULO DE MÉTRICAS POR FLUJO
# =============================================================

def _compute_flow_metrics(key: tuple, flow: dict) -> dict | None:
    """
    Calcula todas las métricas de un flujo a partir de sus paquetes.
    Retorna None si el flujo no tiene datos suficientes.
    """
    fwd_sizes  = flow.get("fwd_pkts", [])
    bwd_sizes  = flow.get("bwd_pkts", [])
    fwd_times  = sorted(flow.get("fwd_times", []))
    bwd_times  = sorted(flow.get("bwd_times", []))
    all_times  = sorted(flow.get("all_times", []))
    flags_list = flow.get("flags", [])
    win_sizes  = flow.get("win_sizes", [])

    total_pkts = len(fwd_sizes) + len(bwd_sizes)
    if total_pkts == 0:
        return None

    # --- Duración del flujo ---
    t_start      = all_times[0]  if all_times else 0
    t_end        = all_times[-1] if all_times else 0
    flow_duration_ms = (t_end - t_start) * 1000.0  # ms

    # --- Métricas Forward ---
    fwd_pkt_count    = len(fwd_sizes)
    fwd_total_bytes  = sum(fwd_sizes)
    fwd_min          = _safe_stat(fwd_sizes, min)
    fwd_max          = _safe_stat(fwd_sizes, max)
    fwd_mean         = _safe_stat(fwd_sizes, np.mean)
    fwd_std          = _safe_stat(fwd_sizes, np.std)

    # IAT forward
    fwd_iats = list(np.diff(fwd_times)) if len(fwd_times) > 1 else []
    fwd_iat_total = sum(fwd_iats)
    fwd_iat_min   = _safe_stat(fwd_iats, min)
    fwd_iat_max   = _safe_stat(fwd_iats, max)
    fwd_iat_mean  = _safe_stat(fwd_iats, np.mean)
    fwd_iat_std   = _safe_stat(fwd_iats, np.std)

    # Header length estimado (20B IP + 20B TCP/UDP)
    fwd_header_len = fwd_pkt_count * 40

    # --- Métricas Backward ---
    bwd_pkt_count    = len(bwd_sizes)
    bwd_total_bytes  = sum(bwd_sizes)
    bwd_min          = _safe_stat(bwd_sizes, min)
    bwd_max          = _safe_stat(bwd_sizes, max)
    bwd_mean         = _safe_stat(bwd_sizes, np.mean)
    bwd_std          = _safe_stat(bwd_sizes, np.std)

    # IAT backward
    bwd_iats = list(np.diff(bwd_times)) if len(bwd_times) > 1 else []
    bwd_iat_total = sum(bwd_iats)
    bwd_iat_min   = _safe_stat(bwd_iats, min)
    bwd_iat_max   = _safe_stat(bwd_iats, max)
    bwd_iat_mean  = _safe_stat(bwd_iats, np.mean)
    bwd_iat_std   = _safe_stat(bwd_iats, np.std)

    # --- Métricas generales ---
    all_sizes        = fwd_sizes + bwd_sizes
    total_bytes      = sum(all_sizes)
    avg_pkt_size     = _safe_stat(all_sizes, np.mean)

    # Throughput
    duration_s = flow_duration_ms / 1000.0
    packets_per_sec = total_pkts / (duration_s + 1e-9)
    bytes_per_sec   = total_bytes / (duration_s + 1e-9)

    # Paquetes activos (con payload > 0)
    act_data_fwd = len([s for s in fwd_sizes if s > 0])

    # Ventana TCP (init)
    init_win_fwd = win_sizes[0]  if win_sizes else 0
    init_win_bwd = win_sizes[-1] if win_sizes else 0

    # --- Flags TCP ---
    parsed_flags = [_parse_tcp_flags(f) for f in flags_list]
    def _sum_flag(name):
        return sum(f.get(name, 0) for f in parsed_flags)

    fin_count = _sum_flag("FIN")
    syn_count = _sum_flag("SYN")
    rst_count = _sum_flag("RST")
    psh_count = _sum_flag("PSH")
    ack_count = _sum_flag("ACK")
    urg_count = _sum_flag("URG")
    ece_count = _sum_flag("ECE")
    cwr_count = _sum_flag("CWR")

    return {
        # Identificadores de flujo
        "Src IP"                      : flow["src_ip"],
        "Dst IP"                      : flow["dst_ip"],
        "Src Port"                    : flow["src_port"],
        "Destination Port"            : flow["dst_port"],
        "Protocol"                    : flow["proto"],

        # Temporales
        "Flow Duration"               : round(flow_duration_ms, 3),
        "Flow Start"                  : round(t_start, 6),
        "Flow End"                    : round(t_end, 6),

        # Totales
        "Total Packets"               : total_pkts,
        "Total Bytes"                 : total_bytes,
        "Average Packet Size"         : round(avg_pkt_size, 3),
        "Packets/s"                   : round(packets_per_sec, 3),
        "Bytes/s"                     : round(bytes_per_sec, 3),

        # Forward
        "Total Fwd Packets"           : fwd_pkt_count,
        "Total Length of Fwd Packets" : fwd_total_bytes,
        "Fwd Packet Length Min"       : round(fwd_min, 3),
        "Fwd Packet Length Max"       : round(fwd_max, 3),
        "Fwd Packet Length Mean"      : round(fwd_mean, 3),
        "Fwd Packet Length Std"       : round(fwd_std, 3),
        "Fwd IAT Total"               : round(fwd_iat_total, 6),
        "Fwd IAT Min"                 : round(fwd_iat_min, 6),
        "Fwd IAT Max"                 : round(fwd_iat_max, 6),
        "Fwd IAT Mean"                : round(fwd_iat_mean, 6),
        "Fwd IAT Std"                 : round(fwd_iat_std, 6),
        "Fwd Header Length"           : fwd_header_len,
        "Subflow Fwd Packets"         : fwd_pkt_count,
        "Subflow Fwd Bytes"           : fwd_total_bytes,
        "Avg Fwd Segment Size"        : round(fwd_mean, 3),
        "act_data_pkt_fwd"            : act_data_fwd,
        "Init_Win_bytes_forward"      : init_win_fwd,

        # Backward
        "Total Bwd Packets"           : bwd_pkt_count,
        "Total Length of Bwd Packets" : bwd_total_bytes,
        "Bwd Packet Length Min"       : round(bwd_min, 3),
        "Bwd Packet Length Max"       : round(bwd_max, 3),
        "Bwd Packet Length Mean"      : round(bwd_mean, 3),
        "Bwd Packet Length Std"       : round(bwd_std, 3),
        "Bwd IAT Total"               : round(bwd_iat_total, 6),
        "Bwd IAT Min"                 : round(bwd_iat_min, 6),
        "Bwd IAT Max"                 : round(bwd_iat_max, 6),
        "Bwd IAT Mean"                : round(bwd_iat_mean, 6),
        "Bwd IAT Std"                 : round(bwd_iat_std, 6),
        "Subflow Bwd Packets"         : bwd_pkt_count,
        "Subflow Bwd Bytes"           : bwd_total_bytes,
        "Avg Bwd Segment Size"        : round(bwd_mean, 3),
        "Init_Win_bytes_backward"     : init_win_bwd,

        # Flags TCP
        "FIN Flag Count"              : fin_count,
        "SYN Flag Count"              : syn_count,
        "RST Flag Count"              : rst_count,
        "PSH Flag Count"              : psh_count,
        "ACK Flag Count"              : ack_count,
        "URG Flag Count"              : urg_count,
        "ECE Flag Count"              : ece_count,
        "CWR Flag Count"              : cwr_count,
    }


# =============================================================
#  FUNCIÓN PRINCIPAL
# =============================================================

def extract_flows(
    pcap_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lee un archivo .pcap o .pcapng y extrae métricas por flujo.

    Usa PcapReader (streaming) en lugar de rdpcap para soportar
    archivos grandes sin cargar todo en memoria a la vez.

    Parámetros
    ----------
    pcap_path  : Ruta al archivo .pcap o .pcapng.
    chunk_size : Paquetes procesados antes de liberar memoria (gc).
    verbose    : Muestra barra de progreso y resumen final.

    Retorna
    -------
    pd.DataFrame con una fila por flujo y todas las métricas calculadas.
    """
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {pcap_path}")
    if path.suffix.lower() not in (".pcap", ".pcapng"):
        raise ValueError(f"Formato no soportado: {path.suffix}. Use .pcap o .pcapng")

    flows: dict = defaultdict(dict)
    stats = {
        "processed"      : 0,
        "skipped_no_ip"  : 0,
        "errors"         : 0,
        "truncated_flows": 0,
    }

    if verbose:
        print(f"[flow_extractor] Leyendo: {path.name}")
        print(f"[flow_extractor] Chunk size: {chunk_size:,} paquetes")

    # --- Lectura por streaming ---
    chunk_counter = 0
    with PcapReader(str(path)) as reader:
        pbar = tqdm(desc="Procesando paquetes", unit="pkt", disable=not verbose)
        for pkt in reader:
            _process_packet(pkt, flows, stats)
            pbar.update(1)
            chunk_counter += 1

            # Cada chunk_size paquetes forzamos GC para liberar memoria
            if chunk_counter % chunk_size == 0:
                gc.collect()

        pbar.close()

    if verbose:
        print(f"[flow_extractor] Paquetes procesados : {stats['processed']:,}")
        print(f"[flow_extractor] Sin capa IP (omitidos): {stats['skipped_no_ip']:,}")
        print(f"[flow_extractor] Errores              : {stats['errors']:,}")
        print(f"[flow_extractor] Flujos truncados     : {stats['truncated_flows']:,}")
        print(f"[flow_extractor] Flujos candidatos    : {len(flows):,}")

    # --- Calcular métricas por flujo ---
    records = []
    failed  = 0

    with tqdm(
        total=len(flows),
        desc="Calculando metricas",
        unit="flujo",
        disable=not verbose,
    ) as pbar:
        for key, flow_data in flows.items():
            try:
                metrics = _compute_flow_metrics(key, flow_data)
                if metrics is not None:
                    records.append(metrics)
            except Exception as e:
                failed += 1
            finally:
                pbar.update(1)

    # Liberar memoria de flows procesados
    del flows
    gc.collect()

    if verbose:
        print(f"[flow_extractor] Flujos validos       : {len(records):,}")
        print(f"[flow_extractor] Flujos con error     : {failed:,}")

    if not records:
        print("[flow_extractor] ADVERTENCIA: No se extrajeron flujos validos.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ordenar por tiempo de inicio del flujo
    if "Flow Start" in df.columns:
        df = df.sort_values("Flow Start").reset_index(drop=True)

    if verbose:
        print(f"[flow_extractor] DataFrame final: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    return df