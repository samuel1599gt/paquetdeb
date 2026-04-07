"""
topology_mapper.py
==================
Construye el grafo de topología de red a partir de los flujos
agrupados y clasifica cada dispositivo según su comportamiento.

Responsabilidades:
    - Construir grafo dirigido (NetworkX) de comunicaciones
    - Clasificar dispositivos: servidor, cliente, gateway,
      router, impresora, IoT, cámara IP, dispositivo móvil,
      workstation/laptop, dominio externo
    - Calcular métricas de centralidad por nodo
    - Detectar subredes presentes en la captura
    - Exportar el grafo en formato JSON para el reporte HTML
    - Generar imagen PNG del grafo para el reporte PDF
"""

import warnings
import ipaddress
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# =============================================================
#  RANGOS Y PUERTOS CONOCIDOS PARA CLASIFICACIÓN
# =============================================================

# Puertos típicos por rol de dispositivo
ROLE_PORT_SIGNATURES = {
    "servidor_web"    : {80, 443, 8080, 8443},
    "servidor_dns"    : {53, 5353},
    "servidor_dhcp"   : {67, 68},
    "servidor_ssh"    : {22},
    "servidor_ftp"    : {20, 21},
    "servidor_db"     : {3306, 5432, 1433, 27017, 6379, 9200},
    "servidor_mail"   : {25, 465, 587, 110, 143, 993, 995},
    "servidor_rdp"    : {3389},
    "servidor_ldap"   : {389, 636},
    "servidor_ntp"    : {123},
    "impresora"       : {9100, 515, 631},
    "camara_ip"       : {554, 8554, 37777, 34567},
    "iot"             : {1883, 8883, 5683, 1900, 5353},
}

# Rangos de IPs privadas
PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]

# Colores por rol para visualización
ROLE_COLORS = {
    "servidor_web"    : "#4A90D9",
    "servidor_dns"    : "#7B68EE",
    "servidor_dhcp"   : "#9B59B6",
    "servidor_ssh"    : "#2ECC71",
    "servidor_ftp"    : "#27AE60",
    "servidor_db"     : "#1ABC9C",
    "servidor_mail"   : "#3498DB",
    "servidor_rdp"    : "#E67E22",
    "servidor_ntp"    : "#F39C12",
    "impresora"       : "#95A5A6",
    "camara_ip"       : "#E74C3C",
    "iot"             : "#FF6B6B",
    "gateway"         : "#F1C40F",
    "router"          : "#E67E22",
    "workstation"     : "#BDC3C7",
    "dispositivo_movil": "#85C1E9",
    "externo"         : "#FADBD8",
    "desconocido"     : "#D5D8DC",
}


# =============================================================
#  HELPERS
# =============================================================

def _is_private(ip: str) -> bool:
    """Retorna True si la IP es privada/local."""
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in PRIVATE_RANGES)
    except ValueError:
        return False


def _get_subnet(ip: str) -> str:
    """Retorna la subred /24 de una IP privada, o 'externo'."""
    try:
        addr = ipaddress.ip_address(ip)
        if _is_private(addr):
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        return "externo"
    except ValueError:
        return "desconocido"


def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


# =============================================================
#  CONSTRUCCIÓN DEL GRAFO
# =============================================================

def _build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Construye un grafo dirigido donde:
    - Nodos = IPs únicas
    - Aristas = comunicaciones entre pares (Src IP → Dst IP)
    - Peso de arista = total_bytes transferidos
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        src = str(row["Src IP"])
        dst = str(row["Dst IP"])

        # Atributos de arista
        edge_attrs = {
            "flow_count"  : int(row.get("flow_count", 1)),
            "total_bytes" : int(row.get("total_bytes", 0)),
            "total_packets": int(row.get("total_packets", 0)),
        }
        if "avg_duration" in row:
            edge_attrs["avg_duration_ms"] = float(row.get("avg_duration", 0))
        if "bytes_ratio_fwd_bwd" in row:
            edge_attrs["bytes_ratio"] = float(row.get("bytes_ratio_fwd_bwd", 1.0))

        if G.has_edge(src, dst):
            # Acumular si ya existe
            G[src][dst]["flow_count"]   += edge_attrs["flow_count"]
            G[src][dst]["total_bytes"]  += edge_attrs["total_bytes"]
            G[src][dst]["total_packets"]+= edge_attrs["total_packets"]
        else:
            G.add_edge(src, dst, **edge_attrs)

        # Atributos de nodo iniciales
        for node in (src, dst):
            if node not in G.nodes:
                G.add_node(node)
            if "is_private" not in G.nodes[node]:
                G.nodes[node]["is_private"] = _is_private(node)
                G.nodes[node]["subnet"]     = _get_subnet(node)

    return G


# =============================================================
#  MÉTRICAS DE NODO
# =============================================================

def _compute_node_metrics(G: nx.DiGraph, df: pd.DataFrame) -> dict:
    """
    Calcula métricas por nodo a partir del grafo y del DataFrame:
    - in_degree / out_degree
    - total_bytes_sent / received
    - unique_dst_contacted / unique_src_received
    - puertos destino usados
    - centralidad (degree, betweenness)
    """
    node_metrics = defaultdict(lambda: {
        "total_bytes_sent"    : 0,
        "total_bytes_recv"    : 0,
        "total_packets_sent"  : 0,
        "total_packets_recv"  : 0,
        "flow_count_out"      : 0,
        "flow_count_in"       : 0,
        "unique_dst"          : set(),
        "unique_src"          : set(),
        "dst_ports_used"      : set(),
        "src_ports_used"      : set(),
        "protocols"           : set(),
    })

    for _, row in df.iterrows():
        src = str(row["Src IP"])
        dst = str(row["Dst IP"])

        bytes_val   = int(row.get("total_bytes", 0))
        packets_val = int(row.get("total_packets", 0))
        flows_val   = int(row.get("flow_count", 1))

        node_metrics[src]["total_bytes_sent"]   += bytes_val
        node_metrics[src]["total_packets_sent"] += packets_val
        node_metrics[src]["flow_count_out"]     += flows_val
        node_metrics[src]["unique_dst"].add(dst)

        node_metrics[dst]["total_bytes_recv"]   += bytes_val
        node_metrics[dst]["total_packets_recv"] += packets_val
        node_metrics[dst]["flow_count_in"]      += flows_val
        node_metrics[dst]["unique_src"].add(src)

        # Puertos
        if _safe_col(df, "Destination Port"):
            port = row.get("Destination Port")
            if pd.notna(port):
                node_metrics[dst]["dst_ports_used"].add(int(port))
        if _safe_col(df, "Src Port"):
            port = row.get("Src Port")
            if pd.notna(port):
                node_metrics[src]["src_ports_used"].add(int(port))

        # Protocolos
        if _safe_col(df, "Protocol"):
            proto = row.get("Protocol")
            if pd.notna(proto):
                node_metrics[src]["protocols"].add(int(proto))
                node_metrics[dst]["protocols"].add(int(proto))

    # Centralidad (costosa en grafos grandes — limitada a 500 nodos)
    degree_centrality     = {}
    betweenness_centrality = {}
    if len(G.nodes) <= 500:
        try:
            degree_centrality      = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            pass

    # Serializar conjuntos a listas/conteos
    result = {}
    for ip, m in node_metrics.items():
        result[ip] = {
            "total_bytes_sent"     : m["total_bytes_sent"],
            "total_bytes_recv"     : m["total_bytes_recv"],
            "total_packets_sent"   : m["total_packets_sent"],
            "total_packets_recv"   : m["total_packets_recv"],
            "flow_count_out"       : m["flow_count_out"],
            "flow_count_in"        : m["flow_count_in"],
            "unique_dst_count"     : len(m["unique_dst"]),
            "unique_src_count"     : len(m["unique_src"]),
            "dst_ports_used"       : sorted(list(m["dst_ports_used"]))[:20],
            "src_ports_used"       : sorted(list(m["src_ports_used"]))[:20],
            "unique_ports_dst"     : len(m["dst_ports_used"]),
            "protocols"            : sorted(list(m["protocols"])),
            "is_private"           : _is_private(ip),
            "subnet"               : _get_subnet(ip),
            "degree_centrality"    : round(degree_centrality.get(ip, 0.0), 4),
            "betweenness_centrality": round(betweenness_centrality.get(ip, 0.0), 4),
        }

    return result


# =============================================================
#  CLASIFICACIÓN DE DISPOSITIVOS
# =============================================================

def _classify_device(ip: str, metrics: dict, df: pd.DataFrame) -> str:
    """
    Clasifica un dispositivo según su comportamiento en la red.

    Lógica por prioridad:
    1. IPs externas → 'externo'
    2. Puertos escuchados → rol de servidor específico
    3. Comportamiento de gateway (muchos dst, alto betweenness)
    4. Solo origina tráfico → workstation / cliente
    5. Desconocido
    """
    if not metrics.get("is_private", True):
        return "externo"

    dst_ports = set(metrics.get("dst_ports_used", []))
    unique_dst = metrics.get("unique_dst_count", 0)
    unique_src = metrics.get("unique_src_count", 0)
    flow_in    = metrics.get("flow_count_in", 0)
    flow_out   = metrics.get("flow_count_out", 0)
    betweenness = metrics.get("betweenness_centrality", 0.0)
    bytes_sent = metrics.get("total_bytes_sent", 0)
    bytes_recv = metrics.get("total_bytes_recv", 0)

    # --- Puertos de servidor específicos ---
    for role, ports in ROLE_PORT_SIGNATURES.items():
        if dst_ports & ports:
            return role

    # --- Gateway / Router ---
    # Un gateway tiene alta centralidad y contacta muchas IPs
    if betweenness > 0.3 or unique_dst > 20:
        if flow_in > 0 and flow_out > 0:
            return "gateway"

    # --- Router (solo reenvía, poco tráfico propio) ---
    if unique_dst > 10 and unique_src > 10 and betweenness > 0.15:
        return "router"

    # --- Dispositivo móvil ---
    # Mucho tráfico saliente a puertos web, pocas conexiones entrantes
    web_ports = {80, 443, 8080}
    if dst_ports & web_ports and flow_in == 0 and unique_dst > 3:
        if bytes_sent > bytes_recv * 0.5:
            return "dispositivo_movil"

    # --- Workstation / Laptop ---
    # Origina tráfico, recibe poco, usa puertos efímeros altos
    high_ports = [p for p in metrics.get("src_ports_used", []) if p > 1024]
    if flow_out > 0 and flow_in <= 2 and len(high_ports) > 2:
        return "workstation"

    # --- Servidor genérico ---
    # Recibe mucho más de lo que envía
    if flow_in > flow_out * 2 and flow_in > 5:
        return "servidor_generico"

    # --- Cliente genérico ---
    if flow_out > 0 and flow_in == 0:
        return "cliente"

    return "desconocido"


def _classify_all_devices(
    node_metrics: dict,
    df: pd.DataFrame,
) -> dict:
    """
    Aplica clasificación a todos los nodos y retorna
    dict {ip: role}.
    """
    return {
        ip: _classify_device(ip, metrics, df)
        for ip, metrics in node_metrics.items()
    }


# =============================================================
#  DETECCIÓN DE SUBREDES
# =============================================================

def _detect_subnets(node_metrics: dict) -> dict:
    """
    Agrupa los dispositivos por subred /24 detectada.
    """
    subnets = defaultdict(list)
    for ip, metrics in node_metrics.items():
        subnet = metrics.get("subnet", "desconocido")
        subnets[subnet].append(ip)
    return dict(subnets)


# =============================================================
#  VISUALIZACIÓN (PNG para PDF)
# =============================================================

def _draw_graph(
    G: nx.DiGraph,
    device_roles: dict,
    output_path: str,
    max_nodes: int = 80,
) -> None:
    """
    Genera imagen PNG del grafo de topología.
    Limita a max_nodes nodos para legibilidad.
    Si el grafo es muy grande, muestra los nodos con más tráfico.
    """
    # Seleccionar subgrafo de los nodos más importantes
    if len(G.nodes) > max_nodes:
        # Ordenar por grado total descendente
        top_nodes = sorted(
            G.nodes,
            key=lambda n: G.in_degree(n) + G.out_degree(n),
            reverse=True,
        )[:max_nodes]
        G_draw = G.subgraph(top_nodes).copy()
    else:
        G_draw = G.copy()

    if len(G_draw.nodes) == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Layout
    try:
        pos = nx.spring_layout(G_draw, k=2.5, iterations=50, seed=42)
    except Exception:
        pos = nx.random_layout(G_draw, seed=42)

    # Colores y tamaños de nodos
    node_colors = []
    node_sizes  = []
    for node in G_draw.nodes:
        role  = device_roles.get(node, "desconocido")
        color = ROLE_COLORS.get(role, ROLE_COLORS["desconocido"])
        node_colors.append(color)
        degree = G_draw.in_degree(node) + G_draw.out_degree(node)
        node_sizes.append(max(300, min(3000, degree * 150)))

    # Pesos de aristas
    edge_weights = []
    max_bytes = max(
        (d.get("total_bytes", 1) for _, _, d in G_draw.edges(data=True)),
        default=1,
    )
    for _, _, data in G_draw.edges(data=True):
        w = data.get("total_bytes", 1) / max_bytes
        edge_weights.append(max(0.3, w * 3))

    # Dibujar aristas
    nx.draw_networkx_edges(
        G_draw, pos,
        width=edge_weights,
        alpha=0.4,
        edge_color="#aaaaaa",
        arrows=True,
        arrowsize=10,
        ax=ax,
    )

    # Dibujar nodos
    nx.draw_networkx_nodes(
        G_draw, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )

    # Etiquetas (solo IPs cortas si hay muchos nodos)
    if len(G_draw.nodes) <= 40:
        labels = {n: n for n in G_draw.nodes}
    else:
        # Mostrar solo los 20 más conectados
        top_20 = sorted(
            G_draw.nodes,
            key=lambda n: G_draw.in_degree(n) + G_draw.out_degree(n),
            reverse=True,
        )[:20]
        labels = {n: n for n in top_20}

    nx.draw_networkx_labels(
        G_draw, pos,
        labels=labels,
        font_size=7,
        font_color="white",
        ax=ax,
    )

    # Leyenda de roles
    unique_roles = set(device_roles.get(n, "desconocido") for n in G_draw.nodes)
    legend_patches = [
        mpatches.Patch(color=ROLE_COLORS.get(r, "#D5D8DC"), label=r.replace("_", " "))
        for r in sorted(unique_roles)
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=8,
        facecolor="#2d2d44",
        edgecolor="#555555",
        labelcolor="white",
        framealpha=0.8,
    )

    title = f"Topología de Red — {len(G_draw.nodes)} nodos"
    if len(G.nodes) > max_nodes:
        title += f" (top {max_nodes} de {len(G.nodes)} totales)"
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# =============================================================
#  EXPORTAR JSON PARA REPORTE HTML
# =============================================================

def _build_topology_json(
    G: nx.DiGraph,
    node_metrics: dict,
    device_roles: dict,
    subnets: dict,
) -> dict:
    """
    Construye el diccionario de topología para exportar a JSON
    y usar en el reporte HTML interactivo.
    """
    nodes = []
    for ip in G.nodes:
        m    = node_metrics.get(ip, {})
        role = device_roles.get(ip, "desconocido")
        nodes.append({
            "id"                   : ip,
            "role"                 : role,
            "color"                : ROLE_COLORS.get(role, "#D5D8DC"),
            "is_private"           : m.get("is_private", False),
            "subnet"               : m.get("subnet", "desconocido"),
            "total_bytes_sent"     : m.get("total_bytes_sent", 0),
            "total_bytes_recv"     : m.get("total_bytes_recv", 0),
            "total_packets_sent"   : m.get("total_packets_sent", 0),
            "total_packets_recv"   : m.get("total_packets_recv", 0),
            "flow_count_out"       : m.get("flow_count_out", 0),
            "flow_count_in"        : m.get("flow_count_in", 0),
            "unique_dst_count"     : m.get("unique_dst_count", 0),
            "unique_src_count"     : m.get("unique_src_count", 0),
            "unique_ports_dst"     : m.get("unique_ports_dst", 0),
            "dst_ports_used"       : m.get("dst_ports_used", [])[:10],
            "degree_centrality"    : m.get("degree_centrality", 0.0),
            "betweenness_centrality": m.get("betweenness_centrality", 0.0),
        })

    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append({
            "source"        : src,
            "target"        : dst,
            "flow_count"    : data.get("flow_count", 1),
            "total_bytes"   : data.get("total_bytes", 0),
            "total_packets" : data.get("total_packets", 0),
            "avg_duration_ms": data.get("avg_duration_ms", 0),
        })

    # Resumen de roles
    role_summary = defaultdict(int)
    for role in device_roles.values():
        role_summary[role] += 1

    return {
        "nodes"        : nodes,
        "edges"        : edges,
        "subnets"      : {k: v for k, v in subnets.items()},
        "role_summary" : dict(role_summary),
        "total_nodes"  : len(nodes),
        "total_edges"  : len(edges),
        "total_subnets": len(subnets),
    }


# =============================================================
#  FUNCIÓN PRINCIPAL
# =============================================================

def map_topology(
    df: pd.DataFrame,
    output_dir: str | None = None,
    graph_png: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Construye la topología de red y clasifica los dispositivos.

    Parámetros
    ----------
    df          : DataFrame de flujos agrupados (salida de flow_grouper.py).
    output_dir  : Directorio para guardar topology.json.
    graph_png   : Ruta para guardar la imagen PNG del grafo.
                  Si es None, no se genera imagen.
    verbose     : Imprime progreso si True.

    Retorna
    -------
    dict con claves:
        nodes        — lista de nodos con métricas y rol
        edges        — lista de aristas con peso
        subnets      — agrupación por subred /24
        role_summary — conteo de dispositivos por rol
        total_nodes  — total de IPs únicas
        total_edges  — total de conexiones únicas
    """
    import json

    if df is None or df.empty:
        raise ValueError("[topology_mapper] DataFrame de entrada vacío.")

    if verbose:
        print(f"[topology_mapper] Pares de flujo : {len(df):,}")

    # --- 1. Construir grafo ---
    if verbose:
        print("[topology_mapper] Construyendo grafo de red...")
    G = _build_graph(df)

    if verbose:
        print(f"[topology_mapper] Nodos (IPs)    : {len(G.nodes):,}")
        print(f"[topology_mapper] Aristas (conex.): {len(G.edges):,}")

    # --- 2. Métricas por nodo ---
    if verbose:
        print("[topology_mapper] Calculando metricas por nodo...")
    node_metrics = _compute_node_metrics(G, df)

    # --- 3. Clasificar dispositivos ---
    if verbose:
        print("[topology_mapper] Clasificando dispositivos...")
    device_roles = _classify_all_devices(node_metrics, df)

    # Agregar rol al grafo
    for ip, role in device_roles.items():
        if ip in G.nodes:
            G.nodes[ip]["role"] = role

    # Resumen de clasificación
    if verbose:
        role_counts = defaultdict(int)
        for role in device_roles.values():
            role_counts[role] += 1
        print("[topology_mapper] Clasificacion:")
        for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {role:<25}: {count}")

    # --- 4. Detectar subredes ---
    if verbose:
        print("[topology_mapper] Detectando subredes...")
    subnets = _detect_subnets(node_metrics)
    if verbose:
        print(f"[topology_mapper] Subredes detectadas: {len(subnets)}")

    # --- 5. Construir JSON de topología ---
    topology = _build_topology_json(G, node_metrics, device_roles, subnets)

    # --- 6. Exportar JSON ---
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / "topology.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topology, f, indent=4, ensure_ascii=False)
        if verbose:
            print(f"[topology_mapper] topology.json guardado en: {json_path}")

    # --- 7. Generar imagen PNG ---
    if graph_png:
        if verbose:
            print(f"[topology_mapper] Generando imagen del grafo...")
        try:
            _draw_graph(G, device_roles, graph_png)
            if verbose:
                print(f"[topology_mapper] Imagen guardada en: {graph_png}")
        except Exception as e:
            print(f"[topology_mapper] No se pudo generar imagen: {e}")

    return topology