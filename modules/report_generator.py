"""
report_generator.py
===================
Genera el reporte final en dos formatos:
    - PDF  : con ReportLab (tablas, gráficas, topología)
    - HTML : interactivo con Jinja2 (filtros, gráficas Chart.js,
             topología de red navegable)

Secciones del reporte:
    1. Resumen ejecutivo
    2. Calidad del dataset (quality_report)
    3. Métricas globales de la captura
    4. Alertas forenses (por severidad)
    5. Topología de red y clasificación de dispositivos
    6. Distribución de protocolos, puertos y flags
    7. Top IPs origen y destino
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

warnings.filterwarnings("ignore")

# =============================================================
#  PALETA DE COLORES
# =============================================================

SEVERITY_COLORS = {
    "CRITICA" : "#C0392B",
    "ALTA"    : "#E67E22",
    "MEDIA"   : "#F1C40F",
    "BAJA"    : "#2ECC71",
}

SEVERITY_COLORS_RL = {
    "CRITICA" : colors.HexColor("#C0392B"),
    "ALTA"    : colors.HexColor("#E67E22"),
    "MEDIA"   : colors.HexColor("#F1C40F"),
    "BAJA"    : colors.HexColor("#27AE60"),
}

CHART_COLORS = [
    "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
    "#E91E63", "#00BCD4",
]

ROLE_COLORS_HTML = {
    "servidor_web"     : "#4A90D9",
    "servidor_dns"     : "#7B68EE",
    "servidor_dhcp"    : "#9B59B6",
    "servidor_ssh"     : "#2ECC71",
    "servidor_ftp"     : "#27AE60",
    "servidor_db"      : "#1ABC9C",
    "servidor_mail"    : "#3498DB",
    "servidor_rdp"     : "#E67E22",
    "servidor_ntp"     : "#F39C12",
    "servidor_generico": "#5D6D7E",
    "impresora"        : "#95A5A6",
    "camara_ip"        : "#E74C3C",
    "iot"              : "#FF6B6B",
    "gateway"          : "#F1C40F",
    "router"           : "#E67E22",
    "workstation"      : "#BDC3C7",
    "cliente"          : "#AED6F1",
    "dispositivo_movil": "#85C1E9",
    "externo"          : "#FADBD8",
    "desconocido"      : "#D5D8DC",
}


# =============================================================
#  HELPERS COMUNES
# =============================================================

def _safe(val, default="N/A"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def _fmt_bytes(b):
    try:
        b = int(b)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"
    except Exception:
        return str(b)


def _fmt_num(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _plots_dir(base_dir: str) -> Path:
    p = Path(base_dir) / "plots_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================
#  GENERACIÓN DE GRÁFICAS (compartidas PDF y HTML)
# =============================================================

def _plot_bar(labels, values, title, outfile, color=None, rotation=35):
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = color if color else CHART_COLORS[:len(labels)]
    ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=12, pad=8)
    ax.tick_params(axis="x", rotation=rotation, labelsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pie(labels, values, title, outfile):
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=CHART_COLORS[:len(labels)],
        pctdistance=0.82,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.legend(
        wedges, labels,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=8,
    )
    ax.set_title(title, fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_hbar(labels, values, title, outfile):
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.45)))
    y_pos = range(len(labels))
    ax.barh(
        list(y_pos), values,
        color=CHART_COLORS[:len(labels)],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=12, pad=8)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def _generate_charts(report: dict, plots_dir: Path) -> dict:
    """
    Genera todas las gráficas y retorna dict {nombre: ruta_png}.
    """
    charts = {}
    gm  = report.get("global_metrics", {})
    topo = report.get("topology", {})
    alerts = report.get("alerts", [])
    alert_summary = report.get("alert_summary", {})

    # --- Distribución de protocolos ---
    proto = gm.get("protocol_distribution", {})
    if proto:
        p = plots_dir / "proto_dist.png"
        _plot_pie(list(proto.keys()), list(proto.values()),
                  "Distribución de Protocolos", str(p))
        charts["proto_dist"] = str(p)

    # --- Flags TCP ---
    flags = gm.get("flags_summary", {})
    if flags:
        p = plots_dir / "tcp_flags.png"
        _plot_bar(
            [k.upper() for k in flags.keys()],
            list(flags.values()),
            "Conteo de Flags TCP",
            str(p),
            color=["#E74C3C", "#3498DB", "#E67E22",
                   "#2ECC71", "#9B59B6", "#1ABC9C",
                   "#F39C12", "#34495E"],
            rotation=0,
        )
        charts["tcp_flags"] = str(p)

    # --- Top IPs origen ---
    top_src = gm.get("top_src_ips", {})
    if top_src:
        items = list(top_src.items())[:8]
        p = plots_dir / "top_src_ips.png"
        _plot_hbar([i[0] for i in items], [i[1] for i in items],
                   "Top IPs Origen (por flujos)", str(p))
        charts["top_src"] = str(p)

    # --- Top IPs destino ---
    top_dst = gm.get("top_dst_ips", {})
    if top_dst:
        items = list(top_dst.items())[:8]
        p = plots_dir / "top_dst_ips.png"
        _plot_hbar([i[0] for i in items], [i[1] for i in items],
                   "Top IPs Destino (por flujos)", str(p))
        charts["top_dst"] = str(p)

    # --- Top puertos destino ---
    top_ports = gm.get("top_dst_ports", {})
    if top_ports:
        items = list(top_ports.items())[:8]
        p = plots_dir / "top_ports.png"
        _plot_bar([str(i[0]) for i in items], [i[1] for i in items],
                  "Top Puertos Destino", str(p))
        charts["top_ports"] = str(p)

    # --- Alertas por severidad ---
    by_sev = alert_summary.get("by_severity", {})
    if by_sev:
        p = plots_dir / "alerts_severity.png"
        sev_order = ["CRITICA", "ALTA", "MEDIA", "BAJA"]
        labels = [s for s in sev_order if s in by_sev]
        values = [by_sev[s] for s in labels]
        colors_sev = [SEVERITY_COLORS[s] for s in labels]
        _plot_bar(labels, values, "Alertas por Severidad", str(p),
                  color=colors_sev, rotation=0)
        charts["alerts_severity"] = str(p)

    # --- Alertas por tipo ---
    by_type = alert_summary.get("by_type", {})
    if by_type:
        items = sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:8]
        p = plots_dir / "alerts_type.png"
        _plot_hbar([i[0] for i in items], [i[1] for i in items],
                   "Alertas por Tipo", str(p))
        charts["alerts_type"] = str(p)

    # --- Roles de dispositivos ---
    role_summary = topo.get("role_summary", {})
    if role_summary:
        items = sorted(role_summary.items(), key=lambda x: x[1], reverse=True)[:10]
        p = plots_dir / "device_roles.png"
        role_colors = [ROLE_COLORS_HTML.get(i[0], "#BDC3C7") for i in items]
        _plot_bar(
            [i[0].replace("_", " ") for i in items],
            [i[1] for i in items],
            "Clasificación de Dispositivos",
            str(p),
            color=role_colors,
        )
        charts["device_roles"] = str(p)

    return charts


# =============================================================
#  GENERADOR PDF
# =============================================================

def _pdf_styles():
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "title", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#1A252F"),
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1", parent=styles["Heading1"],
            fontSize=15, textColor=colors.HexColor("#2C3E50"),
            spaceBefore=14, spaceAfter=4,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2", parent=styles["Heading2"],
            fontSize=12, textColor=colors.HexColor("#34495E"),
            spaceBefore=10, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=styles["Normal"],
            fontSize=9, leading=14,
        ),
        "small": ParagraphStyle(
            "small", parent=styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#666666"),
        ),
        "alert_critica": ParagraphStyle(
            "alert_critica", parent=styles["Normal"],
            fontSize=9, backColor=colors.HexColor("#FDEDEC"),
            borderColor=colors.HexColor("#C0392B"), borderWidth=1,
            borderPad=4, leading=13,
        ),
        "alert_alta": ParagraphStyle(
            "alert_alta", parent=styles["Normal"],
            fontSize=9, backColor=colors.HexColor("#FEF9E7"),
            borderPad=4, leading=13,
        ),
    }
    return styles, custom


def _pdf_table(data, col_widths, header_bg=colors.HexColor("#2C3E50")):
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F7F9FC")]),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t


def _pdf_severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#95A5A6")
    return f'<font color="{color}"><b>[{severity}]</b></font>'


def generate_pdf(report: dict, output_path: str, plots_dir: Path) -> None:
    """Genera el reporte en formato PDF."""

    styles, custom = _pdf_styles()
    doc  = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []
    W = A4[0] - 4*cm  # ancho útil

    gm           = report.get("global_metrics", {})
    quality      = report.get("quality_report", {})
    alerts       = report.get("alerts", [])
    alert_summary= report.get("alert_summary", {})
    topo         = report.get("topology", {})
    charts       = report.get("_charts", {})
    pcap_name    = report.get("pcap_name", "desconocido")

    # ── Portada ──────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("PaketIQ", custom["title"]))
    story.append(Paragraph("Reporte de Análisis Forense de Red", custom["h1"]))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#2C3E50")))
    story.append(Spacer(1, 0.3*cm))

    meta = [
        ["Archivo analizado", pcap_name],
        ["Fecha de análisis", _now()],
        ["Total de flujos",   _fmt_num(gm.get("total_flows", 0))],
        ["IPs únicas",        _fmt_num(gm.get("unique_src_ips", 0))],
        ["Total alertas",     _fmt_num(alert_summary.get("total_alerts", 0))],
    ]
    story.append(_pdf_table(meta, [W*0.4, W*0.6]))
    story.append(Spacer(1, 0.5*cm))

    # ── Resumen ejecutivo ─────────────────────────────────────
    story.append(Paragraph("1. Resumen Ejecutivo", custom["h1"]))

    crit = alert_summary.get("by_severity", {}).get("CRITICA", 0)
    alta = alert_summary.get("by_severity", {}).get("ALTA", 0)
    exec_text = (
        f"Se analizaron <b>{_fmt_num(gm.get('total_flows',0))}</b> flujos de red "
        f"capturados en <b>{pcap_name}</b>. "
        f"Se identificaron <b>{_fmt_num(alert_summary.get('total_alerts',0))}</b> alertas "
        f"de seguridad, de las cuales <b>{crit}</b> son de severidad CRÍTICA "
        f"y <b>{alta}</b> de severidad ALTA. "
        f"Se detectaron <b>{_fmt_num(topo.get('total_nodes',0))}</b> dispositivos "
        f"en <b>{topo.get('total_subnets',0)}</b> subred(es)."
    )
    story.append(Paragraph(exec_text, custom["body"]))
    story.append(Spacer(1, 0.4*cm))

    # ── Calidad del dataset ───────────────────────────────────
    if quality:
        story.append(Paragraph("2. Calidad del Dataset", custom["h1"]))
        q_data = [
            ["Métrica", "Valor"],
            ["Filas originales",      _fmt_num(quality.get("original_rows", 0))],
            ["Filas finales",         _fmt_num(quality.get("final_rows", 0))],
            ["Filas eliminadas",      f"{_fmt_num(quality.get('rows_removed',0))} ({quality.get('rows_removed_pct','0%')})"],
            ["Duplicados eliminados", _fmt_num(quality.get("dropped_duplicates", 0))],
            ["IPs inválidas",         _fmt_num(quality.get("dropped_invalid_ips", 0))],
            ["NaN imputados",         _fmt_num(quality.get("imputed_nan", 0))],
            ["Inf reemplazados",      _fmt_num(quality.get("replaced_inf", 0))],
            ["Negativos corregidos",  _fmt_num(quality.get("clipped_negative", 0))],
        ]
        story.append(_pdf_table(q_data, [W*0.55, W*0.45]))
        story.append(Spacer(1, 0.4*cm))

    # ── Métricas globales ─────────────────────────────────────
    story.append(Paragraph("3. Métricas Globales de la Captura", custom["h1"]))
    flow_dur = gm.get("flow_duration", {})
    pkt_size = gm.get("packet_size", {})
    g_data = [
        ["Métrica", "Valor"],
        ["Total flujos",           _fmt_num(gm.get("total_flows", 0))],
        ["IPs origen únicas",      _fmt_num(gm.get("unique_src_ips", 0))],
        ["IPs destino únicas",     _fmt_num(gm.get("unique_dst_ips", 0))],
        ["Total bytes",            _fmt_bytes(gm.get("total_bytes", 0))],
        ["Total paquetes",         _fmt_num(gm.get("total_packets", 0))],
        ["Avg paquetes/s",         _safe(gm.get("avg_packets_per_sec"))],
        ["Avg bytes/s",            _fmt_bytes(gm.get("avg_bytes_per_sec", 0))],
        ["Duración avg flujo",     f"{_safe(flow_dur.get('avg_ms'), 0):.1f} ms"],
        ["Duración max flujo",     f"{_safe(flow_dur.get('max_ms'), 0):.1f} ms"],
        ["Tamaño avg paquete",     f"{_safe(pkt_size.get('avg'), 0):.1f} B"],
    ]
    story.append(_pdf_table(g_data, [W*0.55, W*0.45]))
    story.append(Spacer(1, 0.3*cm))

    # Gráficas de métricas
    for key in ("proto_dist", "tcp_flags", "top_src", "top_dst", "top_ports"):
        if key in charts and Path(charts[key]).exists():
            story.append(Image(charts[key], width=W, height=W*0.45))
            story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # ── Alertas ───────────────────────────────────────────────
    story.append(Paragraph("4. Alertas Forenses", custom["h1"]))

    # Resumen por severidad
    by_sev = alert_summary.get("by_severity", {})
    if by_sev:
        sev_data = [["Severidad", "Cantidad"]] + [
            [k, str(v)] for k, v in
            sorted(by_sev.items(), key=lambda x: ["CRITICA","ALTA","MEDIA","BAJA"].index(x[0])
                   if x[0] in ["CRITICA","ALTA","MEDIA","BAJA"] else 99)
        ]
        story.append(_pdf_table(
            sev_data, [W*0.6, W*0.4],
            header_bg=colors.HexColor("#922B21"),
        ))
        story.append(Spacer(1, 0.3*cm))

    if "alerts_severity" in charts:
        story.append(Image(charts["alerts_severity"], width=W*0.7, height=W*0.35))
        story.append(Spacer(1, 0.3*cm))
    if "alerts_type" in charts:
        story.append(Image(charts["alerts_type"], width=W, height=W*0.4))
        story.append(Spacer(1, 0.3*cm))

    # Listado de alertas (máx 50 para no inflar el PDF)
    story.append(Paragraph("Detalle de Alertas", custom["h2"]))
    sev_order = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAJA": 3}
    sorted_alerts = sorted(
        alerts,
        key=lambda a: sev_order.get(a.get("severidad", "BAJA"), 4),
    )[:50]

    alert_table_data = [["Severidad", "Tipo", "Src IP", "Dst IP", "Detalle"]]
    for a in sorted_alerts:
        alert_table_data.append([
            a.get("severidad", ""),
            a.get("tipo", ""),
            a.get("src_ip", ""),
            a.get("dst_ip", ""),
            a.get("detalle", "")[:80],
        ])

    if len(alert_table_data) > 1:
        t = Table(alert_table_data, colWidths=[W*0.10, W*0.20, W*0.15, W*0.15, W*0.40])
        style_cmds = [
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2C3E50")),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 7),
            ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ]
        # Color por severidad en columna 0
        for i, a in enumerate(sorted_alerts, start=1):
            sev = a.get("severidad", "BAJA")
            bg  = colors.HexColor(SEVERITY_COLORS.get(sev, "#FFFFFF"))
            style_cmds.append(("BACKGROUND", (0, i), (0, i), bg))
            style_cmds.append(("TEXTCOLOR",  (0, i), (0, i), colors.white))
        t.setStyle(TableStyle(style_cmds))
        story.append(t)

    story.append(PageBreak())

    # ── Topología ─────────────────────────────────────────────
    story.append(Paragraph("5. Topología de Red", custom["h1"]))

    topo_meta = [
        ["Métrica", "Valor"],
        ["Dispositivos detectados", _fmt_num(topo.get("total_nodes", 0))],
        ["Conexiones únicas",       _fmt_num(topo.get("total_edges", 0))],
        ["Subredes detectadas",     _fmt_num(topo.get("total_subnets", 0))],
    ]
    story.append(_pdf_table(topo_meta, [W*0.55, W*0.45]))
    story.append(Spacer(1, 0.3*cm))

    # Imagen del grafo
    topo_img = report.get("_topology_png")
    if topo_img and Path(topo_img).exists():
        story.append(Image(topo_img, width=W, height=W*0.65))
        story.append(Spacer(1, 0.3*cm))

    # Gráfica de roles
    if "device_roles" in charts:
        story.append(Image(charts["device_roles"], width=W, height=W*0.4))
        story.append(Spacer(1, 0.3*cm))

    # Tabla de dispositivos (top 30 por bytes enviados)
    nodes = topo.get("nodes", [])
    if nodes:
        story.append(Paragraph("Dispositivos Detectados (top 30 por tráfico)", custom["h2"]))
        top_nodes = sorted(nodes, key=lambda n: n.get("total_bytes_sent", 0),
                           reverse=True)[:30]
        nd = [["IP", "Rol", "Subred", "Bytes enviados", "Bytes recibidos",
               "Flujos out", "Puertos únicos"]]
        for n in top_nodes:
            nd.append([
                n["id"],
                n["role"].replace("_", " "),
                n.get("subnet", ""),
                _fmt_bytes(n.get("total_bytes_sent", 0)),
                _fmt_bytes(n.get("total_bytes_recv", 0)),
                _fmt_num(n.get("flow_count_out", 0)),
                _fmt_num(n.get("unique_ports_dst", 0)),
            ])
        story.append(Table(
            nd,
            colWidths=[W*0.17, W*0.18, W*0.15, W*0.14, W*0.14, W*0.11, W*0.11],
            style=TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2C3E50")),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                ("FONTSIZE",     (0, 0), (-1, -1), 7),
                ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),
                 [colors.white, colors.HexColor("#F7F9FC")]),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ]),
        ))

    # ── IPs críticas ──────────────────────────────────────────
    critical_ips = alert_summary.get("critical_ips", {})
    if critical_ips:
        story.append(PageBreak())
        story.append(Paragraph("6. IPs con Mayor Actividad Sospechosa", custom["h1"]))
        ci_data = [["IP", "N° Alertas"]] + [
            [ip, str(cnt)]
            for ip, cnt in list(critical_ips.items())[:20]
        ]
        story.append(_pdf_table(ci_data, [W*0.65, W*0.35],
                                header_bg=colors.HexColor("#922B21")))

    doc.build(story)


# =============================================================
#  GENERADOR HTML
# =============================================================

def _escape(text: str) -> str:
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def generate_html(report: dict, output_path: str) -> None:
    """Genera el reporte interactivo en HTML con Chart.js."""

    gm           = report.get("global_metrics", {})
    quality      = report.get("quality_report", {})
    alerts       = report.get("alerts", [])
    alert_summary= report.get("alert_summary", {})
    topo         = report.get("topology", {})
    pcap_name    = report.get("pcap_name", "desconocido")

    # Preparar datos para Chart.js
    proto        = gm.get("protocol_distribution", {})
    flags        = gm.get("flags_summary", {})
    top_src      = gm.get("top_src_ips", {})
    top_dst      = gm.get("top_dst_ips", {})
    top_ports    = gm.get("top_dst_ports", {})
    by_sev       = alert_summary.get("by_severity", {})
    by_type      = alert_summary.get("by_type", {})
    role_summary = topo.get("role_summary", {})
    nodes        = topo.get("nodes", [])
    subnets      = topo.get("subnets", {})

    def _js_labels(d):
        return json.dumps(list(d.keys()))

    def _js_values(d):
        return json.dumps(list(d.values()))

    # Tabla de alertas en HTML
    sev_order    = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAJA": 3}
    sorted_alerts= sorted(
        alerts,
        key=lambda a: sev_order.get(a.get("severidad", "BAJA"), 4),
    )

    alerts_html = ""
    for a in sorted_alerts:
        sev   = a.get("severidad", "BAJA")
        color = SEVERITY_COLORS.get(sev, "#95A5A6")
        alerts_html += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{_escape(sev)}</span></td>
          <td>{_escape(a.get('tipo',''))}</td>
          <td><code>{_escape(a.get('src_ip',''))}</code></td>
          <td><code>{_escape(a.get('dst_ip',''))}</code></td>
          <td>{_escape(a.get('detalle',''))}</td>
        </tr>"""

    # Tabla de nodos
    top_nodes = sorted(nodes, key=lambda n: n.get("total_bytes_sent", 0),
                       reverse=True)[:50]
    nodes_html = ""
    for n in top_nodes:
        role  = n.get("role", "desconocido")
        color = ROLE_COLORS_HTML.get(role, "#BDC3C7")
        nodes_html += f"""
        <tr>
          <td><code>{_escape(n['id'])}</code></td>
          <td><span class="badge" style="background:{color};color:#222">{_escape(role.replace('_',' '))}</span></td>
          <td>{_escape(n.get('subnet',''))}</td>
          <td>{_fmt_bytes(n.get('total_bytes_sent',0))}</td>
          <td>{_fmt_bytes(n.get('total_bytes_recv',0))}</td>
          <td>{_fmt_num(n.get('flow_count_out',0))}</td>
          <td>{_fmt_num(n.get('unique_ports_dst',0))}</td>
          <td>{_fmt_num(n.get('betweenness_centrality',0))}</td>
        </tr>"""

    # Subredes
    subnets_html = ""
    for subnet, ips in subnets.items():
        subnets_html += f"""
        <tr>
          <td><code>{_escape(subnet)}</code></td>
          <td>{_fmt_num(len(ips))}</td>
          <td><small>{_escape(', '.join(ips[:5]))}{' ...' if len(ips)>5 else ''}</small></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaketIQ — Reporte Forense</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --bg2: #1a1d27; --bg3: #22263a;
    --border: #2e3347; --text: #e2e8f0; --text2: #94a3b8;
    --accent: #4A90D9; --danger: #E74C3C; --warn: #F39C12;
    --success: #2ECC71;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }}
  header {{ background: var(--bg2); border-bottom: 1px solid var(--border); padding: 18px 32px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 20px; font-weight: 700; color: var(--accent); }}
  header p  {{ font-size: 12px; color: var(--text2); }}
  nav {{ background: var(--bg2); border-bottom: 1px solid var(--border); padding: 0 32px; display: flex; gap: 4px; overflow-x: auto; }}
  nav a {{ color: var(--text2); text-decoration: none; padding: 10px 16px; font-size: 13px; border-bottom: 2px solid transparent; white-space: nowrap; }}
  nav a:hover, nav a.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  main {{ padding: 28px 32px; max-width: 1400px; margin: 0 auto; }}
  section {{ display: none; }}
  section.active {{ display: block; }}
  h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 18px; color: var(--text); }}
  h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 10px; color: var(--text2); text-transform: uppercase; letter-spacing: .05em; }}
  .grid {{ display: grid; gap: 16px; }}
  .g2 {{ grid-template-columns: repeat(2, 1fr); }}
  .g3 {{ grid-template-columns: repeat(3, 1fr); }}
  .g4 {{ grid-template-columns: repeat(4, 1fr); }}
  @media(max-width:900px) {{ .g2,.g3,.g4 {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  .stat-value {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
  .stat-label {{ font-size: 11px; color: var(--text2); margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }}
  .chart-wrap {{ position: relative; height: 260px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead th {{ background: var(--bg3); color: var(--text2); font-weight: 600; padding: 10px 12px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border); }}
  tbody tr {{ border-bottom: 1px solid var(--border); }}
  tbody tr:hover {{ background: var(--bg3); }}
  tbody td {{ padding: 9px 12px; vertical-align: middle; }}
  code {{ font-family: monospace; font-size: 12px; background: var(--bg3); padding: 2px 6px; border-radius: 4px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; color: #fff; }}
  input[type=text] {{ background: var(--bg3); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 7px 12px; font-size: 13px; width: 260px; outline: none; }}
  input[type=text]:focus {{ border-color: var(--accent); }}
  .filter-bar {{ display: flex; gap: 10px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }}
  select {{ background: var(--bg3); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 7px 10px; font-size: 13px; outline: none; }}
  .tbl-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <div>
    <h1>&#128268; PaketIQ</h1>
    <p>Análisis forense de red · {_escape(pcap_name)} · {_now()}</p>
  </div>
</header>
<nav id="nav">
  <a href="#" class="active" onclick="showSection('resumen',this)">Resumen</a>
  <a href="#" onclick="showSection('alertas',this)">Alertas ({alert_summary.get('total_alerts',0)})</a>
  <a href="#" onclick="showSection('metricas',this)">Métricas</a>
  <a href="#" onclick="showSection('topologia',this)">Topología</a>
  <a href="#" onclick="showSection('calidad',this)">Calidad</a>
</nav>
<main>

<!-- RESUMEN -->
<section id="resumen" class="active">
  <h2>Resumen Ejecutivo</h2>
  <div class="grid g4" style="margin-bottom:20px">
    <div class="card"><div class="stat-value">{_fmt_num(gm.get('total_flows',0))}</div><div class="stat-label">Flujos totales</div></div>
    <div class="card"><div class="stat-value" style="color:var(--danger)">{_fmt_num(alert_summary.get('total_alerts',0))}</div><div class="stat-label">Alertas detectadas</div></div>
    <div class="card"><div class="stat-value">{_fmt_num(topo.get('total_nodes',0))}</div><div class="stat-label">Dispositivos</div></div>
    <div class="card"><div class="stat-value">{_fmt_bytes(gm.get('total_bytes',0))}</div><div class="stat-label">Tráfico total</div></div>
  </div>
  <div class="grid g4" style="margin-bottom:20px">
    <div class="card"><div class="stat-value" style="color:var(--danger)">{by_sev.get('CRITICA',0)}</div><div class="stat-label">Alertas críticas</div></div>
    <div class="card"><div class="stat-value" style="color:{SEVERITY_COLORS['ALTA']}">{by_sev.get('ALTA',0)}</div><div class="stat-label">Alertas altas</div></div>
    <div class="card"><div class="stat-value" style="color:{SEVERITY_COLORS['MEDIA']}">{by_sev.get('MEDIA',0)}</div><div class="stat-label">Alertas medias</div></div>
    <div class="card"><div class="stat-value" style="color:var(--success)">{by_sev.get('BAJA',0)}</div><div class="stat-label">Alertas bajas</div></div>
  </div>
  <div class="grid g2">
    <div class="card"><h3>Alertas por tipo</h3><div class="chart-wrap"><canvas id="chartAlertType"></canvas></div></div>
    <div class="card"><h3>Protocolos</h3><div class="chart-wrap"><canvas id="chartProto"></canvas></div></div>
  </div>
</section>

<!-- ALERTAS -->
<section id="alertas">
  <h2>Alertas Forenses</h2>
  <div class="filter-bar">
    <input type="text" id="alertSearch" placeholder="Buscar IP, tipo, detalle..." oninput="filterAlerts()">
    <select id="alertSev" onchange="filterAlerts()">
      <option value="">Todas las severidades</option>
      <option>CRITICA</option><option>ALTA</option><option>MEDIA</option><option>BAJA</option>
    </select>
    <select id="alertType" onchange="filterAlerts()">
      <option value="">Todos los tipos</option>
      {''.join(f'<option>{_escape(t)}</option>' for t in sorted(by_type.keys()))}
    </select>
  </div>
  <div class="tbl-wrap">
  <table id="alertTable">
    <thead><tr><th>Severidad</th><th>Tipo</th><th>Src IP</th><th>Dst IP</th><th>Detalle</th></tr></thead>
    <tbody>{alerts_html}</tbody>
  </table>
  </div>
</section>

<!-- MÉTRICAS -->
<section id="metricas">
  <h2>Métricas de la Captura</h2>
  <div class="grid g2" style="margin-bottom:20px">
    <div class="card"><h3>Top IPs Origen</h3><div class="chart-wrap"><canvas id="chartSrcIP"></canvas></div></div>
    <div class="card"><h3>Top IPs Destino</h3><div class="chart-wrap"><canvas id="chartDstIP"></canvas></div></div>
  </div>
  <div class="grid g2">
    <div class="card"><h3>Top Puertos Destino</h3><div class="chart-wrap"><canvas id="chartPorts"></canvas></div></div>
    <div class="card"><h3>Flags TCP</h3><div class="chart-wrap"><canvas id="chartFlags"></canvas></div></div>
  </div>
</section>

<!-- TOPOLOGÍA -->
<section id="topologia">
  <h2>Topología de Red</h2>
  <div class="grid g3" style="margin-bottom:20px">
    <div class="card"><div class="stat-value">{topo.get('total_nodes',0)}</div><div class="stat-label">Dispositivos</div></div>
    <div class="card"><div class="stat-value">{topo.get('total_edges',0)}</div><div class="stat-label">Conexiones</div></div>
    <div class="card"><div class="stat-value">{topo.get('total_subnets',0)}</div><div class="stat-label">Subredes</div></div>
  </div>
  <div class="grid g2" style="margin-bottom:20px">
    <div class="card"><h3>Clasificación de dispositivos</h3><div class="chart-wrap"><canvas id="chartRoles"></canvas></div></div>
    <div class="card"><h3>Subredes detectadas</h3>
      <div class="tbl-wrap" style="max-height:260px;overflow-y:auto">
      <table><thead><tr><th>Subred</th><th>Dispositivos</th><th>IPs</th></tr></thead>
      <tbody>{subnets_html}</tbody></table></div></div>
  </div>
  <div class="card">
    <h3>Dispositivos (top 50 por tráfico)</h3>
    <div class="filter-bar">
      <input type="text" id="nodeSearch" placeholder="Buscar IP o rol..." oninput="filterNodes()">
      <select id="nodeRole" onchange="filterNodes()">
        <option value="">Todos los roles</option>
        {''.join(f'<option>{_escape(r)}</option>' for r in sorted(set(n.get("role","") for n in nodes)))}
      </select>
    </div>
    <div class="tbl-wrap">
    <table id="nodeTable">
      <thead><tr><th>IP</th><th>Rol</th><th>Subred</th><th>Bytes enviados</th><th>Bytes recibidos</th><th>Flujos out</th><th>Puertos únicos</th><th>Betweenness</th></tr></thead>
      <tbody>{nodes_html}</tbody>
    </table>
    </div>
  </div>
</section>

<!-- CALIDAD -->
<section id="calidad">
  <h2>Calidad del Dataset</h2>
  <div class="grid g2">
    <div class="card">
      <h3>Resumen de limpieza</h3>
      <table>
        <thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
        <tbody>
          <tr><td>Filas originales</td><td>{_fmt_num(quality.get('original_rows',0))}</td></tr>
          <tr><td>Filas finales</td><td>{_fmt_num(quality.get('final_rows',0))}</td></tr>
          <tr><td>Filas eliminadas</td><td>{_fmt_num(quality.get('rows_removed',0))} ({quality.get('rows_removed_pct','0%')})</td></tr>
          <tr><td>Duplicados eliminados</td><td>{_fmt_num(quality.get('dropped_duplicates',0))}</td></tr>
          <tr><td>IPs inválidas eliminadas</td><td>{_fmt_num(quality.get('dropped_invalid_ips',0))}</td></tr>
          <tr><td>NaN imputados</td><td>{_fmt_num(quality.get('imputed_nan',0))}</td></tr>
          <tr><td>Infinitos reemplazados</td><td>{_fmt_num(quality.get('replaced_inf',0))}</td></tr>
          <tr><td>Negativos corregidos</td><td>{_fmt_num(quality.get('clipped_negative',0))}</td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h3>Protocolos presentes</h3>
      <table>
        <thead><tr><th>Protocolo</th><th>Flujos</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td>{_escape(k)}</td><td>{_fmt_num(v)}</td></tr>" for k,v in quality.get('protocol_counts',{}).items())}
        </tbody>
      </table>
    </div>
  </div>
</section>

</main>
<script>
function showSection(id, el) {{
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if (el) el.classList.add('active');
}}

// Filtro alertas
function filterAlerts() {{
  const q    = document.getElementById('alertSearch').value.toLowerCase();
  const sev  = document.getElementById('alertSev').value;
  const type = document.getElementById('alertType').value;
  document.querySelectorAll('#alertTable tbody tr').forEach(tr => {{
    const txt = tr.textContent.toLowerCase();
    const sevMatch  = !sev  || tr.children[0].textContent.includes(sev);
    const typeMatch = !type || tr.children[1].textContent.includes(type);
    tr.style.display = (txt.includes(q) && sevMatch && typeMatch) ? '' : 'none';
  }});
}}

// Filtro nodos
function filterNodes() {{
  const q    = document.getElementById('nodeSearch').value.toLowerCase();
  const role = document.getElementById('nodeRole').value;
  document.querySelectorAll('#nodeTable tbody tr').forEach(tr => {{
    const txt = tr.textContent.toLowerCase();
    const roleMatch = !role || tr.children[1].textContent.includes(role);
    tr.style.display = (txt.includes(q) && roleMatch) ? '' : 'none';
  }});
}}

// Charts
const COLORS = {json.dumps(CHART_COLORS)};
const chartOpts = (type, labels, data, colors) => ({{
  type,
  data: {{ labels, datasets: [{{ data, backgroundColor: colors || COLORS, borderWidth: 0 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }}, position: 'bottom' }}, tooltip: {{ titleColor:'#e2e8f0', bodyColor:'#94a3b8' }} }},
    scales: type === 'bar' ? {{
      x: {{ ticks: {{ color:'#94a3b8', font:{{size:10}} }}, grid:{{ color:'#2e3347' }} }},
      y: {{ ticks: {{ color:'#94a3b8' }}, grid:{{ color:'#2e3347' }} }}
    }} : undefined,
  }}
}});

new Chart('chartAlertType', chartOpts('bar', {json.dumps([k.replace('_',' ') for k in list(by_type.keys())[:8]])}, {json.dumps(list(by_type.values())[:8])}));
new Chart('chartProto',    chartOpts('doughnut', {_js_labels(proto)}, {_js_values(proto)}));
new Chart('chartSrcIP',    chartOpts('bar', {_js_labels(dict(list(top_src.items())[:8]))}, {_js_values(dict(list(top_src.items())[:8]))}));
new Chart('chartDstIP',    chartOpts('bar', {_js_labels(dict(list(top_dst.items())[:8]))}, {_js_values(dict(list(top_dst.items())[:8]))}));
new Chart('chartPorts',    chartOpts('bar', {_js_labels(dict(list(top_ports.items())[:8]))}, {_js_values(dict(list(top_ports.items())[:8]))}));
new Chart('chartFlags',    chartOpts('bar', {json.dumps([k.upper() for k in flags.keys()])}, {json.dumps(list(flags.values()))}, ['#E74C3C','#3498DB','#E67E22','#2ECC71','#9B59B6','#1ABC9C','#F39C12','#34495E']));
new Chart('chartRoles',    chartOpts('doughnut', {json.dumps([k.replace('_',' ') for k in role_summary.keys()])}, {json.dumps(list(role_summary.values()))}));
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================
#  FUNCIÓN PRINCIPAL
# =============================================================

def generate_report(
    analysis: dict,
    topology: dict,
    quality_report: dict,
    pcap_path: str,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Orquesta la generación del reporte en PDF y HTML.

    Parámetros
    ----------
    analysis      : Salida de flow_analyzer.analyze_flows()
    topology      : Salida de topology_mapper.map_topology()
    quality_report: Salida de preprocessor.preprocess()
    pcap_path     : Ruta del archivo PCAP analizado
    output_dir    : Directorio donde guardar los reportes
    verbose       : Imprime progreso si True

    Retorna
    -------
    dict con rutas:
        pdf  — ruta del PDF generado
        html — ruta del HTML generado
    """
    out_path   = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path = _plots_dir(str(out_path))

    pcap_name  = Path(pcap_path).name

    # Ensamblar reporte completo
    report = {
        "pcap_name"     : pcap_name,
        "global_metrics": analysis.get("global_metrics", {}),
        "alerts"        : analysis.get("alerts", []),
        "alert_summary" : analysis.get("alert_summary", {}),
        "topology"      : topology,
        "quality_report": quality_report,
    }

    # Imagen del grafo de topología
    topo_png = str(out_path / "topology_graph.png")
    report["_topology_png"] = topo_png

    if verbose:
        print("[report_generator] Generando gráficas...")
    charts = _generate_charts(report, plots_path)
    report["_charts"] = charts

    # PDF
    pdf_path = str(out_path / f"{Path(pcap_path).stem}_report.pdf")
    if verbose:
        print("[report_generator] Generando PDF...")
    generate_pdf(report, pdf_path, plots_path)
    if verbose:
        print(f"[report_generator] PDF: {pdf_path}")

    # HTML
    html_path = str(out_path / f"{Path(pcap_path).stem}_report.html")
    if verbose:
        print("[report_generator] Generando HTML...")
    generate_html(report, html_path)
    if verbose:
        print(f"[report_generator] HTML: {html_path}")

    return {"pdf": pdf_path, "html": html_path}