"""
paketiq.py
==========
Entry point CLI de PaketIQ v2.

Uso:
    python paketiq.py <archivo.pcap> -o reporte [opciones]

Ejemplos:
    python paketiq.py captura.pcap -o informe
    python paketiq.py captura.pcap -o informe --out-dir ./reportes --verbose
    python paketiq.py captura.pcap -o informe --no-html --chunk-size 100000
    python paketiq.py captura.pcap -o informe --no-graph
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Verificar dependencias críticas antes de importar módulos
def _check_dependencies():
    missing = []
    required = [
        ("scapy",      "scapy"),
        ("pandas",     "pandas"),
        ("numpy",      "numpy"),
        ("networkx",   "networkx"),
        ("matplotlib", "matplotlib"),
        ("reportlab",  "reportlab"),
        ("tqdm",       "tqdm"),
        ("scipy",      "scipy"),
    ]
    for import_name, pip_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("[ERROR] Dependencias faltantes. Instala con:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

_check_dependencies()

from modules.flow_extractor  import extract_flows
from modules.preprocessor    import preprocess
from modules.flow_grouper    import group_flows
from modules.flow_analyzer   import analyze_flows
from modules.topology_mapper import map_topology
from modules.report_generator import generate_report


# =============================================================
#  HELPERS
# =============================================================

def _banner():
    print("""
  ██████╗  █████╗ ██╗  ██╗███████╗████████╗██╗ ██████╗
  ██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝██║██╔═══██╗
  ██████╔╝███████║█████╔╝ █████╗     ██║   ██║██║   ██║
  ██╔═══╝ ██╔══██║██╔═██╗ ██╔══╝     ██║   ██║██║▄▄ ██║
  ██║     ██║  ██║██║  ██╗███████╗   ██║   ██║╚██████╔╝
  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚══▀▀═╝
  Herramienta de Análisis Forense de Red  v2.0
""")


def _separator(label: str = "") -> None:
    if label:
        pad = (60 - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * 62)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


def _ok(msg: str) -> None:
    print(f"  [✔] {msg}")


def _warn(msg: str) -> None:
    print(f"  [!] {msg}", file=sys.stderr)


def _get_output_dir(base_name: str, custom_dir: str | None) -> Path:
    """
    Determina el directorio de salida.
    Prioridad: --out-dir > ~/Downloads > ~/Descargas > ./reports
    """
    if custom_dir:
        return Path(custom_dir)

    home = Path.home()
    for candidate in ("Downloads", "Descargas"):
        p = home / candidate
        if p.is_dir():
            return p / base_name

    # Fallback: subcarpeta reports/ junto al script
    fallback = Path(__file__).parent / "reports" / base_name
    return fallback


def _cleanup(paths: list[str], verbose: bool) -> None:
    """Elimina archivos/carpetas temporales."""
    for p in paths:
        try:
            path = Path(p)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        except Exception as e:
            if verbose:
                _warn(f"No se pudo limpiar {p}: {e}")


def _print_summary(
    analysis   : dict,
    topology   : dict,
    outputs    : dict,
    elapsed    : float,
) -> None:
    """Imprime el resumen final del análisis."""
    gm      = analysis.get("global_metrics", {})
    summary = analysis.get("alert_summary", {})
    by_sev  = summary.get("by_severity", {})

    _separator("RESULTADO")
    print(f"  Flujos analizados  : {gm.get('total_flows', 0):,}")
    print(f"  Dispositivos       : {topology.get('total_nodes', 0):,}")
    print(f"  Subredes           : {topology.get('total_subnets', 0):,}")
    print(f"  Total alertas      : {summary.get('total_alerts', 0):,}")
    print(f"    CRÍTICAS         : {by_sev.get('CRITICA', 0)}")
    print(f"    ALTAS            : {by_sev.get('ALTA', 0)}")
    print(f"    MEDIAS           : {by_sev.get('MEDIA', 0)}")
    print(f"    BAJAS            : {by_sev.get('BAJA', 0)}")
    _separator()
    if "pdf" in outputs:
        print(f"  PDF  → {outputs['pdf']}")
    if "html" in outputs:
        print(f"  HTML → {outputs['html']}")
    _separator()
    print(f"  Tiempo total: {_fmt_elapsed(elapsed)}")
    print()


# =============================================================
#  ARGPARSE
# =============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paketiq",
        description="PaketIQ — Análisis forense de tráfico de red desde archivos PCAP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  python paketiq.py captura.pcap -o informe
  python paketiq.py captura.pcap -o informe --verbose
  python paketiq.py captura.pcap -o informe --out-dir /ruta/reportes
  python paketiq.py captura.pcap -o informe --no-html --chunk-size 100000
  python paketiq.py captura.pcap -o informe --no-graph --keep-tmp
        """,
    )

    # Posicional obligatorio
    parser.add_argument(
        "pcap",
        type=str,
        help="Ruta al archivo .pcap o .pcapng a analizar.",
    )

    # Opciones principales
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        metavar="NOMBRE",
        help="Nombre base para los archivos de salida (sin extensión).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIRECTORIO",
        help="Directorio donde guardar los reportes. "
             "Por defecto: ~/Downloads/<NOMBRE>.",
    )

    # Formato de salida
    fmt_group = parser.add_argument_group("formato de salida")
    fmt_group.add_argument(
        "--no-pdf",
        action="store_true",
        help="No generar reporte PDF.",
    )
    fmt_group.add_argument(
        "--no-html",
        action="store_true",
        help="No generar reporte HTML.",
    )
    fmt_group.add_argument(
        "--no-graph",
        action="store_true",
        help="No generar imagen del grafo de topología.",
    )

    # Rendimiento
    perf_group = parser.add_argument_group("rendimiento")
    perf_group.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        metavar="N",
        help="Paquetes procesados por chunk (default: 50000). "
             "Reduce si tienes poca RAM.",
    )

    # Debug / salida
    out_group = parser.add_argument_group("salida")
    out_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar progreso detallado de cada etapa.",
    )
    out_group.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Conservar archivos temporales (CSV intermedios) tras el análisis.",
    )
    out_group.add_argument(
        "--export-json",
        action="store_true",
        help="Exportar también el análisis completo en formato JSON.",
    )

    return parser


# =============================================================
#  PIPELINE PRINCIPAL
# =============================================================

def run_pipeline(args: argparse.Namespace) -> int:
    """
    Ejecuta el pipeline completo de análisis.
    Retorna 0 si éxito, 1 si error.
    """
    t_start = time.time()
    TOTAL_STEPS = 6

    # --- Validar entrada ---
    pcap_path = Path(args.pcap)
    if not pcap_path.exists():
        print(f"[ERROR] Archivo no encontrado: {pcap_path}")
        return 1
    if pcap_path.suffix.lower() not in (".pcap", ".pcapng"):
        print(f"[ERROR] Formato no soportado: {pcap_path.suffix}")
        return 1

    # --- Directorios ---
    out_dir   = _get_output_dir(args.output, args.out_dir)
    tmp_dir   = Path(__file__).parent / "data" / "output"
    plots_dir = Path(__file__).parent / "plots_tmp"

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_files = []   # archivos a limpiar al final

    if not args.verbose:
        print(f"  Analizando: {pcap_path.name}")

    try:

        # ── PASO 1: Extracción de flujos ────────────────────────
        _step(1, TOTAL_STEPS, "Extrayendo flujos del PCAP...")
        t = time.time()

        df_raw = extract_flows(
            str(pcap_path),
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        )

        if df_raw.empty:
            print("[ERROR] No se extrajeron flujos válidos del PCAP.")
            return 1

        _ok(f"{len(df_raw):,} flujos extraídos en {_fmt_elapsed(time.time()-t)}")

        # ── PASO 2: Preprocesamiento ────────────────────────────
        _step(2, TOTAL_STEPS, "Preprocesando y limpiando flujos...")
        t = time.time()

        df_clean, quality_report = preprocess(
            df_raw,
            output_dir=str(tmp_dir),
            verbose=args.verbose,
        )
        del df_raw   # liberar memoria

        raw_csv = str(tmp_dir / "raw_flows.csv")
        tmp_files.append(raw_csv)

        _ok(f"{len(df_clean):,} flujos limpios en {_fmt_elapsed(time.time()-t)}")

        # ── PASO 3: Agrupación de flujos ────────────────────────
        _step(3, TOTAL_STEPS, "Agrupando flujos por par IP...")
        t = time.time()

        grouped_csv = str(tmp_dir / "grouped_flows.csv")
        df_grouped  = group_flows(
            df_clean,
            output_path=grouped_csv,
            verbose=args.verbose,
        )
        del df_clean  # liberar memoria

        tmp_files.append(grouped_csv)
        _ok(f"{len(df_grouped):,} pares IP agrupados en {_fmt_elapsed(time.time()-t)}")

        # ── PASO 4: Análisis forense ────────────────────────────
        _step(4, TOTAL_STEPS, "Ejecutando análisis forense...")
        t = time.time()

        json_path = str(tmp_dir / "analysis.json") if args.export_json else None
        analysis  = analyze_flows(
            df_grouped,
            export_json=json_path,
            verbose=args.verbose,
        )

        if json_path:
            tmp_files.append(json_path)
            _ok(f"JSON exportado → {json_path}")

        n_alerts = analysis["alert_summary"]["total_alerts"]
        n_crit   = analysis["alert_summary"]["by_severity"].get("CRITICA", 0)
        _ok(
            f"{n_alerts} alertas detectadas "
            f"({n_crit} críticas) en {_fmt_elapsed(time.time()-t)}"
        )

        # ── PASO 5: Topología ───────────────────────────────────
        _step(5, TOTAL_STEPS, "Mapeando topología de red...")
        t = time.time()

        topo_json = str(tmp_dir / "topology.json")
        graph_png = str(out_dir / "topology_graph.png") \
                    if not args.no_graph else None

        topology = map_topology(
            df_grouped,
            output_dir=str(tmp_dir),
            graph_png=graph_png,
            verbose=args.verbose,
        )
        tmp_files.append(topo_json)

        _ok(
            f"{topology['total_nodes']} dispositivos, "
            f"{topology['total_subnets']} subredes en {_fmt_elapsed(time.time()-t)}"
        )

        # ── PASO 6: Generación de reportes ──────────────────────
        _step(6, TOTAL_STEPS, "Generando reportes...")
        t = time.time()

        outputs = generate_report(
            analysis      = analysis,
            topology      = topology,
            quality_report= quality_report,
            pcap_path     = str(pcap_path),
            output_dir    = str(out_dir),
            verbose       = args.verbose,
        )

        # Filtrar formatos desactivados
        if args.no_pdf and "pdf" in outputs:
            Path(outputs["pdf"]).unlink(missing_ok=True)
            del outputs["pdf"]
        if args.no_html and "html" in outputs:
            Path(outputs["html"]).unlink(missing_ok=True)
            del outputs["html"]

        _ok(f"Reportes generados en {_fmt_elapsed(time.time()-t)}")

        # ── Exportar JSON completo si se pidió ──────────────────
        if args.export_json:
            full_json_path = str(out_dir / f"{args.output}_full.json")
            full_data = {
                "metadata": {
                    "pcap_file"   : str(pcap_path),
                    "analysis_date": datetime.now().isoformat(),
                    "paketiq_version": "2.0",
                },
                "quality_report": quality_report,
                "global_metrics": analysis.get("global_metrics", {}),
                "alert_summary" : analysis.get("alert_summary", {}),
                "alerts"        : analysis.get("alerts", []),
                "topology"      : topology,
            }
            with open(full_json_path, "w", encoding="utf-8") as f:
                json.dump(full_data, f, indent=4, ensure_ascii=False)
            outputs["json"] = full_json_path
            _ok(f"JSON completo → {full_json_path}")

    except KeyboardInterrupt:
        print("\n[INTERRUMPIDO] Análisis cancelado por el usuario.")
        return 1

    except Exception as e:
        print(f"\n[ERROR] {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        # ── Limpieza de temporales ──────────────────────────────
        if not args.keep_tmp:
            _cleanup(tmp_files, args.verbose)
            _cleanup([str(plots_dir)], args.verbose)
            if args.verbose:
                _ok("Archivos temporales eliminados.")

    # ── Resumen final ───────────────────────────────────────────
    _print_summary(analysis, topology, outputs, time.time() - t_start)

    return 0


# =============================================================
#  ENTRY POINT
# =============================================================

def main():
    _banner()
    parser = _build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Validación rápida de argumentos
    if args.no_pdf and args.no_html:
        print("[ERROR] No puedes usar --no-pdf y --no-html al mismo tiempo.")
        sys.exit(1)

    if args.chunk_size < 1000:
        print("[ADVERTENCIA] --chunk-size muy bajo puede afectar el rendimiento.")

    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()