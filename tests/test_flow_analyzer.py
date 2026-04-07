"""
test_flow_analyzer.py
=====================
Tests unitarios para modules/flow_analyzer.py

Cubre:
    - Cada detector individual con datos que DEBEN disparar alerta
    - Cada detector con datos que NO deben disparar alerta (falsos positivos)
    - Cálculo de métricas globales
    - Construcción del resumen de alertas
    - Función principal analyze_flows()
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.flow_analyzer import (
    _detect_port_scan,
    _detect_syn_flood,
    _detect_udp_flood,
    _detect_icmp_sweep,
    _detect_exfiltration,
    _detect_anomalous_flags,
    _detect_long_flows,
    _detect_beaconing,
    _detect_unusual_ports,
    _compute_global_metrics,
    _build_alert_summary,
    analyze_flows,
    THRESHOLDS,
)


# =============================================================
#  FIXTURES
# =============================================================

def _base_df(**kwargs) -> pd.DataFrame:
    """DataFrame mínimo válido."""
    defaults = {
        "Src IP"           : ["192.168.1.1"],
        "Dst IP"           : ["10.0.0.1"],
        "Destination Port" : [80],
        "Protocol"         : [6],
        "Total Packets"    : [10],
        "Total Bytes"      : [1500],
        "Flow Duration"    : [100.0],
        "SYN Flag Count"   : [1],
        "FIN Flag Count"   : [1],
        "RST Flag Count"   : [0],
        "PSH Flag Count"   : [2],
        "ACK Flag Count"   : [5],
        "URG Flag Count"   : [0],
        "Subflow Fwd Bytes": [1000],
        "Subflow Bwd Bytes": [500],
        "Packets/s"        : [10.0],
        "Bytes/s"          : [1500.0],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# =============================================================
#  PORT SCAN
# =============================================================

class TestDetectPortScan(unittest.TestCase):

    def test_vertical_scan_detected(self):
        """Una IP escanea muchos puertos en un mismo destino."""
        n = THRESHOLDS["port_scan_unique_dst_ports"] + 5
        df = pd.DataFrame({
            "Src IP"          : ["192.168.1.100"] * n,
            "Dst IP"          : ["10.0.0.1"] * n,
            "Destination Port": list(range(1, n + 1)),
        })
        alerts = _detect_port_scan(df)
        tipos = [a["tipo"] for a in alerts]
        self.assertIn("Port Scan Vertical", tipos)

    def test_horizontal_scan_detected(self):
        """Una IP escanea muchos hosts en el mismo puerto."""
        n = THRESHOLDS["horizontal_scan_unique_dst_ips"] + 5
        df = pd.DataFrame({
            "Src IP"          : ["192.168.1.100"] * n,
            "Dst IP"          : [f"10.0.0.{i}" for i in range(1, n + 1)],
            "Destination Port": [22] * n,
        })
        alerts = _detect_port_scan(df)
        tipos = [a["tipo"] for a in alerts]
        self.assertIn("Port Scan Horizontal", tipos)

    def test_normal_traffic_no_alert(self):
        """Tráfico normal hacia pocos puertos no genera alerta."""
        df = pd.DataFrame({
            "Src IP"          : ["192.168.1.1"] * 5,
            "Dst IP"          : ["10.0.0.1"] * 5,
            "Destination Port": [80, 443, 8080, 22, 53],
        })
        alerts = _detect_port_scan(df)
        self.assertEqual(len(alerts), 0)

    def test_missing_port_column(self):
        """Sin columna Destination Port no debe fallar."""
        df = pd.DataFrame({"Src IP": ["1.1.1.1"], "Dst IP": ["2.2.2.2"]})
        alerts = _detect_port_scan(df)
        self.assertEqual(alerts, [])


# =============================================================
#  SYN FLOOD
# =============================================================

class TestDetectSynFlood(unittest.TestCase):

    def test_syn_flood_detected(self):
        """Alto ratio de SYN debe disparar alerta."""
        n = THRESHOLDS["syn_flood_min_count"] + 50
        df = pd.DataFrame({
            "Src IP"        : ["10.10.10.10"] * n,
            "Dst IP"        : [f"192.168.1.{i % 254 + 1}" for i in range(n)],
            "SYN Flag Count": [1] * n,
            "FIN Flag Count": [0] * n,
            "RST Flag Count": [0] * n,
        })
        alerts = _detect_syn_flood(df)
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]["tipo"], "SYN Flood")

    def test_normal_handshake_no_alert(self):
        """Handshake normal (SYN + SYN-ACK + ACK + FIN) no dispara."""
        df = pd.DataFrame({
            "Src IP"        : ["192.168.1.1"] * 4,
            "Dst IP"        : ["10.0.0.1"] * 4,
            "SYN Flag Count": [1, 1, 0, 0],
            "FIN Flag Count": [0, 0, 1, 1],
            "RST Flag Count": [0, 0, 0, 0],
        })
        alerts = _detect_syn_flood(df)
        self.assertEqual(len(alerts), 0)

    def test_missing_columns_no_crash(self):
        """Sin columnas de flags no debe lanzar excepción."""
        df = pd.DataFrame({"Src IP": ["1.1.1.1"], "Dst IP": ["2.2.2.2"]})
        alerts = _detect_syn_flood(df)
        self.assertEqual(alerts, [])


# =============================================================
#  UDP FLOOD
# =============================================================

class TestDetectUdpFlood(unittest.TestCase):

    def test_udp_flood_detected(self):
        """Alto pps UDP debe disparar alerta."""
        pps = THRESHOLDS["udp_flood_packets_per_sec"] + 500
        df = pd.DataFrame({
            "Src IP"   : ["10.10.10.10"] * 3,
            "Dst IP"   : ["192.168.1.1"] * 3,
            "Protocol" : [17] * 3,
            "Packets/s": [pps, pps + 100, pps - 50],
        })
        alerts = _detect_udp_flood(df)
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]["tipo"], "UDP Flood")

    def test_tcp_traffic_not_flagged(self):
        """Tráfico TCP con alto pps no se detecta como UDP flood."""
        pps = THRESHOLDS["udp_flood_packets_per_sec"] + 500
        df = pd.DataFrame({
            "Src IP"   : ["10.10.10.10"],
            "Dst IP"   : ["192.168.1.1"],
            "Protocol" : [6],      # TCP
            "Packets/s": [pps],
        })
        alerts = _detect_udp_flood(df)
        self.assertEqual(len(alerts), 0)


# =============================================================
#  ICMP SWEEP
# =============================================================

class TestDetectIcmpSweep(unittest.TestCase):

    def test_ping_sweep_detected(self):
        """ICMP a muchos hosts debe disparar alerta."""
        n = THRESHOLDS["icmp_sweep_unique_dst_ips"] + 5
        df = pd.DataFrame({
            "Src IP"  : ["192.168.1.1"] * n,
            "Dst IP"  : [f"10.0.0.{i}" for i in range(1, n + 1)],
            "Protocol": [1] * n,
        })
        alerts = _detect_icmp_sweep(df)
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]["tipo"], "ICMP Ping Sweep")

    def test_few_icmp_no_alert(self):
        """Pocos pings ICMP no disparan alerta."""
        df = pd.DataFrame({
            "Src IP"  : ["192.168.1.1"] * 3,
            "Dst IP"  : ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            "Protocol": [1, 1, 1],
        })
        alerts = _detect_icmp_sweep(df)
        self.assertEqual(len(alerts), 0)


# =============================================================
#  EXFILTRACIÓN
# =============================================================

class TestDetectExfiltration(unittest.TestCase):

    def test_exfiltration_detected(self):
        """Alto ratio fwd/bwd con volumen suficiente dispara alerta."""
        df = pd.DataFrame({
            "Src IP"           : ["192.168.1.50"],
            "Dst IP"           : ["93.184.216.34"],
            "Subflow Fwd Bytes": [THRESHOLDS["exfil_min_fwd_bytes"] + 100_000],
            "Subflow Bwd Bytes": [1_000],
        })
        alerts = _detect_exfiltration(df)
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]["tipo"], "Posible Exfiltracion")

    def test_symmetric_traffic_no_alert(self):
        """Tráfico simétrico (ratio ~1) no dispara alerta."""
        df = pd.DataFrame({
            "Src IP"           : ["192.168.1.1"],
            "Dst IP"           : ["10.0.0.1"],
            "Subflow Fwd Bytes": [500_000],
            "Subflow Bwd Bytes": [480_000],
        })
        alerts = _detect_exfiltration(df)
        self.assertEqual(len(alerts), 0)

    def test_high_ratio_low_volume_no_alert(self):
        """Alto ratio pero bajo volumen no dispara alerta."""
        df = pd.DataFrame({
            "Src IP"           : ["192.168.1.1"],
            "Dst IP"           : ["10.0.0.1"],
            "Subflow Fwd Bytes": [1_000],   # muy poco volumen
            "Subflow Bwd Bytes": [1],
        })
        alerts = _detect_exfiltration(df)
        self.assertEqual(len(alerts), 0)


# =============================================================
#  FLAGS ANÓMALOS
# =============================================================

class TestDetectAnomalousFlags(unittest.TestCase):

    def test_rst_anomaly_detected(self):
        """Alto ratio de RST dispara alerta RST Anomalo."""
        n = THRESHOLDS["rst_min_count"] + 20
        df = pd.DataFrame({
            "Src IP"        : ["10.10.10.10"] * n,
            "Dst IP"        : ["192.168.1.1"] * n,
            "RST Flag Count": [1] * n,
            "SYN Flag Count": [0] * n,
            "FIN Flag Count": [0] * n,
            "PSH Flag Count": [0] * n,
            "ACK Flag Count": [0] * n,
            "URG Flag Count": [0] * n,
        })
        alerts = _detect_anomalous_flags(df)
        tipos = [a["tipo"] for a in alerts]
        self.assertIn("RST Anomalo", tipos)

    def test_fin_scan_detected(self):
        """FIN sin SYN dispara alerta FIN Scan."""
        n = THRESHOLDS["fin_without_syn_min"] + 10
        df = pd.DataFrame({
            "Src IP"        : ["10.10.10.10"] * n,
            "Dst IP"        : ["192.168.1.1"] * n,
            "FIN Flag Count": [1] * n,
            "SYN Flag Count": [0] * n,
            "RST Flag Count": [0] * n,
            "PSH Flag Count": [0] * n,
            "ACK Flag Count": [0] * n,
            "URG Flag Count": [0] * n,
        })
        alerts = _detect_anomalous_flags(df)
        tipos = [a["tipo"] for a in alerts]
        self.assertIn("FIN Scan", tipos)

    def test_normal_tcp_no_alert(self):
        """Distribución normal de flags no genera alertas."""
        df = pd.DataFrame({
            "Src IP"        : ["192.168.1.1"] * 10,
            "Dst IP"        : ["10.0.0.1"] * 10,
            "SYN Flag Count": [1] * 10,
            "FIN Flag Count": [1] * 10,
            "RST Flag Count": [0] * 10,
            "PSH Flag Count": [2] * 10,
            "ACK Flag Count": [5] * 10,
            "URG Flag Count": [0] * 10,
        })
        alerts = _detect_anomalous_flags(df)
        self.assertEqual(len(alerts), 0)


# =============================================================
#  FLUJOS LARGOS
# =============================================================

class TestDetectLongFlows(unittest.TestCase):

    def test_long_flow_detected(self):
        """Flujos que superan el umbral de duración disparan alerta."""
        duration = THRESHOLDS["long_flow_duration_ms"] + 60_000
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.1"],
            "Dst IP"       : ["185.12.45.67"],
            "Flow Duration": [float(duration)],
            "Total Bytes"  : [50_000],
        })
        alerts = _detect_long_flows(df)
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]["tipo"], "Flujo de Larga Duracion")

    def test_short_flow_no_alert(self):
        """Flujos cortos no disparan alerta."""
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.1"],
            "Dst IP"       : ["10.0.0.1"],
            "Flow Duration": [5000.0],   # 5 segundos
            "Total Bytes"  : [1500],
        })
        alerts = _detect_long_flows(df)
        self.assertEqual(len(alerts), 0)


# =============================================================
#  BEACONING
# =============================================================

class TestDetectBeaconing(unittest.TestCase):

    def test_beaconing_detected(self):
        """Intervalos muy regulares (CV bajo) disparan alerta."""
        n     = THRESHOLDS["beaconing_min_flows"] + 5
        intv  = THRESHOLDS["beaconing_min_interval_ms"] + 1000
        # Simular durations casi uniformes (beaconing)
        durs  = [float(intv * i) for i in range(1, n + 1)]
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.50"] * n,
            "Dst IP"       : ["185.99.10.1"]  * n,
            "Flow Duration": durs,
        })
        alerts = _detect_beaconing(df)
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]["tipo"], "Beaconing / C2 Sospechoso")

    def test_irregular_traffic_no_alert(self):
        """Intervalos muy irregulares (CV alto) no disparan alerta."""
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.1"] * 15,
            "Dst IP"       : ["10.0.0.1"]   * 15,
            "Flow Duration": [10.0, 50000.0, 3.0, 99999.0, 1.0,
                              8000.0, 200.0, 77777.0, 55.0, 12345.0,
                              900.0, 4444.0, 11.0, 66666.0, 333.0],
        })
        alerts = _detect_beaconing(df)
        self.assertEqual(len(alerts), 0)


# =============================================================
#  PUERTOS INUSUALES
# =============================================================

class TestDetectUnusualPorts(unittest.TestCase):

    def test_unusual_port_detected(self):
        """Puerto alto con suficiente actividad dispara alerta."""
        n    = THRESHOLDS["unusual_port_min_count"] + 3
        port = 31337   # puerto no estándar conocido
        df = pd.DataFrame({
            "Src IP"          : [f"192.168.1.{i}" for i in range(1, n + 1)],
            "Dst IP"          : ["10.0.0.1"] * n,
            "Destination Port": [port] * n,
        })
        alerts = _detect_unusual_ports(df)
        self.assertTrue(len(alerts) > 0)
        ports_found = [a["port"] for a in alerts]
        self.assertIn(port, ports_found)

    def test_known_safe_ports_not_flagged(self):
        """Puertos seguros conocidos (80, 443) no generan alertas."""
        df = pd.DataFrame({
            "Src IP"          : ["192.168.1.1"] * 10,
            "Dst IP"          : ["10.0.0.1"]   * 10,
            "Destination Port": [80, 443, 80, 443, 80, 443, 80, 443, 80, 443],
        })
        alerts = _detect_unusual_ports(df)
        ports_flagged = [a.get("port") for a in alerts]
        self.assertNotIn(80,  ports_flagged)
        self.assertNotIn(443, ports_flagged)


# =============================================================
#  MÉTRICAS GLOBALES
# =============================================================

class TestComputeGlobalMetrics(unittest.TestCase):

    def test_basic_metrics(self):
        df = pd.DataFrame({
            "Src IP"        : ["192.168.1.1", "192.168.1.2"],
            "Dst IP"        : ["10.0.0.1",    "10.0.0.2"],
            "Total Bytes"   : [1500, 800],
            "Total Packets" : [10, 5],
            "Protocol"      : [6, 17],
            "Flow Duration" : [100.0, 200.0],
            "Packets/s"     : [10.0, 5.0],
            "Bytes/s"       : [1500.0, 800.0],
        })
        metrics = _compute_global_metrics(df)
        self.assertEqual(metrics["total_flows"], 2)
        self.assertIn("protocol_distribution", metrics)
        self.assertIn("top_src_ips", metrics)
        self.assertIn("top_dst_ips", metrics)

    def test_flags_summary(self):
        df = _base_df(
            **{
                "SYN Flag Count": [5],
                "RST Flag Count": [2],
                "FIN Flag Count": [1],
            }
        )
        metrics = _compute_global_metrics(df)
        self.assertIn("flags_summary", metrics)
        self.assertEqual(metrics["flags_summary"].get("syn"), 5)

    def test_empty_df_does_not_crash(self):
        """DataFrame vacío no debe lanzar excepción."""
        df = pd.DataFrame(columns=["Src IP", "Dst IP", "Protocol"])
        try:
            _compute_global_metrics(df)
        except Exception as e:
            self.fail(f"_compute_global_metrics lanzó excepción: {e}")


# =============================================================
#  RESUMEN DE ALERTAS
# =============================================================

class TestBuildAlertSummary(unittest.TestCase):

    def test_empty_alerts(self):
        summary = _build_alert_summary([])
        self.assertEqual(summary["total_alerts"], 0)
        self.assertEqual(summary["by_severity"], {})
        self.assertEqual(summary["by_type"], {})

    def test_counts_correct(self):
        alerts = [
            {"tipo": "SYN Flood",     "severidad": "CRITICA", "src_ip": "1.1.1.1"},
            {"tipo": "SYN Flood",     "severidad": "CRITICA", "src_ip": "2.2.2.2"},
            {"tipo": "Port Scan Vertical", "severidad": "ALTA", "src_ip": "3.3.3.3"},
        ]
        summary = _build_alert_summary(alerts)
        self.assertEqual(summary["total_alerts"], 3)
        self.assertEqual(summary["by_severity"]["CRITICA"], 2)
        self.assertEqual(summary["by_severity"]["ALTA"], 1)
        self.assertEqual(summary["by_type"]["SYN Flood"], 2)

    def test_critical_ips_ranked(self):
        alerts = [
            {"tipo": "X", "severidad": "ALTA",    "src_ip": "10.0.0.1"},
            {"tipo": "X", "severidad": "CRITICA", "src_ip": "10.0.0.1"},
            {"tipo": "X", "severidad": "MEDIA",   "src_ip": "10.0.0.2"},
        ]
        summary = _build_alert_summary(alerts)
        ips = list(summary["critical_ips"].keys())
        self.assertEqual(ips[0], "10.0.0.1")  # más alertas primero


# =============================================================
#  ANALYZE_FLOWS (integración)
# =============================================================

class TestAnalyzeFlows(unittest.TestCase):

    def _make_clean_df(self):
        return pd.DataFrame({
            "Src IP"           : ["192.168.1.1", "192.168.1.2", "10.0.0.5"],
            "Dst IP"           : ["8.8.8.8",     "8.8.4.4",     "192.168.1.1"],
            "Destination Port" : [53,             443,            80],
            "Protocol"         : [17,             6,              6],
            "Total Packets"    : [5,              20,             8],
            "Total Bytes"      : [300,            5000,           800],
            "Flow Duration"    : [50.0,           200.0,          80.0],
            "SYN Flag Count"   : [0,              1,              1],
            "FIN Flag Count"   : [0,              1,              1],
            "RST Flag Count"   : [0,              0,              0],
            "PSH Flag Count"   : [0,              3,              2],
            "ACK Flag Count"   : [0,              8,              4],
            "URG Flag Count"   : [0,              0,              0],
            "Subflow Fwd Bytes": [200,            3000,           600],
            "Subflow Bwd Bytes": [100,            2000,           200],
            "Packets/s"        : [5.0,            20.0,           8.0],
            "Bytes/s"          : [300.0,          5000.0,         800.0],
        })

    def test_returns_required_keys(self):
        df = self._make_clean_df()
        result = analyze_flows(df)
        self.assertIn("global_metrics", result)
        self.assertIn("alerts", result)
        self.assertIn("alert_summary", result)

    def test_global_metrics_correct(self):
        df = self._make_clean_df()
        result = analyze_flows(df)
        self.assertEqual(result["global_metrics"]["total_flows"], 3)

    def test_alert_summary_structure(self):
        df = self._make_clean_df()
        result = analyze_flows(df)
        summary = result["alert_summary"]
        self.assertIn("total_alerts", summary)
        self.assertIn("by_severity", summary)
        self.assertIn("by_type", summary)
        self.assertIn("critical_ips", summary)

    def test_no_crash_on_minimal_df(self):
        """DataFrame mínimo no debe lanzar excepción."""
        df = pd.DataFrame({
            "Src IP": ["1.1.1.1"],
            "Dst IP": ["2.2.2.2"],
        })
        try:
            analyze_flows(df)
        except Exception as e:
            self.fail(f"analyze_flows lanzó excepción con df mínimo: {e}")

    def test_export_json(self, tmp_path=None):
        """Exportación a JSON produce archivo válido."""
        import tempfile, json, os
        df = self._make_clean_df()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            analyze_flows(df, export_json=path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIn("global_metrics", data)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)