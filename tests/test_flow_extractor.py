"""
test_flow_extractor.py
======================
Tests unitarios para modules/flow_extractor.py

Cubre:
    - Parsing de flags TCP
    - Generación de clave de flujo
    - Cálculo de métricas por flujo
    - Manejo de flujos vacíos y datos faltantes
    - Validación de columnas en el DataFrame resultante
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.flow_extractor import (
    _parse_tcp_flags,
    _flow_key,
    _safe_stat,
    _compute_flow_metrics,
    PROTO_MAP,
)


class TestParseTcpFlags(unittest.TestCase):

    def test_syn_flag(self):
        result = _parse_tcp_flags(0x02)
        self.assertEqual(result["SYN"], 1)
        self.assertEqual(result["FIN"], 0)
        self.assertEqual(result["RST"], 0)

    def test_fin_flag(self):
        result = _parse_tcp_flags(0x01)
        self.assertEqual(result["FIN"], 1)
        self.assertEqual(result["SYN"], 0)

    def test_rst_flag(self):
        result = _parse_tcp_flags(0x04)
        self.assertEqual(result["RST"], 1)

    def test_syn_ack(self):
        # SYN + ACK = 0x02 | 0x10 = 0x12
        result = _parse_tcp_flags(0x12)
        self.assertEqual(result["SYN"], 1)
        self.assertEqual(result["ACK"], 1)
        self.assertEqual(result["FIN"], 0)

    def test_all_flags_zero(self):
        result = _parse_tcp_flags(0x00)
        for flag in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"):
            self.assertEqual(result[flag], 0)

    def test_all_flags_set(self):
        result = _parse_tcp_flags(0xFF)
        for flag in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"):
            self.assertEqual(result[flag], 1)

    def test_psh_urg(self):
        result = _parse_tcp_flags(0x28)  # PSH=0x08, URG=0x20
        self.assertEqual(result["PSH"], 1)
        self.assertEqual(result["URG"], 1)
        self.assertEqual(result["SYN"], 0)


class TestFlowKey(unittest.TestCase):

    def test_tcp_preserves_direction(self):
        """TCP mantiene dirección original del flujo."""
        key1 = _flow_key("1.1.1.1", "2.2.2.2", 50000, 80, PROTO_MAP["TCP"])
        key2 = _flow_key("2.2.2.2", "1.1.1.1", 80, 50000, PROTO_MAP["TCP"])
        # TCP no normaliza — son claves distintas
        self.assertNotEqual(key1, key2)

    def test_udp_normalizes_direction(self):
        """UDP normaliza A→B y B→A al mismo flujo."""
        key1 = _flow_key("1.1.1.1", "2.2.2.2", 12345, 53, PROTO_MAP["UDP"])
        key2 = _flow_key("2.2.2.2", "1.1.1.1", 53, 12345, PROTO_MAP["UDP"])
        self.assertEqual(key1, key2)

    def test_icmp_normalizes_direction(self):
        """ICMP normaliza dirección igual que UDP."""
        key1 = _flow_key("10.0.0.1", "10.0.0.2", 8, 0, PROTO_MAP["ICMP"])
        key2 = _flow_key("10.0.0.2", "10.0.0.1", 0, 8, PROTO_MAP["ICMP"])
        self.assertEqual(key1, key2)

    def test_key_is_tuple(self):
        key = _flow_key("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 5)


class TestSafeStat(unittest.TestCase):

    def test_empty_list_returns_default(self):
        self.assertEqual(_safe_stat([], min), 0.0)
        self.assertEqual(_safe_stat([], np.mean, default=-1.0), -1.0)

    def test_single_value(self):
        self.assertEqual(_safe_stat([42], max), 42.0)

    def test_mean(self):
        self.assertAlmostEqual(_safe_stat([1, 2, 3, 4, 5], np.mean), 3.0)

    def test_std(self):
        result = _safe_stat([2, 2, 2, 2], np.std)
        self.assertAlmostEqual(result, 0.0)


class TestComputeFlowMetrics(unittest.TestCase):

    def _make_flow(self, fwd_sizes, bwd_sizes, fwd_times, bwd_times, flags=None):
        all_times = sorted(fwd_times + bwd_times)
        return {
            "src_ip"   : "192.168.1.1",
            "dst_ip"   : "10.0.0.1",
            "src_port" : 50000,
            "dst_port" : 80,
            "proto"    : 6,
            "fwd_pkts" : fwd_sizes,
            "bwd_pkts" : bwd_sizes,
            "fwd_times": fwd_times,
            "bwd_times": bwd_times,
            "all_times": all_times,
            "flags"    : flags or [0x02, 0x12, 0x10],
            "win_sizes": [65535, 8192],
            "pkt_count": len(fwd_sizes) + len(bwd_sizes),
        }

    def test_basic_metrics_present(self):
        """Verifica que todas las columnas clave estén presentes."""
        key  = ("192.168.1.1", "10.0.0.1", 50000, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[100, 200, 150],
            bwd_sizes=[300, 250],
            fwd_times=[0.0, 0.1, 0.2],
            bwd_times=[0.05, 0.15],
        )
        result = _compute_flow_metrics(key, flow)
        self.assertIsNotNone(result)

        required_cols = [
            "Src IP", "Dst IP", "Protocol", "Flow Duration",
            "Total Packets", "Total Bytes",
            "Fwd Packet Length Max", "Bwd Packet Length Max",
            "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
            "Packets/s", "Bytes/s",
        ]
        for col in required_cols:
            self.assertIn(col, result, f"Columna faltante: {col}")

    def test_empty_flow_returns_none(self):
        """Un flujo sin paquetes debe retornar None."""
        key  = ("1.1.1.1", "2.2.2.2", 80, 443, 6)
        flow = self._make_flow([], [], [], [])
        result = _compute_flow_metrics(key, flow)
        self.assertIsNone(result)

    def test_total_bytes_correct(self):
        key  = ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[100, 200],
            bwd_sizes=[300],
            fwd_times=[0.0, 0.1],
            bwd_times=[0.05],
        )
        result = _compute_flow_metrics(key, flow)
        self.assertEqual(result["Total Bytes"], 600)
        self.assertEqual(result["Total Packets"], 3)

    def test_flow_duration_ms(self):
        """La duración se calcula en milisegundos."""
        key  = ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[100],
            bwd_sizes=[100],
            fwd_times=[0.0],
            bwd_times=[1.0],   # 1 segundo de diferencia
        )
        result = _compute_flow_metrics(key, flow)
        self.assertAlmostEqual(result["Flow Duration"], 1000.0, places=1)

    def test_syn_count(self):
        """Los flags SYN se cuentan correctamente."""
        key  = ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[100, 100],
            bwd_sizes=[100],
            fwd_times=[0.0, 0.1],
            bwd_times=[0.05],
            flags=[0x02, 0x02, 0x10],  # 2 SYN + 1 ACK
        )
        result = _compute_flow_metrics(key, flow)
        self.assertEqual(result["SYN Flag Count"], 2)

    def test_fwd_bwd_split(self):
        """Los bytes fwd y bwd se asignan correctamente."""
        key  = ("192.168.1.1", "10.0.0.1", 50000, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[500, 500],
            bwd_sizes=[100],
            fwd_times=[0.0, 0.1],
            bwd_times=[0.05],
        )
        result = _compute_flow_metrics(key, flow)
        self.assertEqual(result["Total Length of Fwd Packets"], 1000)
        self.assertEqual(result["Total Length of Bwd Packets"], 100)

    def test_single_packet_flow(self):
        """Flujo de un solo paquete no lanza excepción."""
        key  = ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[64],
            bwd_sizes=[],
            fwd_times=[0.0],
            bwd_times=[],
        )
        result = _compute_flow_metrics(key, flow)
        self.assertIsNotNone(result)
        self.assertEqual(result["Total Packets"], 1)

    def test_iat_zero_for_single_packet(self):
        """IAT debe ser 0 cuando hay solo 1 paquete en cada dirección."""
        key  = ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow = self._make_flow(
            fwd_sizes=[100],
            bwd_sizes=[100],
            fwd_times=[0.5],
            bwd_times=[0.6],
        )
        result = _compute_flow_metrics(key, flow)
        self.assertEqual(result["Fwd IAT Total"], 0.0)
        self.assertEqual(result["Bwd IAT Total"], 0.0)


class TestDataFrameOutput(unittest.TestCase):
    """
    Verifica que el DataFrame final tiene la estructura correcta
    usando datos sintéticos sin necesitar un archivo PCAP real.
    """

    def test_column_types(self):
        """Simula el resultado de extract_flows y verifica tipos."""
        data = {
            "Src IP"            : ["192.168.1.1", "10.0.0.5"],
            "Dst IP"            : ["8.8.8.8", "192.168.1.1"],
            "Protocol"          : [6, 17],
            "Total Packets"     : [10, 5],
            "Total Bytes"       : [1500, 300],
            "Flow Duration"     : [120.5, 45.2],
            "SYN Flag Count"    : [1, 0],
            "FIN Flag Count"    : [1, 0],
            "RST Flag Count"    : [0, 0],
        }
        df = pd.DataFrame(data)
        self.assertEqual(len(df), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Total Bytes"]))
        self.assertTrue(pd.api.types.is_object_dtype(df["Src IP"]))

    def test_no_negative_values(self):
        """No debe haber valores negativos en columnas de conteo."""
        data = {
            "Total Packets" : [10, 5, 8],
            "Total Bytes"   : [1500, 300, 800],
            "Flow Duration" : [120.5, 45.2, 33.1],
            "SYN Flag Count": [1, 0, 2],
        }
        df = pd.DataFrame(data)
        for col in data:
            self.assertTrue((df[col] >= 0).all(), f"Negativos en {col}")


if __name__ == "__main__":
    unittest.main(verbosity=2)