"""
test_topology_mapper.py
=======================
Tests unitarios para modules/topology_mapper.py

Cubre:
    - Detección de IPs privadas vs públicas
    - Cálculo de subred /24
    - Construcción del grafo
    - Cálculo de métricas por nodo
    - Clasificación de dispositivos
    - Detección de subredes
    - Función principal map_topology()
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.topology_mapper import (
    _is_private,
    _get_subnet,
    _build_graph,
    _compute_node_metrics,
    _classify_device,
    _classify_all_devices,
    _detect_subnets,
    _build_topology_json,
    map_topology,
    ROLE_PORT_SIGNATURES,
)


# =============================================================
#  FIXTURES
# =============================================================

def _make_df(**overrides) -> pd.DataFrame:
    """DataFrame mínimo para topology_mapper."""
    defaults = {
        "Src IP"           : ["192.168.1.10", "192.168.1.20", "10.0.0.5"],
        "Dst IP"           : ["192.168.1.1",  "8.8.8.8",      "192.168.1.10"],
        "Destination Port" : [80,              443,             22],
        "Src Port"         : [50001,           51000,           22],
        "Protocol"         : [6,               6,               6],
        "flow_count"       : [5,               3,               2],
        "total_bytes"      : [5000,            3000,            800],
        "total_packets"    : [20,              15,              8],
        "avg_duration"     : [200.0,           150.0,           80.0],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# =============================================================
#  IP HELPERS
# =============================================================

class TestIsPrivate(unittest.TestCase):

    def test_rfc1918_10(self):
        self.assertTrue(_is_private("10.0.0.1"))
        self.assertTrue(_is_private("10.255.255.254"))

    def test_rfc1918_172(self):
        self.assertTrue(_is_private("172.16.0.1"))
        self.assertTrue(_is_private("172.31.255.254"))

    def test_rfc1918_192(self):
        self.assertTrue(_is_private("192.168.0.1"))
        self.assertTrue(_is_private("192.168.255.254"))

    def test_loopback(self):
        self.assertTrue(_is_private("127.0.0.1"))

    def test_public_ip(self):
        self.assertFalse(_is_private("8.8.8.8"))
        self.assertFalse(_is_private("1.1.1.1"))
        self.assertFalse(_is_private("93.184.216.34"))

    def test_invalid_ip(self):
        """IPs inválidas no deben lanzar excepción."""
        result = _is_private("no_es_ip")
        self.assertFalse(result)


class TestGetSubnet(unittest.TestCase):

    def test_private_returns_slash24(self):
        self.assertEqual(_get_subnet("192.168.1.50"),  "192.168.1.0/24")
        self.assertEqual(_get_subnet("10.0.0.100"),    "10.0.0.0/24")
        self.assertEqual(_get_subnet("172.16.5.200"),  "172.16.5.0/24")

    def test_public_returns_externo(self):
        self.assertEqual(_get_subnet("8.8.8.8"),       "externo")
        self.assertEqual(_get_subnet("1.1.1.1"),        "externo")

    def test_invalid_returns_desconocido(self):
        self.assertEqual(_get_subnet("not_an_ip"),     "desconocido")


# =============================================================
#  CONSTRUCCIÓN DEL GRAFO
# =============================================================

class TestBuildGraph(unittest.TestCase):

    def test_nodes_created(self):
        df = _make_df()
        G  = _build_graph(df)
        all_ips = set(df["Src IP"]) | set(df["Dst IP"])
        for ip in all_ips:
            self.assertIn(ip, G.nodes)

    def test_edges_created(self):
        df = _make_df()
        G  = _build_graph(df)
        self.assertGreater(len(G.edges), 0)

    def test_edge_attributes(self):
        df = _make_df()
        G  = _build_graph(df)
        for _, _, data in G.edges(data=True):
            self.assertIn("flow_count",    data)
            self.assertIn("total_bytes",   data)
            self.assertIn("total_packets", data)

    def test_node_private_attribute(self):
        df = _make_df()
        G  = _build_graph(df)
        self.assertTrue(G.nodes["192.168.1.10"]["is_private"])
        self.assertFalse(G.nodes["8.8.8.8"]["is_private"])

    def test_duplicate_edges_accumulate(self):
        """Dos filas con el mismo par IP deben acumular bytes."""
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.1", "192.168.1.1"],
            "Dst IP"       : ["10.0.0.1",    "10.0.0.1"],
            "total_bytes"  : [1000,           2000],
            "total_packets": [10,             20],
            "flow_count"   : [2,              3],
        })
        G = _build_graph(df)
        edge = G["192.168.1.1"]["10.0.0.1"]
        self.assertEqual(edge["total_bytes"],   3000)
        self.assertEqual(edge["total_packets"], 30)
        self.assertEqual(edge["flow_count"],    5)


# =============================================================
#  MÉTRICAS POR NODO
# =============================================================

class TestComputeNodeMetrics(unittest.TestCase):

    def test_bytes_sent_received(self):
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.1"],
            "Dst IP"       : ["10.0.0.1"],
            "total_bytes"  : [5000],
            "total_packets": [20],
            "flow_count"   : [3],
        })
        G       = _build_graph(df)
        metrics = _compute_node_metrics(G, df)

        self.assertEqual(metrics["192.168.1.1"]["total_bytes_sent"], 5000)
        self.assertEqual(metrics["10.0.0.1"]["total_bytes_recv"],    5000)

    def test_unique_dst_count(self):
        df = pd.DataFrame({
            "Src IP"       : ["192.168.1.1"] * 3,
            "Dst IP"       : ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            "total_bytes"  : [100, 200, 300],
            "total_packets": [1, 2, 3],
            "flow_count"   : [1, 1, 1],
        })
        G       = _build_graph(df)
        metrics = _compute_node_metrics(G, df)
        self.assertEqual(metrics["192.168.1.1"]["unique_dst_count"], 3)

    def test_dst_ports_tracked(self):
        df = pd.DataFrame({
            "Src IP"          : ["192.168.1.1", "192.168.1.1"],
            "Dst IP"          : ["10.0.0.1",    "10.0.0.1"],
            "Destination Port": [80,             443],
            "total_bytes"     : [100, 200],
            "total_packets"   : [1, 2],
            "flow_count"      : [1, 1],
        })
        G       = _build_graph(df)
        metrics = _compute_node_metrics(G, df)
        ports   = set(metrics["10.0.0.1"]["dst_ports_used"])
        self.assertIn(80,  ports)
        self.assertIn(443, ports)


# =============================================================
#  CLASIFICACIÓN DE DISPOSITIVOS
# =============================================================

class TestClassifyDevice(unittest.TestCase):

    def _metrics(self, **kwargs):
        base = {
            "is_private"             : True,
            "total_bytes_sent"       : 1000,
            "total_bytes_recv"       : 1000,
            "total_packets_sent"     : 10,
            "total_packets_recv"     : 10,
            "flow_count_out"         : 3,
            "flow_count_in"          : 3,
            "unique_dst_count"       : 2,
            "unique_src_count"       : 2,
            "dst_ports_used"         : [],
            "src_ports_used"         : [],
            "unique_ports_dst"       : 0,
            "protocols"              : [6],
            "subnet"                 : "192.168.1.0/24",
            "degree_centrality"      : 0.1,
            "betweenness_centrality" : 0.05,
        }
        base.update(kwargs)
        return base

    def test_externo(self):
        m    = self._metrics(is_private=False)
        role = _classify_device("8.8.8.8", m, pd.DataFrame())
        self.assertEqual(role, "externo")

    def test_servidor_web(self):
        m    = self._metrics(dst_ports_used=[80, 443])
        role = _classify_device("192.168.1.100", m, pd.DataFrame())
        self.assertEqual(role, "servidor_web")

    def test_servidor_dns(self):
        m    = self._metrics(dst_ports_used=[53])
        role = _classify_device("192.168.1.50", m, pd.DataFrame())
        self.assertEqual(role, "servidor_dns")

    def test_servidor_db(self):
        m    = self._metrics(dst_ports_used=[3306])
        role = _classify_device("192.168.1.20", m, pd.DataFrame())
        self.assertEqual(role, "servidor_db")

    def test_impresora(self):
        m    = self._metrics(dst_ports_used=[9100])
        role = _classify_device("192.168.1.30", m, pd.DataFrame())
        self.assertEqual(role, "impresora")

    def test_camara_ip(self):
        m    = self._metrics(dst_ports_used=[554])
        role = _classify_device("192.168.1.40", m, pd.DataFrame())
        self.assertEqual(role, "camara_ip")

    def test_iot(self):
        m    = self._metrics(dst_ports_used=[1883])  # MQTT
        role = _classify_device("192.168.1.60", m, pd.DataFrame())
        self.assertEqual(role, "iot")

    def test_gateway(self):
        m    = self._metrics(
            unique_dst_count=25,
            flow_count_in=50,
            flow_count_out=50,
            betweenness_centrality=0.5,
            dst_ports_used=[],
        )
        role = _classify_device("192.168.1.1", m, pd.DataFrame())
        self.assertEqual(role, "gateway")

    def test_workstation(self):
        m    = self._metrics(
            flow_count_out=10,
            flow_count_in=0,
            src_ports_used=[50001, 51234, 52000, 53100],
            dst_ports_used=[],
        )
        role = _classify_device("192.168.1.50", m, pd.DataFrame())
        self.assertEqual(role, "workstation")


# =============================================================
#  DETECCIÓN DE SUBREDES
# =============================================================

class TestDetectSubnets(unittest.TestCase):

    def test_groups_by_subnet(self):
        node_metrics = {
            "192.168.1.10" : {"subnet": "192.168.1.0/24"},
            "192.168.1.20" : {"subnet": "192.168.1.0/24"},
            "10.0.0.5"     : {"subnet": "10.0.0.0/24"},
            "8.8.8.8"      : {"subnet": "externo"},
        }
        subnets = _detect_subnets(node_metrics)
        self.assertIn("192.168.1.0/24", subnets)
        self.assertIn("10.0.0.0/24", subnets)
        self.assertEqual(len(subnets["192.168.1.0/24"]), 2)

    def test_external_grouped_as_externo(self):
        node_metrics = {
            "8.8.8.8" : {"subnet": "externo"},
            "1.1.1.1" : {"subnet": "externo"},
        }
        subnets = _detect_subnets(node_metrics)
        self.assertIn("externo", subnets)
        self.assertEqual(len(subnets["externo"]), 2)


# =============================================================
#  MAP_TOPOLOGY (integración)
# =============================================================

class TestMapTopology(unittest.TestCase):

    def test_returns_required_keys(self):
        df     = _make_df()
        result = map_topology(df)
        for key in ("nodes", "edges", "subnets", "role_summary",
                    "total_nodes", "total_edges", "total_subnets"):
            self.assertIn(key, result)

    def test_total_nodes_correct(self):
        df     = _make_df()
        result = map_topology(df)
        all_ips = set(df["Src IP"]) | set(df["Dst IP"])
        self.assertEqual(result["total_nodes"], len(all_ips))

    def test_nodes_have_required_fields(self):
        df     = _make_df()
        result = map_topology(df)
        for node in result["nodes"]:
            for field in ("id", "role", "color", "is_private",
                          "subnet", "total_bytes_sent", "total_bytes_recv"):
                self.assertIn(field, node, f"Campo faltante en nodo: {field}")

    def test_edges_have_required_fields(self):
        df     = _make_df()
        result = map_topology(df)
        for edge in result["edges"]:
            for field in ("source", "target", "flow_count", "total_bytes"):
                self.assertIn(field, edge, f"Campo faltante en arista: {field}")

    def test_role_summary_populated(self):
        df     = _make_df()
        result = map_topology(df)
        self.assertGreater(len(result["role_summary"]), 0)

    def test_empty_df_raises(self):
        """DataFrame vacío debe lanzar ValueError."""
        with self.assertRaises(ValueError):
            map_topology(pd.DataFrame())

    def test_export_json(self):
        """Exportación a JSON produce archivo válido."""
        import tempfile, json, os
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            map_topology(df, output_dir=tmpdir)
            json_path = os.path.join(tmpdir, "topology.json")
            self.assertTrue(os.path.exists(json_path))
            with open(json_path) as f:
                data = json.load(f)
            self.assertIn("nodes", data)
            self.assertIn("edges", data)

    def test_no_graph_png_skipped(self):
        """Con graph_png=None no se intenta generar imagen."""
        df = _make_df()
        try:
            map_topology(df, graph_png=None)
        except Exception as e:
            self.fail(f"map_topology con graph_png=None lanzó: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)