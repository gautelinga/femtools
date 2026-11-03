"""`femtools.xdmf.grids`: where the arrays of every grid of an XDMF file are; and
`parse_xdmf`, partractools' interface to the same.

    python3 -m pytest tests/test_xdmf_grids.py

A time series as dolfin writes it, each step after the first taking its mesh by
xi:include; the older layout with the times listed once for the collection; a single
grid; and data held in the XML, which is refused. The XML is written out here, so only
numpy is needed.
"""
import os

import pytest

from femtools.xdmf import grids, parse_xdmf

SERIES = """<?xml version="1.0"?>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries_u" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Topology NumberOfElements="2" TopologyType="Triangle" NodesPerElement="3">
          <DataItem Dimensions="2 3" NumberType="UInt" Format="HDF">u.h5:/Mesh/0/mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XY">
          <DataItem Dimensions="4 2" Format="HDF">u.h5:/Mesh/0/mesh/geometry</DataItem>
        </Geometry>
        <Time Value="0.000000000000000e+00" />
        <Attribute Name="u" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="4 1" Format="HDF">u.h5:/VisualisationVector/0</DataItem>
        </Attribute>
      </Grid>
      <Grid>
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_u&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
        <Time Value="5.000000000000000e-01" />
        <Attribute Name="u" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="4 1" Format="HDF">u.h5:/VisualisationVector/1</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

LISTED = """<?xml version="1.0"?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List">
        <DataItem Format="XML" Dimensions="2"> 0.25 0.75</DataItem>
      </Time>
      <Grid Name="grid_1" GridType="Uniform">
        <Topology NumberOfElements="1" TopologyType="Tetrahedron">
          <DataItem Format="HDF" Dimensions="1 4">sub/m.h5:/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Format="HDF" Dimensions="4 3">sub/m.h5:/coordinates</DataItem>
        </Geometry>
        <Attribute Name="p" AttributeType="Scalar" Center="Cell">
          <DataItem Format="HDF" Dimensions="1">sub/m.h5:/p0</DataItem>
        </Attribute>
      </Grid>
      <Grid Name="grid_2" GridType="Uniform">
        <Topology NumberOfElements="1" TopologyType="Tetrahedron">
          <DataItem Format="HDF" Dimensions="1 4">sub/m.h5:/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Format="HDF" Dimensions="4 3">sub/m.h5:/coordinates</DataItem>
        </Geometry>
        <Attribute Name="p" AttributeType="Scalar" Center="Cell">
          <DataItem Format="HDF" Dimensions="1">sub/m.h5:/p1</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""


def write(tmp_path, text, name="f.xdmf"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_a_dolfin_time_series_carries_the_mesh_to_every_step(tmp_path):
    path = write(tmp_path, SERIES)
    h5 = os.path.join(str(tmp_path), "u.h5")
    got = grids(path)
    assert [g["time"] for g in got] == [0.0, 0.5]
    for i, g in enumerate(got):
        assert g["geometry"] == (h5, "/Mesh/0/mesh/geometry")
        assert g["topology"] == (h5, "/Mesh/0/mesh/topology")
        assert g["attributes"] == {"u": (h5, "/VisualisationVector/{}".format(i))}
        assert g["centers"] == {"u": "Node"}


def test_times_listed_for_the_collection_go_to_its_grids(tmp_path):
    path = write(tmp_path, LISTED)
    h5 = os.path.join(str(tmp_path), "sub", "m.h5")
    got = grids(path)
    assert [g["time"] for g in got] == [0.25, 0.75]
    assert [g["attributes"]["p"] for g in got] == [(h5, "/p0"), (h5, "/p1")]
    assert got[1]["centers"] == {"p": "Cell"}


def test_a_single_grid_has_no_time(tmp_path):
    text = LISTED.split('<Grid Name="grid_1"', 1)[1].split("</Grid>", 1)[0]
    path = write(tmp_path, '<?xml version="1.0"?>\n<Xdmf><Domain><Grid Name="grid_1"'
                 + text + "</Grid></Domain></Xdmf>\n")
    (g,) = grids(path)
    assert g["time"] is None and g["topology"][1] == "/topology"


def test_data_in_the_xml_is_refused(tmp_path):
    path = write(tmp_path, SERIES.replace(
        '<DataItem Dimensions="4 1" Format="HDF">u.h5:/VisualisationVector/1</DataItem>',
        '<DataItem Dimensions="4 1" Format="XML">0 1 2 3</DataItem>'))
    with pytest.raises(ValueError, match="u is not in an HDF5 file"):
        grids(path)


def test_parse_xdmf_gives_each_time_its_dataset_and_the_mesh(tmp_path):
    path = write(tmp_path, SERIES)
    h5 = os.path.join(str(tmp_path), "u.h5")
    dsets, topology, geometry = parse_xdmf(path, get_mesh_address=True)
    assert dsets == [(0.0, [h5, "/VisualisationVector/0"]), (0.5, [h5, "/VisualisationVector/1"])]
    assert topology == [h5, "/Mesh/0/mesh/topology"]
    assert geometry == [h5, "/Mesh/0/mesh/geometry"]
    assert parse_xdmf(path) == dsets


def test_parse_xdmf_takes_listed_times_and_keeps_a_relative_path_relative(tmp_path, monkeypatch):
    write(tmp_path, LISTED)
    monkeypatch.chdir(tmp_path)
    assert parse_xdmf("f.xdmf") == [(0.25, ["sub/m.h5", "/p0"]), (0.75, ["sub/m.h5", "/p1"])]


def test_parse_xdmf_reads_a_single_grid_as_one_step_without_a_time(tmp_path):
    text = LISTED.split('<Grid Name="grid_1"', 1)[1].split("</Grid>", 1)[0]
    path = write(tmp_path, '<?xml version="1.0"?>\n<Xdmf><Domain><Grid Name="grid_1"'
                 + text + "</Grid></Domain></Xdmf>\n")
    h5 = os.path.join(str(tmp_path), "sub", "m.h5")
    assert parse_xdmf(path, get_mesh_address=True) == (
        [(None, [h5, "/p0"])], [h5, "/topology"], [h5, "/coordinates"])
