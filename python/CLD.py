"""
    CLD specialized functions for bib-studies.

    Adapted from ALLEGRO.py for the CLD_o2_vXX detector concept.
    Covers the full silicon tracker (Vertex + IT + OT). Calorimeters use the
    default recursive get_cells() walk.

    --- Why this exists ---

    The default CLD.py only had the recursive walk and gave wrong cell counts
    (=1 per layer) for all endcap sub-detectors, plus missing sensor areas.

    Looking at the DD4hep drivers for the endcaps:

      * TrackerEndcap_o2_v06_geo.cpp (lines ~175-215) and
      * VertexEndcap_o1_v06_geo.cpp  (lines ~155-190)

    both create a DetElement per module DIRECTLY under the sub-detector, with
    flat names of the form 'layer{L}_module{M}_sensor{K}_pos' or '_neg'.
    There is NO 'layer N' DetElement above them. The reflected -z side is
    also instantiated explicitly with the '_neg' suffix.

    So the right thing to do for endcaps is to:
      - iterate sub_det.children() (all modules at flat level)
      - parse name with regex to extract (layer, side)
      - aggregate sensor and area counts per signed layer
      - apply the +1 layer shift to remove layer 0 degeneracy

    For BARRELS the original recursive descent works (as confirmed by the
    JSON output already produced for VertexBarrel / IT-Barrel / OT-Barrel).

    --- CellID encoding ---

    All CLD silicon trackers use:
        GlobalTrackerReadoutID = "system:5,side:-2,layer:6,module:11,sensor:8"
    + per-readout pixel fields, e.g. for VertexBarrel: ",y:32:-16,z:-16".
    So the full cellID encodes pixel position natively.

    For drawhits.py occupancy interpretation see the README at the bottom.
"""

from collections import defaultdict
import re

from helpers import get_cells, is_endcap


# ---------------------------------------------------------------------------
#  Endcap DetElement-name pattern (set in the DD4hep drivers)
# ---------------------------------------------------------------------------

#  TrackerEndcap_o2_v06_geo.cpp line 178:
#    string m_base = _toString(l_id, "layer%d") + _toString(mod_num, "_module%d")
#                  + _toString(k, "_sensor%d");
#  VertexEndcap_o1_v06_geo.cpp line 158: same pattern.
_ENDCAP_NAME_RE = re.compile(r"layer(\d+)_module(\d+)_sensor(\d+)_(pos|neg)")


# ---------------------------------------------------------------------------
#  Sensor area helper (handles the actual TGeo shapes used by CLD drivers)
# ---------------------------------------------------------------------------

def _solid_area_mm2(solid):
    """
    Active face area of a TGeo solid, in mm^2.

    CLD drivers use:
      - Box        (TrackerEndcap_o2_v06): module_component is a Box(dx, dy, c_thick/2)
                   with active face dx*dy in (x,y) plane.
      - Trapezoid  (VertexEndcap_o1_v06):  module_component is a Trapezoid(x1,x2,y1,y2,z)
                   with active face (x1+x2) * 2*z in (x,z) plane.
      - Box        (TrackerBarrel_o1_v06): module envelope and components.
      - Box        (ZPlanarTracker):       ladder is a Box.

    Returns 0 if the shape is unknown.
    """
    if solid is None:
        return 0.0
    name = solid.ClassName()
    CM2MM = 10.0
    try:
        if name == "TGeoBBox":
            # A silicon sensor is always thin in one dimension. The active face
            # is therefore the LARGEST of the three Box faces, regardless of
            # which TGeo axis carries the thickness. Robust against driver
            # orientation conventions (ZPlanarTracker: thickness on X;
            # TrackerBarrel/Endcap: thickness on Z; etc.).
            a = 2 * solid.GetDX() * CM2MM
            b = 2 * solid.GetDY() * CM2MM
            c = 2 * solid.GetDZ() * CM2MM
            return max(a * b, a * c, b * c)
        if name == "TGeoTrd1":
            # TGeoTrd1: half-lengths Dx1, Dx2 along x; Dy along y; Dz along z.
            # Active face is the (x,z) trapezoid: area = (Dx1 + Dx2) * 2*Dz
            return (solid.GetDx1() + solid.GetDx2()) * CM2MM * (2 * solid.GetDz() * CM2MM)
        if name == "TGeoTrd2":
            # Active face = (x,z) trapezoid (Dy is the thickness)
            return (solid.GetDx1() + solid.GetDx2()) * CM2MM * (2 * solid.GetDz() * CM2MM)
    except Exception:
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
#  Endcap walk: iterate flat module DetElements, aggregate per (layer, side)
# ---------------------------------------------------------------------------

def _walk_endcap_subdet(sub_det):
    """
    Build per-layer maps for an endcap sub-detector by parsing the flat
    DetElement names that the TrackerEndcap / VertexEndcap drivers produce.

    The signed layer key follows the +1 shift convention from ALLEGRO.py
    (so that layer 0 on +z and -z don't collapse to the same int):
        signed_layer = (l_unsigned + 1) * side,    side in {+1, -1}

    Returns:
        (cells_map, sensor_size_map, sensors_per_module_map)
    """
    cells_map = defaultdict(int)
    layer_area_sum = defaultdict(float)   # signed_layer -> sum of per-module areas
    layer_area_n = defaultdict(int)       # number of areas summed (denominator)
    layer_module_set = defaultdict(set)   # signed_layer -> set of module (= ring) indices

    n_unmatched = 0

    for de_name, de in sub_det.children():
        name = str(de_name)
        m = _ENDCAP_NAME_RE.match(name)
        if not m:
            n_unmatched += 1
            continue

        l_unsigned = int(m.group(1))
        mod_idx = int(m.group(2))
        side = +1 if m.group(4) == "pos" else -1
        signed_layer = (l_unsigned + 1) * side

        # one sensor (= one module placement in the ring) per matched DetElement
        cells_map[signed_layer] += 1
        layer_module_set[signed_layer].add(mod_idx)

        # accumulate sensor area for THIS module so we can average over the
        # whole layer afterwards. This matters for ITEndcap where each ring
        # uses a different module size: the per-layer "average" sensor area
        # is much more meaningful than a single arbitrary first-encountered
        # ring's module.
        try:
            area = 0.0
            kids = list(de.children())
            if len(kids) > 0:
                # The module DetElement has the sensitive component as a child
                _, comp_de = kids[0]
                area = _solid_area_mm2(comp_de.volume().solid())
            if area == 0.0:
                # Fallback: read the area from the module envelope itself
                area = _solid_area_mm2(de.volume().solid())
            if area > 0:
                layer_area_sum[signed_layer] += area
                layer_area_n[signed_layer] += 1
        except Exception:
            pass

    if n_unmatched > 0:
        print(f"         [endcap walk] skipped {n_unmatched} non-module DetElements")

    # convert defaultdict and stringify keys
    out_cells = {}
    out_size = {}
    out_spm = {}
    for sl, n in cells_map.items():
        key = f"layer{sl}"
        out_cells[key] = n
        if layer_area_n[sl] > 0:
            # average sensor area per module on this layer
            out_size[key] = layer_area_sum[sl] / layer_area_n[sl]
        else:
            out_size[key] = 0.0
        n_modules = len(layer_module_set[sl])
        out_spm[key] = int(n / n_modules) if n_modules > 0 else n

    return out_cells, out_size, out_spm


# ---------------------------------------------------------------------------
#  Barrel walk (works fine: layer DetElement -> ladder/module DetElement)
# ---------------------------------------------------------------------------

def _walk_barrel_subdet(sub_det):
    """
    Iterate layer DetElements, count sensitive leaves, sample sensor area.

    Works for ZPlanarTracker (VertexBarrel: layer -> ladder) and
    TrackerBarrel_o1_v06 (IT/OT Barrel: layer -> module).
    """
    cells_map = {}
    sensor_size_map = {}
    sensors_per_module_map = {}

    def _count_leaves(node, depth=0, max_depth=4):
        leaves = []
        kids = list(node.children())
        if len(kids) == 0 or depth >= max_depth:
            return [node]
        for _, k in kids:
            leaves.extend(_count_leaves(k, depth + 1, max_depth))
        return leaves

    for de_name, de in sub_det.children():
        layer_name = str(de_name)
        modules = sum(1 for _ in de.children())
        leaves = _count_leaves(de)
        sensors = len(leaves)

        area = 0.0
        if sensors > 0:
            area = _solid_area_mm2(leaves[0].volume().solid())

        cells_map[layer_name] = sensors
        sensor_size_map[layer_name] = area
        sensors_per_module_map[layer_name] = int(sensors / modules) if modules > 0 else sensors

    return cells_map, sensor_size_map, sensors_per_module_map


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

_SILICON_BARRELS = ("VertexBarrel", "InnerTrackerBarrel", "OuterTrackerBarrel")
_SILICON_ENDCAPS = ("VertexEndcap", "InnerTrackerEndcap", "OuterTrackerEndcap")


def get_cells_map(detector, sub_det, name,
                  skip_pattern=r"(supportTube)|(cryo)|(Support)|(Cable)|(Interlink)|(BeampipeShell)|(VerticalCable)"):
    """
    Per-layer cell maps for a CLD sub-detector.

    Returns:
        (cells_map, sensor_size_map, sensors_per_module_map)
    where cells_map is the SENSOR COUNT per layer (i.e. number of MAPS wafers
    / module placements). For pixel-level cell counts, multiply by:
        pixels_per_sensor[layer] = sensor_size_map[layer] / pixel_area
    using the per-sub-detector pixel area (in mm^2):
        VertexBarrel/Endcap, InnerTrackerEndcap layer +/-1: 0.020 * 0.020
        InnerTrackerBarrel, IT/OT Endcap layers >=2, OT*  : 0.050 * 0.300
    See the assumptions JSON for these numbers.
    """
    re_skip = re.compile(skip_pattern)

    if name in _SILICON_BARRELS:
        cells_map, sensor_size_map, sensors_per_module_map = _walk_barrel_subdet(sub_det)
        return cells_map, sensor_size_map, sensors_per_module_map

    if name in _SILICON_ENDCAPS:
        cells_map, sensor_size_map, sensors_per_module_map = _walk_endcap_subdet(sub_det)
        return cells_map, sensor_size_map, sensors_per_module_map

    # default: calorimeters / yoke / etc.
    cells_map = defaultdict(int)
    sensor_size_map = defaultdict(float)
    sensors_per_module_map = defaultdict(int)
    for de_name, de in sub_det.children():
        if re_skip.match(str(de_name)):
            print("         Skipping sub-detector:", de_name)
            continue
        cells_map[str(de_name)] = get_cells(de)

    return dict(cells_map), dict(sensor_size_map), dict(sensors_per_module_map)


# ---------------------------------------------------------------------------
#  Layer / module / sensor decoding
# ---------------------------------------------------------------------------

def get_layer(cell_id, decoder, detector, dtype):
    """
    Decode a cell ID into a signed layer index, matching the convention used
    by get_cells_map / _walk_endcap_subdet.
    """
    match detector:

        case "VertexEndcap" | "InnerTrackerEndcap" | "OuterTrackerEndcap":
            layer = decoder.get(cell_id, "layer") + 1
            side = decoder.get(cell_id, "side")
            return layer * side

        case _:
            layer = decoder.get(cell_id, "layer")
            side = 0
            if is_endcap(dtype):
                try:
                    side = decoder.get(cell_id, "side")
                except Exception:
                    side = 0
            if side != 0:
                layer *= side
            return layer


def get_module(cell_id, decoder, detector, dtype):
    """
    For endcaps, the cellID 'module' field is the RING index (0..nrings-1),
    and 'sensor' is the position within the ring. See the encoder block in
    TrackerEndcap_o2_v06_geo.cpp lines 200-220. Together (module, sensor)
    uniquely identifies a wafer within a layer.
    """
    try:
        return decoder.get(cell_id, "module")
    except Exception:
        return 0


def get_sensor(cell_id, decoder, detector, dtype):
    try:
        return decoder.get(cell_id, "sensor")
    except Exception:
        return 0