"""Planar-graph topology for snapped ceiling plans.

A snapped plan is a planar partition of the room into faces (one per ceiling
region + the main ceiling). Each pair of adjacent faces shares a single
boundary edge, so editing that edge moves both faces together — the way an
architectural drawing wants it.

The graph is built from the per-pixel ``assignment_room`` image that
``api_snap`` already produces. Edges are traced at corner resolution (so
the two faces that share a boundary really do see the *same* polyline,
not two parallel near-duplicates), then RDP-simplified once per edge.

JSON shape (stored at ``plan["topology"]``):

    {
      "vertices": [[x_world, z_world], ...],
      "edges": [
        {
          "id": int,
          "vertices": [vid, vid, ...],   # polyline of vertex indices
          "faces": [fid_left, fid_right]  # right is null on the room boundary
        }, ...
      ],
      "faces": [
        {
          "id": int,                      # 0 = main, 1..N = regions (matches the assignment label)
          "kind": "main" | "region",
          "ring": [{"edge": int, "rev": bool}, ...],   # CCW in world coords
          "polygon": [[x, z], ...]        # derived: ring resolved to a vertex loop
          # plus: label, notes, tint, stats, heatmap_*, relative_y — populated
          # by api_snap, not by this module
        }, ...
      ]
    }

The "outside" face is implicit; edges with ``faces[1] = null`` lie on the
room boundary. Faces are walked with the face on the LEFT, so the resulting
``ring`` is counter-clockwise in world coordinates (positive area).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .planes import PlanGrid


OUTSIDE = -1
SIMPLIFY_M = 0.05  # 5 cm — same tolerance as the previous per-polygon mask trace


# ─── Public API ───────────────────────────────────────────────────────────────


def build_from_assignment(
    assignment: np.ndarray,
    grid: PlanGrid,
    *,
    simplify_m: float = SIMPLIFY_M,
) -> dict[str, Any]:
    """Build the topology JSON from a per-pixel face assignment image.

    ``assignment`` is ``H × W`` int with -1 outside the room and 0..N-1 for
    each face (0 = main, 1..N-1 = regions in the order ``api_snap`` packs
    them).

    The result is the dict described at the top of this module — without
    label/notes/tint/stats/heatmap fields on faces; ``api_snap`` decorates
    the dict with those after running per-face analysis.
    """
    pixel_edges = _collect_pixel_edges(assignment)
    if not pixel_edges:
        return {"vertices": [], "edges": [], "faces": []}

    polylines = _trace_polylines(pixel_edges)
    out_vertices, out_edges = _simplify_and_pool(polylines, grid, simplify_m)
    out_faces = _build_face_rings(out_edges, out_vertices, assignment)

    # Resolve each face's ring back into an [[x, z], ...] polygon so the
    # frontend can render without re-implementing the edge-walk.
    for f in out_faces:
        f["polygon"] = _ring_to_polygon(f["ring"], out_edges, out_vertices)

    return {
        "vertices": [list(v) for v in out_vertices],
        "edges": [
            {
                "id": e["id"],
                "vertices": list(e["vertices"]),
                "faces": [
                    e["face_left"] if e["face_left"] != OUTSIDE else None,
                    e["face_right"] if e["face_right"] != OUTSIDE else None,
                ],
            }
            for e in out_edges
        ],
        "faces": out_faces,
    }


def face_polygon(topology: dict[str, Any], face_id: int) -> list[list[float]]:
    """Resolve one face's edge-ring into a closed [[x, z], ...] polygon."""
    edges_by_id = {e["id"]: e for e in topology["edges"]}
    vertices = topology["vertices"]
    face = next((f for f in topology["faces"] if f["id"] == face_id), None)
    if face is None:
        return []
    return _ring_to_polygon(face["ring"], list(edges_by_id.values()), vertices)


# ─── Edits ───────────────────────────────────────────────────────────────────
# These mutate the topology dict in place. Callers (the server endpoints)
# re-derive each face's polygon + re-run analysis afterwards. They do NOT
# update polygons here — the topology dict is the structural truth, the
# polygons are a derived view.


def insert_vertex_on_edge(
    topology: dict[str, Any], edge_id: int, position: tuple[float, float],
) -> int:
    """Split ``edge_id`` at ``position`` (which should lie on or near the
    edge's polyline). Returns the new vertex's id.

    The original edge is replaced by two new edges with fresh ids; both
    incident faces' rings are updated to reference the pair instead of
    the original.
    """
    edges = topology["edges"]
    vertices = topology["vertices"]
    faces = topology["faces"]

    edge_idx = next((i for i, e in enumerate(edges) if e["id"] == edge_id), -1)
    if edge_idx < 0:
        raise ValueError(f"unknown edge {edge_id}")
    edge = edges[edge_idx]
    poly_vids = edge["vertices"]
    if len(poly_vids) < 2:
        raise ValueError(f"edge {edge_id} is degenerate")

    # Find the segment closest to ``position`` and the projected point on it.
    px, pz = float(position[0]), float(position[1])
    best_k = 0
    best_d2 = float("inf")
    best_proj = (0.0, 0.0)
    for k in range(len(poly_vids) - 1):
        ax, az = vertices[poly_vids[k]]
        bx, bz = vertices[poly_vids[k + 1]]
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz
        if L2 < 1e-12:
            continue
        t = ((px - ax) * dx + (pz - az) * dz) / L2
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_z = az + t * dz
        d2 = (proj_x - px) ** 2 + (proj_z - pz) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_k = k
            best_proj = (proj_x, proj_z)

    # Add the new vertex at the projected point.
    new_vid = len(vertices)
    vertices.append([best_proj[0], best_proj[1]])

    # Build two replacement edges. New ids are max+1, max+2 so they never
    # collide with anything currently in the topology.
    next_id = max(e["id"] for e in edges) + 1
    edge_a = {
        "id": next_id,
        "vertices": list(poly_vids[: best_k + 1]) + [new_vid],
        "faces": list(edge["faces"]),
    }
    edge_b = {
        "id": next_id + 1,
        "vertices": [new_vid] + list(poly_vids[best_k + 1:]),
        "faces": list(edge["faces"]),
    }

    # Splice into the edges list (keep ordering stable for diffs).
    edges[edge_idx:edge_idx + 1] = [edge_a, edge_b]

    # Patch every face ring that referenced the old edge. Forward direction
    # → [a, b]; reversed direction → [b_rev, a_rev] (the walk now traverses
    # b backwards, then a backwards).
    for face in faces:
        new_ring = []
        for h in face["ring"]:
            if h["edge"] != edge_id:
                new_ring.append(h)
                continue
            if h["rev"]:
                new_ring.append({"edge": edge_b["id"], "rev": True})
                new_ring.append({"edge": edge_a["id"], "rev": True})
            else:
                new_ring.append({"edge": edge_a["id"], "rev": False})
                new_ring.append({"edge": edge_b["id"], "rev": False})
        face["ring"] = new_ring

    return new_vid


def delete_vertex(topology: dict[str, Any], vertex_id: int) -> None:
    """Remove a vertex from the topology.

    Two cases are supported:

    - **Interior vertex** (appears as a non-endpoint of exactly one edge's
      polyline) — simply spliced out of that edge.
    - **Degree-2 endpoint** (endpoint of exactly two edges that share the
      same face pair) — the two edges are merged into one.

    Junction vertices (degree ≥ 3) and endpoints whose two edges have
    different face pairs are rejected — deleting them would change the
    planar partition's combinatorics, which is a different operation
    (delete-face) handled separately.

    The vertex pool keeps the entry at ``vertex_id`` even after deletion
    (orphaned but cheap) so that every other edge's vertex indices stay
    valid. Callers can compact the pool later if desired.
    """
    edges = topology["edges"]
    faces = topology["faces"]

    # Classify: which edges have vertex_id in the interior, and which as an
    # endpoint?
    interior_in: list[int] = []
    endpoint_in: list[int] = []
    for i, e in enumerate(edges):
        vids = e["vertices"]
        if vertex_id in vids:
            if vids[0] == vertex_id or vids[-1] == vertex_id:
                endpoint_in.append(i)
            # A vertex can be both an endpoint AND an interior of the same
            # edge in a closed-loop edge (start == end). For the interior
            # check, only count strict-interior occurrences.
            if any(vids[k] == vertex_id for k in range(1, len(vids) - 1)):
                interior_in.append(i)

    # Interior case: vertex is purely an RDP corner inside one edge. Remove
    # it from that edge's polyline.
    if not endpoint_in and len(interior_in) == 1:
        e = edges[interior_in[0]]
        e["vertices"] = [v for v in e["vertices"] if v != vertex_id]
        if len(e["vertices"]) < 2:
            raise ValueError(
                f"removing vertex {vertex_id} would collapse edge {e['id']}"
            )
        return

    # Endpoint, degree-2: merge the two incident edges if and only if they
    # share the same face pair (otherwise this vertex is a junction in the
    # planar-graph sense, not just a polyline corner).
    if not interior_in and len(endpoint_in) == 2:
        e1 = edges[endpoint_in[0]]
        e2 = edges[endpoint_in[1]]
        # Compare face pairs irrespective of orientation: {fa, fb} sets.
        if set(_face_pair(e1)) != set(_face_pair(e2)):
            raise ValueError(
                f"vertex {vertex_id} sits at a junction between faces "
                f"{e1['faces']} and {e2['faces']} — delete the face instead"
            )
        # Orient both edges so vertex_id is the join point.
        v1 = list(e1["vertices"])
        v2 = list(e2["vertices"])
        if v1[-1] != vertex_id:
            v1.reverse()
            e1_reversed_for_merge = True
        else:
            e1_reversed_for_merge = False
        if v2[0] != vertex_id:
            v2.reverse()
            e2_reversed_for_merge = True
        else:
            e2_reversed_for_merge = False
        # The merged edge follows e1 forward then e2 forward (both as
        # oriented above), with vertex_id appearing once.
        merged_vids = v1 + v2[1:]
        # Decide the face_left / face_right of the merged edge. We picked
        # e1's orientation as the "forward" reference; if we reversed e1
        # for the merge, swap its faces accordingly.
        if e1_reversed_for_merge:
            merged_faces = [e1["faces"][1], e1["faces"][0]]
        else:
            merged_faces = list(e1["faces"])
        # Sanity-check: if e2 was also reversed, its forward face pair (as
        # we are walking it) should match merged_faces.
        e2_walk_faces = (
            [e2["faces"][1], e2["faces"][0]] if e2_reversed_for_merge
            else list(e2["faces"])
        )
        if e2_walk_faces != merged_faces:
            raise ValueError(
                f"vertex {vertex_id}: edges {e1['id']} and {e2['id']} have "
                f"inconsistent face orientation; cannot merge"
            )

        next_id = max(e["id"] for e in edges) + 1
        merged = {"id": next_id, "vertices": merged_vids, "faces": merged_faces}

        # Replace e1 and e2 with the merged edge. Drop e2 first so e1's
        # index stays valid for the substitution.
        for idx in sorted(endpoint_in, reverse=True):
            edges.pop(idx)
        edges.append(merged)

        # Patch face rings: any reference to e1 or e2 becomes a single
        # reference to the merged edge, with rev flag respecting the
        # orientations chosen above.
        for face in faces:
            new_ring = []
            i = 0
            while i < len(face["ring"]):
                h = face["ring"][i]
                if h["edge"] not in (e1["id"], e2["id"]):
                    new_ring.append(h)
                    i += 1
                    continue
                # Found one of the two — by construction the next entry in
                # the ring is the OTHER one (rings traverse contiguous
                # edges around the face). Replace the pair with the merged
                # edge. The rev flag is whichever orientation walks face
                # with face on the left — derive from the merged edge's
                # face pair vs. this face's id.
                merged_rev = (face["id"] == merged["faces"][1])
                new_ring.append({"edge": merged["id"], "rev": merged_rev})
                # Skip the partner — it must be the next ring entry.
                if i + 1 < len(face["ring"]) and face["ring"][i + 1]["edge"] in (e1["id"], e2["id"]):
                    i += 2
                else:
                    # Defensive: ring is malformed; leave as-is, log via
                    # exception so the caller can roll back.
                    raise ValueError(
                        f"face {face['id']}: ring references e1/e2 "
                        f"non-contiguously; cannot merge"
                    )
            face["ring"] = new_ring
        return

    raise ValueError(
        f"vertex {vertex_id} has interior_in={interior_in} endpoint_in={endpoint_in} "
        f"— not an interior point or a degree-2 endpoint, cannot delete"
    )


def _face_pair(edge: dict[str, Any]) -> tuple[Any, Any]:
    """Edge's face pair, normalised (None survives as None)."""
    return (edge["faces"][0], edge["faces"][1])


# ─── Step 1: Collect pixel-edges ──────────────────────────────────────────────


def _collect_pixel_edges(assignment: np.ndarray):
    """Return a dict ``{(c1, c2): (face_left, face_right)}`` for every unit
    boundary segment between two adjacent pixels of different labels.

    Corners are addressed in *padded* coordinates: the assignment is wrapped
    in a 1-pixel border of OUTSIDE so the room boundary appears as edges
    against the outside face.
    """
    H, W = assignment.shape
    P = np.full((H + 2, W + 2), OUTSIDE, dtype=np.int32)
    P[1:-1, 1:-1] = assignment

    edges: dict[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]] = {}

    # Vertical pixel-edges: separate columns j and j+1.
    # Walking direction (top → bottom): from corner (i, j+1) to (i+1, j+1).
    # In image coords (y-down), the LEFT side of that walk is +col (east) =
    # P[i, j+1]; the RIGHT side is -col (west) = P[i, j].
    diff_v = P[:, :-1] != P[:, 1:]
    rows, cols = np.where(diff_v)
    for i, j in zip(rows.tolist(), cols.tolist()):
        c1 = (i, j + 1)
        c2 = (i + 1, j + 1)
        edges[(c1, c2)] = (int(P[i, j + 1]), int(P[i, j]))

    # Horizontal pixel-edges: separate rows i and i+1.
    # Walking direction (left → right): from corner (i+1, j) to (i+1, j+1).
    # Left side = -row (north) = P[i, j]; right side = +row = P[i+1, j].
    diff_h = P[:-1, :] != P[1:, :]
    rows, cols = np.where(diff_h)
    for i, j in zip(rows.tolist(), cols.tolist()):
        c1 = (i + 1, j)
        c2 = (i + 1, j + 1)
        edges[(c1, c2)] = (int(P[i, j]), int(P[i + 1, j]))

    return edges


# ─── Step 2: Trace polylines ──────────────────────────────────────────────────


def _trace_polylines(pixel_edges):
    """Trace runs of pixel-edges between junction corners.

    Each polyline is a sequence of corners ``[c0, c1, ..., cN]`` with
    constant ``(face_left, face_right)`` along its length. Endpoints are
    junctions (corners with ≥ 3 incident pixel-edges) or, for closed loops
    with no junction, an arbitrary corner repeated at the start and end.
    """
    # Build corner adjacency: for each corner, list of (neighbour, fl, fr)
    # where (fl, fr) is the face pair when walking *toward* the neighbour.
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int, int]]] = {}
    for (c1, c2), (fl, fr) in pixel_edges.items():
        adj.setdefault(c1, []).append((c2, fl, fr))
        # Reversed direction: left/right swap.
        adj.setdefault(c2, []).append((c1, fr, fl))

    junctions = {c for c, nbs in adj.items() if len(nbs) >= 3}

    walked: set[frozenset[tuple[int, int]]] = set()

    def edge_key(a, b):
        return frozenset((a, b))

    polylines: list[dict] = []

    def walk(start, first_nb, fl, fr):
        path = [start, first_nb]
        walked.add(edge_key(start, first_nb))
        prev, cur = start, first_nb
        while cur not in junctions:
            nbs = adj[cur]
            if len(nbs) != 2:
                break  # defensive; non-junction corners always have 2
            # Take the neighbour that isn't where we came from.
            nxt = nbs[0][0] if nbs[0][0] != prev else nbs[1][0]
            walked.add(edge_key(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
        return {"corners": path, "face_left": fl, "face_right": fr}

    # Walks anchored on junctions.
    for j in junctions:
        for (nb, fl, fr) in adj[j]:
            if edge_key(j, nb) in walked:
                continue
            polylines.append(walk(j, nb, fl, fr))

    # Closed loops with no junctions (e.g. an island region inside main).
    for (c1, c2), (fl, fr) in pixel_edges.items():
        if edge_key(c1, c2) in walked:
            continue
        path = [c1, c2]
        walked.add(edge_key(c1, c2))
        prev, cur = c1, c2
        while cur != c1:
            nbs = adj[cur]
            if len(nbs) != 2:
                break
            nxt = nbs[0][0] if nbs[0][0] != prev else nbs[1][0]
            walked.add(edge_key(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
        polylines.append({"corners": path, "face_left": fl, "face_right": fr})

    return polylines


# ─── Step 3: Simplify and pool vertices ───────────────────────────────────────


def _corner_to_world(c, grid: PlanGrid):
    # Corner (i, j) in *padded* coordinates → world (x, z).
    # Padding offset: the original assignment corner is (i-1, j-1). PlanGrid's
    # world_to_px does (max - coord) * ppm; invert that.
    i, j = c
    row = i - 1
    col = j - 1
    x = grid.max_x - col / grid.pixels_per_metre
    z = grid.max_z - row / grid.pixels_per_metre
    return (float(x), float(z))


def _rdp_simplify(points, tolerance_m):
    """Ramer–Douglas–Peucker. Operates on a list of (x, z) world points."""
    if len(points) <= 2:
        return list(points)

    def perp_dist(p, a, b):
        ax, az = a
        bx, bz = b
        dx, dz = bx - ax, bz - az
        L = math.hypot(dx, dz)
        if L < 1e-9:
            return math.hypot(p[0] - ax, p[1] - az)
        # Distance from p to the line through a, b.
        return abs(dz * (p[0] - ax) - dx * (p[1] - az)) / L

    a, b = points[0], points[-1]
    max_d = 0.0
    max_i = 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], a, b)
        if d > max_d:
            max_d, max_i = d, i

    if max_d > tolerance_m:
        left = _rdp_simplify(points[: max_i + 1], tolerance_m)
        right = _rdp_simplify(points[max_i:], tolerance_m)
        return left[:-1] + right
    else:
        return [a, b]


def _simplify_and_pool(polylines, grid: PlanGrid, simplify_m: float):
    """Convert each polyline to world coords, simplify, and pool vertices.

    Returns ``(vertex_pool, edges)`` where each edge is a dict
    ``{id, vertices: [vid, ...], face_left, face_right}``.
    """
    vertex_pool: list[tuple[float, float]] = []
    vertex_index: dict[tuple[int, int], int] = {}

    def add_vertex(pt):
        # Quantise to 0.1 mm so junctions on different polylines hash to the
        # same pool entry.
        key = (int(round(pt[0] * 10000)), int(round(pt[1] * 10000)))
        if key in vertex_index:
            return vertex_index[key]
        idx = len(vertex_pool)
        vertex_pool.append(pt)
        vertex_index[key] = idx
        return idx

    out_edges: list[dict] = []
    for poly in polylines:
        wpts = [_corner_to_world(c, grid) for c in poly["corners"]]
        simp = _rdp_simplify(wpts, simplify_m)
        if len(simp) < 2:
            continue
        vids = [add_vertex(p) for p in simp]
        # Strip immediate duplicates that survive quantisation.
        deduped = [vids[0]]
        for vid in vids[1:]:
            if vid != deduped[-1]:
                deduped.append(vid)
        if len(deduped) < 2:
            continue
        out_edges.append({
            "id": len(out_edges),
            "vertices": deduped,
            "face_left": poly["face_left"],
            "face_right": poly["face_right"],
        })

    return vertex_pool, out_edges


# ─── Step 4: Build face rings ─────────────────────────────────────────────────


def _build_face_rings(edges, vertex_pool, assignment):
    """For every face id present in ``edges``, build a CCW ring of half-edges.

    A "half-edge" is ``(edge_id, reverse: bool)``. Walking with
    ``reverse=False`` keeps ``edge.face_left`` on the left; with
    ``reverse=True``, ``edge.face_right`` is on the left. So the boundary of
    face ``f`` is the chain of half-edges that put ``f`` on the left.

    At junctions where multiple half-edges of ``f`` meet, the next half-edge
    is the one immediately CW from the reverse-incoming direction (standard
    DCEL face traversal).
    """
    face_ids = sorted({
        f for e in edges
        for f in (e["face_left"], e["face_right"])
        if f != OUTSIDE
    })

    # Pre-compute outgoing direction angle for every half-edge (atan2 over
    # the first segment of the polyline in walking direction).
    def outgoing_angle(eid_rev):
        e = edges[eid_rev[0]]
        rev = eid_rev[1]
        verts = e["vertices"]
        if rev:
            v0 = vertex_pool[verts[-1]]
            v1 = vertex_pool[verts[-2]]
        else:
            v0 = vertex_pool[verts[0]]
            v1 = vertex_pool[verts[1]]
        return math.atan2(v1[1] - v0[1], v1[0] - v0[0])

    # Ending vertex (in walking direction) of a half-edge.
    def end_vertex(eid_rev):
        e = edges[eid_rev[0]]
        verts = e["vertices"]
        return verts[0] if eid_rev[1] else verts[-1]

    # Tangent at the end of a half-edge, as seen from end_vertex looking
    # back along the polyline (used as the "incoming reverse direction").
    def reverse_incoming_angle(eid_rev):
        e = edges[eid_rev[0]]
        verts = e["vertices"]
        if eid_rev[1]:
            v_prev = vertex_pool[verts[1]]
            v_cur = vertex_pool[verts[0]]
        else:
            v_prev = vertex_pool[verts[-2]]
            v_cur = vertex_pool[verts[-1]]
        return math.atan2(v_prev[1] - v_cur[1], v_prev[0] - v_cur[0])

    out_faces: list[dict] = []

    for fid in face_ids:
        # Half-edges that have face `fid` on the left.
        halves = []
        for e in edges:
            if e["face_left"] == fid:
                halves.append((e["id"], False))
            if e["face_right"] == fid:
                halves.append((e["id"], True))
        if not halves:
            continue

        # Group by start vertex for fast next-edge lookup.
        starting_at: dict[int, list[tuple[int, bool]]] = {}
        for h in halves:
            e = edges[h[0]]
            v_start = e["vertices"][-1] if h[1] else e["vertices"][0]
            starting_at.setdefault(v_start, []).append(h)

        used = set()
        # A face may have multiple rings (outer + holes). Walk all of them
        # so half-edges are consumed exactly once.
        rings = []
        for start in halves:
            if start in used:
                continue
            ring = [start]
            used.add(start)
            cur = start
            while True:
                v_end = end_vertex(cur)
                candidates = [h for h in starting_at.get(v_end, []) if h not in used]
                if not candidates:
                    # Only valid termination: the next edge would close the loop.
                    candidates_all = starting_at.get(v_end, [])
                    closes = [h for h in candidates_all if h == start]
                    if closes:
                        break
                    # Otherwise the ring is broken — bail out (defensive).
                    break

                if len(candidates) == 1:
                    nxt = candidates[0]
                else:
                    # Rotational sort: pick the half-edge immediately CW
                    # from the reverse-incoming direction. CW rotation from
                    # angle `a` to `b` is (a - b) mod 2π.
                    theta_in_rev = reverse_incoming_angle(cur)
                    nxt = min(
                        candidates,
                        key=lambda h: (theta_in_rev - outgoing_angle(h)) % (2 * math.pi),
                    )

                if nxt == start:
                    break
                ring.append(nxt)
                used.add(nxt)
                cur = nxt
            rings.append(ring)

        # Sort rings by signed area (in world coords). The outer ring of a
        # face is the largest CCW one; holes are smaller and CW. For the
        # MVP we only emit the outer ring (sorted by absolute area, take
        # the largest); holes get logged as a warning by the caller.
        rings_with_area = [(_signed_ring_area(r, edges, vertex_pool), r) for r in rings]
        rings_with_area.sort(key=lambda a_r: abs(a_r[0]), reverse=True)
        outer_ring = rings_with_area[0][1] if rings_with_area else []

        kind = "main" if fid == 0 else "region"
        out_faces.append({
            "id": int(fid),
            "kind": kind,
            "ring": [{"edge": h[0], "rev": h[1]} for h in outer_ring],
        })

    return out_faces


def _signed_ring_area(ring, edges, vertex_pool):
    """Shoelace area of the polygon obtained by walking a ring of half-edges."""
    pts = []
    for h in ring:
        e = edges[h["edge"] if isinstance(h, dict) else h[0]]
        rev = h["rev"] if isinstance(h, dict) else h[1]
        verts = list(reversed(e["vertices"])) if rev else e["vertices"]
        # Append every vertex except the last (it's the next edge's first).
        pts.extend(verts[:-1])
    if len(pts) < 3:
        return 0.0
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, z0 = vertex_pool[pts[i]]
        x1, z1 = vertex_pool[pts[(i + 1) % n]]
        a += x0 * z1 - x1 * z0
    return 0.5 * a


def _ring_to_polygon(ring, edges, vertex_pool):
    """Resolve a face ring into a closed [[x, z], ...] polygon."""
    edges_by_id = {e["id"]: e for e in edges}
    pts: list[list[float]] = []
    for h in ring:
        eid = h["edge"]
        rev = h["rev"]
        e = edges_by_id[eid]
        verts = list(reversed(e["vertices"])) if rev else e["vertices"]
        for vid in verts[:-1]:
            x, z = vertex_pool[vid]
            pts.append([float(x), float(z)])
    return pts
