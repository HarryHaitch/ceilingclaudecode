# Polycam coordinate system

Source: original derivation in the 13 Farrington render-compare work,
plus Polycam's own `polyform` reference.

## Three things that must be correct

### 1. Camera matrix convention

`corrected_cameras/*.json` stores a **cam-to-world POSE matrix**, not a
world-to-camera extrinsic. Fields `t_ij` mean row `i`, column `j`,
forming a 3×4 matrix:

```
[ t_00  t_01  t_02  t_03 ]
[ t_10  t_11  t_12  t_13 ]
[ t_20  t_21  t_22  t_23 ]
```

The 3×3 block is rotation **R** (columns are camera axes in world). The
last column `(t_03, t_13, t_23)` is the camera position in world.

To project a world point `P` into camera space:

```
P_cam = R^T @ (P - t)
```

### 2. Camera looks along -Z (ARKit)

Points in front have `P_cam[2] < 0`. Depth = `-P_cam[2]` (positive when
visible).

```
u =  fx * P_cam[0] / depth + cx     # X right, OpenCV-style
v = -fy * P_cam[1] / depth + cy     # Y flip: ARKit Y up, image v down
```

Y is up in Polycam/ARKit world space.

### 3. Mesh must be transformed before use

The OBJ lives in axis-aligned mesh space, **different** from camera-pose
space. They are related by `alignmentTransform` in `mesh_info.json`,
stored as 16 elements in **column-major** order.

```python
M_align     = np.array(info["alignmentTransform"]).reshape(4, 4, order="F")
M_align_inv = np.linalg.inv(M_align)
verts_h     = np.hstack([verts, np.ones((N, 1))])
verts_world = (M_align_inv @ verts_h.T).T[:, :3]
```

Apply the **inverse** to the mesh — never to the cameras.

## What does not work

- Using the camera matrix as a world-to-cam extrinsic (`P_cam = R @ P + t`)
  → camera ends up in an impossible pose.
- Keeping `P_cam[2] > 0` (OpenCV) → discards everything visible.
- Skipping `M_align_inv` on the mesh → cameras and mesh are in
  different frames; renders show the wrong area, off by ~43° about Y.
- Applying `M_align` to the cameras instead of `M_align_inv` to the mesh
  → equivalent in theory but mesh-side is what we use.

## References

- Polyform repo: https://github.com/PolyCam/polyform
- Polycam Help Centre: https://learn.poly.cam/hc/en-us/articles/38276871185044
