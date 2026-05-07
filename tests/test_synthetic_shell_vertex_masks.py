import numpy as np
import pytest

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    GeometryError,
    SyntheticShellVertexMasks,
    classify_synthetic_shell_vertices,
    create_subdivided_synthetic_bottle_shell,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    create_gaussian_dimple_pattern,
    compute_vertex_displacement_amounts,
)


_OUTER_RADIUS = 30.0
_WALL_THICKNESS = 1.0
_HEIGHT = 120.0
_SECTIONS = 64
_HEIGHT_SEGMENTS = 12


def _build_shell():
    return create_subdivided_synthetic_bottle_shell(
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        sections=_SECTIONS,
        height_segments=_HEIGHT_SEGMENTS,
    )


def _classify(mesh, **overrides) -> SyntheticShellVertexMasks:
    kwargs = dict(
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    kwargs.update(overrides)
    return classify_synthetic_shell_vertices(mesh, **kwargs)


def _build_uniform_risk_map(resolution: tuple[int, int]) -> RiskMap:
    nv, nu = int(resolution[0]), int(resolution[1])
    risk = np.ones((nv, nu), dtype=float)
    total = float(risk.sum())
    return RiskMap(
        risk_map=risk,
        probability_map=risk / total,
        active_mask=np.ones((nv, nu), dtype=bool),
        total_risk=total,
        active_count=nv * nu,
        epsilon=0.0,
        threshold=None,
    )


def test_returns_synthetic_shell_vertex_masks() -> None:
    masks = _classify(_build_shell())
    assert isinstance(masks, SyntheticShellVertexMasks)


def test_all_masks_have_vertex_count_shape() -> None:
    mesh = _build_shell()
    masks = _classify(mesh)
    n = int(len(mesh.vertices))
    assert masks.vertex_count == n
    for arr in (
        masks.outer_wall_mask,
        masks.inner_wall_mask,
        masks.z_boundary_mask,
        masks.outer_lateral_interior_mask,
        masks.inner_lateral_interior_mask,
    ):
        assert arr.shape == (n,)
        assert arr.dtype == bool


def test_outer_wall_mask_has_positive_count() -> None:
    masks = _classify(_build_shell())
    assert int(masks.outer_wall_mask.sum()) > 0


def test_inner_wall_mask_has_positive_count() -> None:
    masks = _classify(_build_shell())
    assert int(masks.inner_wall_mask.sum()) > 0


def test_outer_and_inner_wall_masks_are_disjoint() -> None:
    masks = _classify(_build_shell())
    overlap = masks.outer_wall_mask & masks.inner_wall_mask
    assert int(overlap.sum()) == 0


def test_outer_lateral_interior_mask_has_positive_count() -> None:
    masks = _classify(_build_shell(), boundary_epsilon=0.02)
    assert int(masks.outer_lateral_interior_mask.sum()) > 0


def test_inner_lateral_interior_mask_has_positive_count() -> None:
    masks = _classify(_build_shell(), boundary_epsilon=0.02)
    assert int(masks.inner_lateral_interior_mask.sum()) > 0


def test_z_boundary_mask_has_positive_count_when_epsilon_positive() -> None:
    masks = _classify(_build_shell(), boundary_epsilon=0.02)
    assert int(masks.z_boundary_mask.sum()) > 0


def test_z_boundary_mask_is_empty_when_epsilon_zero() -> None:
    masks = _classify(_build_shell(), boundary_epsilon=0.0)
    assert int(masks.z_boundary_mask.sum()) == 0


def test_outer_lateral_interior_excludes_z_boundary() -> None:
    masks = _classify(_build_shell(), boundary_epsilon=0.02)
    overlap = masks.outer_lateral_interior_mask & masks.z_boundary_mask
    assert int(overlap.sum()) == 0


def test_inner_lateral_interior_excludes_z_boundary() -> None:
    masks = _classify(_build_shell(), boundary_epsilon=0.02)
    overlap = masks.inner_lateral_interior_mask & masks.z_boundary_mask
    assert int(overlap.sum()) == 0


def test_outer_wall_selected_vertices_have_outer_radius() -> None:
    mesh = _build_shell()
    masks = _classify(mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    r = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
    selected = r[masks.outer_wall_mask]
    assert selected.size > 0
    assert np.allclose(selected, masks.outer_radius, atol=1e-6)


def test_inner_wall_selected_vertices_have_inner_radius() -> None:
    mesh = _build_shell()
    masks = _classify(mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    r = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
    selected = r[masks.inner_wall_mask]
    assert selected.size > 0
    assert np.allclose(selected, masks.inner_radius, atol=1e-6)


def test_invalid_outer_radius_raises() -> None:
    mesh = _build_shell()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="outer_radius"):
            classify_synthetic_shell_vertices(
                mesh,
                outer_radius=bad,
                wall_thickness=_WALL_THICKNESS,
                height=_HEIGHT,
            )


def test_invalid_wall_thickness_raises() -> None:
    mesh = _build_shell()
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="wall_thickness"):
            classify_synthetic_shell_vertices(
                mesh,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=bad,
                height=_HEIGHT,
            )
    with pytest.raises(GeometryError, match="wall_thickness"):
        classify_synthetic_shell_vertices(
            mesh,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_OUTER_RADIUS,
            height=_HEIGHT,
        )


def test_invalid_height_raises() -> None:
    mesh = _build_shell()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="height"):
            classify_synthetic_shell_vertices(
                mesh,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=_WALL_THICKNESS,
                height=bad,
            )


def test_invalid_radial_tolerance_raises() -> None:
    mesh = _build_shell()
    for bad in (0.0, -1e-6, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="radial_tolerance"):
            classify_synthetic_shell_vertices(
                mesh,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=_WALL_THICKNESS,
                height=_HEIGHT,
                radial_tolerance=bad,
            )


def test_invalid_boundary_epsilon_raises() -> None:
    mesh = _build_shell()
    for bad in (-0.1, 0.5, 0.7, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="boundary_epsilon"):
            classify_synthetic_shell_vertices(
                mesh,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=_WALL_THICKNESS,
                height=_HEIGHT,
                boundary_epsilon=bad,
            )


def test_invalid_mesh_type_raises() -> None:
    with pytest.raises(GeometryError, match="trimesh.Trimesh"):
        classify_synthetic_shell_vertices(
            object(),  # type: ignore[arg-type]
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_WALL_THICKNESS,
            height=_HEIGHT,
        )


def test_mesh_not_mutated() -> None:
    mesh = _build_shell()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    _classify(mesh, boundary_epsilon=0.02)
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_integration_with_compute_vertex_displacement_amounts() -> None:
    mesh = _build_shell()
    surface_map = create_vertex_surface_coordinates(mesh)
    masks = _classify(mesh, boundary_epsilon=0.02)
    risk_map = _build_uniform_risk_map((16, 32))
    pattern = create_gaussian_dimple_pattern(
        risk_map,
        count=10,
        amplitude=1.0,
        sigma_u=0.05,
        sigma_v=0.05,
        max_depth=0.25,
        seed=7,
    )
    displacement = compute_vertex_displacement_amounts(
        mesh, surface_map, pattern,
        active_threshold=0.01,
        exclude_v_boundary_epsilon=0.02,
        include_mask=masks.outer_lateral_interior_mask,
    )

    assert int(displacement.active_mask.sum()) > 0
    nonzero = displacement.normalized_depth > 0.0
    assert not bool((nonzero & ~masks.outer_lateral_interior_mask).any())
    assert (
        float(np.abs(displacement.normalized_depth[masks.inner_wall_mask]).max())
        == 0.0
    )
    assert (
        float(np.abs(displacement.physical_depth[masks.inner_wall_mask]).max())
        == 0.0
    )
