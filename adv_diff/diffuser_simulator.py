import numpy as np
import festim as F
from dolfinx.log import set_log_level, LogLevel
from autoemulate.simulations.base import Simulator
import torch
import dolfinx


class Diffuser(Simulator):
    def _forward(self, x):
        c_inlet = x[:, 0]
        velocity_magnitude = x[:, 1]

        # convert to float
        c_inlet = c_inlet.item() if isinstance(c_inlet, torch.Tensor) else c_inlet
        velocity_magnitude = (
            velocity_magnitude.item()
            if isinstance(velocity_magnitude, torch.Tensor)
            else velocity_magnitude
        )

        assert isinstance(c_inlet, float), (
            f"Expected c_inlet to be a float, got {type(c_inlet)}"
        )
        assert isinstance(velocity_magnitude, float), (
            f"Expected velocity_magnitude to be a float, got {type(velocity_magnitude)}"
        )
        result = make_festim_model(c_inlet, velocity_magnitude)
        return torch.tensor(result, dtype=torch.float32).unsqueeze(0)


def make_festim_model(c_inlet: float, velocity_magnitude: float) -> float:
    # print(
    #     f"Running simulation with c_inlet={c_inlet} and velocity_magnitude={velocity_magnitude}"
    # )
    # set_log_level(LogLevel.INFO)

    my_model = F.HydrogenTransportProblemDiscontinuous()

    mesh_fenics = make_mesh()
    my_model.mesh = F.Mesh(mesh_fenics)

    top_boundary = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.1))
    left_boundary = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0.0))
    right_boundary = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 1.0))
    eps = 0.001
    top_volume = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0),
        locator=lambda x: x[1] >= 1.0 - eps,
    )
    bottom_volume = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0),
        locator=lambda x: x[1] <= 1.0 + eps,
    )

    my_model.subdomains = [
        top_boundary,
        left_boundary,
        right_boundary,
        top_volume,
        bottom_volume,
    ]

    my_model.temperature = 500

    H = F.Species("H", subdomains=[top_volume, bottom_volume])
    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(species=H, subdomain=top_boundary, value=0),
        F.FixedConcentrationBC(species=H, subdomain=left_boundary, value=c_inlet),
    ]

    my_model.interfaces = [
        F.Interface(id=4, subdomains=[top_volume, bottom_volume], penalty_term=1000)
    ]

    my_model.surface_to_volume = {
        top_boundary: top_volume,
        left_boundary: bottom_volume,
        right_boundary: bottom_volume,
    }

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    from basix.ufl import element

    el = element(
        "Lagrange",
        mesh_fenics.topology.cell_name(),
        2,
        shape=(mesh_fenics.geometry.dim,),
    )

    V = dolfinx.fem.functionspace(mesh_fenics, el)

    velocity = dolfinx.fem.Function(V)

    velocity.interpolate(
        lambda x: (-velocity_magnitude * x[1] * (x[1] - 1), np.full_like(x[0], 0.0))
    )

    advection_term = F.AdvectionTerm(
        velocity=velocity,
        subdomain=bottom_volume,
        species=H,
    )

    my_model.advection_terms = [advection_term]

    top_flux = F.SurfaceFlux(field=H, surface=top_boundary)
    my_model.exports = [top_flux]

    my_model.initialise()

    my_model.run()
    # post_processing(my_model)
    return top_flux.data[-1]


def make_mesh():
    from dolfinx.mesh import create_rectangle
    from mpi4py import MPI

    # creating a mesh with FEniCS
    nx = 20
    ny = 110
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        points=[[0, 0], [1, 1.1]],
        n=[nx, ny],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    return mesh


def post_processing(model: F.HydrogenTransportProblemDiscontinuous):
    import pyvista
    from dolfinx import plot

    u_plotter = pyvista.Plotter()

    for subdomain in model.volume_subdomains:
        hydrogen_concentration = model.species[0].subdomain_to_post_processing_solution[
            subdomain
        ]

        topology, cell_types, geometry = plot.vtk_mesh(
            hydrogen_concentration.function_space
        )
        u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        u_grid.point_data["c"] = hydrogen_concentration.x.array.real
        u_grid.set_active_scalars("c")
        u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()

    if not pyvista.OFF_SCREEN:
        u_plotter.show()
    else:
        figure = u_plotter.screenshot("concentration_diff_only.png")
