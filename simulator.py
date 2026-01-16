import festim as F
import numpy as np

from dolfinx.mesh import create_unit_square
from mpi4py import MPI


def make_model(x):
    fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

    festim_mesh = F.Mesh(fenics_mesh)

    material_top = F.Material(D_0=1, E_D=0)
    material_bot = F.Material(D_0=2, E_D=0)

    top_volume = F.VolumeSubdomain(
        id=1, material=material_top, locator=lambda x: x[1] >= 0.5
    )
    bottom_volume = F.VolumeSubdomain(
        id=2, material=material_bot, locator=lambda x: x[1] <= 0.5
    )

    top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
    bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = festim_mesh
    my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = 400

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=top_surface, value=x[0], species=H),
        F.FixedConcentrationBC(subdomain=bottom_surface, value=x[1], species=H),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [
        F.TotalVolume(field=H, volume=top_volume),
        F.TotalVolume(field=H, volume=bottom_volume),
    ]

    return my_model


def simulate(x):
    my_model = make_model(x)
    my_model.initialise()
    my_model.run()

    total_H = np.sum(my_model.exports[0].data[-1] + my_model.exports[1].data[-1])

    return total_H
