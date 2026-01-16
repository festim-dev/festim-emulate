import festim as F
import numpy as np


def make_model(x):
    my_model = F.HydrogenTransportProblem()

    vertices = np.concatenate(
        [
            np.linspace(0, 30e-9, num=200),
            np.linspace(30e-9, 3e-6, num=300),
            np.linspace(3e-6, 20e-6, num=200),
        ]
    )

    my_model.mesh = F.Mesh1D(vertices)

    tungsten = F.Material(
        D_0=4.1e-07,  # m2/s
        E_D=0.39,  # eV
    )

    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 20e-6], material=tungsten)
    left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
    right_boundary = F.SurfaceSubdomain1D(id=2, x=20e-6)

    my_model.subdomains = [
        volume_subdomain,
        left_boundary,
        right_boundary,
    ]

    w_atom_density = 6.3e28  # atom/m3

    H = F.Species("H")
    trapped_H1 = F.Species("trapped_H1", mobile=False)
    trapped_H2 = F.Species("trapped_H2", mobile=False)
    empty_trap1 = F.ImplicitSpecies(n=x[0] * w_atom_density, others=[trapped_H1])
    empty_trap2 = F.ImplicitSpecies(n=x[2] * w_atom_density, others=[trapped_H2])
    my_model.species = [H, trapped_H1, trapped_H2]

    trapping_reaction_1 = F.Reaction(
        reactant=[H, empty_trap1],
        product=[trapped_H1],
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=x[1],
        volume=volume_subdomain,
    )
    trapping_reaction_2 = F.Reaction(
        reactant=[H, empty_trap2],
        product=[trapped_H2],
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=x[3],
        volume=volume_subdomain,
    )

    my_model.reactions = [
        trapping_reaction_1,
        trapping_reaction_2,
    ]

    import ufl

    implantation_time = 400  # s
    incident_flux = 2.5e19  # H/m2/s

    def ion_flux(t):
        return ufl.conditional(t <= implantation_time, incident_flux, 0)

    def gaussian_distribution(x, center, width):
        return (
            1
            / (width * (2 * ufl.pi) ** 0.5)
            * ufl.exp(-0.5 * ((x[0] - center) / width) ** 2)
        )

    source_term = F.ParticleSource(
        value=lambda x, t: ion_flux(t) * gaussian_distribution(x, 4.5e-9, 2.5e-9),
        volume=volume_subdomain,
        species=H,
    )

    my_model.sources = [source_term]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
        F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
    ]

    implantation_temp = 300  # K
    temperature_ramp = 8  # K/s

    start_tds = implantation_time + 50  # s

    def temp_fun(t):
        if t <= start_tds:
            return implantation_temp
        else:
            return implantation_temp + temperature_ramp * (t - start_tds)

    my_model.temperature = temp_fun

    my_model.settings = F.Settings(atol=1e11, rtol=1e-10, final_time=500)
    my_model.settings.stepsize = F.Stepsize(
        initial_value=0.5,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=4,
        max_stepsize=lambda t: 0.5 if t > start_tds else None,
        milestones=[implantation_time, start_tds, start_tds + 50],
    )

    left_flux = F.SurfaceFlux(surface=left_boundary, field=H)
    right_flux = F.SurfaceFlux(surface=right_boundary, field=H)
    total_mobile_H = F.TotalVolume(field=H, volume=volume_subdomain)
    total_trapped_H1 = F.TotalVolume(field=trapped_H1, volume=volume_subdomain)
    total_trapped_H2 = F.TotalVolume(field=trapped_H2, volume=volume_subdomain)

    my_model.exports = [
        total_mobile_H,
        total_trapped_H1,
        total_trapped_H2,
        left_flux,
        right_flux,
    ]

    return my_model


def simulate(x):
    my_model = make_model(x)
    my_model.initialise()
    my_model.run()

    desorption = np.array(my_model.exports[-1].data) + np.array(
        my_model.exports[-2].data
    )
    time = my_model.exports[-1].t

    return desorption, time
