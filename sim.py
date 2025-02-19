"""
sim.py
==============
Simulation of basic DPD system using OpenMM. This code utilizes the MDCraft
package to simplify the process of setting up the simulation system and topology.
In addition, MDCraft contains the CustomNonbondedForce for the DPD conservative
force.

.. author:: Sam Varner
.. inspiration:: Dr. Benjamin B. Ye

Last updated: February 19, 2025
"""
import logging
import os
import platform
import sys
from sys import stdout
import numpy as np
import openmm
from openmm import app, unit

# TODO: Path to the MDHelper package (if not installed via conda or pip)
from mdcraft.openmm import pair, reporter, system as s, topology as t, unit as u
from mdcraft.openmm.unit import get_lj_scale_factors

ROOM_TEMP = 300.0 * unit.kelvin        # Room temperature
MW = 1.0 * unit.amu                     # Water molar mass
DIAMETER = 1.0 * unit.nanometer         # Water molecule size

def md(N: int, ts: int, *, 
        temp: unit.Quantity = ROOM_TEMP,
        size: unit.Quantity = DIAMETER,
        mass: unit.Quantity = MW,
        dt_md: float = 0.025, gam=4.5, 
        every: int = 10_000,
        path: str = None, verbose: bool = True,
        Aii: float = 25.0, L: float = 20.0,
        restart = False, restartFile = None,
        fname = None, device: int = 0) -> None:
    
    # Set up logger
    logging.basicConfig(format="{asctime} | {levelname:^8s} | {message}", 
                        style="{", 
                        level=logging.INFO if verbose else logging.WARNING)
    
    # Change to the data directory
    if path is None:
        path = ORIG_PATH
    if not os.path.isdir(path):
        os.makedirs(path)
    os.chdir(path)

    scales = get_lj_scale_factors({
        "energy": (unit.BOLTZMANN_CONSTANT_kB * temp).in_units_of(unit.kilojoule),
        "length": size,
        "mass": mass
    })
    if verbose:
        logging.info(f"Fundamental quantities:\n"
                f"\tMolar energy: {scales['molar_energy']}\n"
                f"\tLength: {scales['length']}\n"
                f"\tMass: {scales['mass']}\n"
                f"\tTime: {scales['time']}")
    Aii *= (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temp).in_units_of(unit.kilojoules_per_mole) / size
    A_ij = np.array((
        (Aii,),
    ))
    # Set up the DPD potential
    pair_dpd = pair.dpd(cutoff = scales["length"],
                        cutoff_dpd = scales["length"],
                        mix="A12=A(type1,type2);",
                        per_params=("type",),
                        tab_funcs = {"A": A_ij})

    # Determine the system dimensions
    dims = np.array((L,L,L)) * size

    # Initialize simulation system and topology
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        (L * size, 0, 0),
        (0, L * size, 0),
        (0, 0, L * size)
    )
    topology = app.Topology()
    topology.setUnitCellDimensions(dims)
    if verbose:
        logging.info("Created simulation system and topology with "
                f"dimensions {dims[0]} x {dims[1]} x {dims[2]}.")

    # Assign arbitrary particle identities
    element_A = app.Element.getBySymbol("N")

    for _ in range(N):
        chain = topology.addChain()
        s.register_particles(
            system, topology, 1, 1.0 * unit.amu, chain=chain,
            element=element_A, name="A", resname="A",
            cnbforces={pair_dpd: (0,)}
        )
    if verbose:
        logging.info(f"Registered {N:,} particles to the force field.")

        
    # Register force field to simulation
    # system.addForce(pair_dpd)
    if verbose:
        logging.info("Registered the force field to the simulation.")

    # Determine the filename prefix
    if fname is None:
        fname = "sim"

    # Create OpenMM CUDA Platform
    plat = openmm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed", "DeviceIndex": f"{device}",
                  "UseBlockingSync": "false"}
    
    dt = dt_md * scales["time"]
    fric = gam / scales["time"]
    if verbose:
        logging.info(f"OpenMM {plat.getOpenMMVersion()} is using the "
                f"{plat.getName()} Platform on {platform.node()}.")

    # Generate initial particle positions for polymers
    pos = t.create_atoms(dims, N, N_p=1, length=size, randomize=True)

    # integrator = openmm.LangevinMiddleIntegrator(temp, fric, dt)
    integrator = openmm.DPDIntegrator(temp, fric, size, dt)
    simulation = app.Simulation(topology, system, integrator, plat, properties)

    simulation.context.reinitialize(True)

    if not restart:
        if verbose:
            logging.info("Starting system relaxation...")

        # Perform NVT energy minimization
        if verbose:
            logging.info("Starting system relaxation...")
        simulation.context.setPositions(pos)
        simulation.minimizeEnergy()
        if verbose:
            logging.info("Local energy minimization completed.")
    else:
        if os.path.exists(f"{restartFile}"):
            simulation.loadState(f"{restartFile}")

    pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    # Write topology file
    with open(f"{fname}.cif", "w") as f:
        app.PDBxFile.writeFile(topology, pos, f, keepIds=True)

    with open(f"{fname}-initial.pdb", "w") as f:
        app.PDBFile.writeFile(topology, pos, f, keepIds=True)

    simulation.context.reinitialize(True)
    mdsteps = ts * every
    if verbose:
        logging.info("Starting NPT anisotropic simulation...")
    simulation.reporters.append(reporter.NetCDFReporter(f"{fname}.nc", every, subset=range(N)))
    for o in [sys.stdout, f"{fname}.log"]:
        simulation.reporters.append(
            app.StateDataReporter(
                o, reportInterval=every, step=True, temperature=True, volume=True,
                potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                remainingTime=True, speed=True, totalSteps=mdsteps 
            )
        )
    for i in range(ts):
        simulation.step(every)
        simulation.saveState(f"{fname}.xml")

        pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

        with open(f"{fname}-final.pdb", "w") as f:
            app.PDBFile.writeFile(topology, pos, f, keepIds=True)

    simulation.reporters = []
    pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    vel = simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
    dims = simulation.context.getState().getPeriodicBoxVectors()

    simulation.context.setPositions(pos)
    simulation.context.setVelocities(vel)
    simulation.context.setPeriodicBoxVectors(dims[0],dims[1],dims[2])
    simulation.saveState(f"{fname}.xml")
    
if __name__ == "__main__":

    path: str = os.getcwd()
    device: int = 0

    rho: float = 3.00
    L: float = 20.00
    N: int = int(rho * L * L * L)
    L = (N / rho) ** (1/3)
    Aii: float = 25.0
    ts = 100
    every = 1000

    md(N, ts,
        path=path,
        every=every,
        dt_md=0.01,
        gam=4.5,
        Aii=Aii,
        L=L,
        restart=False,
        restartFile=None,
        fname='sim',
        device=device)