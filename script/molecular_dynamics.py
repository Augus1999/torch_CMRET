# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
MD test
"""
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, Trajectory
from cmret import trained_model
from cmret.utils import CMRETCalculator


model = trained_model("ani1x")
cal = CMRETCalculator(model)
atoms = read("edta.sdf")  # use your own input file here!
# you can define the total spin and/or charge as following:
# atoms.info: Dict[str, float] = {"S": 0, "Q": 0}
atoms.calc = cal
atoms.get_potential_energy()  # call this method first

steps = 0


def printenergy(a=atoms):
    """
    Print the potential, kinetic and total energy.
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()
    global steps
    steps += 1
    with open("md.log", "a") as f:
        f.write(
            f"Steps={steps:12.3f} Etot={epot + ekin:12.3f} Epot={epot:12.3f} Ekin={ekin:12.3f} temperature={temp:8.2f}\n"
        )


MaxwellBoltzmannDistribution(atoms, temperature_K=350)
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=350, friction=0.1)
dyn.attach(printenergy, interval=1)

traj = Trajectory("md.traj", "w", atoms)
dyn.attach(traj.write, interval=10)
dyn.run(10000)
