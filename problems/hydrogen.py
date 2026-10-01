import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit

from base import QuantumProblem


class HydrogenVQEProblem(QuantumProblem):
    def __init__(self, xyz_file: str = "h2.xyz", basis: str = "sto3g"):
        self.atoms, self.coords = self._parse_xyz(xyz_file)
        self.bond_distance = float(np.linalg.norm(
            np.array(self.coords[1]) - np.array(self.coords[0])
        ))
        self.basis = basis

        atom_str = "; ".join(
            f"{a} {c[0]} {c[1]} {c[2]}"
            for a, c in zip(self.atoms, self.coords)
        )
        driver = PySCFDriver(
            atom=atom_str,
            basis=basis,
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )

        es_problem = driver.run()
        second_q_op = es_problem.hamiltonian.second_q_op()
        self.nuclear_repulsion_energy = es_problem.hamiltonian.nuclear_repulsion_energy
        self.num_spatial_orbitals = es_problem.num_spatial_orbitals
        self.num_particles = es_problem.num_particles

        mapper = JordanWignerMapper()
        self.hamiltonian = mapper.map(second_q_op)

        hf_init = HartreeFock(
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_particles=self.num_particles,
            qubit_mapper=mapper,
        )
        self.ansatz = UCCSD(
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_particles=self.num_particles,
            qubit_mapper=mapper,
            initial_state=hf_init,
        )

        mat = self.hamiltonian.to_matrix()
        dense = mat.toarray() if hasattr(mat, "toarray") else mat
        electronic_energy = float(np.linalg.eigvalsh(dense)[0])
        self.exact_energy = electronic_energy + self.nuclear_repulsion_energy

    @staticmethod
    def _parse_xyz(path: str):
        with open(path, "r") as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        atoms, coords = [], []
        for line in lines[2 : 2 + n_atoms]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return atoms, coords

    def get_ansatz(self):
        return self.ansatz

    def get_hamiltonian(self):
        return self.hamiltonian

    def get_initial_point(self, initialization_type: str = "small"):
        n = self.ansatz.num_parameters
        if initialization_type == "random":
            return 0.01 * 2 * np.pi * np.random.random(n)
        elif initialization_type == "small":
            return 0.01 * np.ones(n)
        elif initialization_type == "zeros":
            return np.zeros(n)
        else:
            raise ValueError(
                f"Unknown initialization type: {initialization_type}. "
                f"Choose from: 'random', 'small', 'zeros'"
            )
