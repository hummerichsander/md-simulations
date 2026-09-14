import os
from pathlib import Path

from openmm.unit import kilojoules_per_mole

from md_simulations.config import AwsemConfig
from md_simulations.engines.base import BuiltSystem, OpenMMEngine


class AwsemEngine(OpenMMEngine):
    """OpenAWSEM coarse-grained MD.

    The input PDB must be (or be prepared into) 3-bead CG format (Cα, Cβ, O
    per residue). AWSEM uses implicit solvent: water-mediated interactions are
    encoded analytically in the contact-term gamma matrices. Requires the
    ``awsem`` optional dependency (``openawsem``)."""

    config: AwsemConfig

    def _prepare_cg_pdb(self, input_pdb: str) -> str:
        """Convert an all-atom PDB to the 3-bead CG PDB required by AWSEM.

        :param input_pdb: Path to the all-atom input PDB.
        :return: Path to the 3-bead CG PDB."""
        from openawsem.openAWSEM import prepare_pdb

        cfg = self.config
        self.logger.info(f"Preparing all-atom PDB for AWSEM: {input_pdb}  (chains: {cfg.chains})")
        cg_pdb, cleaned_pdb = prepare_pdb(
            input_pdb,
            chains_to_simulate=list(cfg.chains),
            use_cis_proline=cfg.cis_proline,
            keepIds=True,
            removeHeterogens=True,
        )
        self.logger.info(f"Cleaned all-atom PDB written: {cleaned_pdb}")
        self.logger.info(f"3-bead CG PDB ready:          {cg_pdb}")
        return cg_pdb

    def _build_memories(self, nres: int) -> list[tuple]:
        """Assemble the memories list consumed by ``associative_memory_term``.

        Each entry is a tuple ``(pdb_file, chain, target_start, fragment_start,
        length, weight)``. Missing per-memory options fall back to defaults
        (chain A, weight 0.5, start 1, full length).

        :param nres: Number of residues in the simulated protein.
        :return: List of memory tuples."""
        cfg = self.config
        n = len(cfg.memory_pdbs)

        def _pad(lst: list, default) -> list:
            return (lst + [default] * n)[:n]

        chains = _pad(cfg.memory_chains, "A")
        weights = _pad(cfg.memory_weights, 0.5)
        t_starts = _pad(cfg.memory_target_starts, 1)
        f_starts = _pad(cfg.memory_fragment_starts, 1)
        lengths = _pad(cfg.memory_lengths, nres)

        memories = []
        for pdb, chain, weight, t_start, f_start, length in zip(
            cfg.memory_pdbs, chains, weights, t_starts, f_starts, lengths
        ):
            pdb = str(self.config.resolve(self.data_root, pdb))
            if not os.path.isfile(pdb):
                raise FileNotFoundError(
                    f"Memory PDB not found: '{pdb}'. Pass a valid path with memory_pdbs."
                )
            memories.append((pdb, chain, t_start, f_start, length, weight))
            self.logger.info(
                f"  Memory: {pdb}  chain={chain}  "
                f"target={t_start}  fragment={f_start}  length={length}  weight={weight}"
            )
        return memories

    def build_system(self) -> BuiltSystem:
        import openawsem as _openawsem
        import openawsem.functionTerms as terms
        from openawsem import OpenMMAWSEMSystem

        cfg = self.config
        input_pdb = str(cfg.resolve(self.data_root, cfg.input_pdb))
        if cfg.prepare:
            input_pdb = self._prepare_cg_pdb(input_pdb)

        if cfg.parameters_dir is not None:
            _openawsem.data_path.parameters = Path(cfg.parameters_dir)
            self.logger.info(f"Parameters dir: {cfg.parameters_dir}")
        param_path = str(_openawsem.data_path.parameters)

        self.logger.info(f"Loading CG PDB: {input_pdb}  (chains: {cfg.chains})")
        oa = OpenMMAWSEMSystem(input_pdb, chains=cfg.chains, k_awsem=cfg.k_awsem)

        force_list = [
            terms.basicTerms.con_term(oa),
            terms.basicTerms.chain_term(oa),
            terms.basicTerms.chi_term(oa),
            terms.basicTerms.excl_term(oa),
        ]

        if cfg.use_rama:
            self.logger.info("Adding Ramachandran backbone term...")
            force_list += [
                terms.basicTerms.rama_term(oa),
                terms.basicTerms.rama_proline_term(oa),
            ]

        if cfg.use_contact:
            self.logger.info(f"Adding contact term (k_contact={cfg.k_contact} kJ/mol)...")
            force_list.append(
                terms.contactTerms.contact_term(
                    oa, k_contact=cfg.k_contact, parametersLocation=param_path
                )
            )

        if cfg.use_burial and not cfg.use_contact:
            self.logger.info("Adding standalone burial free-energy term...")
            force_list.append(terms.contactTerms.burial_term(oa, parametersLocation=param_path))

        if cfg.use_beta:
            self.logger.info(f"Adding beta-sheet H-bond terms 1/2/3 (k={cfg.k_beta} kJ/mol)...")
            k_beta_qty = cfg.k_beta * kilojoules_per_mole
            force_list += [
                terms.hydrogenBondTerms.beta_term_1(oa, k=k_beta_qty),
                terms.hydrogenBondTerms.beta_term_2(oa, k=k_beta_qty),
                terms.hydrogenBondTerms.beta_term_3(oa, k=k_beta_qty),
            ]

        if cfg.use_helical:
            self.logger.info("Adding helical H-bond term...")
            force_list.append(terms.hydrogenBondTerms.helical_term(oa))

        if cfg.use_electrostatics:
            self.logger.info("Adding Debye-Hückel electrostatics...")
            force_list.append(terms.debyeHuckelTerms.debye_huckel_term(oa))

        if cfg.use_associative_memory:
            if not cfg.memory_pdbs:
                raise ValueError(
                    "use_associative_memory requires at least one entry in memory_pdbs."
                )
            self.logger.info(
                f"Adding associative memory term (k_am={cfg.k_am} kJ/mol, "
                f"seq_sep=[{cfg.am_min_seq_sep},{cfg.am_max_seq_sep}], "
                f"well_width={cfg.am_well_width} nm)..."
            )
            memories = self._build_memories(nres=oa.nres)
            force_list.append(
                terms.templateTerms.associative_memory_term(
                    oa,
                    memories=memories,
                    k_am=cfg.k_am,
                    min_seq_sep=cfg.am_min_seq_sep,
                    max_seq_sep=cfg.am_max_seq_sep,
                    am_well_width=cfg.am_well_width,
                )
            )

        self.logger.info(f"Registering {len(force_list)} force term(s) with the system...")
        for f in force_list:
            oa.system.addForce(f)

        self.logger.info(
            f"System ready: {oa.system.getNumParticles()} CG beads, {oa.nres} residues"
        )
        return BuiltSystem(oa.pdb.topology, oa.system, oa.pdb.positions)
