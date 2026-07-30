"""Engine-specific ASE calculators -- the extension point for new engines.

:class:`~md_simulations.torch_potentials.potential.Potential` and the bridge under
it are engine-agnostic: they only ever call ``atoms.get_potential_energy()`` and
``atoms.get_forces()``. Supporting a new engine therefore means adding one module
here, and touching neither ``Potential`` nor ``bridge``.

Nothing is re-exported: each calculator pulls in its own engine, so importing this
package must not drag every engine along. Import the one you need explicitly::

    from md_simulations.torch_potentials.calculators.openmm import OpenMMCalculator"""
