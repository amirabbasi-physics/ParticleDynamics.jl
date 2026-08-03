# Experimental and high-level reference data for benzene crystal I

Values used to judge the MACE-OFF vs Orb-v3 head-to-head. Where references
disagree, the spread is reported rather than a single number.

## Structure

| quantity | value | source |
|---|---|---|
| space group, Z | Pbca, Z = 4 (orthorhombic) | standard for benzene I |
| starting cell (150 K, ambient p) | a = 6.914, b = 7.476, c = 9.563 Å, V = 494.30 Å³ | COD 7238223; Nayak, Sathishkumar & Guru Row, *CrystEngComm* **12**, 3112 (2010) |
| C–C bond (X-ray, 150 K) | 1.379 Å | same entry |
| C–H bond (X-ray) | 0.930 Å | same entry — X-ray foreshortening artifact |
| C–H bond (neutron) | ~1.08 Å | neutron diffraction; X-ray systematically underestimates C–H |

Note on the cell: this entry's density is 1.050 g/cm³, on the low side of
reported values for solid benzene (historical X-ray at 251 K gives
a = 7.44, b = 9.65, c = 6.81 Å, V ≈ 489 Å³). A 0 K classical relaxation should
therefore land *below* 494 Å³, both from thermal contraction and from the
absence of zero-point expansion. Reported low-temperature cells cluster around
462–475 Å³, so the volume comparison carries a few-percent reference
uncertainty and is treated as secondary to the energy comparison.

## Lattice energy (magnitudes, 0 K)

| method | value (kJ/mol) | source |
|---|---|---|
| **experiment**, back-corrected to 0 K | **55.3 ± 2.2** | quoted in Hasanein et al., arXiv:1606.08178, from ref. 15 below |
| CCSD(T) / many-electron wavefunction | 55.90 ± 0.76 | Yang et al., *Science* **345**, 640 (2014) — "sub-kJ/mol accuracy" |
| DMC | 50.6 ± 0.5 | Hasanein et al., arXiv:1606.08178 |
| DMC (X23 set) | 49.8 ± 0.2 | Della Pia et al., *Phys. Rev. Lett.* **133**, 046401 (2024) |

The high-level references themselves span ~6 kJ/mol, which is worth stating
plainly: experiment and CCSD(T) agree near 55–56 kJ/mol, while the two DMC
studies sit near 50 kJ/mol. Experimentally derived sublimation enthalpies for
benzene scatter over 41.7–53.9 kJ/mol across historical measurements
(Della Pia et al., supplementary), so any claim of sub-kJ/mol agreement from an
MLIP would be over-reading the reference data.

## Vibrational frequencies (Raman, cm⁻¹)

| mode | experimental | note |
|---|---|---|
| ring breathing | 992 | strongest benzene Raman line |
| C–C stretch | 1586 | |
| aromatic C–H stretch | 3062 | |

Crystalline benzene additionally shows librational lattice modes below
~130 cm⁻¹ (single-crystal Raman at 140 K).

## Why no NPT / pressure comparison

While an external potential is attached, the engine's provider contract
supplies no virial (`src/simulation/ExternalPotential.jl`), so pressure
observables and NPT are unsupported. The equilibrium volume is therefore
obtained from an energy–volume scan at fixed cells rather than from a barostat.
