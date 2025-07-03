# ReactorPy

A general-purpose, symbolic + numerical chemical reactor modeling library for Python.

ReactorPy lets you define, simulate, and analyze **CSTR**, **batch**, **PFR**, and **custom hybrid reactors** using clear, unit-aware syntax. Supports symbolic derivation and numerical integration of arbitrary reaction systems.

---

## Features

- Define reactions using simple strings or formulas
- Supports batch, CSTR, PFR, and semi-batch modes
- Handles multiple species and parallel reactions
- Symbolic modeling via SymPy
- Numerical simulation via SciPy or JAX
- Optional unit support using Pint
- Built-in plotting for concentrations, rates, conversions
- Extensible: define custom rate laws, mass transfer, energy balances

---

## Example

```python
from reactorpy import Reactor, Reaction, Species

A = Species("A", initial=1.0)  # mol/L
B = Species("B", initial=2.0)
C = Species("C", initial=0.0)

r1 = Reaction("A + B -> C", rate="k * A * B", parameters={"k": 0.5})
rxn = Reactor(type="batch", reactions=[r1], species=[A, B, C])

t, results = rxn.simulate(time=10)
rxn.plot()

## To-do List

- [ ] Core API for `Species`, `Reaction`, and `Reactor`
- [ ] Implement batch reactor model
- [ ] Add CSTR model (steady-state and dynamic)
- [ ] Add PFR model (1D spatial discretization)
- [ ] Parse symbolic rate expressions using SymPy
- [ ] Implement numerical ODE solver backend (SciPy)
- [ ] Optional: Add JAX or Numba backend for speed
- [ ] Create unit integration using Pint
- [ ] Add plotting for concentration vs. time/space
- [ ] Build JSON/DSL model input format
- [ ] Export capability (CSV, LaTeX, etc.)
- [ ] Develop simple CLI or Jupyter-based UI
- [ ] Integrate with ThermoProps for enthalpy/heat balance
