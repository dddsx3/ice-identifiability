# ice-identifiability

A sea-ice thickness product ships a declared uncertainty. Before that number is
used in an error budget, two questions have to be answered: is the declared
quantity the one the product delivers, and is the validation reference
independent of it? This toolkit answers them and, when it cannot, says so — a
gate that cannot be evaluated is never a pass.

    from ice_identifiability import run
    verdict = run("path/to/product.nc", "path/to/reference.nc")
    print(verdict.code, verdict.reason)

## What is here

    src/ice_identifiability/   the package
    examples/                  runnable examples and a synthetic fixture
      data/                    four 60x60 fixtures, twelve monthly steps each,
                               plus synth_truth.json with the target parameter,
                               the correlation length and a SHA-256 per file
    tests/                     29 tests, including the gate regressions
    scripts/, pyproject.toml, requirements-lock.txt

## Run it

    pip install -e .
    python -m pytest tests -q                  # 29 tests
    python examples/gate_calibration.py        # the gate matrix
    python examples/discriminate.py            # A/B discrimination on the fixture
    python examples/bootstrap_ci.py            # interval on the estimand

Five synthetic inputs return the codes 0, 0, 2, 2 and 3; the examples print them
in that order.

## What it refuses, and why that is the point

The three gate states are pass, fail and *not testable*. The third is the one
that matters: a product with no declared thickness uncertainty is refused with
`INPUT_NOT_ASSESSABLE`, not silently scored. One real archive product is
assessable on its declared posterior channel and refused on its freeboard
channel, where the declared freeboard error lies a factor of four below the same
nuisance floor.

## Licence and citation

MIT (see LICENSE). The synthetic fixtures under `examples/data` are CC BY 4.0.
Cite the article and this repository (CITATION.cff).
