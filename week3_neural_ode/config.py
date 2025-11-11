from data.sine import get_sine_data
from data.lynx_hare import load_lynx_hare_data
from data.covid19 import load_covid_data
import diffrax

LEARNING_RATE = .005
NUM_STEPS = 50_000
PRINT_EVERY = 1000
WIDTH = 32
DEPTH = 2
T_FINAL = 10.0
DT = 0.05
BATCH_SIZE = 32
SEED = 42

datasets = {
        "sine": get_sine_data,
        "lynx_hare": load_lynx_hare_data,
        "covid": load_covid_data,
        }

SOLVER_CONFIG = {
    "train": {
        "solver": diffrax.Euler(),
        "substeps": 4,
        "controller": diffrax.ConstantStepSize(),
    },
    "eval": {
        "solver": diffrax.Tsit5(),
        "substeps": 1,
        "controller": diffrax.PIDController(rtol=1e-3, atol=1e-6),
    },
}
