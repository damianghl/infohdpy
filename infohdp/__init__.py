# Import core functionalities
from .core import entropy_true, entropy_x_true, entropy_y_true, mutual_information_true

# Import estimators
from .estimators import naive, nsb, binary_infohdp, multiclass_infohdp

# Import generators
from .generators import probability, sample

# Import utilities
from .utils import freq_of_frequencies, count_nxy_binary, count_nxy_multiclass

# Define version
__version__ = "0.1.1"

# Define all exported symbols
__all__ = [
    "naive",
    "nsb",
    "binary_infohdp",
    "multiclass_infohdp",
    "probability",
    "sample",
]
