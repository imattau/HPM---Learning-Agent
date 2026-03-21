from .wikipedia import WikipediaSubstrate
from .pypi import PyPISubstrate
from .local_file import LocalFileSubstrate
from .linguistic import LinguisticSubstrate
from .math import MathSubstrate
from .bridge import SubstrateBridgeAgent

__all__ = [
    'WikipediaSubstrate',
    'PyPISubstrate',
    'LocalFileSubstrate',
    'LinguisticSubstrate',
    'MathSubstrate',
    'SubstrateBridgeAgent',
]
