"""schema-compat: make one tool/function schema work with Anthropic, OpenAI and Gemini.

>>> from schema_compat import transform_schema
>>> result = transform_schema({"type": "object", "properties": {"n": {"const": 1}}}, "gemini")
>>> result.schema
{'type': 'object', 'properties': {'n': {'enum': ['1']}}}
"""

from ._api import check_schema, check_tools, transform_schema
from ._model import TARGETS, Issue, SchemaResult, Target, ToolsResult
from ._restore import restore_arguments
from ._tools import name_is_valid, sanitize_name, transform_tools

__version__ = "0.1.0"

__all__ = [
    "TARGETS",
    "Issue",
    "SchemaResult",
    "Target",
    "ToolsResult",
    "__version__",
    "check_schema",
    "check_tools",
    "name_is_valid",
    "restore_arguments",
    "sanitize_name",
    "transform_schema",
    "transform_tools",
]
