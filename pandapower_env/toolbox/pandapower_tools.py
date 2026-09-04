"""
Helper functions for writing streamlined code that applies to e.g. both lines and trafos.

This should help shorten your code.
"""

ELEMENT_TYPES = ["line", "trafo", "ext_grid", "load", "gen", "sgen"]


def get_from_bus_str(element_type: str) -> str | None:
    """
    For elements that have an analogue to "from_bus" or "to_bus", get the corresponding "from_bus" logical string.

    This function is for iterating over different types of elements (e.g. lines and trafos) in the same code.

    :param element_type: Which element to consider (e.g. "trafo", "line", etc.)
    :type element_type: str
    :return: corresponding "from_bus" style string (e.g. "lv_bus" for trafo)
    :rtype: str
    """
    return {"line": "from_bus",
            "trafo": "lv_bus",
            "ext_grid": "bus",
            "gen": "bus",
            "sgen": "bus",
            "load": "bus"}.get(element_type)


def get_to_bus_str(element_type: str) -> str | None:
    """
    For elements that have an analogue to "from_bus" or "to_bus", get the corresponding "to_bus" logical string.

    This function is for iterating over different types of elements (e.g. lines and trafos) in the same code.

    :param element_type: Which element to consider (e.g. "trafo", "line", etc.)
    :type element_type: str
    :return: corresponding "to_bus" style string (e.g. "hv_bus" for trafo)
    :rtype: str
    """
    return {"line": "to_bus",
            "trafo": "hv_bus",
            "ext_grid": "bus",
            "gen": "bus",
            "sgen": "bus",
            "load": "bus"}.get(element_type)


def get_element_type_string(element_type: str) -> str | None:
    """
    Return the "shortened" version of an element type.

    Certain elements are represented by string. This function can be called
    to make sure that the strings are harmonized in this package, and to be able to change strings centrally.

    :param element_type: Which element to consider (e.g. "trafo", "line", etc.)
    :type element_type: str
    :return: corresponding "shortened" string (e.g. "ext" for ext_grid)
    :rtype: str
    """
    return {"line": "line",
            "trafo": "trafo",
            "ext_grid": "ext",
            "gen": "gen",
            "sgen": "sgen",
            "load": "load"}.get(element_type)
