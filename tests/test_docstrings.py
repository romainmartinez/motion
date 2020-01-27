import inspect
import re

import pytest

import motion


class DocStringError(Exception):
    pass


methods = [
    method_obj
    for class_name, class_obj in inspect.getmembers(motion, inspect.isclass)
    for method_name, method_obj in inspect.getmembers(class_obj)
    if (inspect.isfunction(method_obj) or inspect.ismethod(method_obj))
    and method_name[0] != "_"
    and class_name != "MecaDataArrayAccessor"
]
ignored_args = ["cls", "self"]


@pytest.mark.parametrize("method", methods)
def test_has_docstring(method):
    if not method.__doc__:
        raise DocStringError(f"Missing docstring in `{method}`")


@pytest.mark.parametrize("method", methods)
def test_docstring_has_example(method):
    if "```python" not in method.__doc__:
        raise DocStringError(f"Missing example in `{method}` docstring")


@pytest.mark.parametrize("method", methods)
def test_docstring_example(method):
    exec(extract_code_from_docstring(method.__doc__), {}, {})


@pytest.mark.parametrize("method", methods)
def test_docstring_return(method):
    if function_has_return(method):
        if "Returns:" not in method.__doc__:
            raise DocStringError(f"Missing returns in `{method}` docstring")
        if "return" not in inspect.getfullargspec(method).annotations:
            raise DocStringError(
                f"Type annotation missing for the `return` type in {method} docstring"
            )


@pytest.mark.parametrize("method", methods)
def test_docstring_parameters(method):
    argspec = inspect.getfullargspec(method)
    if argspec.args and "Arguments:" not in method.__doc__:
        raise DocStringError(f"`Arguments` block missing in `{method}` docstring")
    for arg in argspec.args:
        if arg in ignored_args:
            continue
        if arg not in method.__doc__:
            raise DocStringError(f"{arg} not described in {method} docstring")
        if arg not in argspec.annotations:
            raise DocStringError(f"{arg} not type annotated in {method}")


def function_has_return(func):
    """Caution: this will return True if the function contains the word 'return'"""
    lines, _ = inspect.getsourcelines(func)
    return any("return" in line for line in lines)


def extract_code_from_docstring(
    docstring: str, start_code_block: str = "```python", end_code_block: str = "```"
) -> str:
    return inspect.cleandoc(
        "\n".join(
            re.findall(
                fr"{start_code_block}(.*?){end_code_block}", docstring, re.DOTALL
            )
        )
    )
