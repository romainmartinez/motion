import inspect

import black
import pytest

import motion
from tests.utils import extract_code_blocks_from_md, function_has_return


class DocStringError(Exception):
    pass


methods = [
    method_obj
    for class_name, class_obj in inspect.getmembers(motion, inspect.isclass)
    for method_name, method_obj in inspect.getmembers(class_obj)
    if (inspect.isfunction(method_obj) or inspect.ismethod(method_obj))
    and method_name[0] != "_"
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
    code_block = extract_code_blocks_from_md(method.__doc__).replace(
        "plt.show()",
        f'plt.savefig("docs/images/api/{method.__name__}.svg", bbox_inches="tight")\nplt.figure()',
    )
    exec(
        code_block, {}, {},
    )


@pytest.mark.parametrize("method", methods)
def test_docstring_lint_code_blocks(method):
    code_blocks = extract_code_blocks_from_md(method.__doc__)
    if code_blocks:
        code_blocks = f"{code_blocks}\n"
        assert code_blocks == black.format_str(code_blocks, mode=black.FileMode())


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
    args = [a for a in argspec.args if a not in ignored_args]
    if args and "Arguments:" not in method.__doc__:
        raise DocStringError(f"`Arguments` block missing in `{method}` docstring")
    for arg in args:
        if arg in ignored_args:
            continue
        if arg not in method.__doc__:
            raise DocStringError(f"{arg} not described in {method} docstring")
        if arg not in argspec.annotations:
            raise DocStringError(f"{arg} not type annotated in {method}")
