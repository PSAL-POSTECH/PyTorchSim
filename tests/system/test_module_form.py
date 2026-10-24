"""The form two extension modules keep, so an addition follows it: post_grad_pass.py lists every
rewrite in its docstring in registration order, each under a section comment naming it;
registration.py's install() calls every _register_* function in file order, each under a section
comment, and its docstring has one line per registry. Read with ast and tokenize, no device."""
import ast
import io
import os
import re
import sys
import tokenize

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


def _module_tree(module):
    source = open(module.__file__).read()
    comments = [(tok.start[0], tok.string) for tok in
                tokenize.generate_tokens(io.StringIO(source).readline)
                if tok.type == tokenize.COMMENT and tok.start[1] == 0]
    return ast.parse(source), comments


def _section_comment(node, comments):
    """The nearest column-0 comment above `node`'s first line (decorators included)."""
    first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    above = [text for line, text in comments if line < first]
    return above[-1] if above else ""


def _check(name, ok):
    check_result(name, torch.tensor([bool(ok)]), torch.tensor([True]))


def test_post_grad_pass_form():
    from pytorchsim.frontend.inductor import post_grad_pass as pg
    names = [*pg.REWRITES, "complex_real_pairs"]
    doc_words = re.findall(r"\w+", pg.__doc__)
    listed = [w for w in doc_words if w in names]
    order = list(dict.fromkeys(listed))
    tree, comments = _module_tree(pg)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing_section = [n for n in names if n not in defs or n not in _section_comment(defs[n], comments)]
    print("rewrites:", names, "docstring order:", order, "without a section:", missing_section)
    _check("PostGradPassForm", order == names and not missing_section)


def test_registration_form():
    from pytorchsim.frontend.inductor import registration as reg
    tree, comments = _module_tree(reg)
    registers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name.startswith("_register_")]
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install")
    called = [c.func.id for c in ast.walk(install)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    missing_section = [n.name for n in registers if not _section_comment(n, comments)
                       or _section_comment(n, comments) == _section_comment(install, comments)]
    doc_lines = [line for line in reg.__doc__.splitlines() if line.startswith("    ")]
    print("registers:", [n.name for n in registers], "install calls:", called,
          "docstring lines:", len(doc_lines), "without a section:", missing_section)
    _check("RegistrationForm", called == [n.name for n in registers]
           and len(doc_lines) == len(registers) and not missing_section)


if __name__ == "__main__":
    test_post_grad_pass_form()
    test_registration_form()
