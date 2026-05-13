import ast
from pathlib import Path

SOURCE_ROOTS = (Path("src"), Path("apps"))


def test_source_code_uses_logging_instead_of_print() -> None:
    offenders: list[str] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                is_print_call = (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                )
                if is_print_call:
                    offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
