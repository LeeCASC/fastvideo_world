"""Check the public architecture lookup without CUDA or model imports."""
import ast
from pathlib import Path


def test_registered_text_encoder_still_resolves_to_the_existing_class():
    root = Path(__file__).resolve().parents[2]
    registry = ast.parse((root / 'fastvideo/models/registry.py').read_text())
    assignment = next(node for node in registry.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == '_TEXT_ENCODER_MODELS'
                              for target in node.targets))
    models = ast.literal_eval(assignment.value)
    name = 'Mistral3ForConditional' 'Generation'
    assert models[name] == ('encoders', 'mistral3', name)
    implementation = ast.parse((root / 'fastvideo/models/encoders/mistral3.py').read_text())
    assert any(isinstance(node, ast.ClassDef) and node.name == name for node in implementation.body)
    parity = ast.parse((root / 'tests/local_tests/flux2/test_flux2_component_parity.py').read_text())
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr'
               and len(node.args) == 2 and isinstance(node.args[0], ast.Name)
               and node.args[0].id == 'mistral3_encoder' and isinstance(node.args[1], ast.Constant)
               and node.args[1].value == name for node in ast.walk(parity))
