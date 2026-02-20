from types import SimpleNamespace

from core.perception import PerceptionEngine


def _engine():
    eng = PerceptionEngine.__new__(PerceptionEngine)
    eng.TOOL_NAME = "extract_perceptions"
    return eng


def test_extract_function_args_from_top_level_function_call():
    engine = _engine()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="extract_perceptions",
                arguments='{"entities": [{"entity_type":"place","entity":"park","dimension_values":{"quietness":0.7},"confidence":0.8}]}'
            )
        ]
    )

    args = engine._extract_function_args(response)
    assert len(args["entities"]) == 1
    assert args["entities"][0]["entity"] == "park"


def test_extract_function_args_from_nested_tool_call():
    engine = _engine()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="tool_call",
                        name="extract_perceptions",
                        arguments={"entities": []}
                    )
                ]
            )
        ]
    )

    args = engine._extract_function_args(response)
    assert args == {"entities": []}
