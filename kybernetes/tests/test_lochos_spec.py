import pytest
from kybernetes.lochos.spec import RigSpec, PodSpec, SeatSpec, EdgeSpec, create_starter_rig


def test_starter_rig_creation():
    rig = create_starter_rig("metis-squad")
    assert rig.name == "metis-squad"
    assert len(rig.pods) == 1
    assert len(rig.list_seats()) == 2

    coder = rig.get_seat("coder")
    assert coder is not None
    assert coder.id == "coder"
    assert coder.full_address(rig.name) == "coder@metis-squad"
    assert coder.permission_policy == "strict"

    reviewer = rig.get_seat("reviewer")
    assert reviewer is not None
    assert reviewer.id == "reviewer"
    assert reviewer.permission_policy == "audit_only"


def test_rig_yaml_serialization_roundtrip():
    rig = create_starter_rig("test-squad")
    yaml_text = rig.to_yaml()
    assert "name: test-squad" in yaml_text
    assert "coder" in yaml_text
    assert "reviewer" in yaml_text

    loaded = RigSpec.from_yaml(yaml_text)
    assert loaded.name == rig.name
    assert len(loaded.list_seats()) == 2
    assert loaded.get_seat("coder").model == "qwen2.5-coder-32b"


def test_invalid_edge_raises():
    with pytest.raises(ValueError, match="Invalid edge kind"):
        EdgeSpec.from_dict({
            "from_seat": "a",
            "to_seat": "b",
            "kind": "illegal_relationship",
        })
