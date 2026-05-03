from hpm_ai_v5.adapter import AdapterPacket
from hpm_ai_v5.adapter.action_sequence_unpacker import ActionSequenceUnpacker
from hpm_ai_v5.adapter.connected_components import ConnectedComponentsAdapter
from hpm_ai_v5.adapter.delta_buffer import DeltaBufferAdapter
from hpm_ai_v5.adapter.flatten_grid import FlattenGridAdapter
from hpm_ai_v5.adapter.grid_postprocessor import GridPostprocessor
from hpm_ai_v5.adapter.recent_buffer import RecentBufferAdapter
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core import Action


def test_recent_buffer_adapter_emits_recent_history() -> None:
    adapter = RecentBufferAdapter(buffer_size=3, default=-1.0)

    packet = adapter.run(AdapterPacket(raw=1.0))
    assert packet.states[-1].value == (-1.0, -1.0, 1.0)

    packet = adapter.run(AdapterPacket(raw=2.0))
    assert packet.states[-1].value == (-1.0, 1.0, 2.0)


def test_delta_buffer_adapter_emits_recent_deltas() -> None:
    adapter = DeltaBufferAdapter(buffer_size=2)

    packet = adapter.run(AdapterPacket(raw=1.0))
    assert packet.states[-1].value == (0.0, 0.0)

    packet = adapter.run(AdapterPacket(raw=4.0))
    assert packet.states[-1].value == (0.0, 3.0)


def test_flatten_grid_adapter_flattens_2d_grid() -> None:
    adapter = FlattenGridAdapter()

    packet = adapter.run(AdapterPacket(raw=((1, 2), (3, 4))))
    assert packet.states[-1].value == (1, 2, 3, 4)
    assert packet.states[-1].context["height"] == 2
    assert packet.states[-1].context["width"] == 2


def test_connected_components_adapter_extracts_signatures() -> None:
    adapter = ConnectedComponentsAdapter(background=0)

    packet = adapter.run(AdapterPacket(raw=((0, 1, 1), (0, 0, 1), (2, 0, 0))))
    assert packet.states[-1].value
    assert packet.states[-1].context["component_count"] == 2


def test_grid_postprocessor_reshapes_flat_grid() -> None:
    adapter = GridPostprocessor()
    action = Action(action_type="apply_delta", value=(1, 2, 3, 4), confidence=1.0)

    assert adapter.postprocess(action, context={"height": 2, "width": 2}) == [[1, 2], [3, 4]]


def test_action_sequence_unpacker_unpacks_macros() -> None:
    adapter = ActionSequenceUnpacker()
    action = Action(action_type="execute_sequence", value=("up", "up", "down"), confidence=1.0)

    assert adapter.postprocess(action) == ("up", "up", "down")


def test_validation_only_adapter_passes_through_value() -> None:
    adapter = ValidationOnlyAdapter()
    action = Action(action_type="apply_delta", value=7.0, confidence=1.0)

    assert adapter.postprocess(action) == 7.0
