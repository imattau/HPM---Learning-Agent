from hpm_ai_v5.arc import ArcSolver


def test_arc_solver_infers_translation_and_recolour() -> None:
    raw_task = {
        "task_id": "toy_translate_recolour",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            },
        ],
        "test": [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)

    assert packet.context["arc"]["route"] == "translate_recolour"
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
    assert packet.context["arc"]["hypotheses"]
    assert packet.states
    assert any(view["name"] == "object_polygraph" for view in packet.views)
    assert any(view["name"] == "geometry_polygraph" for view in packet.views)
    assert any(view["name"] == "arc_hypothesis" for view in packet.views)
    assert any(entry["role"] == "adapter" for entry in packet.trace)
    assert any(entry["role"] == "agent" for entry in packet.trace)


def test_arc_solver_handles_rotation() -> None:
    raw_task = {
        "task_id": "toy_rotate_90",
        "train": [
            {
                "input": [
                    [0, 1, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                ],
                "output": [
                    [1, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0],
                ],
            },
            {
                "input": [
                    [0, 2, 0],
                    [0, 2, 0],
                    [2, 2, 0],
                ],
                "output": [
                    [2, 0, 0],
                    [2, 2, 2],
                    [0, 0, 0],
                ],
            },
        ],
        "test": [
            [
                [0, 3, 0],
                [0, 3, 0],
                [3, 3, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)

    assert packet.context["arc"]["route"] == "rotate_90_recolour"
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == [
        [3, 0, 0],
        [3, 3, 3],
        [0, 0, 0],
    ]


def test_arc_solver_handles_horizontal_reflection() -> None:
    raw_task = {
        "task_id": "toy_mirror_horizontal",
        "train": [
            {
                "input": [
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                ],
                "output": [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                ],
            },
            {
                "input": [
                    [0, 2, 0],
                    [2, 0, 0],
                    [2, 2, 0],
                ],
                "output": [
                    [0, 2, 0],
                    [0, 0, 2],
                    [0, 2, 2],
                ],
            },
        ],
        "test": [
            [
                [0, 3, 0],
                [3, 0, 0],
                [3, 3, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)

    assert packet.context["arc"]["route"] == "mirror_horizontal_recolour"
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == [
        [0, 3, 0],
        [0, 0, 3],
        [0, 3, 3],
    ]
    assert packet.context["arc"]["symmetry_hypotheses"]
    assert any(view["name"] == "arc_symmetry_completion" for view in packet.views)


def test_arc_solver_extends_horizontal_line() -> None:
    raw_task = {
        "task_id": "toy_extend_horizontal_line",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 4, 4, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [4, 4, 4, 4, 4],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            }
        ],
        "test": [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 4, 4, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)

    assert packet.context["arc"]["route"] == "extend_line"
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert packet.context["arc"]["line_extension_hypotheses"]
    assert any(view["name"] == "arc_line_extension" for view in packet.views)


def test_arc_solver_extends_vertical_line() -> None:
    raw_task = {
        "task_id": "toy_extend_vertical_line",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 5, 0, 0, 0],
                    [0, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 5, 0, 0, 0],
                    [0, 5, 0, 0, 0],
                    [0, 5, 0, 0, 0],
                    [0, 5, 0, 0, 0],
                    [0, 5, 0, 0, 0],
                ],
            }
        ],
        "test": [
            [
                [0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)

    assert packet.context["arc"]["route"] == "extend_line"
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == [
        [0, 5, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 5, 0, 0, 0],
    ]
    assert packet.context["arc"]["line_extension_hypotheses"]


def test_arc_solver_crops_object() -> None:
    raw_task = {
        "task_id": "toy_crop_object",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 6, 6, 0],
                    [0, 0, 6, 6, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [
                    [7, 7],
                    [7, 7],
                ],
            }
        ],
        "test": [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 6, 6, 0],
                [0, 0, 6, 6, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)

    assert packet.context["arc"]["route"] == "crop_object"
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == [
        [7, 7],
        [7, 7],
    ]


def test_arc_solver_emits_richer_structural_views() -> None:
    raw_task = {
        "task_id": "toy_structural_views",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            }
        ],
        "test": [
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
    }

    solver = ArcSolver()
    packet = solver.solve(raw_task)
    arc = packet.context["arc"]

    assert arc["edges"]
    assert arc["objects"]
    assert arc["shapes"]
    assert arc["relations"]
    assert arc["colours"]
    assert arc["patterns"]
    assert arc["grid_deltas"]
    assert arc["object_deltas"]
    assert arc["colour_deltas"]
    assert arc["structural_deltas"]
    assert any(view["name"] == "image_polygraph" for view in packet.views)
    assert any(view["name"] == "transformation_polygraph" for view in packet.views)
