from traininglib.paths import svg
from traininglib.paths import pathutils as utils

import torch


def test_svg_export_import():
    paths = [
        [
            [
                (10.,50.),
                (20, 70),
                (30, 60),
                (40, 40),
            ],
            [
                (20.,70.),
                (10,60),
                (12,65),
            ]
        ]
    ]

    exported = svg.export_paths_as_svg(paths, (1000, 1000))
    print(exported)

    parsed = svg.parse_svg_string(exported)
    print(parsed)

    assert paths == parsed.paths


def test_svg_parse():
    parsed = svg.parse_svg('tests/assets/mockpaths.svg')
    paths  = parsed.paths 
    assert len(paths) == 2
    assert len(paths[0]) == 2
    assert len(paths[1]) == 3
    assert len(paths[1][0]) == 6


def test_convert_tensor_to_svg_path():
    pathstensor = torch.tensor([
        (10.0, 50.0, 0),
        (20.0, 70.0, 0),
        (30.0, 60.0, 0),
        (40.0, 40.0, 0),
        (20.0, 70.0, 1),
        (10.0, 60.0, 1),
        (12.0, 65.0, 1),
    ])
    pathssvg = utils.convert_multiple_paths_tensor_to_svg(pathstensor)

    expected = [
        [
            [
                (10.,50.),
                (20, 70),
                (30, 60),
                (40, 40),
            ],
        ],
        [
            [
                (20.,70.),
                (10,60),
                (12,65),
            ]
        ]
    ]

    assert pathssvg == expected

