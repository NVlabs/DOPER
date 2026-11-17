# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from rendering.config import load_render_config, render_config_from_mapping


@pytest.mark.parametrize("document", ["false", "0", "[]", "[studio]", "true", "text", "null", ""])
def test_yaml_root_must_be_a_mapping(tmp_path, document):
    path = tmp_path / "render.yaml"
    path.write_text(document)
    with pytest.raises(ValueError, match="render config.*mapping"):
        load_render_config(str(path))


@pytest.mark.parametrize("raw", [False, 0, [], ["studio"], True, "text", None])
def test_mapping_entrypoint_rejects_non_mapping_roots(raw, tmp_path):
    with pytest.raises(ValueError, match="render config.*mapping"):
        render_config_from_mapping(raw, str(tmp_path))


def test_background_mode_list_normalizes_like_comma_separated_string(tmp_path):
    modes = [" HDRI ", "Dark", " RANDOM ", " STUDIO ", "", "unknown"]
    listed = render_config_from_mapping({"scene": {"world_background_modes": modes}}, str(tmp_path))
    comma = render_config_from_mapping(
        {"scene": {"world_background_modes": ",".join(modes)}}, str(tmp_path)
    )
    assert (
        listed.world_background_modes
        == comma.world_background_modes
        == ("hdri", "dark", "random", "studio")
    )
