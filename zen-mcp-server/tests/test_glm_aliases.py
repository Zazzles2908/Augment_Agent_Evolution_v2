from providers.zhipu.provider import GLMModelProvider


def test_glm_aliases_map_to_supported_models():
    prov = GLMModelProvider(api_key="test")
    aliases = [
        "GLM-4-Air-250414",
        "GLM-4-Flash-250414",
        "GLM-4.5-Air",
        "GLM-4.5-Flash",
        "GLM-4.5",
        "glm-4-32b-0414-128k",
        "glm-4.5-airx",
        "glm-4.5-x",
        "glm-4.5v",
        "glm-air",
        "glm-plus",
        "glm",
        "z-ai",
        "zhipu",
    ]
    for alias in aliases:
        caps = prov.get_capabilities(alias)
        assert caps.model_name in {
            "glm-4",
            "glm-4-plus",
            "glm-4-air",
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-flash",
            "GLM-4",
        }

