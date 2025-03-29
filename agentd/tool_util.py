def call_tool_with_introspection(tool, provided_params: dict):
    model_cls = tool.get_model()
    valid_fields = set(model_cls.model_fields.keys())
    valid_params = {k: v for k, v in provided_params.items() if k in valid_fields}
    for field_name, field_info in model_cls.model_fields.items():
        if field_info.is_required() and field_name not in valid_params:
            raise ValueError(f"Missing required parameter: {field_name}")
    instance = model_cls(**valid_params)
    return tool.call(instance)
