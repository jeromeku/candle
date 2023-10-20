import torch


def _make_module_map(model: torch.nn.Module):
    module_map = {}

    for k, v in model.named_modules():
        module_cls = v.__class__.__name__
        if module_cls in module_map:
            module_map[module_cls] = module_map[module_cls] + [k]
        else:
            module_map[module_cls] = [k]

    return module_map


def get_module(name: str, model: torch.nn.Module):
    """Look up module by module class name

    Calling module.named_modules() yields weight path (str) to module mapping.  This function
    looks up the module by *module class name* instead.

    For example, given this model / module:
        MistralForCausalLM(
            (model): MistralModel(
            (embed_tokens): Embedding(32000, 4096)
            (layers): ModuleList(
            (0-31): 32 x MistralDecoderLayer(
                (self_attn): MistralAttention(
                (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (rotary_emb): MistralRotaryEmbedding()
                )
                (mlp): MistralMLP(
                (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
                (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
                (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
                (act_fn): SiLUActivation()
                )
                (input_layernorm): MistralRMSNorm()
                (post_attention_layernorm): MistralRMSNorm()
            )
            )
            (norm): MistralRMSNorm()
        )
            (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
        )

    Calling get_module('MistralAttention', model) will return the MistralAttention module along with its weight path.

    In the case that there are multiple modules of the same class, the weight path will be a list of paths.

    Args:
        name (str): module class name
        model (torch.nn.Module): model to search

    Returns:
        path (list[str]): weight path to module
        module (torch.nn.Module): module
    """

    module_map = _make_module_map(model)
    modules = dict(model.named_modules())

    path = module_map[name]

    key = path[0]
    fmt_path = path[0] + ", ... ," + path[-1] if len(path) > 1 else path[0]

    module = modules.get(key)
    print(f"{fmt_path}: {module}")

    return path, module
