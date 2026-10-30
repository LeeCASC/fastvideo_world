from fastvideo.train.utils.lora import _is_target_layer


def test_lora_target_matches_through_activation_checkpoint_wrapper() -> None:
    wrapped_name = "transformer_blocks.42._checkpoint_wrapped_module.attn.to_q"

    assert _is_target_layer(wrapped_name, ["transformer_blocks.42.attn"])
    assert not _is_target_layer(wrapped_name, ["transformer_blocks.41.attn"])
