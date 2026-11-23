"""Compatibility guards for broken optional packages in shared base images."""


def disable_broken_peft_bitsandbytes() -> None:
    import peft.import_utils as peft_import_utils

    try:
        import bitsandbytes as _bitsandbytes  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        clear_cache = getattr(peft_import_utils.is_bnb_available, 'cache_clear', None)
        if clear_cache is not None:
            clear_cache()
        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        try:
            from peft.tuners.lora import model as peft_lora_model
        except ImportError:
            return
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False
