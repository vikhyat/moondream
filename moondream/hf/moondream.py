"""Hugging Face integration layer for Moondream.

Restores the ``moondream.hf`` namespace (removed in the 2.5 refactor) with:

* ``Moondream`` -- a legacy-compatible wrapper around the torch model with a
  working :meth:`Moondream.from_pretrained` that downloads checkpoints from
  the Hugging Face Hub and loads them into the model.
* ``from_pretrained`` -- module-level convenience alias.
* ``HfMoondream`` / ``HfConfig`` -- re-exports of the transformers
  ``PreTrainedModel`` / ``PretrainedConfig`` classes used by
  ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.

All heavy imports (torch, transformers, huggingface_hub) happen lazily so the
package can be imported without a GPU stack installed.
"""

from .util import LATEST_REVISION, detect_device


class Moondream:
    """Hugging Face-facing Moondream model.

    Wraps the torch :class:`~moondream.torch.moondream.MoondreamModel` and
    provides the legacy moondream 2.x API surface (``from_pretrained``,
    list-based ``caption``/``encode_image``, ``answer_question`` with a result
    queue, ``generate``, etc.).
    """

    def __init__(self, config=None):
        from ..torch.hf_moondream import HfConfig, HfMoondream

        self.config = config if config is not None else HfConfig()
        self._hf = HfMoondream(self.config)
        self.model = self._hf.model
        self._is_kv_cache_setup = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        revision=None,
        cache_dir=None,
        torch_dtype=None,
        device=None,
        **kwargs,
    ):
        """Download a Moondream checkpoint from the Hugging Face Hub and
        load it into the model.

        Args:
            model_id: Hugging Face Hub model identifier (e.g.
                ``"vikhyatk/moondream2"``).
            revision: Optional revision (tag, branch, or commit) to use.
            cache_dir: Optional directory to cache downloaded files in.
            torch_dtype: Optional torch dtype to cast the model to.
            device: Optional torch device to move the model to.
            **kwargs: Accepted for API compatibility (``trust_remote_code``,
                ``device_map``, etc.) and ignored.

        Returns:
            A :class:`Moondream` instance with weights loaded.
        """
        import torch
        from dataclasses import replace

        from huggingface_hub import snapshot_download

        from ..torch.config import MoondreamConfig
        from ..torch.moondream import MoondreamModel
        from ..torch.weights import load_weights_into_model

        repo_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.pt"],
        )
        weights = _find_weights(repo_dir)

        # The published checkpoints are non-MoE; the default config builds an
        # MoE text model, so detect the architecture from the checkpoint and
        # disable MoE when it isn't present.
        config = MoondreamConfig()
        if not _has_moe_blocks(weights[0]):
            config = replace(config, text=replace(config.text, moe=None))

        model = cls.__new__(cls)
        model.config = None
        model._hf = None
        model.model = MoondreamModel(config, setup_caches=False)
        model._is_kv_cache_setup = False

        for weights_file in weights:
            load_weights_into_model(weights_file, model.model)

        if torch_dtype is not None:
            model.model = model.model.to(dtype=torch_dtype)
        if device is not None:
            model.model = model.model.to(device=device)
        return model

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _setup_caches(self):
        if not self._is_kv_cache_setup:
            self.model._setup_caches()
            self._is_kv_cache_setup = True

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        self._is_kv_cache_setup = False
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    @property
    def device(self):
        return self.model.device

    # ------------------------------------------------------------------
    # Skills (legacy-compatible)
    # ------------------------------------------------------------------

    def encode_image(self, image, settings=None):
        self._setup_caches()
        if isinstance(image, (list, tuple)):
            return [self.model.encode_image(img, settings) for img in image]
        return self.model.encode_image(image, settings)

    def load_encoded_image(self, encoded_image):
        self._setup_caches()
        self.model.load_encoded_image(encoded_image)

    def caption(
        self,
        images,
        tokenizer=None,
        length="normal",
        stream=False,
        settings=None,
    ):
        """Caption one or more images.

        For backwards compatibility, a list of images returns a list of
        caption strings (the ``tokenizer`` argument is accepted and ignored).
        """
        self._setup_caches()
        if isinstance(images, (list, tuple)):
            return [
                self.model.caption(
                    image, length=length, stream=stream, settings=settings
                )["caption"]
                for image in images
            ]
        return self.model.caption(
            images, length=length, stream=stream, settings=settings
        )

    def query(
        self,
        image=None,
        question=None,
        reasoning=True,
        spatial_refs=None,
        stream=False,
        settings=None,
    ):
        self._setup_caches()
        return self.model.query(
            image=image,
            question=question,
            reasoning=reasoning,
            spatial_refs=spatial_refs,
            stream=stream,
            settings=settings,
        )

    def detect(self, image, object, settings=None):
        self._setup_caches()
        return self.model.detect(image, object, settings=settings)

    def point(self, image, object, settings=None):
        self._setup_caches()
        return self.model.point(image, object, settings=settings)

    def detect_gaze(self, image, eye=None, face=None, unstable_settings={}):
        self._setup_caches()
        return self.model.detect_gaze(
            image, eye=eye, face=face, unstable_settings=unstable_settings
        )

    # ------------------------------------------------------------------
    # Legacy generation API
    # ------------------------------------------------------------------

    def answer_question(
        self,
        image_embeds,
        question,
        tokenizer=None,
        chat_history="",
        result_queue=None,
        max_new_tokens=256,
        **kwargs,
    ):
        answer = self.query(
            image_embeds, question, reasoning=False
        )["answer"].strip()

        if result_queue is not None:
            result_queue.put(answer)
        return answer

    def batch_answer(self, images, prompts, tokenizer=None, **kwargs):
        answers = []
        for image, prompt in zip(images, prompts):
            answers.append(self.query(image, prompt, reasoning=False)["answer"].strip())
        return answers

    def generate(self, image_embeds, prompt, tokenizer, max_new_tokens=128, **kwargs):
        """
        Function definition remains unchanged for backwards compatibility.
        Be aware that tokenizer, max_new_takens, and kwargs are ignored.
        """
        import torch

        from ..torch.hf_moondream import extract_question

        self._setup_caches()
        prompt_extracted = extract_question(prompt)
        if prompt_extracted is not None:
            answer = self.query(
                image_embeds, prompt_extracted, reasoning=False
            )["answer"]
        else:
            if not hasattr(image_embeds, "pos"):
                image_embeds = self.encode_image(image_embeds)
            self.model.load_encoded_image(image_embeds)
            prompt_tokens = torch.tensor(
                [self.model.tokenizer.encode(prompt).ids],
                device=self.device,
            )
            settings = {"max_tokens": max_new_tokens}
            if "settings" in kwargs:
                settings.update(kwargs["settings"])
            answer = "".join(
                self.model._generate_answer(
                    prompt_tokens, image_embeds.pos, settings=settings
                )
            )

        return [answer]


def from_pretrained(
    model_id,
    revision=None,
    cache_dir=None,
    torch_dtype=None,
    device=None,
    **kwargs,
):
    """Module-level convenience wrapper around
    :meth:`Moondream.from_pretrained`."""
    return Moondream.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device=device,
        **kwargs,
    )


def _find_weights(repo_dir):
    """Locate checkpoint files in a downloaded repo directory.

    Prefers a single ``model.safetensors``, then sharded
    ``model-<n>-of-<m>.safetensors`` files, then ``pytorch_model.bin``,
    then any other ``.safetensors`` / ``.pt`` files.
    """
    import glob
    import os

    single = os.path.join(repo_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    shards = sorted(
        glob.glob(os.path.join(repo_dir, "model-*-of-*.safetensors"))
    )
    if shards:
        return shards

    bin_file = os.path.join(repo_dir, "pytorch_model.bin")
    if os.path.exists(bin_file):
        return [bin_file]

    candidates = sorted(
        glob.glob(os.path.join(repo_dir, "*.safetensors"))
        + glob.glob(os.path.join(repo_dir, "*.pt"))
    )
    if candidates:
        return candidates

    raise ValueError(
        f"No checkpoint found in {repo_dir!r}: expected a .safetensors, "
        "pytorch_model.bin, or .pt file."
    )


def _has_moe_blocks(weights_file):
    """Return True if the checkpoint contains MoE text blocks."""
    from ..torch.weights import safetensors_open

    if not weights_file.endswith(".safetensors"):
        return False
    try:
        with safetensors_open(weights_file) as get_tensor:
            keys = get_tensor.keys()
    except Exception:
        return False
    return any(
        "mlp.gate.weight" in key or "mlp.experts.weight" in key for key in keys
    )