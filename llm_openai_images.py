from enum import Enum
import openai
from llm import hookimpl, KeyModel, AsyncKeyModel, Prompt, Response, Options
from llm.utils import simplify_usage_dict
from pydantic import Field
from typing import Iterator, AsyncGenerator, Optional


def _set_usage(response: Response, usage):
    if not usage:
        return
    # if it's a Pydantic model
    if not isinstance(usage, dict):
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        else:
            return
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    # drop the raw fields
    usage.pop("input_tokens", None)
    usage.pop("output_tokens", None)
    usage.pop("total_tokens", None)
    response.set_usage(
        input=input_tokens,
        output=output_tokens,
        details=simplify_usage_dict(usage),
    )


@hookimpl
def register_models(register):
    register(
        OpenAIImageModel("gpt-image-1"),
        AsyncOpenAIImageModel("gpt-image-1"),
    )

class QualityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class SizeEnum(str, Enum):
    square = "square"
    portrait = "portrait"
    landscape = "landscape"

    @property
    def dimensions(self) -> str:
        return {
            SizeEnum.square: "1024x1024",
            SizeEnum.portrait: "1024x1536",
            SizeEnum.landscape: "1536x1024",
        }[self]

class ImageOptions(Options):
    quality: Optional[QualityEnum] = Field(
        description=(
            "The quality of the image that will be generated."
            "high, medium and low are supported for gpt-image-1."
        ),
        default=None,
    )
    size: Optional[SizeEnum] = Field(
        description=(
            "The size of the generated images. One of "
            "square (1024x1024), "
            "landscape (1536x1024), "
            "portrait (1024x1536)"
        ),
        default=None,
    )

class _BaseOpenAIImageModel(KeyModel):
    needs_key = "openai"
    key_env_var = "OPENAI_API_KEY"
    can_stream = False
    supports_schema = False
    attachment_types = {"image/png", "image/jpeg", "image/webp"}

    # Assign the pre-defined Options class
    Options = ImageOptions

    def __init__(self, model_name: str):
        self.model_id = f"openai/{model_name}"
        self.model_name = model_name

    def __str__(self):
        return f"OpenAI: {self.model_id}"
        
    def _build_api_kwargs(self, prompt: Prompt) -> dict:
        """Build the dictionary of arguments for the OpenAI API call."""
        if not prompt.prompt:
            raise ValueError("Prompt text is required for image generation/editing.")

        # Access options from the prompt object
        size = (prompt.options.size or SizeEnum.square).dimensions
        quality = (prompt.options.quality or QualityEnum.medium).value

        kwargs = {
            "model": self.model_name,
            "prompt": prompt.prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            # output_format="png",
            # "moderation": "low", # only used by generate, rejected by edit
        }

        return kwargs

class OpenAIImageModel(_BaseOpenAIImageModel, KeyModel):
    """
    Sync model for OpenAI image generation/editing (gpt-image-1).
    """

    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation,
        key: str | None,
    ) -> Iterator[str]:
        client = openai.OpenAI(api_key=self.get_key(key))
        kwargs = self._build_api_kwargs(prompt)

        imgs = [
            a for a in (prompt.attachments or [])
            if a.resolve_type() in self.attachment_types
        ]

        result_b64 = None
        usage = None

        if not imgs:
            api_response = client.images.generate(**kwargs)
        else:
            for img in imgs:
                if not img.path:
                    raise ValueError(f"Attachment must be a local file for editing: {img!r}")

            files = [open(img.path, "rb") for img in imgs]

            try:
                api_response = client.images.edit(
                image=files[0] if len(files) == 1 else files,
                    # mask=
                    **kwargs
                )
            finally:
                for f in files:
                    f.close()

        # pull out the base64 result
        result_b64 = api_response.data[0].b64_json
        if hasattr(api_response, "usage") and api_response.usage:
            usage = api_response.usage

        # store the JSON + usage
        response.response_json = {"result_b64_json": result_b64}
        if usage:
            _set_usage(response, usage)

        yield result_b64


class AsyncOpenAIImageModel(_BaseOpenAIImageModel, AsyncKeyModel):
    """
    Async model for OpenAI image generation/editing (gpt-image-1).
    """

    async def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation,
        key: str | None,
    ) -> AsyncGenerator[str, None]:
        client = openai.AsyncOpenAI(api_key=self.get_key(key))
        kwargs = self._build_api_kwargs(prompt)

        imgs = [
            a for a in (prompt.attachments or [])
            if a.resolve_type() in self.attachment_types
        ]

        if not imgs:
            api_response = await client.images.generate(**kwargs)
        else:
            for img in imgs:
                if not img.path:
                    raise ValueError(f"Attachment must be a local file for editing: {img!r}")

            files = [open(img.path, "rb") for img in imgs]

            try:
                api_response = client.images.edit(
                image=files[0] if len(files) == 1 else files,
                    # mask=
                    **kwargs
                )
            finally:
                for f in files:
                    f.close()

        result_b64 = api_response.data[0].b64_json
        if hasattr(api_response, "usage") and api_response.usage:
            _set_usage(response, api_response.usage)

        response.response_json = {"result_b64_json": result_b64}
        yield result_b64