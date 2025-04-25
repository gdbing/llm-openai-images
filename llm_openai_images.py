from enum import Enum
import openai
from llm import hookimpl, KeyModel, AsyncKeyModel, Prompt, Response, Options
from llm.utils import simplify_usage_dict
from pydantic import Field, create_model
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
    auto = "auto"
    low = "low"
    medium = "medium"
    high = "high"

class SizeEnum(str, Enum):
    auto = "auto"
    square = "1024x1024"
    portrait = "1024x1536"
    landscape = "1536x1024"

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
            "1024x1024 (default), "
            "1536x1024 (landscape), "
            "1024x1536 (portrait)"
        ),
        default=None,
    )

class OpenAIImageModel(KeyModel):
    """
    Sync model for OpenAI image generation/editing (gpt-image-1).
    """
    needs_key = "openai"
    key_env_var = "OPENAI_API_KEY"
    can_stream = False
    supports_schema = False
    attachment_types = {"image/png", "image/jpeg", "image/webp"}

    # no custom options for now
    Options = create_model("ImageOptions", __base__=(Options,))

    def __init__(self, model_name: str):
        self.model_id = f"openai/{model_name}"
        self.model_name = model_name

    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation,
        key: str | None,
    ) -> Iterator[str]:
        client = openai.OpenAI(api_key=self.get_key(key))
        imgs = [
            a for a in (prompt.attachments or [])
            if a.resolve_type() in self.attachment_types
        ]

        if not prompt.prompt:
            raise ValueError("Prompt text is required for image generation/editing.")

        result_b64 = None
        usage = None

        if not imgs:
            # generate
            api_response = client.images.generate(
                model=self.model_name,
                prompt=prompt.prompt,
                n=1,
                size="1536x1024", #"1024x1024",
                quality="medium",
                # output_format="png",
                moderation="low",
            )
        else:
            # edit, only first image supported here
            img = imgs[0]
            if not img.path:
                raise ValueError(f"Attachment must be a local file: {img!r}")
            with open(img.path, "rb") as f:
                api_response = client.images.edit(
                    model=self.model_name,
                    image=f,
                    prompt=prompt.prompt,
                    n=1,
                    size="1536x1024", #"1024x1024",
                    quality="medium",
                    # output_format="png",
                    moderation="low",
                )

        # pull out the base64 result
        result_b64 = api_response.data[0].b64_json
        if hasattr(api_response, "usage") and api_response.usage:
            usage = api_response.usage

        # store the JSON + usage
        response.response_json = {"result_b64_json": result_b64}
        if usage:
            _set_usage(response, usage)

        yield result_b64


class AsyncOpenAIImageModel(AsyncKeyModel):
    """
    Async model for OpenAI image generation/editing (gpt-image-1).
    """
    needs_key = "openai"
    key_env_var = "OPENAI_API_KEY"
    can_stream = False
    supports_schema = False
    attachment_types = {"image/png", "image/jpeg", "image/webp"}

    Options = create_model("ImageOptions", __base__=(Options,))

    def __init__(self, model_name: str):
        self.model_id = f"openai/{model_name}"
        self.model_name = model_name

    async def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation,
        key: str | None,
    ) -> AsyncGenerator[str, None]:
        client = openai.AsyncOpenAI(api_key=self.get_key(key))
        imgs = [
            a for a in (prompt.attachments or [])
            if a.resolve_type() in self.attachment_types
        ]

        if not prompt.prompt:
            raise ValueError("Prompt text is required for image generation/editing.")

        if not imgs:
            api_response = await client.images.generate(
                model=self.model_name,
                prompt=prompt.prompt,
                n=1,
                size="1536x1024", #"1024x1024",
                quality="medium",
                # output_format="png",
                moderation="low",
            )
        else:
            img = imgs[0]
            if not img.path:
                raise ValueError(f"Attachment must be a local file: {img!r}")
            with open(img.path, "rb") as f:
                api_response = await client.images.edit(
                    model=self.model_name,
                    image=f,
                    prompt=prompt.prompt,
                    n=1,
                    size="1536x1024", #"1024x1024",
                    quality="medium",
                    # output_format="png",
                    moderation="low",
                )

        result_b64 = api_response.data[0].b64_json
        if hasattr(api_response, "usage") and api_response.usage:
            _set_usage(response, api_response.usage)

        response.response_json = {"result_b64_json": result_b64}
        yield result_b64