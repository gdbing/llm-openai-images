# llm-openai-images

[![PyPI](https://img.shields.io/pypi/v/llm-openai-images.svg)](https://pypi.org/project/llm-openai-images/)

An [LLM](https://llm.datasette.io/) plugin providing access to `gpt-image-1`, OpenAI's latest image generation and editing model.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-openai-images
```

## API Key Setup

You will need an OpenAI API key to use this plugin. Set it using:

```bash
llm keys set openai
```
Enter your key when prompted. You can obtain a key from the [OpenAI Platform](https://platform.openai.com/api-keys).

## Understanding the Output (Base64)

The plugin outputs raw base64 encoded data directly from the OpenAI API. This provides the unmodified data but requires an extra step to view or save.

You typically need to pipe the output through `base64 --decode` (available on most Linux/macOS systems) and redirect it to a file:

```bash
llm -m openai/gpt-image-1 "Prompt..." | base64 --decode > my_image.png
```

## Usage

This plugin adds the `openai/gpt-image-1` model to LLM.

### Basic Image Generation

To generate an image from a text prompt:

```bash
llm -m openai/gpt-image-1 "A cat wearing sunglasses, riding a skateboard" \
  | base64 --decode > cat_skateboard.png
```

This command generates the image based on the prompt and saves the decoded PNG data to `cat_skateboard.png`.

### Generation Options

You can control the image size and quality using options (`-o`):

*   `-o size <value>`: Set the image dimensions.
    *   `square` (default): 1024x1024
    *   `portrait`: 1024x1792
    *   `landscape`: 1792x1024
*   `-o quality <value>`: Set the image quality/detail.
    *   `high`: ~$0.26
    *   `medium` (default): ~$0.06
    *   `low`: ~$0.015

Example with options:

```bash
llm -m openai/gpt-image-1 "Impressionist painting of a harbor at sunset" \
  -o size landscape -o quality high \
  | base64 --decode > harbor_sunset_hd_landscape.png
```

### Editing an Image

To edit an existing image, provide the image file as an attachment using the `-a` or `--attach` flag. The prompt should describe the desired *changes* or additions to the image.

```bash
# First, generate an image or use an existing one (e.g., cat_skateboard.png from above)

# Now, edit it:
llm -m openai/gpt-image-1 "Add a small blue bird perched on the cat's head" \
  -a cat_skateboard.png \
  | base64 --decode > cat_skateboard_with_bird.png
```

### Combining Multiple Images

```bash
llm -m openai/gpt-image-1 "A photo of me dressed in these pants and top" \
  -a maison-martin-margiela-ss16-blouse.jpg \
  -a dior-homme-19cm-mij.jpg \
  -a me.jpg \
  | base64 --decode > my_fabulous_self.png
```

## Development

To set up this plugin locally, first checkout the code. Then install it in editable mode:

```bash
cd llm-openai-images
llm install -e .
```

See the [LLM plugin documentation](https://llm.datasette.io/en/stable/plugins/) for more details on plugin development.