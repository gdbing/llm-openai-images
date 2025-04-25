# llm-openai-images

[LLM](https://llm.datasette.io/) plugin for [gpt-image-1](https://platform.openai.com/docs/api-reference/images).

## Usage

To run a prompt against `gpt-image-1` do this:

```bash
llm -m openai/gpt-image-1 "SVG image of a pelican riding a bicycle" -o size landscape -o quality low|base64 --decode > ~/Pictures/pelican-svg.png
```

## Development

To set up this plugin locally, see [the documentation on plugins for llm](https://llm.datasette.io/en/stable/plugins/tutorial-model-plugin.html#installing-your-plugin-to-try-it-out)