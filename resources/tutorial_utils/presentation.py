"""Plain, fully visible lesson text for notebook sources and rich outputs."""
import re
from html import escape

from bs4 import BeautifulSoup


_EMOJI = re.compile(r"[\U0001f000-\U0001faff\u2600-\u27bf\u23f1\u25b6\ufe0f]")
_LITERAL = re.compile(r"(```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$)")


def interpretation_html(text):
    """Render one plain, escaped result sentence immediately after a table."""
    return f'<p class="table-reading">{escape(text)}</p>'


def takeaway_html(text):
    """Render the one-sentence, plain-language 'Interpretation:' line that closes a result."""
    return f'<p class="result-interpretation">Interpretation: {escape(text)}</p>'


def show_interpretation(text):
    """Display the 'Interpretation:' line after a table, graph, or other result."""
    from IPython.display import HTML, display
    display(HTML(takeaway_html(text)))


def _plain_words(text):
    return _EMOJI.sub("", text)


def plain_html(text):
    """Expose disclosure contents and remove decorative emphasis, keeping tables."""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all("style"):
        tag.decompose()
    for tag in soup.find_all(True):
        tag.attrs.pop("style", None)
        if tag.name == "details":
            tag.name = "section"
            tag.attrs.pop("open", None)
        elif tag.name == "summary":
            tag.name = "h4"
        elif tag.name == "blockquote":
            tag.name = "div"
        elif tag.name in {"b", "strong", "em", "i", "mark"}:
            tag.unwrap()
    for node in list(soup.find_all(string=True)):
        if node.find_parent(["pre", "code", "script"]):
            continue
        node.replace_with(_plain_words(str(node)))
    return str(soup)


def plain_markdown(text):
    """Remove prose styling while preserving code, math, links, and identifiers."""
    parts = _LITERAL.split(text)
    for index in range(0, len(parts), 2):
        part = _plain_words(parts[index])
        part = re.sub(r"<details\b[^>]*>\s*<summary\b[^>]*>(.*?)</summary>",
                      r"\n\n#### \1\n\n", part, flags=re.I | re.S)
        part = re.sub(r"</details\s*>", "\n", part, flags=re.I)
        part = re.sub(r"</?(?:strong|b|em|i|mark)\b[^>]*>", "", part, flags=re.I)
        part = re.sub(r"<(\/?)blockquote\b[^>]*>", r"<\1div>", part, flags=re.I)
        part = part.replace("**", "").replace("__", "")
        part = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", part)
        part = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", part)
        parts[index] = re.sub(r"^[ \t]*> ?", "", part, flags=re.M)
    return "".join(parts)


def install_plain_display():
    """Apply the same display policy in fresh or shared notebook kernels."""
    from IPython import get_ipython
    from IPython.display import HTML, Markdown
    from pandas.io.formats.style import Styler

    shell = get_ipython()
    if shell is None:
        return
    formatters = shell.display_formatter.formatters
    formatters["text/html"].for_type(HTML, lambda value: plain_html(value.data))
    formatters["text/html"].for_type(Styler, lambda value: plain_html(value.to_html()))
    formatters["text/markdown"].for_type(Markdown, lambda value: plain_markdown(value.data))
