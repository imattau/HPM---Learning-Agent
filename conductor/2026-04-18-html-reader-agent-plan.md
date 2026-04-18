# Plan: HtmlReaderAgent – Fractal HTML Understanding

**Goal:** Implement a specialized `HtmlReaderAgent` that parses raw HTML into a hierarchical HFN node structure, representing tags, attributes, and text as a fractal graph.

**Strategic Intent:** This agent will provide a deeper structural understanding of web resources by treating every syntactic element as an HFN node, enabling more sophisticated information retrieval and synthesis compared to plain-text stripping.

## Changes

### 1. HtmlReaderAgent Implementation (`hpm_ai_v2/agents/html_reader_agent.py`)
- [x] Implement `HtmlReaderAgent(ReaderAgent)`:
    - [x] `__init__`: Accepts `config`, `forest`, `web_agent`, etc.
    - [x] `ingest_html(html, url, webpage_node=None) -> HFN`: Entry point for HTML ingestion.
    - [x] `_build_attribute_node(name, value) -> HFN`: Creates an HFN node for an HTML attribute.
    - [x] `_build_element_node(tag, attrs, children) -> HFN`: Creates an HFN node for an HTML element.
    - [x] `_ingest_text_fragment(text) -> HFN`: Wrapper around `build_sentence_node` to handle inline text content.
- [x] Implement `_HtmlToHFNParser(HTMLParser)`:
    - [x] `handle_starttag(tag, attrs)`: Create element node and add to stack.
    - [x] `handle_endtag(tag)`: Pop from stack.
    - [x] `handle_data(data)`: Ingest text and add to parent element.

### 2. Mixin HTML Processing (Optional Refactoring)
- Included in `HtmlReaderAgent` for now.

### 3. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_web2_html_fractal.py`)
- [x] Implement verification script:
    1. Define a simple HTML string (e.g., `<div class="container"><h1>Title</h1><p>Hello world</p></div>`).
    2. Initialize `HtmlReaderAgent` with a shared forest.
    3. Call `ingest_html`.
    4. Verify the forest contains:
        - `html_document` root node.
        - `html_element` nodes for `div`, `h1`, `p`.
        - `html_attribute` node for `class="container"`.
        - Word macros for `div`, `h1`, `p`, `class`, `container`.
        - `sentence` nodes for text content.
    5. Test basic structural queries.

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_web2_html_fractal.py`.
- [x] Verify that the HFN hierarchy correctly mirrors the HTML structure.
- [x] Confirm that word macros are shared between HTML structure and text content.
