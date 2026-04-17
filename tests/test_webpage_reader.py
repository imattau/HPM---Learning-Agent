from hpm_ai_v2.utils.text_fetcher import fetch_passages, strip_html

def test_strip_html_removes_tags():
    html = "<p>Hello <b>world</b></p>"
    result = strip_html(html)
    assert "Hello world" in result
    assert "<" not in result

def test_strip_html_removes_scripts():
    html = "<script>alert('x')</script><p>Content</p>"
    result = strip_html(html)
    assert "alert" not in result
    assert "Content" in result

def test_fetch_passages_splits_paragraphs():
    text = "First sentence. Second sentence.\n\nNew paragraph here."
    passages = fetch_passages(text=text, min_length=5)
    assert len(passages) >= 2
    assert all(isinstance(p, str) for p in passages)

def test_fetch_passages_filters_short():
    text = "Hi.\n\nThis is a longer and more meaningful passage."
    passages = fetch_passages(text=text, min_length=20)
    assert len(passages) == 1
    assert "meaningful" in passages[0]
