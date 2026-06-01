from core.profile_loader import ProfileDocument, split_profile_document


def test_split_profile_document_keeps_heading_context():
    document = ProfileDocument(
        source="sample.md",
        content="# Resume\n\n## Skills\n\n- Python\n- RAG\n\n## Project\n\nBuilt a document QA system.",
    )

    chunks = split_profile_document(document)

    assert chunks[0].source == "sample.md"
    assert "Skills" in chunks[0].content
    assert "Python" in chunks[0].content
    assert "Project" in chunks[1].content

