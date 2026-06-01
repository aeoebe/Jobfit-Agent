from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter


@dataclass
class ProfileDocument:
    source: str
    content: str


def load_profile_documents(paths: list[Path]) -> list[ProfileDocument]:
    documents = []

    for path in paths:
        if not path.exists():
            continue
        documents.append(
            ProfileDocument(
                source=path.name,
                content=path.read_text(encoding="utf-8"),
            )
        )

    return documents


def split_profile_document(document: ProfileDocument) -> list[ProfileDocument]:
    chunks = []
    current_heading = ""

    for block in document.content.split("\n\n"):
        clean_block = block.strip()
        if not clean_block:
            continue

        if clean_block.startswith("#"):
            current_heading = clean_block.strip("# ").strip()
            continue

        content = f"{current_heading}\n{clean_block}".strip()
        chunks.append(ProfileDocument(source=document.source, content=content))

    return chunks

# def split_profile_document(document: ProfileDocument) -> list[ProfileDocument]:
#     splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on = [
#             ("#", "section"),
#             ("##", "subsection")
#         ]
#     )

#     docs = splitter.split_text(document.content)

#     chunks = []

#     for doc in docs:
#         section = doc.metadata.get("section", "")
#         subsection = doc.metadata.get("subsection", "")

#         content = "\n".join(
#             part
#             for part in [
#                 section,
#                 subsection,
#                 doc.page_content,
#             ]
#             if part
#         )

#         chunks.append(
#             ProfileDocument(
#                 source=document.source,
#                 content=content,
#             )
#         ) 
    
#     return chunks


def split_profile_documents(documents: list[ProfileDocument]) -> list[ProfileDocument]:
    chunks = []
    for document in documents:
        chunks.extend(split_profile_document(document))
    return chunks

