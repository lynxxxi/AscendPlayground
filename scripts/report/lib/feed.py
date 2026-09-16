#!/usr/bin/env python3
"""极简 RSS 2.0 / Atom 1.0 解析器（标准库实现，替代 feedparser）。"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

from .util import parse_datetime, strip_html

ATOM_NS = "{http://www.w3.org/2005/Atom}"
RSS_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"


@dataclass
class FeedEntry:
    title: str = ""
    link: str = ""
    summary: str = ""
    published: Optional[object] = None
    author: str = ""
    tags: list[str] = field(default_factory=list)
    guid: str = ""


@dataclass
class Feed:
    title: str = ""
    link: str = ""
    entries: list[FeedEntry] = field(default_factory=list)


def _text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _first(node: ET.Element, *names: str) -> Optional[ET.Element]:
    for name in names:
        found = node.find(name)
        if found is not None:
            return found
    return None


def _atom_link(node: ET.Element) -> str:
    fallback = ""
    for link in node.findall(f"{ATOM_NS}link"):
        rel = link.get("rel", "alternate")
        href = (link.get("href") or "").strip()
        if not href:
            continue
        if rel == "alternate":
            return href
        if not fallback:
            fallback = href
    return fallback


def _rss_link(node: ET.Element) -> str:
    link = _text(node.find("link"))
    if link:
        return link
    guid = node.find("guid")
    if guid is not None and (guid.get("isPermaLink", "true").lower() != "false"):
        return _text(guid)
    # 部分中文源把链接塞在 atom:link
    atom = node.find(f"{ATOM_NS}link")
    if atom is not None:
        return (atom.get("href") or "").strip()
    return ""


def _entry_published(node: ET.Element) -> Optional[object]:
    candidates = (
        f"{ATOM_NS}published",
        f"{ATOM_NS}updated",
        "pubDate",
        "published",
        "updated",
        f"{DC_NS}date",
        "date",
    )
    for name in candidates:
        value = _text(node.find(name))
        if value:
            parsed = parse_datetime(value)
            if parsed:
                return parsed
    # arXiv / 部分 Atom 把时间放在 <published>，兜底扫一遍
    for child in node:
        tag = child.tag.split("}")[-1].lower()
        if tag in {"published", "updated", "date", "pubdate", "created"}:
            parsed = parse_datetime(_text(child))
            if parsed:
                return parsed
    return None


def _entry_summary(node: ET.Element) -> str:
    for name in (
        f"{ATOM_NS}summary",
        f"{ATOM_NS}content",
        "description",
        f"{RSS_CONTENT_NS}encoded",
        "summary",
        "content",
    ):
        raw = _text(node.find(name))
        if raw:
            return strip_html(raw)
    return ""


def _entry_title(node: ET.Element) -> str:
    for name in (f"{ATOM_NS}title", "title"):
        value = _text(node.find(name))
        if value:
            return strip_html(value)
    return ""


def _entry_author(node: ET.Element) -> str:
    author = _first(node, f"{ATOM_NS}author", "author", f"{DC_NS}creator")
    if author is None:
        return ""
    name = _text(author.find(f"{ATOM_NS}name"))
    return strip_html(name or _text(author))


def _entry_tags(node: ET.Element) -> list[str]:
    tags: list[str] = []
    for category in node.findall("category"):
        value = (category.get("term") or _text(category)).strip()
        if value:
            tags.append(value)
    for category in node.findall(f"{ATOM_NS}category"):
        value = (category.get("term") or _text(category)).strip()
        if value:
            tags.append(value)
    return tags


def parse_feed(xml_text: str) -> Feed:
    """解析 RSS/Atom 文本；无法解析时返回空 Feed 而不是抛异常。"""
    feed = Feed()
    if not xml_text or not xml_text.strip():
        return feed
    # 去掉 BOM 与 XML 声明前的空白，并清理非法控制字符
    cleaned = xml_text.lstrip("\ufeff \r\n\t")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        # 尝试截断到最后一个可闭合标签以求部分可用
        try:
            root = ET.fromstring(cleaned + "</feed>")
        except ET.ParseError:
            return feed

    tag = root.tag.split("}")[-1].lower()
    channel = root.find("channel") if tag == "rss" else None
    container = channel if channel is not None else root

    feed.title = _entry_title(container) if container is not root else _entry_title(root)
    if tag == "feed":
        feed.title = _text(root.find(f"{ATOM_NS}title")) or feed.title
        feed.link = _atom_link(root)
        nodes = root.findall(f"{ATOM_NS}entry") or root.findall("entry")
    else:
        feed.link = _rss_link(container) if container is not None else ""
        nodes = container.findall("item") if container is not None else root.findall(".//item")

    for node in nodes:
        entry = FeedEntry(
            title=_entry_title(node),
            link=_atom_link(node) if tag == "feed" else _rss_link(node),
            summary=_entry_summary(node),
            published=_entry_published(node),
            author=_entry_author(node),
            tags=_entry_tags(node),
            guid=_text(node.find("guid")) or _text(node.find(f"{ATOM_NS}id")),
        )
        if entry.title or entry.link:
            feed.entries.append(entry)
    return feed
