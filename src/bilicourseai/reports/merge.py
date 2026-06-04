from __future__ import annotations

from bilicourseai.models import VideoReport


def merge_fetched_report(existing: VideoReport, fetched: VideoReport, part_page: int | None) -> VideoReport:
    existing.aid = fetched.aid
    existing.title = fetched.title
    existing.owner_name = fetched.owner_name
    existing.source_url = fetched.source_url

    if part_page is not None:
        fetched_by_page = {part.page: part for part in fetched.parts}
        fetched_part = fetched_by_page.get(part_page)
        if fetched_part is None:
            return existing

        merged_parts = []
        replaced = False
        for old_part in existing.parts:
            if old_part.page == part_page:
                fetched_part.blocks = old_part.blocks
                merged_parts.append(fetched_part)
                replaced = True
            else:
                merged_parts.append(old_part)
        if not replaced:
            merged_parts.append(fetched_part)
            merged_parts.sort(key=lambda part: part.page)
        existing.parts = merged_parts
        return existing

    existing_by_page = {part.page: part for part in existing.parts}
    merged_parts = []
    for fetched_part in fetched.parts:
        old_part = existing_by_page.get(fetched_part.page)
        if old_part is not None:
            merged_parts.append(old_part)
            continue
        merged_parts.append(fetched_part)
    existing.parts = merged_parts
    return existing
