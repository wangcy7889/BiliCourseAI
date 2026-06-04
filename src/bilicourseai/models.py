from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TranscriptLine(BaseModel):
    start: float
    end: float
    text: str


class SubtitleTrack(BaseModel):
    lan: str = ""
    lan_doc: str = ""
    is_ai: bool = False
    subtitle_url: str = ""


class VisualRequest(BaseModel):
    id: str
    part_page: int
    block_id: str
    timestamp: float
    reason: str
    prompt: str
    candidate_timestamps: list[float] = Field(default_factory=list)
    section_id: str | None = None


class FrameArtifact(BaseModel):
    request_id: str
    part_page: int
    block_id: str
    timestamp: float
    path: str
    source: str
    error: str | None = None


class VisualAnalysis(BaseModel):
    request_id: str
    part_page: int
    block_id: str
    timestamp: float
    image_path: str
    summary: str
    observed_elements: list[str] = Field(default_factory=list)
    learning_value: str = ""
    guidance: str = ""
    pitfalls: list[str] = Field(default_factory=list)
    confidence: str = "unknown"


class NoteSection(BaseModel):
    id: str
    title: str
    body: str
    visual_requests: list[VisualRequest] = Field(default_factory=list)
    frames: list[FrameArtifact] = Field(default_factory=list)
    visual_analyses: list[VisualAnalysis] = Field(default_factory=list)


class KnowledgeBlock(BaseModel):
    id: str
    title: str
    start: float
    end: float
    summary: str
    node_type: str = "leaf"
    status: str = "expanded"
    expandable: bool = False
    depth: int = 0
    source_part_page: int | None = None
    granularity: str = ""
    should_expand: bool = False
    expand_reason: str = ""
    boundary_confidence: str = ""
    split_hints: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    transcript: list[TranscriptLine] = Field(default_factory=list)
    children: list["KnowledgeBlock"] = Field(default_factory=list)
    sections: list[NoteSection] = Field(default_factory=list)
    visual_requests: list[VisualRequest] = Field(default_factory=list)
    frames: list[FrameArtifact] = Field(default_factory=list)
    visual_analyses: list[VisualAnalysis] = Field(default_factory=list)


class VideoPart(BaseModel):
    page: int
    cid: int
    title: str
    duration: int | None = None
    subtitle_tracks: list[SubtitleTrack] = Field(default_factory=list)
    selected_subtitle_track: SubtitleTrack | None = None
    subtitle_errors: list[str] = Field(default_factory=list)
    transcript: list[TranscriptLine] = Field(default_factory=list)
    blocks: list[KnowledgeBlock] = Field(default_factory=list)


class VideoReport(BaseModel):
    bvid: str
    aid: int | None = None
    title: str
    owner_name: str | None = None
    source_url: str
    parts: list[VideoPart]
    llm_notes: list[str] = Field(default_factory=list)


class ReportArtifacts(BaseModel):
    report_dir: Path
    json_path: Path
    html_path: Path
