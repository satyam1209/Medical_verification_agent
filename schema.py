"""SQLAlchemy models for the medical evidence database."""

from sqlalchemy import String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MedicalRecord(Base):
    __tablename__ = "medical_records"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    publish_date: Mapped[str | None] = mapped_column(String, nullable=True)
    last_updated: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingestion_date: Mapped[str] = mapped_column(String)

    def __repr__(self):
        return f"<MedicalRecord(id={self.id!r}, source={self.source!r}, type={self.type!r})>"