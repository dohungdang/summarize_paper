from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func
from database import Base

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(20), nullable=False, default="text")  # text / pdf
    paper_id = Column(String(100), nullable=True)

    input_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)

    length = Column(String(10), nullable=False, default="medium")
    model_name = Column(String(100), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<Summary id={self.id} source={self.source_type} length={self.length}>"
