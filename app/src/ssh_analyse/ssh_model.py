from pydantic import BaseModel, Field

class TableInfo(BaseModel):
  """Information regarding a structured table."""
  table_name: str = Field(..., description="table_name (must be underscores and NO spaces)")
  table_summary: str = Field(..., description="short, concise summary/caption of the table")