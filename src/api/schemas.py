"""Pydantic schemas for the geo-search API."""
from pydantic import BaseModel, Field


class CityResult(BaseModel):
    geoname_id:  int
    ascii_name:  str
    country_code: str
    admin1_code: str | None
    latitude:    float
    longitude:   float
    population:  int
    score:       float


class SearchResponse(BaseModel):
    query:   str
    results: list[CityResult]
    total:   int = Field(description="Number of results returned")
