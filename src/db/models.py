from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    Double,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Geoname(Base):
    """A populated place record from GeoNames (feature_class = 'P')."""

    __tablename__ = "geoname"

    geonameid: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    asciiname: Mapped[str] = mapped_column(Text, nullable=False)
    feature_class: Mapped[str] = mapped_column(String(1), nullable=False, default="P")
    feature_code: Mapped[str | None] = mapped_column(String(10))
    country_code: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    admin1_code: Mapped[str | None] = mapped_column(String(20))
    admin2_code: Mapped[str | None] = mapped_column(String(80))
    population: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    latitude: Mapped[float | None] = mapped_column(Double)
    longitude: Mapped[float | None] = mapped_column(Double)
    timezone: Mapped[str | None] = mapped_column(String(40))
    modification_date: Mapped[Date | None] = mapped_column(Date)

    alternate_names: Mapped[list["AlternateName"]] = relationship(
        back_populates="geoname", cascade="all, delete-orphan"
    )


class AlternateName(Base):
    """An alternate name variant for a GeoNames place."""

    __tablename__ = "alternate_name"

    alternate_name_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False
    )
    geonameid: Mapped[int] = mapped_column(
        Integer, ForeignKey("geoname.geonameid", ondelete="CASCADE"), nullable=False, index=True
    )
    isolanguage: Mapped[str | None] = mapped_column(String(7), index=True)
    alternate_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_preferred_name: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_short_name: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_colloquial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_historic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # 'geonames' for original data, 'generated' for our enrichments
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="geonames")

    geoname: Mapped["Geoname"] = relationship(back_populates="alternate_names")
