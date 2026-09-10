"""Strict management input; no mass-assigned roles or model embeddings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProfileInput(StrictInput):
    name: str = Field(min_length=1, max_length=150)
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):
        return value.lower()


class CreateUserInput(ProfileInput):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    images: list[str] = Field(min_length=2, max_length=10)
    consent: Literal[True]

    @field_validator("images")
    @classmethod
    def bounded_images(cls, images):
        if any(len(image) > 11_200_000 for image in images) or sum(map(len, images)) > 30_000_000:
            raise ValueError("Enrollment images exceed request budget")
        return images


class UpdateUserInput(StrictInput):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    email: str | None = Field(
        default=None, min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
    )


class RoleInput(StrictInput):
    role: Literal["USER", "ADMIN"]


class StatusInput(StrictInput):
    status: Literal["ACTIVE", "DISABLED"]
