"""Canonical database roles; normalization is for trusted local/migration inputs."""

from enum import Enum


class Role(str, Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    USER = "USER"


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    DELETED = "DELETED"


ROLES = frozenset(role.value for role in Role)


def normalize_role(value):
    if isinstance(value, Role):
        return value.value
    if not isinstance(value, str) or value.upper() not in ROLES:
        raise ValueError("Invalid role")
    return value.upper()
