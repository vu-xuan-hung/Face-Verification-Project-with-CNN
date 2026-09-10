"""User administration routes: role and managed-account scope enforced independently of UI."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from vshield.api import sessions, user_store
from vshield.api.user_schemas import CreateUserInput, RoleInput, StatusInput, UpdateUserInput
from vshield.core.embedder import EmbeddingError
from vshield.core.face_preprocessor import FacePreprocessingError
from vshield.core.identity_index_support import IdentityIndexError

router = APIRouter()


def execute(operation):
    try:
        return operation()
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, "Account not found") from exc
    except FacePreprocessingError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise HTTPException(
            409,
            "Account or face conflicts with an existing registration"
            if isinstance(exc, sqlite3.IntegrityError)
            else str(exc),
        ) from exc
    except (EmbeddingError, IdentityIndexError, sqlite3.OperationalError, OSError) as exc:
        raise HTTPException(503, "Enrollment or account storage unavailable") from exc


def create(data, request, actor, role):
    # Imported lazily to keep the application transport's existing decode contract.
    from vshield.api.app import decode_image

    service = request.app.state.user_management_service
    if service is None:
        raise HTTPException(503, "Enrollment service unavailable")
    try:
        images = [decode_image(image) for image in data.images]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return execute(
        lambda: service.create(actor["id"], data.model_dump(exclude={"images"}), images, role)
    )


@router.post("/users", status_code=201)
def create_user(data: CreateUserInput, request: Request, actor=Depends(sessions.require_admin)):
    return create(data, request, actor, "USER")


@router.post("/admins", status_code=201)
def create_admin(
    data: CreateUserInput, request: Request, actor=Depends(sessions.require_super_admin)
):
    return create(data, request, actor, "ADMIN")


@router.get("/users")
def list_users(actor=Depends(sessions.require_admin)):
    return execute(lambda: user_store.list_users(actor["id"]))


@router.patch("/users/{user_id}")
def update_user(user_id: int, data: UpdateUserInput, actor=Depends(sessions.require_admin)):
    if not data.model_fields_set or any(
        getattr(data, key) is None for key in data.model_fields_set
    ):
        raise HTTPException(422, "Provide non-null name and/or email")
    return execute(
        lambda: user_store.mutate_user(actor["id"], user_id, **data.model_dump(exclude_unset=True))
    )


@router.patch("/users/{user_id}/role")
def change_role(user_id: int, data: RoleInput, actor=Depends(sessions.require_super_admin)):
    return execute(lambda: user_store.mutate_user(actor["id"], user_id, role=data.role))


@router.patch("/users/{user_id}/status")
def change_status(user_id: int, data: StatusInput, actor=Depends(sessions.require_admin)):
    return execute(lambda: user_store.mutate_user(actor["id"], user_id, status=data.status))


@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, actor=Depends(sessions.require_admin)):
    execute(lambda: user_store.mutate_user(actor["id"], user_id, delete=True))
    return Response(status_code=204)
