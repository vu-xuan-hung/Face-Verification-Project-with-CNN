"""Explicit local account enrollment and revocation; never derives roles from files."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    from vshield.api import database

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    enrollment = commands.add_parser(
        "enroll", help="Enroll consented photos; restart backend after"
    )
    enrollment.add_argument("--username", required=True)
    enrollment.add_argument("--role", choices=["USER", "ADMIN", "user", "admin"], default="USER")
    enrollment.add_argument("--images", type=Path, nargs="+", required=True)
    enrollment.add_argument(
        "--consent",
        action="store_true",
        help="Confirm subject consent and correct account/photo association",
    )
    disable = commands.add_parser(
        "disable", help="Disable account and revoke all sessions immediately"
    )
    disable.add_argument("--username", required=True)
    role = commands.add_parser("role", help="Change role and revoke all sessions immediately")
    role.add_argument("--username", required=True)
    role.add_argument("--role", required=True, choices=["USER", "ADMIN", "user", "admin"])
    bootstrap = commands.add_parser(
        "bootstrap-super-admin", help="Explicitly designate the first enrolled owner; only once"
    )
    bootstrap.add_argument("--username", required=True)
    commands.add_parser("list", help="List accounts without biometric templates")
    commands.add_parser("sync", help="Reconcile local Chroma; stop backend first, then restart")
    args = parser.parse_args()
    database.init_db()
    try:
        if args.command == "enroll":
            from vshield.services.enrollment import enroll

            result = enroll(
                PROJECT_ROOT / "data" / "authorization",
                args.username,
                args.images,
                role=args.role,
                consent=args.consent,
            )
        elif args.command == "sync":
            from vshield.core.managed_identity_index import ManagedIdentityIndex

            root = PROJECT_ROOT / "data" / "authorization"
            count = ManagedIdentityIndex(root).refresh()
            result = {"synced_templates": count, "restart_required": True}
        elif args.command == "bootstrap-super-admin":
            from vshield.api.bootstrap import bootstrap_super_admin

            result = bootstrap_super_admin(args.username)
        elif args.command == "list":
            result = [
                {k: row[k] for k in ("id", "username", "name", "email", "role", "status")}
                for row in database.list_accounts()
            ]
        else:
            database.change_account(
                args.username, role=getattr(args, "role", None), disable=args.command == "disable"
            )
            result = {
                "username": args.username,
                "sessions_revoked": True,
                "restart_for_vector_cleanup": args.command == "disable",
            }
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
