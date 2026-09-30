"""Command-line interface for validated M2M-AGDA campaigns."""

from __future__ import annotations

import argparse
import json

from .config import expand_campaign, load_campaign
from .runner import campaign_manifest, run_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m2m-agda",
        description="Train the focused M2M-AGDA implementation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate a campaign and print its expanded manifest"
    )
    validate.add_argument("config", help="YAML campaign file")
    validate.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="non-negative offset added to each configured target accuracy",
    )

    run = subparsers.add_parser("run", help="execute a YAML campaign")
    run.add_argument("config", help="YAML campaign file")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--device", default=None, help="override training.device")
    run.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="non-negative offset added to each configured target accuracy",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print expanded runs without training or writing results",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        campaign = load_campaign(args.config)
        if args.command == "validate":
            manifest = campaign_manifest(campaign, margin=args.margin)
            print(
                json.dumps(
                    {
                        "campaign": campaign.name,
                        "runs": len(expand_campaign(campaign)),
                        "manifest": manifest,
                    },
                    indent=2,
                )
            )
            return
        run_campaign(
            campaign,
            output_dir=args.output_dir,
            device=args.device,
            margin=args.margin,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
