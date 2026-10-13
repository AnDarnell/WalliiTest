"""Import a copied HSGuru Battlegrounds season leaderboard TSV into history JSON."""
import argparse
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "leaderboard_history.json"


def label_for(season_id):
    return f"bS{season_id}" if season_id <= 5 else f"S{season_id - 5}"


def clean_player(value):
    value = "".join(
        c for c in value
        if not unicodedata.category(c).startswith(("S", "C"))
        and not ("\ufe00" <= c <= "\ufe0f")
    )
    parts = value.split()
    # HSGuru prefixes can contain a country, region, and flag; battletags
    # themselves are a single word, so the final token is the player name.
    return parts[-1].strip() if parts else ""


def read_rows(lines):
    records = []
    current_rank = None
    current_fields = []
    for line in lines:
        match = re.match(r"^\s*(\d+)\s*\t(.*)$", line)
        if match:
            if current_rank is not None:
                records.append((current_rank, current_fields))
            current_rank = int(match.group(1))
            current_fields = match.group(2).split("\t")
            continue
        match = re.match(r"^\s*(\d+)\s+(.*)$", line)
        if match:
            if current_rank is not None:
                records.append((current_rank, current_fields))
            current_rank = int(match.group(1))
            current_fields = [match.group(2).strip()]
        elif current_rank is not None and line.strip():
            current_fields.extend(line.split("\t"))
    if current_rank is not None:
        records.append((current_rank, current_fields))

    rows = []
    for rank, fields in records:
        rating_index = next(
            (i for i, field in enumerate(fields) if re.fullmatch(r"\s*\d{1,6}\s*", field)),
            None,
        )
        if rating_index is None:
            raise ValueError(f"Could not find rating for rank {rank}: {fields!r}")
        player = clean_player(" ".join(fields[:rating_index]))
        if not player:
            print(f"Skipping rank {rank}: HSGuru row has no Battletag")
            continue
        rows.append({
            "rank": rank, "player": player.lower(),
            "rating": int(fields[rating_index].strip()),
        })
    if not rows:
        raise ValueError("No leaderboard rows found")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--region", required=True, choices=("EU", "US", "AP", "CN"))
    parser.add_argument("--season-id", type=int,
                        help="Required only when the file has no seasonId=... section headings")
    args = parser.parse_args()

    lines = args.file.read_text(encoding="utf-8-sig").splitlines()
    sections = []
    current_id = None
    current_lines = []
    for line in lines:
        heading = re.fullmatch(r"\s*seasonId\s*=\s*(\d+)\s*", line, re.IGNORECASE)
        if heading:
            if current_id is not None:
                sections.append((current_id, current_lines))
            current_id = int(heading.group(1))
            current_lines = []
        elif current_id is not None:
            current_lines.append(line)
    if current_id is not None:
        sections.append((current_id, current_lines))
    elif args.season_id is not None:
        sections.append((args.season_id, lines))
    else:
        parser.error("No seasonId headings found; pass --season-id for a single-season file")

    imports = []
    for season_id, section_lines in sections:
        try:
            rows = read_rows(section_lines)
        except ValueError as exc:
            raise ValueError(f"seasonId={season_id}: {exc}") from exc
        if len({row["rank"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate ranks in seasonId={season_id}")
        season = season_id if season_id <= 5 else season_id - 5
        imports.extend({
            "region": args.region,
            "season": season,
            "season_id": season_id,
            "season_label": label_for(season_id),
            **row,
        } for row in rows)

    existing = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else []
    existing = [
        row for row in existing
        if not (row.get("region", "").upper() == args.region
                and row.get("season_id") in {season_id for season_id, _ in sections})
    ]
    existing.extend(imports)
    HISTORY.write_text(json.dumps(existing, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"Imported {len(imports)} rows across {len(sections)} {args.region} seasons")


if __name__ == "__main__":
    main()
