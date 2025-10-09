#!/usr/bin/env python3
"""
Sort overlapping city files between output_inside and output_outside.

This script identifies cities that appear in output_inside and moves their
corresponding files from output_outside to output_inside_additional.

Usage:
    python scripts/sort_overlapping_cities.py --data-type squares [--dry-run]
    python scripts/sort_overlapping_cities.py --data-type addresses [--dry-run]
"""

import argparse
import shutil
from pathlib import Path
from typing import Set, List, Tuple
import re


def extract_city_name_from_inside_file(filename: str, data_type: str) -> str:
    """
    Extract city name from output_inside filename.
    
    Pattern: umap_{data_type}_{City}_{number}_htwk_{type}.geojson
    Example: umap_squares_Aachen_87_htwk_hochburg.geojson -> Aachen
    Example: umap_squares_Bad_Homburg_123_htwk_prekaer.geojson -> Bad_Homburg
    
    Args:
        filename: Name of the file
        data_type: 'squares' or 'addresses'
        
    Returns:
        City name extracted from filename
    """
    prefix = f"umap_{data_type}_"
    if not filename.startswith(prefix):
        return None
    
    # Remove prefix and .geojson suffix
    rest = filename[len(prefix):]
    if rest.endswith('.geojson'):
        rest = rest[:-8]
    
    # Find the pattern: city name followed by _{number}_htwk_
    # We need to find where the number starts after the city name
    match = re.match(r'^(.+?)_(\d+)_htwk_', rest)
    if match:
        return match.group(1)
    
    return None


def extract_city_name_from_outside_file(filename: str, data_type: str) -> str:
    """
    Extract city name from output_outside filename.
    
    Pattern: umap_{data_type}_{City}_heating_shape.geojson
    Example: umap_squares_Aachen_heating_shape.geojson -> Aachen
    Example: umap_squares_Bad_Homburg_heating_shape.geojson -> Bad_Homburg
    
    Args:
        filename: Name of the file
        data_type: 'squares' or 'addresses'
        
    Returns:
        City name extracted from filename
    """
    prefix = f"umap_{data_type}_"
    suffix = "_heating_shape.geojson"
    
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    
    # Extract the city name between prefix and suffix
    city_name = filename[len(prefix):-len(suffix)]
    return city_name


def get_unique_cities_from_inside(inside_dir: Path, data_type: str) -> Set[str]:
    """
    Get all unique city names from output_inside directory.
    
    Args:
        inside_dir: Path to output_inside/{data_type} directory
        data_type: 'squares' or 'addresses'
        
    Returns:
        Set of unique city names
    """
    cities = set()
    
    if not inside_dir.exists():
        print(f"⚠️  Directory does not exist: {inside_dir}")
        return cities
    
    for file_path in inside_dir.glob("*.geojson"):
        city_name = extract_city_name_from_inside_file(file_path.name, data_type)
        if city_name:
            cities.add(city_name)
    
    return cities


def find_matching_files_in_outside(
    outside_dir: Path,
    city_names: Set[str],
    data_type: str
) -> List[Tuple[Path, str]]:
    """
    Find files in output_outside that match city names from output_inside.
    
    Args:
        outside_dir: Path to output_outside/{data_type} directory
        city_names: Set of city names to match
        data_type: 'squares' or 'addresses'
        
    Returns:
        List of (file_path, city_name) tuples for matching files
    """
    matching_files = []
    
    if not outside_dir.exists():
        print(f"⚠️  Directory does not exist: {outside_dir}")
        return matching_files
    
    for file_path in outside_dir.glob("*.geojson"):
        city_name = extract_city_name_from_outside_file(file_path.name, data_type)
        if city_name and city_name in city_names:
            matching_files.append((file_path, city_name))
    
    return matching_files


def move_files_to_additional(
    files_to_move: List[Tuple[Path, str]],
    destination_dir: Path,
    dry_run: bool = False
) -> int:
    """
    Move files from output_outside to output_inside_additional.
    
    Args:
        files_to_move: List of (file_path, city_name) tuples
        destination_dir: Path to output_inside_additional/{data_type} directory
        dry_run: If True, only simulate the move without actually moving files
        
    Returns:
        Number of files moved
    """
    if not files_to_move:
        print("\n✅ No files to move.")
        return 0
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    for file_path, city_name in files_to_move:
        destination_path = destination_dir / file_path.name
        
        if dry_run:
            print(f"  [DRY RUN] Would move: {file_path.name} -> {destination_path}")
        else:
            try:
                shutil.move(str(file_path), str(destination_path))
                print(f"  ✓ Moved: {file_path.name}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Error moving {file_path.name}: {e}")
    
    return moved_count


def main():
    parser = argparse.ArgumentParser(
        description="Sort overlapping city files between output_inside and output_outside"
    )
    parser.add_argument(
        "--data-type",
        choices=["squares", "addresses"],
        required=True,
        help="Type of data to sort (squares or addresses)"
    )
    parser.add_argument(
        "--inside-dir",
        type=str,
        default=None,
        help="Path to inside directory (default: output/output_inside)"
    )
    parser.add_argument(
        "--outside-dir",
        type=str,
        default=None,
        help="Path to outside directory (default: output/output_outside)"
    )
    parser.add_argument(
        "--additional-dir",
        type=str,
        default=None,
        help="Path to additional directory (default: output/output_inside_additional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the operation without actually moving files"
    )
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    
    if args.inside_dir:
        inside_dir = Path(args.inside_dir)
    else:
        inside_dir = base_dir / "output" / "output_inside" / args.data_type
    
    if args.outside_dir:
        outside_dir = Path(args.outside_dir)
    else:
        outside_dir = base_dir / "output" / "output_outside" / args.data_type
    
    if args.additional_dir:
        additional_dir = Path(args.additional_dir)
    else:
        # Derive from inside_dir if custom paths are used
        if args.inside_dir:
            inside_parent = Path(args.inside_dir).parent
            inside_name = Path(args.inside_dir).name
            additional_dir = inside_parent / f"{inside_name}_additional"
        else:
            additional_dir = base_dir / "output" / "output_inside_additional" / args.data_type
    
    print("=" * 80)
    print(f"🔍 SORTING OVERLAPPING CITIES - {args.data_type.upper()}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be moved\n")
    
    # Step 1: Get unique cities from output_inside
    print(f"\n📂 Step 1: Scanning {inside_dir}")
    inside_cities = get_unique_cities_from_inside(inside_dir, args.data_type)
    
    if not inside_cities:
        print(f"⚠️  No cities found in {inside_dir}")
        return
    
    print(f"✓ Found {len(inside_cities)} unique cities in output_inside")
    print(f"  Cities: {', '.join(sorted(inside_cities)[:10])}{'...' if len(inside_cities) > 10 else ''}")
    
    # Step 2: Find matching files in output_outside
    print(f"\n📂 Step 2: Searching for matches in {outside_dir}")
    matching_files = find_matching_files_in_outside(
        outside_dir,
        inside_cities,
        args.data_type
    )
    
    if not matching_files:
        print("✓ No overlapping cities found in output_outside")
        return
    
    print(f"✓ Found {len(matching_files)} files in output_outside that match cities from output_inside")
    
    # Group by city for better reporting
    files_by_city = {}
    for file_path, city_name in matching_files:
        if city_name not in files_by_city:
            files_by_city[city_name] = []
        files_by_city[city_name].append(file_path.name)
    
    print(f"\n  Overlapping cities ({len(files_by_city)}):")
    for city_name in sorted(files_by_city.keys()):
        print(f"    - {city_name}: {len(files_by_city[city_name])} file(s)")
    
    # Step 3: Move files to output_inside_additional
    print(f"\n📦 Step 3: Moving files to {additional_dir}")
    
    moved_count = move_files_to_additional(
        matching_files,
        additional_dir,
        dry_run=args.dry_run
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"  Cities in output_inside: {len(inside_cities)}")
    print(f"  Overlapping cities found: {len(files_by_city)}")
    print(f"  Files {'that would be moved' if args.dry_run else 'moved'}: {len(matching_files)}")
    
    if not args.dry_run and moved_count > 0:
        print(f"\n✅ Successfully moved {moved_count} files to {additional_dir}")
    elif args.dry_run:
        print(f"\n💡 Run without --dry-run to actually move the files")
    
    print()


if __name__ == "__main__":
    main()

