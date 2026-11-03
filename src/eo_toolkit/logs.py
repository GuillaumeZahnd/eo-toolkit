from pathlib import Path


def log_orbit_audit(
    year: int, 
    all_counts: list[tuple[int, int]], 
    selected_orbit: int, 
    log_path: Path
) -> None:
    """Helper function to append orbit distribution and selection results to a text log."""
    log_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Year {year} (Selected orbit: {selected_orbit}):\n")
        for orb, cnt in all_counts:
            f.write(f"  - Orbit {orb}: {cnt} passes\n")
        f.write(f"\n\n")
