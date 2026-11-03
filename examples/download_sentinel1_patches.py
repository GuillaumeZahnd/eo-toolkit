import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from pathlib import Path

from eo_toolkit import (
    get_s1_manifest,
    fetch_s1_image,
    filter_manifest_by_best_orbit,
    log_orbit_audit
)    

@hydra.main(version_base=None, config_path="config", config_name="config")
def download_sentinel1_patches(cfg: DictConfig) -> None:

    cfg.longitude, cfg.latitude, cfg.size_m = 7.85, 48.83, 3000
    YEARS = range(2014, 2026)  # Sentinel-1A was launched on April 3, 2014.
    REQUIRED_BANDS = ("vv", "vh")
    LOG_NAME = f"sentinel1_orbitals_{YEARS[0]}_{YEARS[-1]}_{cfg.longitude}_{cfg.latitude}_{cfg.size_m}.txt"
    DATA_COLLECTION_NAME = f"sentinel1_lat_{cfg.latitude}_lon_{cfg.longitude}_size_{cfg.size_m}"
       
    log_file_path = cfg.cfg.path_to_logs / LOG_NAME
    if log_file_path.exists():
        log_file_path.unlink()

    patch_dir = cfg.cfg.path_to_data / DATA_COLLECTION_NAME
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the complet multi-year manifest
    raw_manifest = get_s1_manifest(cfg.longitude, cfg.latitude, cfg.size_m, f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31")

    # Process year by year
    for year in YEARS:
    
        # Filter the manifest items belonging strictly to this calendar year
        year_items = [item for item in raw_manifest if pd.Timestamp(item["date"]).year == year]        
        if not year_items:
            continue
            
        # Find the most consistent orbit for this specific year
        clean_year_manifest, best_orbit, all_counts = filter_manifest_by_best_orbit(
            year_items,
            required_bands=REQUIRED_BANDS,
        )

        # Log the breakdown for auditing
        log_orbit_audit(year, all_counts, best_orbit, log_file_path)
        print(f"Processed Year {year}: Locked to Orbit {best_orbit} ({len(clean_year_manifest)} acquisitions)")

        # Fetch and save each acquisition to disk (with caching logic)
        for entry in clean_year_manifest:
            file_path = patch_dir / f"{entry['date']}.npz"
            
            if file_path.exists():
                continue  # Skip downloading if already cached locally
                
            arr, exact_date = fetch_s1_image(entry["_item_obj"], cfg.longitude, cfg.latitude, cfg.size_m)
            
            np.savez_compressed(
                file_path,
                data=arr,
                orbit_number=np.array([best_orbit]),
                date=np.array([exact_date], dtype=str)
            )

    print(f"Download complete.\nLogs saved to {log_file_path}.\nRaster patches saved to {patch_dir}.")
    
    
if __name__ == "__main__":
    download_sentinel1_patches()

