import numpy as np
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer
import rioxarray


def download_sample_sentinel2_chip():
    """
    Query Microsoft's Planetary Computer STAC API for a cloud-free
    Sentinel-2 L2A scene and download a small bounding box window.
    """
    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # [xmin, ymin, xmax, ymax]
    bbox = [1.40, 43.50, 1.50, 43.60]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2023-06-01/2023-08-31",
        query={"eo:cloud_cover": {"lt": 5}}
    )

    # Fixed: use search.items() instead of search.item_search()
    items = list(search.items())
    if not items:
        raise ValueError("No matching Sentinel-2 scenes found for the criteria.")

    item = items[0]
    print(f"Found scene ID: {item.id} acquired on {item.datetime}")

    # Sentinel-2 bands of interest: Red (B04), Green (B03), Blue (B02), Near Infrared (B08) - all 10m resolution
    bands = ["B04", "B03", "B02", "B08"]

    # Open and clip using rioxarray
    print("Downloading and clipping the target region...")
    band_ds_list = []
    for b in bands:
        href = item.assets[b].href
        # Read the asset via rioxarray and clip to our bounding box (EPSG:4326)
        da = rioxarray.open_rasterio(href, masked=True)
        da_clipped = da.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326")
        band_ds_list.append(da_clipped.squeeze(drop=True))

    # Stack bands together into a multi-channel raster (Channels, Height, Width)
    min_h = min([ds.shape[0] for ds in band_ds_list])
    min_w = min([ds.shape[1] for ds in band_ds_list])

    stacked_data = np.stack([ds.values[:min_h, :min_w] for ds in band_ds_list], axis=0)

    # Fill any NaN values with 0 for toy processing
    stacked_data = np.nan_to_num(stacked_data, nan=0.0)

    print(f"Successfully downloaded stack shape (Channels, Height, Width): {stacked_data.shape}")
    return stacked_data


def extract_patches(image_stack, patch_size=64):
    """
    Slices a large (C, H, W) raster array into non-overlapping spatial patches
    (mimicking tokens/chips for a vision model or patch embedding layer).
    """
    C, H, W = image_stack.shape
    patches = []

    coords = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = image_stack[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

    patches_array = np.array(patches) # Shape: (Num_Patches, Channels, Patch_Size, Patch_Size)
    print(f"Extracted {patches_array.shape[0]} patches of shape {patch_size}x{patch_size}.")
    return patches_array, coords

def main():
    # Step 1: Programmatically get a small region array
    # If offline or restricted, you can comment this out and use: image_stack = np.random.rand(4, 512, 512) * 10000
    try:
        image_stack = download_sample_sentinel2_chip()
    except Exception as e:
        print(f"Network/Download skipped or failed ({e}). Falling back to a synthetic array for local testing.")
        image_stack = np.random.randint(0, 4000, size=(4, 512, 512), dtype=np.uint16)

    # Step 2: Basic Normalization (Sentinel-2 L2A data scale is typically 0-10000)
    normalized_stack = image_stack.astype(np.float32) / 10000.0
    normalized_stack = np.clip(normalized_stack, 0.0, 1.0)

    # Step 3: Slice into patches (Toy tokens)
    PATCH_SIZE = 64
    patches, coordinates = extract_patches(normalized_stack, patch_size=PATCH_SIZE)

    print(f"Final tensor batch shape ready for model pipeline: {patches.shape}")

    # Step 4: Quick visualization of the first patch (RGB composite: bands index 2=Blue, 1=Green, 0=Red)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # True color preview of first patch
    rgb_patch = np.stack([patches[0, 0], patches[0, 1], patches[0, 2]], axis=-1) # R, G, B
    axes[0].imshow(rgb_patch)
    axes[0].set_title(f"Patch 0 (RGB True Color)")

    # NIR band view
    nir_patch = patches[0, 3] # Index 3 is B08 (NIR)
    im = axes[1].imshow(nir_patch, cmap='gray')
    axes[1].set_title("Patch 0 (Band 8 - NIR)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Histogram of values
    axes[2].hist(patches[0].flatten(), bins=30, color='teal', alpha=0.7)
    axes[2].set_title("Pixel Value Distribution")

    plt.tight_layout()
    fig.savefig(
        "s2.png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none"
    )

if __name__ == "__main__":
    main()
