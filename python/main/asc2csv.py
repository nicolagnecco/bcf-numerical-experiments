import argparse
import os

import numpy as np
import pandas as pd
import rasterio


def extract_data_from_asc(file_path, varname):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # read the first and only band

        nodata_val = src.nodata  # get the nodata value from the raster file
        mask = data != nodata_val  # create a mask of valid data values

        # Generate lat and lon from the file's affine transformation
        lon, lat = (
            np.meshgrid(np.arange(src.width) + 0.5, np.arange(src.height) + 0.5)
            * src.transform
        )

        # Flatten lat, lon and data arrays and return as DataFrame
        df = pd.DataFrame(
            {
                "lat": lat[mask].ravel(),
                "lon": lon[mask].ravel(),
                varname: data[mask].ravel(),
            }
        )
        return df


def main(dir: str, csvpath: str, varname: str):
    # Iterate over .asc files and concatenate to a single DataFrame
    directory = dir
    dfs = []  # to store individual DataFrames

    for filename in os.listdir(directory):
        if filename.endswith(".asc"):
            file_path = os.path.join(directory, filename)
            dfs.append(extract_data_from_asc(file_path, varname))

    # Concatenate all DataFrames
    final_df = pd.concat(objs=dfs, ignore_index=True)

    # Filter for California based on latitude and longitude ranges
    california_df = final_df[
        (final_df["lat"] >= 32)
        & (final_df["lat"] <= 42)
        & (final_df["lon"] >= -125)
        & (final_df["lon"] <= -114)
    ]

    # Save DataFrame to csv
    california_df.to_csv(f"{csvpath}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert asc file to csv with latitude and longitude for  California."
    )
    parser.add_argument("dir", type=str, help="Directory where asc files are located.")
    parser.add_argument(
        "csvpath",
        type=str,
        help="Path (including name and '.csv' extension) of the csv result.",
    )
    parser.add_argument(
        "varname",
        type=str,
        help="Name of the variable of interest. It will be used as name of the column in the csv file.",
    )
    args = parser.parse_args()

    main(args.dir, args.csvpath, args.varname)
