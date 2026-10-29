import zipfile, os

zip_path = r"C:\744-proj\lidar\744-lidar\cache\datasets--nvidia--PhysicalAI-Autonomous-Vehicles\snapshots\2ae73f49ffd2b5db43b404201beb7b92889f7afc\lidar\lidar_top_360fov\lidar_top_360fov.chunk_0588.zip"
out_dir  = r"C:\744-proj\lidar\744-lidar\cache"

# with zipfile.ZipFile(zip_path) as zf:
#     print("Contents:", zf.namelist())   # see all clip parquets inside

with zipfile.ZipFile(zip_path) as zf:
    # Replace with whichever name appeared in the list above
    target = "a17e9bc4-4c9a-4e70-bbb5-da4ed7e87d76.lidar_top_360fov.parquet"
    zf.extract(target, path=out_dir)
    print(f"Extracted to: {os.path.join(out_dir, target)}")