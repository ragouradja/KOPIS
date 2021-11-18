import fiftyone as fo



# # Create a dataset from a glob pattern of images
# dataset = fo.Dataset.from_images_patt(r"C:\Users\Radja\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\ragou\repo\github\Protein_peeling\data\19HCA\data\*.png")

dir = r"C:\Users\Radja\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\ragou\repo\github\Protein_peeling\data\19HCA\dataset_dir"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dir,
    dataset_type=fo.types.VOCDetectionDataset,
)



# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())
session = fo.launch_app(dataset, port=5151)
session.wait() 
