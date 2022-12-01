# OAKINK

 OakInk is a dataset containing 50.000 distinct affordance-aware and intent-oriented hand-object interactions.



# Download Dataset

1. Create a proper directory using `HOPE-Generator/_SOURCE_DATA/OakInk`. Download the OakInk dataset (containing the Image and Shape subsets) from the [project site](http://www.oakink.net). Arrange all zip files into a folder: `/_SOURCE_DATA/OakInk` as follow:

   ```
    .
    ├── image
    │   ├── anno.zip
    │   ├── obj.zip
    │   └── stream_zipped
    │       ├── oakink_image_v2.z01
    │       ├── ...
    │       ├── oakink_image_v2.z10
    │       └── oakink_image_v2.zip
    └── shape
        ├── metaV2.zip
        ├── OakInkObjectsV2.zip
        ├── oakink_shape_v2.zip
        └── OakInkVirtualObjectsV2.zip
   ```

2. Extract the files.

- For the `image/anno.zip`, `image/obj.zip` and `shape/*.zip`, you can simply _unzip_ it at the same level of the `.zip` file:
  ```Shell
  $ unzip obj.zip
  ```
- For the 11 split zip files in `image/stream_zipped`, you need to _cd_ into the `image/` directory, run:
  ```Shell
  $ zip -F ./stream_zipped/oakink_image_v2.zip --out single-archive.zip
  ```
  This will combine the split zip files into a single .zip, at `image/single-archive.zip`. Finally, _unzip_ the combined archive:
  ```Shell
  $ unzip single-archive.zip
  ```
  After all the extractions are finished, you will have a your `/_SOURCE_DATA/OakInk` as the following structure:
  ```
  .
  ├── image
  │   ├── anno
  │   ├── obj
  │   └── stream_release_v2
  │       ├── A01001_0001_0000
  │       ├── A01001_0001_0001
  │       ├── A01001_0001_0002
  │       ├── ....
  │
  └── shape
      ├── metaV2
      ├── OakInkObjectsV2
      ├── oakink_shape_v2
      └── OakInkVirtualObjectsV2
  ```

3. Set the environment variable `$OAKINK_DIR` to your dataset folder:

   ```Shell
   $ export OAKINK_DIR=/_SOURCE_DATA/OakInk
   ```

## Visualize

## Load Dataset and Visualize

we provide three scripts to provide basic usage for data loading and visualizing:

1. visualize OakInk-Image set on sequence level:
   ```Shell
   $ python scripts/viz_oakink_image_seq.py (--help)
   ```
2. use OakInkImage to load data_split: `all` and visualize:

   ```Shell
   $ python scripts/viz_oakink_image.py (--help)
   ```

3. visualize OakInk-Shape set with object category and subject intent
   ```Shell
   $ python scripts/viz_oakink_shape.py --categories teapot --intent_mode use (--help)
   ```
