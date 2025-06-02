# W_Segmentation-micro-sam

## Features
- Processes 5D images (TZCYX format) by converting to a standardized format
- Handles multidimensional data with time points, z-slices, and multiple channels
- Uses micro-sam's pattern matching feature for efficient batch processing
- Configurable parameters for selecting specific time points, z-slices, and scale factors

## Local testing
```
docker build -t microsam_biomero .
```

Models can be stored externally by mounting a volume to the model cache folder ```/opt/microsam_cache/opt/microsam_cache)```

```
docker run --rm --gpus=all -v E:\\tmp\\micro_sam:/micro_sam -v E:\\tmp\\micro_sam\\model_cache:/opt/microsam_cache microsam_biomero --local --infolder /micro_sam/in --outfolder /micro_sam/out --gtfolder /micro_sam/gt --ndim 2 --model_type vit_b_lm --segmentation_mode ais -nmc
```

## Advanced Parameters

The workflow supports these additional parameters:

- `time_series`: Process specific time point (start at 0), -1 to process all time points
- `z_slices`: Process specific z-slice (start at 0), -1 to process all z-slices
- `scale_factor`: Scale the input image by this factor for processing (values <1 for large nuclei, >1 for small nuclei)

## Processing Pipeline

1. Input images are converted to a standardized 5D format (TZCYX)
2. Images are sliced according to specified time, z, and channel parameters
3. Slices are grouped by channel and processed in batches using the pattern matching feature
4. Results are assembled back into a 5D output with proper metadata
5. Final outputs are saved as TIFF files with the original filename

