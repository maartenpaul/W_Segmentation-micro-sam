# W_Segmentation-micro-sam

Instance segmentation using Micro-SAM (Segment Anything Model) with automatic mask generation. This container processes multi-dimensional images (TZCYX) and handles channel selection, time points, and z-slices.

## References
- Micro-SAM: Segment Anything for Microscopy  
  Anwai Archit, Sushmita Nair, Nabeel Khalid, Paul Hilt, Vikas Rajashekar, Marei Freitag, Sagnik Gupta, Andreas Dengel, Sheraz Ahmed, Constantin Pape  
  arXiv:2306.08219; https://doi.org/10.48550/arXiv.2306.08219

## Build Container
```bash
docker build -t microsam_biomero .
```

## Key Parameters
- `model_type`: SAM model variant (default: 'vit_b_lm', options: 'vit_t', 'vit_b', 'vit_l', 'vit_h', 'vit_b_lm')
- `segmentation_mode`: Segmentation mode (default: 'ais', options: 'amg', 'ais')
- `ndim`: Input dimensionality (default: 2, options: 2 for 2D/RGB, 3 for 3D)
- `channel`: Channel to segment (0-based, -1 for all)
- `tile_shape`: Tile shape for tiled prediction (e.g., '[512,512]')
- `halo`: Overlap between tiles (e.g., '[64,64]')

## Test Locally
```bash
docker build -t microsam_biomero .
docker run --rm --gpus=all -v E:\\tmp\\micro_sam:/micro_sam -v E:\\tmp\\micro_sam\\models:/tmp/models microsam_biomero --local --infolder /micro_sam/in --outfolder /micro_sam/out --gtfolder /micro_sam/gt --ndim 2 --model_type vit_b_lm --segmentation_mode ais -nmc
```

## Complete Parameter Documentation

All parameters below are actively used in the segmentation workflow. No obsolete parameters exist in the current version.

### Core Segmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | String | vit_b_lm | SAM model variant (vit_t, vit_b, vit_l, vit_h, vit_b_lm) |
| `segmentation_mode` | String | ais | Segmentation mode ('amg' or 'ais') |
| `ndim` | Number | 2 | Input dimensionality (2 for 2D/RGB, 3 for 3D) |
| `channel` | Number | 0 | Channel to segment (0-based, -1 for all channels) |

### Image Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_series` | Number | -1 | Process specific time point (0-based, -1 for all) |
| `z_slices` | Number | -1 | Process specific z-slice (0-based, -1 for all) |
| `scale_factor` | Number | 1.0 | Scale factor (<1 for large objects, >1 for small objects) |

### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_shape` | String | "" | Tile shape for processing (e.g., '[512,512]') |
| `halo` | String | "" | Overlap between tiles (e.g., '[64,64]') |
| `batch_size` | Number | 1 | Number of tiles/planes to process in parallel |

## Saving Micro-SAM Models   

By default micro-sam downloads models to `/tmp/models/microsam_cache`. By mounting this folder to the container, you can store model files externally and reload them at the next run.

For BIOMERO, define a custom binding in the 'slurm_data_bind_path' by adding e.g., `/data1/models:/tmp/models`. Models will be saved at `/data1/models/micro-sam` on the HPC.

## Processing Pipeline

1. **Input Standardization**: Images are converted to 5D format (TZCYX)
2. **Slice Generation**: Images are sliced according to specified time, z, and channel parameters
3. **Scaling**: Optional scaling applied for object size optimization
4. **Batch Processing**: Slices are processed using micro-sam's pattern matching feature
5. **Result Assembly**: Results are assembled back into 5D output with proper metadata
6. **Output**: Final outputs saved as OME-TIFF files with original filename

## Features
- Processes 5D images (TZCYX format) by converting to a standardized format
- Handles multidimensional data with time points, z-slices, and multiple channels
- Uses micro-sam's pattern matching feature for efficient batch processing
- Configurable parameters for selecting specific time points, z-slices, and scale factors
- Support for both AIS and AMG segmentation modes
- Automatic model downloading and caching
- GPU acceleration support

