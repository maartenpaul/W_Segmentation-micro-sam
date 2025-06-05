# filepath: d:\Code\W_Segmentation-micro-sam\run.py
import sys
import os
import shutil
import subprocess
import time
import numpy as np
import tifffile
from tifffile import imwrite
from skimage.transform import rescale
import itertools

# Cytomine / BIAFLOWS related imports
from cytomine.models import Job
from biaflows import CLASS_OBJSEG # Assuming Object Segmentation problem
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics, get_discipline

# Define the name of the Conda environment for micro-sam
MICROSAM_ENV_NAME = "microsam_env"

def convert_to_5d_from_tifffile(volume, axes, target="XYZCT"):
    """
    Convert a numpy array from TiffFile to 5D dimensions suitable for OMERO
    
    Parameters
    ----------
    volume : numpy.ndarray
        Image data from tifffile's asarray()
    axes : str
        Axes string from tifffile (e.g., 'TZCYX', 'YX', etc.)
    target : str, optional
        String specifying the desired dimension order, default is "XYZCT"
        
    Returns
    -------
    img_5d : numpy.ndarray or tuple
        5D numpy array with dimensions ordered according to target
        When unpacked as a tuple, returns (img_5d, target)
    """
    # Validate input volume is a numpy array
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input volume must be a numpy.ndarray")
    
    # Standardize to uppercase
    axes = axes.upper()
    target = target.upper()
    
    # Validate axes dimensions match array dimensions
    if len(axes) != volume.ndim:
        raise ValueError(f"Axes string '{axes}' does not match array dimensions {volume.ndim}")
    
    # Some TIFF files use 'S' for samples/channels, convert to 'C' for consistency
    axes = axes.replace('S', 'C')
        
    # Validate target dimensions
    if len(target) != 5:
        raise ValueError(f"Target dimensions must have exactly 5 dimensions, got '{target}'")
    
    if set(target) != set("XYZCT"):
        raise ValueError("Target dimensions must contain letters X, Y, Z, C, and T exactly once")
    
    # Create a 5D array by adding missing dimensions
    img_5d = volume
    current_order = axes
    
    # Add missing dimensions
    for dim in "XYZCT":
        if dim not in current_order:
            img_5d = np.expand_dims(img_5d, axis=-1)
            current_order += dim
    
    # Reorder dimensions if needed
    if current_order != target:
        # Create list of current positions for each dimension
        current_positions = []
        for dim in target:
            current_positions.append(current_order.index(dim))
        
        # Rearrange dimensions
        img_5d = np.moveaxis(img_5d, current_positions, range(len(target)))
    
    # Return both the array and target, allowing for flexible unpacking
    class ReturnValue(tuple):
        """Custom return class to allow both direct access and unpacking"""
        def __new__(cls, img, axes):
            return tuple.__new__(cls, (img, axes))
            
        def __repr__(self):
            return repr(self[0])
            
        # Make the first element (the image) accessible directly
        def __array__(self, dtype=None):
            return np.asarray(self[0], dtype=dtype)
    
    return ReturnValue(img_5d, target)

def guess_axes(shape):
    """Guess the axes string based on the shape of an array"""
    ndim = len(shape)
    if ndim == 2:
        return "YX"
    elif ndim == 3:
        return "ZYX"  # Assume Z stack for 3D
    elif ndim == 4:
        return "CZYX"  # Assume channel is first for 4D
    elif ndim == 5:
        return "TCZYX"  # Assume full 5D
    else:
        return "".join(["Q"] * (ndim - 2)) + "YX"  # Unknown dims + YX

def main(argv):
    with BiaflowsJob.from_cli(argv) as bj:
        # Set problem class, adjust if needed
        problem_cls = get_discipline(bj, default=CLASS_OBJSEG)

        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")

        # 1. Prepare data for workflow
        # is_2d=None allows handling both 2D and 3D if micro_sam supports it via ndim
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=None, **bj.flags)        # Make tmp_path unique if running multiple instances concurrently
        unique_tmp_suffix = f"_{int(time.time() * 1000)}"
        tmp_path = os.path.join(tmp_path, f"microsam_run{unique_tmp_suffix}")
        os.makedirs(tmp_path, exist_ok=True)

        # Final assembled results directory (shared across all images)
        ms_final_path = os.path.join(tmp_path, "ms_assembled")
        os.makedirs(ms_final_path, exist_ok=True)
        
        # 2. Run image analysis workflow (micro-sam)
        bj.job.update(progress=5, statusComment="Launching Micro-SAM segmentation...")

        # --- Parameters for micro_sam ---
        # Mandatory / Core Parameters
        model_type = bj.parameters.model_type # e.g., "vit_b", "vit_l", "vit_h"
        segmentation_mode = bj.parameters.segmentation_mode # "amg" or "ais"
        
        # Optional Parameters
        ndim = getattr(bj.parameters, 'ndim', 2) # Default to 2D for micro-sam
        tile_shape_str = getattr(bj.parameters, 'tile_shape', None)
        halo_str = getattr(bj.parameters, 'halo', None)        # Parameters for multidimensional handling
        channel = getattr(bj.parameters, 'channel', 0)  # Default to channel 0 (set to -1 to process all channels)
        batch_size = getattr(bj.parameters, 'batch_size', 1)
        z_slices = getattr(bj.parameters, 'z_slices', -1)  # Default to all z-slices (-1)
        time_series = getattr(bj.parameters, 'time_series', -1)  # Default to all time points (-1)
        scale_factor = getattr(bj.parameters, 'scale_factor', 1.0)
          # --- Process each image ---
        total_images = len(in_imgs)
        for i, bfimg in enumerate(in_imgs):
            progress = 5 + int(60 * (i / total_images))
            bj.job.update(progress=progress, statusComment=f"Processing image {i+1}/{total_images}: {bfimg.filename}")
            
            # Input/output file paths
            in_file_path = os.path.join(in_path, bfimg.filename)
            
            # Create unique temporary directories for this specific image
            img_tmp_path = os.path.join(tmp_path, f"img_{i}")
            img_ms_prep_path = os.path.join(img_tmp_path, "ms_prep")
            img_ms_out_path = os.path.join(img_tmp_path, "ms_output")
            img_ms_emb_path = os.path.join(img_tmp_path, "ms_embeddings")
            
            # Create directories for this image
            for path in [img_tmp_path, img_ms_prep_path, img_ms_out_path, img_ms_emb_path]:
                os.makedirs(path, exist_ok=True)
            
            try:
                # Load the input image
                img = tifffile.imread(in_file_path)
                axes = None
                with tifffile.TiffFile(in_file_path) as tif:
                    if hasattr(tif.series[0], 'axes'):
                        axes = tif.series[0].axes
                        
                # Convert to standardized 5D format (TZCYX)
                bj.job.update(progress=progress, statusComment=f"Processing image with original axes: {axes}")
                axes = axes if axes else guess_axes(img.shape)
                img_5d, axes_5d = convert_to_5d_from_tifffile(img, axes, target="TZCYX")
                
                # Get image dimensions
                dims = {
                    'T': img_5d.shape[0],  # Time
                    'Z': img_5d.shape[1],  # Depth/slices
                    'C': img_5d.shape[2],  # Channels
                    'Y': img_5d.shape[3],  # Height
                    'X': img_5d.shape[4]   # Width
                }
                bj.job.update(progress=progress, statusComment=f"Dimensions (TZCYX): {dims}")                # Determine which slices to process based on parameters
                # Time points
                if time_series == -1:
                    time_points = list(range(dims['T']))
                else:
                    time_points = [time_series] if time_series < dims['T'] else [0]
                    
                # Z-slices
                if z_slices == -1:
                    z_indices = list(range(dims['Z']))
                else:
                    z_indices = [z_slices] if z_slices < dims['Z'] else [0]
                    
                # Channels - handle similar to Cellpose
                if channel == -1:
                    # Process all channels
                    channels = list(range(dims['C']))
                elif isinstance(channel, (list, tuple)):
                    channels = list(channel)
                else:
                    # Single channel specified (default is 0)
                    channels = [channel] if channel < dims['C'] else [0]
                    
                # Handle case of no channels (shouldn't happen but defensive programming)
                if len(channels) == 0:
                    img_5d = np.expand_dims(img_5d, axis=2)
                    dims['C'] = 1
                    channels = [0]
                  # Log which dimensions are being processed
                total_slices_to_process = len(time_points) * len(z_indices) * len(channels)
                bj.job.update(progress=progress, 
                            statusComment=f"Processing {len(time_points)} time points, {len(z_indices)} z-slices, {len(channels)} channels ({total_slices_to_process} total slices)")
                print(f"Time points: {time_points}")
                print(f"Z-slices: {z_indices}")
                print(f"Channels: {channels}")
                if len(channels) == 1 and dims['C'] > 1:
                    print(f"Note: Only processing channel {channels[0]} out of {dims['C']} available channels. Set channel=-1 to process all channels.")
                
                # Create a 5D output array matching ONLY the selected dimensions
                # This is a key change - only make space for channels we're actually processing
                output_shape = (
                    len(time_points),
                    len(z_indices),
                    len(channels),
                    int(dims['Y'] * scale_factor) if scale_factor != 1.0 else dims['Y'],
                    int(dims['X'] * scale_factor) if scale_factor != 1.0 else dims['X']
                )
                
                # Use uint16 for label mask output
                output_5d = np.zeros(output_shape, dtype=np.uint16)
                
                # Create mapping from original indices to output array indices
                index_mapping = {
                    'T': {t: idx for idx, t in enumerate(time_points)},
                    'Z': {z: idx for idx, z in enumerate(z_indices)},
                    'C': {c: idx for idx, c in enumerate(channels)}
                }
                
                # Group slices by channel for batch processing
                slice_info = []
                
                # First, create all the 2D slices needed and track their info
                for t, z, c in itertools.product(time_points, z_indices, channels):
                    # Extract 2D slice
                    slice_2d = img_5d[t, z, c, :, :]
                    
                    # Apply scaling if needed
                    if scale_factor != 1.0:
                        slice_2d = rescale(slice_2d, scale_factor, order=1, 
                                         preserve_range=True, channel_axis=None, 
                                         anti_aliasing=True).astype(slice_2d.dtype)                    # Create a simple unique filename for this slice
                    slice_fname = f"slice_t{t}_z{z}_c{c}.tif"
                    slice_path = os.path.join(img_ms_prep_path, slice_fname)
                    
                    # Save the 2D slice
                    tifffile.imwrite(slice_path, slice_2d)
                    
                    # Store slice information for later processing
                    slice_info.append({
                        'filename': slice_fname,
                        't': t,
                        'z': z,
                        'c': c,
                        'path': slice_path
                    })                # Track number of slices created
                print(f"Created {len(slice_info)} slices for processing")
                  # Debug: Print all created filenames
                print("Created slice filenames:")
                for info in slice_info:
                    print(f"  - {info['filename']}")
                
                # Process all slices in a single batch (no need for channel grouping)
                if len(slice_info) > 0:
                    bj.job.update(progress=progress + 10, 
                                statusComment=f"Processing {len(slice_info)} slices with micro-sam")
                    
                    # Use simple wildcard pattern to process all slice files at once
                    pattern = "*"
                      # Debug: Print pattern and matching files
                    print(f"Using pattern: {pattern}")
                    print(f"Files in prep directory: {os.listdir(img_ms_prep_path)}")
                    
                    # Build micro-sam command for batch processing
                    cmd = [
                        "conda", "run", "-n", MICROSAM_ENV_NAME,
                        "micro_sam.automatic_segmentation",
                        "--input_path", img_ms_prep_path,
                        "--output_path", img_ms_out_path,
                        "--embedding_path", img_ms_emb_path,
                        "--pattern", pattern,
                        "--model_type", model_type,
                        "--mode", segmentation_mode
                    ]
                    
                    # Add optional arguments
                    if ndim is not None:
                        cmd.extend(["--ndim", str(ndim)])
                    if tile_shape_str:
                        cmd.extend(["--tile_shape", tile_shape_str])
                    if halo_str:
                        cmd.extend(["--halo", halo_str])
                    if batch_size > 1:
                        cmd.extend(["--batch_size", str(batch_size)])
                    
                    # Add verbose flag
                    cmd.extend(["--verbose"])
                    
                    print(f"Running command: {' '.join(cmd)}")
                    
                    # Execute micro-sam for all slices
                    try:
                        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
                        print("Command output:", result.stdout)
                          # Process each slice result
                        for slice_dict in slice_info:
                            slice_out_path = os.path.join(img_ms_out_path, slice_dict['filename'])
                            t, z, c = slice_dict['t'], slice_dict['z'], slice_dict['c']
                            
                            # Check primary location (multiple files case)
                            if os.path.exists(slice_out_path):
                                slice_result = tifffile.imread(slice_out_path)
                            # Check fallback location (single file case - micro-sam outputs to directory.tif)
                            elif len(slice_info) == 1 and os.path.exists(img_ms_out_path + '.tif'):
                                slice_result = tifffile.imread(img_ms_out_path + '.tif')
                            else:
                                print(f"Warning: No output file found at {slice_out_path}")
                                if len(slice_info) == 1:
                                    print(f"Also checked fallback location: {img_ms_out_path + '.tif'}")
                                continue
                            
                            # Handle case where output is relabeled (starts from 1)
                            if np.any(slice_result > 0):
                                # Get indices in the reduced output array using our mapping
                                t_idx = index_mapping['T'][t]
                                z_idx = index_mapping['Z'][z]
                                c_idx = index_mapping['C'][c]
                                
                                # Copy slice result to output array with correct indexing
                                output_5d[t_idx, z_idx, c_idx, :slice_result.shape[0], :slice_result.shape[1]] = slice_result
                            
                    except subprocess.CalledProcessError as e:
                        # Log the error but continue processing other images
                        error_message = f"Micro-SAM failed with code {e.returncode}"
                        print(f"ERROR: {error_message}")
                        print(f"Stderr: {e.stderr}")
                        print(f"Stdout: {e.stdout}")
                        bj.job.update(progress=progress + 25, statusComment=f"Warning: {error_message}")
                        # Continue with next image                # Save the assembled 5D result
                # CRITICAL: Use ome=True to ensure proper OME-TIFF metadata is written
                # This prevents dimension flattening and ensures correct SizeT/SizeZ values
                final_out_path = os.path.join(ms_final_path, bfimg.filename)
                imwrite(final_out_path, output_5d, 
                       metadata={
                           'axes': 'TZCYX',
                           # Add explicit mapping to show which original indices were processed
                           'dimension_mapping': {
                               'T': time_points,
                               'Z': z_indices,
                               'C': channels
                           },
                           'microsam_params': {
                               'model_type': model_type,
                               'segmentation_mode': segmentation_mode,
                               'scale_factor': scale_factor
                           }
                       },
                       photometric='minisblack',
                       ome=True,  # Essential for proper 5D dimension preservation
                       description='Processed with Micro-SAM, standardized to TZCYX format')
                
                # Copy the final result to the output directory
                final_dest_path = os.path.join(out_path, bfimg.filename)
                shutil.copyfile(final_out_path, final_dest_path)
                  # Log objects counted
                num_objects = len(np.unique(output_5d)) - 1  # Subtract 1 for background (0)
                bj.job.update(progress=progress + int(60 * (i+1) / total_images),
                            statusComment=f"Completed {bfimg.filename}: Found {num_objects} objects")
                
                # Clean up temporary files for this image to prevent accumulation
                shutil.rmtree(img_tmp_path, ignore_errors=True)
                
            except Exception as e:
                error_message = f"Error processing {bfimg.filename}: {str(e)}"
                print(error_message)
                import traceback
                traceback.print_exc()
                # Clean up temporary files even on error
                shutil.rmtree(img_tmp_path, ignore_errors=True)
                # Continue with next image instead of terminating
                bj.job.update(progress=progress, statusComment=f"Warning: {error_message[:500]}")
                continue

        # 3. Upload data to BIAFLOWS
        bj.job.update(progress=70, statusComment="Uploading segmentation results...")
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 70, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})

        # 4. Compute and upload metrics
        bj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        # Make sure gt_path is valid if metrics are needed
        if gt_imgs:
             upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)
        else:
             print("No ground truth images found, skipping metrics calculation.")
             bj.job.update(progress=95, statusComment="Skipped metrics calculation (no GT).")

        # 5. Pipeline finished
        # Clean up temporary directory
        shutil.rmtree(tmp_path, ignore_errors=True)
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
