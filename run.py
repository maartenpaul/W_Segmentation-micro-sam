import sys
import os
import shutil
import subprocess
import time
import numpy as np
import tifffile # Keep for potential metadata or checks if needed

# Cytomine / BIAFLOWS related imports
from cytomine.models import Job
from biaflows import CLASS_OBJSEG # Assuming Object Segmentation problem
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics, get_discipline

# Define the name of the Conda environment for micro-sam
MICROSAM_ENV_NAME = "microsam_env"

def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    with BiaflowsJob.from_cli(argv) as bj:
        # Set problem class, adjust if needed
        problem_cls = get_discipline(bj, default=CLASS_OBJSEG)

        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")

        # 1. Prepare data for workflow
        # is_2d=None allows handling both 2D and 3D if micro_sam supports it via ndim
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=None, **bj.flags)

        # Make tmp_path unique if running multiple instances concurrently
        # (Consider if tmp_path from prepare_data is already sufficiently unique)
        unique_tmp_suffix = f"_{int(time.time() * 1000)}"
        tmp_path = os.path.join(tmp_path, f"microsam_run{unique_tmp_suffix}")
        os.makedirs(tmp_path, exist_ok=True)

        # Define paths for micro_sam within the temporary directory
        # input files will be linked/copied here if needed, or used directly from in_path
        ms_out_path = os.path.join(tmp_path, "ms_output") # Dir for individual outputs
        ms_emb_path = os.path.join(tmp_path, "ms_embeddings") # Dir for embeddings cache
        os.makedirs(ms_out_path, exist_ok=True)
        os.makedirs(ms_emb_path, exist_ok=True)

        # 2. Run image analysis workflow (micro-sam)
        bj.job.update(progress=5, statusComment="Launching Micro-SAM segmentation...")

        # --- Parameters for micro_sam ---
        # Mandatory / Core Parameters (Add these to your descriptor.json)
        model_type = bj.parameters.model_type # e.g., "vit_b", "vit_l", "vit_h" (or path to custom?)
        segmentation_mode = bj.parameters.segmentation_mode # "amg" or "ais"

        # Optional Parameters (Add relevant ones to descriptor.json with defaults)
        # Using getattr with default values
        ndim = getattr(bj.parameters, 'ndim', None) # Let micro_sam infer if None, or set (e.g., 2 for RGB)
        tile_shape_str = getattr(bj.parameters, 'tile_shape', None) # e.g., "512,512"
        halo_str = getattr(bj.parameters, 'halo', None) # e.g., "64,64"
        pred_iou_thresh = getattr(bj.parameters, 'pred_iou_thresh', None) # e.g., 0.88 for AMG
        stability_score_thresh = getattr(bj.parameters, 'stability_score_thresh', None) # e.g., 0.95 for AMG
        # Add other relevant generate_kwargs from micro_sam docs as needed (e.g., for AIS mode)

        # --- Build and Run Command for each image ---
        total_images = len(in_imgs)
        for i, bfimg in enumerate(in_imgs):
            progress = 5 + int(60 * (i / total_images))
            bj.job.update(progress=progress, statusComment=f"Processing image {i+1}/{total_images}: {bfimg.filename}")

            in_file_path = os.path.join(in_path, bfimg.filename)
            # Define where micro_sam should write the output mask for this image
            out_file_path = os.path.join(ms_out_path, bfimg.filename)

            # Base command - Uses 'conda run' to execute in the specified environment
            cmd = [
                "conda", "run", "-n", MICROSAM_ENV_NAME,
                "micro_sam.automatic_segmentation",
                "--input_path", in_file_path,
                "--output_path", out_file_path,
                "--embedding_path", ms_emb_path, # Use shared embedding cache
                "--model_type", model_type,
                "--mode", segmentation_mode
            ]

            # Add optional arguments if provided
            if ndim is not None:
                cmd.extend(["--ndim", str(ndim)])
            if tile_shape_str: # Assuming format "y,x" like "512,512"
                cmd.extend(["--tile_shape", tile_shape_str])
            if halo_str: # Assuming format "y,x" like "64,64"
                cmd.extend(["--halo", halo_str])

            # Add generate_kwargs if provided
            if pred_iou_thresh is not None:
                 cmd.extend(["--pred_iou_thresh", str(pred_iou_thresh)])
            if stability_score_thresh is not None:
                 cmd.extend(["--stability_score_thresh", str(stability_score_thresh)])
            # Add other generate_kwargs here...

            print(f"Running command: {' '.join(cmd)}")

            # Execute the command
            try:
                # Set run in shell=False for security and better argument handling
                # check=True will raise CalledProcessError if micro_sam fails
                subprocess.run(cmd, check=True, text=True, capture_output=True)
                 # Can check stdout/stderr from result if needed:
                 # result = subprocess.run(...)
                 # print("stdout:", result.stdout)
                 # print("stderr:", result.stderr)

                # Copy the generated mask to the final BIAFLOWS output directory
                # Ensure the filename matches the original input filename
                final_dest_path = os.path.join(out_path, bfimg.filename)
                shutil.copyfile(out_file_path, final_dest_path)

            except subprocess.CalledProcessError as e:
                # Log error and terminate
                error_message = f"Micro-SAM execution failed for {bfimg.filename} with code {e.returncode}."
                error_detail = f"Stderr: {e.stderr}\nStdout: {e.stdout}"
                print(error_message)
                print(error_detail)
                bj.job.update(status=Job.FAILED, progress=progress,
                              statusComment=f"{error_message}\n{error_detail[:500]}") # Limit length
                # Optional: clean up tmp_path before exiting on error
                # shutil.rmtree(tmp_path, ignore_errors=True)
                sys.exit(1) # Stop processing further images
            except FileNotFoundError:
                 # Handle case where conda or python command itself is not found
                 error_message = f"Error: Command 'conda run' or underlying python/script not found. Is the '{MICROSAM_ENV_NAME}' Conda environment set up correctly in the Docker image?"
                 print(error_message)
                 bj.job.update(status=Job.FAILED, progress=progress, statusComment=error_message)
                 sys.exit(1)


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