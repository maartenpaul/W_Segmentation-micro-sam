# W_Segmentation-micro-sam

## local testing
 ```
 docker build -t microsam_biomero .
```

models can be stored externally by mounting a volume to the model cache folder ```/opt/microsam_cache/opt/microsam_cache)```

```
docker run --rm --gpus=all -v E:\\tmp\\micro_sam:/micro_sam -v E:\\tmp\\micro_sam\\model_cache:/opt/microsam_cache microsam_biomero --local --infolder /micro_sam/in --outfolder /micro_sam/out --gtfolder /micro_sam/gt --ndim 2 --model_type vit_b_lm --segmentation_mode ais -nmc
```

