#!/bin/bash
docker run -it --rm \
    -v $PWD:/workspaces/scenic_ins_data_analysis \
    --user $(id -u):$(id -g) \
    scenic-ins-data-analysis \
    python -m src.scenicins.run_pc_analysis