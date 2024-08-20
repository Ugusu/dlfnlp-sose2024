#!/bin/bash

# Submit the scripts using sbatch
sbatch run_grid_search_extra_layer_non_regularized.sh
sbatch run_grid_search_extra_layer_regularized.sh
sbatch run_grid_search_no_extra_layer_non_regularized.sh
sbatch run_grid_search_no_extra_layer_regularized.sh

echo "All scripts have been submitted."
