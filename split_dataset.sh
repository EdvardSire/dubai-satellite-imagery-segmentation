#!/usr/bin/env bash

group_all_images() {
  dataset_dir="/home/user/repos/datasets/Semantic-segmentation-dataset"
  train_dir="$dataset_dir/train"
  val_dir="$dataset_dir/val"
  echo 1
}

# val_split=0.15  # 20% for validation
#
# # Create train and val directories if they don't exist
# mkdir -p "$train_dir" "$val_dir"
#
# # List all tile directori
# tiles=($(ls -d $dataset_dir/tile*))
#
# # Shuffle the tiles list and calculate number for validation
# num_tiles=${#tiles[@]}
# num_val=$(echo "$num_tiles * $val_split" | bc | awk '{print int($1)}')
#
# # Randomly shuffle tiles
# shuffled_tiles=($(shuf -e "${tiles[@]}"))
#
# # Split tiles into train and val
# val_tiles=("${shuffled_tiles[@]:0:$num_val}")
# train_tiles=("${shuffled_tiles[@]:$num_val}")
#
# # Move selected tiles to train and val directories
# for tile in "${train_tiles[@]}"; do
#     tile_name=$(basename "$tile")
#     cp -r "$tile" "$train_dir/$tile_name"
# done
#
# for tile in "${val_tiles[@]}"; do
#     tile_name=$(basename "$tile")
#     cp -r "$tile" "$val_dir/$tile_name"
# done
#
# # Output the split
# echo "Train tiles: ${train_tiles[@]}"
# echo "Val tiles: ${val_tiles[@]}"
#
#
group_all_images

