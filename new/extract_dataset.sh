#!/usr/bin/env bash


dataset_zip_name='Semantic-segmentation-dataset-1.zip'
new_folder_name='dataset'

cleanup () {
  rm -rf $new_folder_name
}

extract_from_zip () {
  unzip $dataset_zip_name &>/dev/null
  mv 'Semantic segmentation dataset' $new_folder_name
  mkdir "${new_folder_name}/train"
  mkdir "${new_folder_name}/test"
  cd $new_folder_name

  for dir in */; do
    if [[ "$dir" == Tile* ]]; then
      tile_id="${dir//[^0-9]/}"

      cd "${dir}images"
      for image in *; do
        extension=$(echo "$image" | sed -E 's/.*\.([^.]+)$/\1/')
        cp $image "../../train/tile${tile_id}_part${image//[^0-9]/}.${extension}"
      done;

      cd - &>/dev/null

      cd "${dir}masks"
      for image in *; do
        extension=$(echo "$image" | sed -E 's/.*\.([^.]+)$/\1/')
        cp $image "../../train/tile${tile_id}_part${image//[^0-9]/}.${extension}"
      done;

      cd - &>/dev/null
    fi
  done;
}


ROOT_START_DIR="$(pwd)"
cleanup
extract_from_zip

