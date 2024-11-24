#!/usr/bin/env bash


dataset_zip_name='Semantic-segmentation-dataset-1.zip'
new_folder_name='dataset'
train_name='train'
val_name='val'

cleanup () {
  cd $ROOT_START_DIR
  rm -rf $new_folder_name
}

extract_from_zip_into_one_dir () {
  cd $ROOT_START_DIR
  unzip $dataset_zip_name &>/dev/null
  mv 'Semantic segmentation dataset' $new_folder_name
  mkdir "${new_folder_name}/${train_name}"
  mkdir "${new_folder_name}/${val_name}"
  cd $new_folder_name

  # Copy and rename to single dir
  for dir in */; do
    if [[ "$dir" == Tile* ]]; then
      tile_id="${dir//[^0-9]/}"

      cd "${dir}images"
      for image in *; do
        extension=$(echo "$image" | sed -E 's/.*\.([^.]+)$/\1/')
        cp $image "../../${train_name}/tile${tile_id}_part${image//[^0-9]/}.${extension}"
      done;

      cd - &>/dev/null

      cd "${dir}masks"
      for image in *; do
        extension=$(echo "$image" | sed -E 's/.*\.([^.]+)$/\1/')
        cp $image "../../${train_name}/tile${tile_id}_part${image//[^0-9]/}.${extension}"
      done;

      cd - &>/dev/null
      rm -rf "${dir}"
    fi
  done;
}

shuffle_from_train_to_val () {
  cd $ROOT_START_DIR
  num_total_images=$(ls ${new_folder_name}/${train_name} | wc -l)
  num_image_mask_pairs=$((num_total_images / 2))
  percentage=10
  num_image_mask_pairs_for_val=$(( (num_image_mask_pairs * percentage + 50) / 100 ))
  echo $num_image_mask_pairs_for_val

  for image in $( find dataset/train/ -name "*.png" | shuf | head -n $num_image_mask_pairs_for_val ); do
    mv "${image%.*}".{jpg,png} "${new_folder_name}/${val_name}"
  done;
}



ROOT_START_DIR="$(pwd)"
cleanup
extract_from_zip_into_one_dir
shuffle_from_train_to_val
