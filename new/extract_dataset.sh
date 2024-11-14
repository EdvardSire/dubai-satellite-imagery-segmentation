#!/usr/bin/env bash


dataset_zip_name='Semantic-segmentation-dataset-1.zip'
new_folder_name='dataset'

cleanup () {
  rm -rf $new_folder_name
}

extract_from_zip () {
  unzip $dataset_zip_name
  mv 'Semantic segmentation dataset' $new_folder_name
  cd $new_folder_name
  for dir in */; do
    id=''
    ls "$dir"
  done
}



ROOT_START_DIR="$(pwd)"
cleanup
extract_from_zip




