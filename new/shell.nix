{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = (with pkgs.python312Packages; [
      numpy
      # matplotlib
      # pandas
      # albumentations
      torch
      # tensorflow
      # keras
      # notebook
      # nbformat
  ]);
}

