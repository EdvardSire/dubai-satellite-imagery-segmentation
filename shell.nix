{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = (with pkgs.python311Packages; [
      numpy
      scipy
      matplotlib
      pandas
      albumentations
      # torch
      tensorflow
      keras
      notebook
      nbformat
  ]);
}
