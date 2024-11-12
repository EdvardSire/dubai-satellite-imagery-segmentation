{ pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs.git";
    rev = "057f9aecfb71c4437d2b27d3323df7f93c010b7e";
    ref = "release-23.11";
  }) {} }:

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

