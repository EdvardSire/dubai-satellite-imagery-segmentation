#{ pkgs ? import (builtins.fetchGit {
#    url = "https://github.com/NixOS/nixpkgs.git";
#    rev = "057f9aecfb71c4437d2b27d3323df7f93c010b7e";
#    ref = "release-23.11";
#  }) {} }:
{ pkgs ? import <nixpkgs> {}}:
let
  segmentation_models = pkgs.python311Packages.buildPythonPackage rec {
    pname = "segmentation-models-pytorch";
    version = "0.3.4";
    src = pkgs.fetchFromGitHub {
			owner = "qubvel-org";
			repo = "segmentation_models.pytorch";
			rev = "v0.3.4";
			sha256 = "M/7c/bItUe69dBiD47LFhhuD44648/R68iBXvAT0Jmc=";
		};
  };
  pretrainedmodels = pkgs.python311Packages.buildPythonPackage rec {
    pname = "pretrainedmodels";
    version = "0.7.4";
    src = pkgs.fetchFromGitHub {
			owner = "Cadene";
			repo = "pretrained-models.pytorch";
			rev = "8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0";
			sha256 = "OK865VBFRbsSZbEGHe1wLdkioj595YmLwaztwx2R6tE=";
		};
  };
  efficientnet = pkgs.python311Packages.buildPythonPackage rec {
    pname = "efficientnet-pytorch";
    version = "0.7.1";
    src = pkgs.fetchFromGitHub {
			owner = "lukemelas";
			repo = "EfficientNet-PyTorch";
			rev = "e047e4eb9e3ac1cb11e3efa69694c150293b16b1";
			sha256 = "RGOVhxjt0dFv3valneHjzZaF7m9JtC1MNkbh7MUGogo=";
		};
  };
in
pkgs.mkShell {
  buildInputs = (with pkgs.python311Packages; [
      numpy
      scipy
      matplotlib
      pandas
      albumentations
      torch
      torchvision
      tensorflow
      keras
      notebook
      nbformat
      timm
      segmentation_models
      pretrainedmodels
      efficientnet
  ]);
}

