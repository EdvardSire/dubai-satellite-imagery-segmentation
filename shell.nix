let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "8.6" ]; # nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    };
  };
  expose-cuda = pkgs.callPackage ./expose-cuda.nix {};
  cuda-python312 = pkgs.callPackage ./cuda-python312.nix { pkgs=pkgs; expose-cuda=expose-cuda; };

  segmentation_models = pkgs.python312Packages.buildPythonPackage rec {
    pname = "segmentation-models-pytorch";
    version = "0.3.4";
    src = pkgs.fetchFromGitHub {
			owner = "qubvel-org";
			repo = "segmentation_models.pytorch";
			rev = "v0.3.4";
			sha256 = "M/7c/bItUe69dBiD47LFhhuD44648/R68iBXvAT0Jmc=";
		};
  };
  pretrainedmodels = pkgs.python312Packages.buildPythonPackage rec {
    pname = "pretrainedmodels";
    version = "0.7.4";
    src = pkgs.fetchFromGitHub {
      owner = "Cadene";
      repo = "pretrained-models.pytorch";
      rev = "8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0";
      sha256 = "OK865VBFRbsSZbEGHe1wLdkioj595YmLwaztwx2R6tE=";
    };
  };
  efficientnet = pkgs.python312Packages.buildPythonPackage rec {
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
  nativeBuildInputs = with pkgs.buildPackages; [
    cuda-python312
    python312Packages.torchWithCuda
    python312Packages.torchvision
    python312Packages.ipython
    python312Packages.tensorboard
    python312Packages.tqdm
    (python312.withPackages (ps: with ps; [
                             (ps.opencv4.override {
                              enableGtk3 = true;
                              enableUnfree = true;
                              })
    ]))
    ###
    segmentation_models
    pretrainedmodels
    efficientnet
    python312Packages.timm
  ];
}
