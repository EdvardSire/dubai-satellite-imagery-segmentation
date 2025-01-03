let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
  expose-cuda = pkgs.callPackage ./expose-cuda.nix {};
  cuda-python = pkgs.callPackage ./cuda-python.nix { inherit expose-cuda; };
in
pkgs.mkShell rec {
  nativeBuildInputs = with pkgs.buildPackages; [
    cuda-python
    python3Packages.torch
  ];
}
