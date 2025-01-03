let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
  expose-cuda = pkgs.callPackage ./expose-cuda.nix {};
  cuda-python312 = pkgs.callPackage ./cuda-python312.nix { expose-cuda=expose-cuda; };
in
pkgs.mkShell rec {
  nativeBuildInputs = with pkgs.buildPackages; [
    cuda-python312
    python312Packages.torchWithCuda
  ];
}
