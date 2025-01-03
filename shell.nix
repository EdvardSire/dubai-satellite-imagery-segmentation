let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
  expose-cuda = pkgs.callPackage ./expose-cuda.nix {};
  cuda-python = pkgs.callPackage ./cuda-python.nix { inherit expose-cuda; };
  env = import ./env.nix;
in
  env { pkgs = pkgs; cuda-python = cuda-python; }
