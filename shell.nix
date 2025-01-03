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
in
pkgs.mkShell {
  nativeBuildInputs = with pkgs.buildPackages; [
    cuda-python312
    python312Packages.torchWithCuda
    # python312Packages.torchvision
    (python312.withPackages (ps: with ps; [
                             (ps.opencv4.override {
                              enableGtk3 = true;
                              enableUnfree = true;
                              })
    ]))
  ];
}
