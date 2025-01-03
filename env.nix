{ pkgs, cuda-python }:

pkgs.mkShell rec {
  nativeBuildInputs = with pkgs.buildPackages; [
    cuda-python
    python3Packages.torch
  ];

  shellHook = ''
    echo -e "PyTorch/CUDA environment active.\n"
  '';
}
