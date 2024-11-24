{ pkgs ? import <nixpkgs> {config = {allowUnfree = true; }; } }:

pkgs.mkShell {
  buildInputs = (with pkgs; [
      (python312.withPackages (ps: with ps; [
                               (ps.opencv4.override {
                                enableGtk3 = true;
                                enableUnfree = true;
                                })
      ]))


  ]) ++ (with pkgs.python312Packages; [
    numpy
    torch
    torchvision
    onnx
    timm
    ipython
    albumentations
  ]);
}

