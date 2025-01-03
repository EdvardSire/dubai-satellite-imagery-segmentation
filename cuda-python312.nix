{ stdenv, pkgs, expose-cuda }:

stdenv.mkDerivation rec {

  name    = "cuda-python-${version}";
  version = "1.0";
  src     = ./.;

  nativeBuildInputs = with pkgs; [
    makeWrapper
  ];

  buildInputs = with pkgs; [
    python312
    expose-cuda
	];

  buildPhase = ''
  '';

  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/lib

    ln -s ${pkgs.python3}/bin/python $out/bin
    ln -s ${pkgs.python3}/bin/python3 $out/bin
  '';

  postFixup = ''
    wrapProgram $out/bin/python --suffix LD_LIBRARY_PATH ':' ${expose-cuda}/lib
    wrapProgram $out/bin/python3 --suffix LD_LIBRARY_PATH ':' ${expose-cuda}/lib
  '';
}
