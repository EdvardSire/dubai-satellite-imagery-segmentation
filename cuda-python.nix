{ stdenv, pkgs, expose-cuda }:

stdenv.mkDerivation rec {

  name    = "cuda-python-${version}";
  version = "1.0";
  src     = ./.;

  nativeBuildInputs = with pkgs; [
    makeWrapper
  ];

  buildInputs = with pkgs; [
    python3
    expose-cuda
	];

  buildPhase = ''
  '';

  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/lib
    cp -p ${pkgs.python3}/bin/python $out/bin
  '';

  postFixup = ''
    wrapProgram $out/bin/python --suffix LD_LIBRARY_PATH ':' ${expose-cuda}/lib
  '';
}
