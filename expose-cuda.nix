{ stdenv }:

stdenv.mkDerivation rec {

  name    = "expose-cuda-${version}";
  version = "1.0";
  src = builtins.filterSource (path: type: false) ./.; #https://github.com/NixOS/nixpkgs/issues/23099

  installPhase = ''
    mkdir -p $out/lib
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $out/lib
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $out/lib
    # ln -s /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libGLESv1_CM_nvidia.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so.2 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-allocator.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-allocator.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-api.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-cfg.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-cfg.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-egl-gbm.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-egl-gbm.so.1.1.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-fbc.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-fbc.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-glsi.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-pkcs11-openssl3.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-pkcs11.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvidia-tls.so.560.35.03 $out/lib, ln -s /usr/lib/x86_64-linux-gnu/libnvoptix.so.1 $out/lib, 
  '';
}
