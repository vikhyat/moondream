{
    lib,
    stdenv,
    python3,
    python3Packages,
    symlinkJoin, 
    buildPythonPackage,
    setuptools-scm,
    cmake,
    ninja,
    pybind11,
}:

buildPythonPackage rec {
    pname = "moondream-ggml-python";
    version = "0.1.0";
    src = ./.;  # Assuming the setup.py is in the same directory as flake.nix
    format = "pyproject";
    dontUseCmakeConfigure = true;
    doCheck = false;
    
    buildInputs = [
        stdenv.cc.cc.lib # This is required for libstdc++.so
    ];

    nativeBuildInputs = [
        cmake
        ninja
        pybind11
        setuptools-scm
    ];

    propagatedBuildInputs = [
        pybind11
    ];
    
    preBuild = ''
        cd bindings/python
    '';

    preFixup = ''
        patchelf --set-rpath "${lib.makeLibraryPath buildInputs}" \
            $out/lib/python${python3.pythonVersion}/site-packages/moondream_ggml/*.so
    '';

}
