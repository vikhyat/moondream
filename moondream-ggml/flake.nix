{
    description = "Moondream GGML devshell";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/24.05";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = inputs@{ 
        self, 
        nixpkgs, 
        flake-utils 
    }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
        let inherit (nixpkgs) lib;
        in {
            devShells = {
                default = let
                    pkgs = import nixpkgs { inherit system; };
                    python = (pkgs.python311.withPackages (ps: with ps; [ pybind11 ]));
                in pkgs.mkShell {
                    name = "moondream-ggml";
                    buildInputs = with pkgs; [
                        python
                        gcc
                        cmake
                    ];
                    shellHook = ''
                        export PYTHONPATH=${python}/lib/${python.executable}/site-packages
                    '';
                };

                python = let
                    overlay = final: prev: {
                        python3 = prev.python3.override {
                            packageOverrides = python-final: python-prev: {
                                moondream_ggml_python = python-final.callPackage ./moondream-ggml-python.nix {
                                    inherit (final) cmake ninja;
                                };
                            };
                        };
                    };
                    pkgs = import nixpkgs { 
                        inherit system; 
                        overlays = [ overlay ];
                    };
                in pkgs.mkShell {
                    name = "moondream-ggml-python";
                    buildInputs = with pkgs; [
                        (python3.withPackages (ps: with ps; [
                            moondream_ggml_python
                        ]))
                    ];
                };
            };
        }
    );
}
