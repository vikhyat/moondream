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
        let
            inherit (nixpkgs) lib;
            pkgs = import nixpkgs { inherit system; };
            python = (pkgs.python311.withPackages (ps: with ps; [ pybind11 ]));
        in {
            devShells.default = pkgs.mkShell {
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
        }
    );
}
