{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = (import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      });
      program = pkgs.python3Packages.buildPythonApplication {
        pname = "equivariant-transformers";
        version = "1.0";

        propagatedBuildInputs = with pkgs.python3Packages; [ numpy pytorch ];

        src = ./.;
      };
    in {
      packages = {
        default = program;
      };

      devShell.x86_64-linux = pkgs.mkShellNoCC {
        buildInputs = with pkgs; [
          helix
          (python3.withPackages (p: [ p.python-lsp-server p.numpy p.pytorch ]))
        ];
      };

      sing = pkgs.singularity-tools.buildImage {
        name = "equivariant-posteriors";
        diskSize = 1024*20;
        memSize = 1024*8;
        contents = [program];
      };

    };
}
