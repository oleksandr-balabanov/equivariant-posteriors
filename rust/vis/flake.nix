{
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/02f05fc";
  inputs.helix-pkg.url = "github:helix-editor/helix";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  inputs.fenix = {
    url = "github:nix-community/fenix";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = { self, nixpkgs, helix-pkg, fenix, ... }:
    # outputs = { self, nixpkgs, ... }:
    let
      system = "";
      pkgs = (import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      });
      rustToolchain = fenix.packages."${system}".stable;
      helixmaster = helix-pkg.packages.${system}.default;

      devinputs = with pkgs; [
        (rustToolchain.withComponents [
          "cargo"
          "rustc"
          "rust-src"
          "rustfmt"
          "clippy"
        ])
        fenix.packages."${system}".rust-analyzer
        nil
        postgresql
        helixmaster
        nixfmt
        jq
        yazi
        nerdfonts
        (ueberzugpp.override {
          enableWayland = false;
          enableOpencv = false;
        })
      ];
    in {
      packages.aarch64-darwin.default = program;
      # devShells.x86_64-linux.default = pkgs.mkShellNoCC {
      #devShells.aarch64-darwin.default =
      #  (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_14.stdenv; }) {
      #    buildInputs = devinputs;
      #    shellHook = ''
      #      export POSTGRES=${pkgs.postgresql}
      #      export EDITOR=hx
      #    '';
      #  };
      devShells.aarch64-darwin.default =
        pkgs.mkShellNoCC { buildInputs = devinputs; };


    };
}
