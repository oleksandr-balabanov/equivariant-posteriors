{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/02f05fc";
  inputs.helix-pkg.url = "github:helix-editor/helix";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  inputs.fenix = {
    url = "github:nix-community/fenix";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = { self, nixpkgs, helix-pkg, fenix, ... }:
    # outputs = { self, nixpkgs, ... }:
    let
      disableCudaEnvFlag = builtins.getEnv "DISABLE_CUDA";
      system = "x86_64-linux";
      pkgs = (import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = disableCudaEnvFlag != "1";
        };
      });
      rustToolchain = fenix.packages."${system}".stable;
      helixmaster = helix-pkg.packages.${system}.default;
      datasets = pkgs.python3Packages.buildPythonPackage rec {
        pname = "datasets";
        version = "2.14.7";
        format = "setuptools";

        # disabled = pythonOlder "3.8";

        src = pkgs.fetchFromGitHub {
          owner = "huggingface";
          repo = pname;
          rev = "refs/tags/${version}";
          hash = "sha256-Q8cSgupfj6xKD0bYgL6bvYBwdYDdNaiWEWWUrRvwc4g";
        };

        propagatedBuildInputs = [
          pkgs.python3Packages.aiohttp
          pkgs.python3Packages.dill
          pkgs.python3Packages.fsspec
          pkgs.python3Packages.huggingface-hub
          pkgs.python3Packages.multiprocess
          pkgs.python3Packages.numpy
          pkgs.python3Packages.packaging
          pkgs.python3Packages.pandas
          pkgs.python3Packages.pyarrow
          pkgs.python3Packages.requests
          pkgs.python3Packages.responses
          pkgs.python3Packages.tqdm
          pkgs.python3Packages.xxhash
          pyarrow_hotfix
        ]; # ++ pkgs.lib.optionals (pkgs.pythonOlder "3.8") [
        #pkgs.python3Packages.importlib-metadata
        #];

        # Tests require pervasive internet access
        doCheck = false;

        # Module import will attempt to create a cache directory
        postFixup = "export HF_MODULES_CACHE=$TMPDIR";

        pythonImportsCheck = [ "datasets" ];

        meta = with pkgs.lib; {
          description =
            "Open-access datasets and evaluation metrics for natural language processing";
          homepage = "https://github.com/huggingface/datasets";
          changelog =
            "https://github.com/huggingface/datasets/releases/tag/${version}";
          license = licenses.asl20;
          platforms = platforms.unix;
          maintainers = with maintainers; [ ];
        };
      };
      pyarrow_hotfix = pkgs.python3Packages.buildPythonPackage rec {
        pname = "pyarrow_hotfix";
        version = "1.0";

        src = ./pyarrow_hotfix; # Path to your local dummy_package directory
        format = "other";

        buildPhase = "runHook postBuild";

        installPhase = ''
          # mkdir -p $out/${pkgs.python3.sitePackages}
          # cp -r $src/* $out/${pkgs.python3.sitePackages}
          mkdir -p $out/${pkgs.python3.sitePackages}/pyarrow_hotfix
          cp -r $src/* $out/${pkgs.python3.sitePackages}/pyarrow_hotfix
        '';
        # No dependencies for this dummy package
        propagatedBuildInputs = [ ];

        # No tests to run
        doCheck = false;

        meta = with pkgs.lib; {
          description = "A dummy Python package";
          homepage = "https://example.com/dummy_package";
          license = licenses.mit;
        };
      };

      healpix = pkgs.python3Packages.buildPythonPackage rec {
        pname = "healpix";
        version = "2023.1.13";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-JU8AUgck7F8X2+0zk+O/F8iSt+laxt9S75p/a2jjhOE="; # TODO
        };
        propagatedBuildInputs = [ pkgs.python3Packages.numpy ];
        doCheck = false;
      };
      cdsapi = pkgs.python3Packages.buildPythonPackage rec {
        pname = "cdsapi";
        version = "0.6.1";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-fUDFjj/T51qKzc3IHqtO+bb3Y7KQK6AdfRc482UqWjA="; # TODO
        };
        propagatedBuildInputs = [
          pkgs.python3Packages.requests
          pkgs.python3Packages.tqdm

        ];
        doCheck = false;
      };
      eccodes3 = pkgs.python3Packages.buildPythonPackage rec {
        pname = "eccodes";
        version = "1.6.0";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-WTkwQLz4nYBIEnQQkmWCqv/GiOF0vr1GRxkwe7xdnhU="; # TODO
        };
        propagatedBuildInputs = [
          pkgs.python3Packages.numpy
          pkgs.eccodes
          pkgs.python3Packages.cffi
          pkgs.python3Packages.attrs
          findlibs
        ];
        doCheck = false;
      };
      cfgrib = pkgs.python3Packages.buildPythonPackage rec {
        pname = "cfgrib";
        version = "0.9.10.3";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-wQgGBYyAxIYQwgG/BespJAGAeAarlCOrSWWuI9u2tSE="; # TODO
        };
        propagatedBuildInputs = [
          pkgs.python3Packages.numpy
          eccodes3
          pkgs.python3Packages.click
          pkgs.python3Packages.attrs
          pkgs.python3Packages.xarray
        ];
        doCheck = false;
      };
      findlibs = pkgs.python3Packages.buildPythonPackage rec {
        pname = "findlibs";
        version = "0.0.5";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-eoAVcemZ0O6D+bksu1mMIfhh7ibKnbp0zqiVi6QzXn4="; # TODO
        };
        propagatedBuildInputs = [ ];
        doCheck = false;
      };
      program = pkgs.python3Packages.buildPythonApplication {
        pname = "equivariant-transformers";
        version = "1.0";

        # buildInputs = [pythonWithPackages];
        propagatedBuildInputs = pythonPackages; # tqdm

        src = ./.;

        checkPhase = ''
          # ${pkgs.python3Packages.pytorch}/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 test.py
          export TORCH_DEVICE="cpu"
          python test.py
        '';
      };

      pythonPackages = with pkgs.python3Packages; [
        psutil
        jupyter
        healpix
        cdsapi
        cfgrib
        netcdf4
        eccodes
        eccodes3
        xarray
        einops
        timm
        ipdb
        dill
        filelock
        gitpython
        tqdm
        python-lsp-server
        python-lsp-ruff
        numpy
        transformers
        peft
        datasets
        pytorch
        torchvision
        onnxruntime
        plotext
        torchmetrics
        ipython
        black
        flake8
        wandb
        snakeviz
        pandas
        matplotlib
        psycopg
        psycopg2
        pytest
        sqlalchemy
      ];

      pythonWithPackages = pkgs.python3.withPackages (p: pythonPackages);
      devinputs = with pkgs; [
      lazygit
      linuxPackages_latest.perf
        glfw3
        glm
        xorg.libX11
        xorg.libpthreadstubs
        xorg.libXau
        xorg.libXdmcp
        xorg.libXrandr
        xorg.libXinerama
        xorg.libXcursor
        xorg.libXi
        xclip
        libGL
        libGLU

        pkg-config
        openssl
        (rustToolchain.withComponents [
          "cargo"
          "rustc"
          "rust-src"
          "rustfmt"
          "clippy"
        ])
        fenix.packages."${system}".rust-analyzer
        nil
        julia
        postgresql
        ruff
        helixmaster
        nixfmt
        jq
        yazi
        nerdfonts
        poppler
        fzf
        (ueberzugpp.override {
          enableWayland = false;
          enableOpencv = false;
        })
        (rWrapper.override {
          packages = with rPackages; [
            ggplot2
            dplyr
            latex2exp
            patchwork
            reticulate
            Hmisc
            RPostgreSQL
            plotly
          ];
        })
        (rstudioWrapper.override {
          packages = with rPackages; [
            ggplot2
            dplyr
            patchwork
            reticulate
            Hmisc
            RPostgreSQL
            plotly
            esquisse
            matlab
            ggExtra
            ggpubr
          ];
        })
        # cudatoolkit
        # python
        pythonWithPackages
      ];
    in {
      packages.x86_64-linux.default = program;
      packages.aarch64-darwin.default = program;
      # devShells.x86_64-linux.default = pkgs.mkShellNoCC {
      devShells.x86_64-linux.default =
        (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_14.stdenv; }) {
          buildInputs = devinputs;
          nativeBuildInputs = [ pkgs.cudatoolkit ];
          shellHook = ''
            export POSTGRES=${pkgs.postgresql}
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib/"
            export EDITOR=hx
            export PYTHON=${pythonWithPackages}/bin/python
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib/:${
              pkgs.lib.makeLibraryPath ([
                pkgs.libGL
                pkgs.libGLU
                pkgs.xorg.libX11
                pkgs.xorg.libXcursor
                pkgs.xorg.libXrandr
                pkgs.xorg.libXi
                pkgs.pipewire
              ])
            }"
            export X11_X11_INCLUDE_PATH="${pkgs.xorg.libX11}/include"
            export X11_X11_LIB=${
              pkgs.lib.makeLibraryPath ([ pkgs.xorg.libX11 ])
            }

            # export CUDA_PATH=${pkgs.cudatoolkit}
          '';
        };
      devShells.aarch64-darwin.default =
        pkgs.mkShellNoCC { buildInputs = devinputs; };

      sing = pkgs.singularity-tools.buildImage {
        name = "equivariant-posteriors";
        diskSize = 1024 * 200;
        memSize = 1024 * 15;
        contents = [
          (pkgs.buildEnv {
            name = "root";
            paths = [
              pkgs.bashInteractive
              pkgs.coreutils
              pkgs.findutils
              pkgs.gnugrep
              program
              devinputs
            ];
            pathsToLink = [ "/bin" "/share" ];
          })
          program
          devinputs
          pkgs.gnugrep
          pkgs.findutils
          pkgs.vim
          pkgs.nix
          pkgs.glibcLocales
          pkgs.which
        ];
        # [ program (pkgs.python3.withPackages (p: [ p.numpy p.pytorch ])) ];
        runScript = ''
          #!${pkgs.stdenv.shell}
          export POSTGRES=${pkgs.postgresql}
          export LOCALE_ARCHIVE=${pkgs.glibcLocales}/lib/locale/locale-archive
          exec /bin/sh $@'';
        runAsRoot =
          "   #!${pkgs.stdenv.shell}\n   ${pkgs.dockerTools.shadowSetup}\n";
      };

    };
}
