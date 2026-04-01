{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-25.11";
    utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nmattia/naersk";
    naersk.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      naersk,
      rust-overlay,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rust = pkgs.rust-bin.stable."1.93.1".default.override {
          targets = [ "x86_64-unknown-linux-musl" ];
          extensions = [
            "llvm-tools-preview"
            "rust-analyzer"
          ];
        };

        naersk-lib = naersk.lib."${system}".override {
          cargo = rust;
          rustc = rust;
        };
      in
      rec {
        packages.h5ad-inspect = naersk-lib.buildPackage {
          pname = "h5ad-inspect";
          root = ./h5ad_inspect;
          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs = with pkgs; [ hdf5 ];
          release = true;
          CARGO_PROFILE_RELEASE_debug = "0";
        };

        packages.h5ad-inspect_other_linux =
          (naersk-lib.buildPackage {
            pname = "h5ad-inspect";
            root = ./h5ad_inspect;
            nativeBuildInputs = with pkgs; [ pkg-config patchelf ];
            buildInputs = with pkgs; [ hdf5 ];
            release = true;
            CARGO_PROFILE_RELEASE_debug = "0";
          }).overrideAttrs
            {
              postInstall = ''
                patchelf $out/bin/h5ad-inspect --set-interpreter "/lib64/ld-linux-x86-64.so.2"
              '';
            };

        packages.h5ad-inspect-docker =
          let
            binary = packages.h5ad-inspect_other_linux;
          in
          pkgs.dockerTools.buildLayeredImage {
            name = "h5ad-inspect";
            tag = "latest";
            contents = [
              pkgs.busybox
              pkgs.glibc
              pkgs.hdf5
              binary
            ];
            config = {
              Env = [ "PATH=/usr/local/bin:/bin" ];
              Entrypoint = [ "/bin/h5ad-inspect" ];
              WorkingDir = "/work";
            };
          };

        packages.check = naersk-lib.buildPackage {
          src = ./h5ad_inspect;
          mode = "check";
          name = "h5ad-inspect";
          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs = with pkgs; [ hdf5 ];
        };

        packages.test = naersk-lib.buildPackage {
          pname = "h5ad-inspect";
          root = ./h5ad_inspect;
          mode = "test";
          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs = with pkgs; [ hdf5 ];
        };

        defaultPackage = packages.h5ad-inspect;

        apps.h5ad-inspect = utils.lib.mkApp { drv = packages.h5ad-inspect; };
        defaultApp = apps.h5ad-inspect;

        devShell = pkgs.mkShell {
          shellHook = ''
            #export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
          '';
          nativeBuildInputs = [
            pkgs.bacon
            pkgs.cargo-nextest
            pkgs.hdf5
            pkgs.mold
            pkgs.pkg-config
            pkgs.ripgrep
            rust
          ];
        };
      }
    );
}
