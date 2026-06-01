{
  description = "Meeting transcription with speaker diarization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311
            ffmpeg
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi
            source .venv/bin/activate

            if ! python -c "import faster_whisper" 2>/dev/null; then
              echo "Installing Python dependencies (this may take a few minutes)..."
              pip install -q -r requirements.txt
            fi

            echo ""
            echo "Transcription environment ready."
            echo "Usage: python transcribe.py <audio_or_video_file>"
            echo "       HF_TOKEN=<token> python transcribe.py <file>"
            echo ""
          '';
        };
      }
    );
}
