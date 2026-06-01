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

            req_hash=$(sha256sum requirements.txt | cut -d' ' -f1)
            stamp=.venv/.requirements-hash
            if [ ! -f "$stamp" ] || [ "$(cat $stamp)" != "$req_hash" ]; then
              echo "Installing Python dependencies (this may take a few minutes)..."
              pip install -q -r requirements.txt && echo "$req_hash" > "$stamp"
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
