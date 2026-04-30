"""Tests for ``mhcflurry.common.write_generate_sh``.

The helper drops a snapshot script next to every artifact directory
written by a train/select/calibrate command. The script's default
action is to tar.gz the directory; ``regenerate`` re-runs the original
mhcflurry invocation first.
"""
import os
import subprocess
import tempfile

import mhcflurry.common as common


def test_writes_executable_generate_sh():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "models.combined")
        os.makedirs(out_dir)

        path = common.write_generate_sh(
            out_dir,
            argv=["mhcflurry-class1-train-pan-allele-models",
                  "--data", "/in path/data.csv",
                  "--out-models-dir", out_dir, "--num-jobs", "0"],
            mhcflurry_version="9.9.9-test",
        )
        assert path == os.path.join(out_dir, "GENERATE.sh")
        assert os.path.exists(path)
        assert os.access(path, os.X_OK)

        contents = open(path).read()
        assert "mhcflurry-class1-train-pan-allele-models" in contents
        # Spaces in args must round-trip correctly via shlex.quote.
        assert "/in path/data.csv" in contents
        assert "9.9.9-test" in contents


def test_default_run_produces_tar_gz_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "models.combined")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "manifest.csv"), "w") as f:
            f.write("model_name,fold\nNet0,0\n")

        common.write_generate_sh(out_dir, argv=["echo", "noop"])
        result = subprocess.run(
            ["bash", os.path.join(out_dir, "GENERATE.sh")],
            capture_output=True, text=True, check=True)

        tarball = os.path.join(tmp, "models.combined.tar.gz")
        assert os.path.exists(tarball), result.stdout + "\n" + result.stderr

        listing = subprocess.run(
            ["tar", "tzf", tarball],
            capture_output=True, text=True, check=True)
        assert "models.combined/manifest.csv" in listing.stdout


def test_regenerate_reruns_command_then_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "models.combined")
        os.makedirs(out_dir)
        # The "regenerate" path will rm -rf the dir first, so put the
        # marker file outside the dir to confirm the recorded command
        # actually ran.
        marker = os.path.join(tmp, "regenerated_marker.txt")
        # Use `bash -c` to embed both the dir recreate and the marker
        # write in one command, so the recorded ORIGINAL_COMMAND
        # restores out_dir (otherwise the trailing tarball step fails
        # because the dir was rm -rf'd).
        argv = ["bash", "-c",
                f"mkdir -p {out_dir} && touch {marker}"]
        common.write_generate_sh(out_dir, argv=argv)

        result = subprocess.run(
            ["bash", os.path.join(out_dir, "GENERATE.sh"), "regenerate"],
            capture_output=True, text=True, check=True)
        assert os.path.exists(marker), result.stdout + "\n" + result.stderr
        assert os.path.exists(os.path.join(tmp, "models.combined.tar.gz"))


if __name__ == "__main__":
    test_writes_executable_generate_sh()
    test_default_run_produces_tar_gz_snapshot()
    test_regenerate_reruns_command_then_snapshots()
    print("PASS")
