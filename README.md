# Asevoxel-Compute-Binaries

Independently rebuilt, checksummed copies of the two native binaries
[AseVoxel-Dev](https://github.com/matiasman1/Asevoxel-Dev)'s Compute render
mode ships: `asevoxel_compute.exe` (the OpenCL compute worker) and
`asevoxel_native.dll` (the Lua native-render bridge).

## Why this repo exists

Both files are small, unsigned, frequently-rebuilt Windows binaries — exactly
the profile antivirus heuristics flag by default, signed or not. AseVoxel
doesn't (yet) have a code-signing certificate, so instead of asking you to
just trust an unsigned `.exe`, this repo lets you verify it yourself:

1. A GitHub Actions workflow in this repo checks out the exact source of
   [Asevoxel-Dev](https://github.com/matiasman1/Asevoxel-Dev) at a given
   commit, on a clean, disposable, GitHub-hosted Windows runner — no local
   toolchain state, no manual steps.
2. It builds both binaries using the *same* MSYS2/g++ commands
   [`create_extension.ps1`](https://github.com/matiasman1/Asevoxel-Dev/blob/main/create_extension.ps1)
   uses to build the ones actually shipped.
3. It publishes the binaries, their SHA-256 checksums, and a signed
   [build-provenance attestation](https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds)
   (via `actions/attest-build-provenance`) as a GitHub Release here, tagged
   with the exact source commit it was built from.

If the checksum of the copy you have (from the extension .zip, itch.io, or
wherever you got it) matches the checksum published here for the
corresponding source commit, you have independent confirmation it was built
from the public source and hasn't been tampered with in between — you didn't
have to trust the developer's own machine to produce it.

## How to verify your copy

1. Find which build this corresponds to. The extension reports its
   `package.json` version in-app; find the matching
   [release here](https://github.com/matiasman1/Asevoxel-Compute-Binaries/releases)
   (tagged `build-<short-commit-sha>`).
2. Download `CHECKSUMS.sha256` from that release.
3. Hash your local copy and compare:

   **Windows (PowerShell):**
   ```powershell
   Get-FileHash .\asevoxel_compute.exe -Algorithm SHA256
   ```

   **Linux/macOS:**
   ```sh
   sha256sum asevoxel_compute.exe
   ```
4. Compare the hash against the matching line in `CHECKSUMS.sha256`. Match
   means your copy is byte-identical to a binary built directly from the
   public source on a clean runner — safe to add an antivirus exclusion for.
   Mismatch means don't trust it — please
   [open an issue](https://github.com/matiasman1/Asevoxel-Dev/issues) on the
   main repo right away.

For stronger, cryptographic confirmation (not just "this repo says the hash
matches"), you can also verify the attestation directly:

```sh
gh attestation verify asevoxel_compute.exe --repo matiasman1/Asevoxel-Compute-Binaries
```

This checks a signed record, published by GitHub itself, tying those exact
bytes to this exact workflow run and source commit — independent of anything
this README claims.

## Adding a safe antivirus exception

Once you've verified a match, scope any exclusion as narrowly as possible —
exclude the specific file or the extension's install folder
(`%APPDATA%\Aseprite\extensions\asevoxel-viewer`), not your whole Downloads
folder or a disabled real-time protection setting.

## What this doesn't cover

- This repo builds from source *as published*; it can't tell you whether the
  source itself is trustworthy — that's what reading
  [Asevoxel-Dev](https://github.com/matiasman1/Asevoxel-Dev)'s code is for.
- Builds run on a schedule and on demand, not on every single commit — if
  your copy's version isn't published here yet, check back shortly or
  trigger a run manually via the Actions tab.
