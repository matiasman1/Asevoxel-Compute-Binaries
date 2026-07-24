# Mirror provenance

The `source/` directory in this repo is a mirror of exactly the files needed
to build `asevoxel_compute.exe` and `asevoxel_native.dll`, copied out of the
private `Asevoxel-Dev` repository:

- `source/src/` — mirrors `Asevoxel-Dev/src/` (the OpenCL compute worker)
- `source/asevoxel_native.cpp` — mirrors `Asevoxel-Dev/asevoxel_native.cpp`
  (the Lua native-render bridge)

The rest of Asevoxel-Dev (the Lua extension itself, art assets, tests, design
docs) stays private; only the C/C++ source that actually compiles into the
two binaries antivirus software tends to flag is mirrored here, so anyone can
read and independently rebuild exactly what produces those binaries without
needing access to the full private repo.

This mirror is kept in sync automatically: pushes to `Asevoxel-Dev`'s `main`
branch that touch `src/**`, `asevoxel_native.cpp`, or the build commands in
`create_extension.ps1` re-push the affected files here via a scheduled
mirror workflow in the private repo.

Last manually-synced from Asevoxel-Dev commit: `cd5686bff2113c3d99e83b3dac2976e9b101182f`
