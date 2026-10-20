# TriWeave data snapshot

The full server-side `data/` tree is preserved as assets of the private GitHub
Release [`data-snapshot-20261020-v1`](https://github.com/idealism5114/TriWeave/releases/tag/data-snapshot-20261020-v1).
It is deliberately excluded from Git history. No virtual environment, model
checkpoint directory, runtime log directory, or experiment directory is included.

The six archives preserve `data/raw`, `data/standardized`, `data/derived`,
`data/manifests`, `data/audit`, and `data/sealed`, including existing dataset
versions and research hold states. This backup does not change any data protocol
or certify historical point-in-time availability or futures roll definitions.

`manifest.json` records the size and SHA256 of every original data file and every
archive. All archive members were independently read back and checked against
their source-file hashes before upload. `SHA256SUMS` verifies the downloaded
archives and manifest. The server's original files are left in place.

## Restore

From a clone of this private repository, with GitHub CLI authenticated:

```bash
mkdir -p .upload-work/download
gh release download data-snapshot-20261020-v1 \
  --repo idealism5114/TriWeave \
  --dir .upload-work/download
```

Verify before extraction:

```bash
cd .upload-work/download
sha256sum -c SHA256SUMS
cd ../..
```

Extract into an empty directory first, avoiding changes to an existing database:

```bash
mkdir -p .upload-work/restored
for archive in .upload-work/download/triweave-data-*.tar.gz; do
  tar -xzf "$archive" -C .upload-work/restored
done
```

The restored tree is `.upload-work/restored/data/`. Build a fresh Python virtual
environment separately; the server environment is not part of this backup.
