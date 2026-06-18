# Processed data artifacts

This directory is reserved for local processed data artifacts required by the inference services.

The files in this directory may contain private data and must not be committed to git.

## Expected local file for Stage 2

```text
cv_normalized.parquet
```

This file should be generated locally by the preprocessing pipeline and placed here:

```text
data/processed/v1/cv_normalized.parquet
```

Docker Compose mounts this directory into the Stage 2 ranking service as:

```text
/app/artifacts/cv
```

