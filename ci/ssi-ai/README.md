# SSi-AI CI tooling

This directory contains the reusable causal local sparse-attention implementation, daily runner, and rolling-median performance gate used by GitHub Actions.

The CUDA workflow is scheduled/manual and runs on a self-hosted runner labeled `self-hosted,linux,x64,cuda,gpu`. It stores recent benchmark JSON files on the `ssi-ai-history` branch and uploads the current JSON, CSV, plot, and regression report as an Actions artifact.
