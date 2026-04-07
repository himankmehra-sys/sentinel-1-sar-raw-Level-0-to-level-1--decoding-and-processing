# Sentinel-1 SAR RAW Data Processing

This project implements an end-to-end pipeline for decoding and processing Sentinel-1 Level-0 RAW SAR data into focused SAR images.

## 🚀 Features

- Sentinel-1 RAW packet decoding
- BAQ (Block Adaptive Quantization) decoding
- Ephemeris extraction from sub-commutated data
- Range-Doppler Algorithm implementation
- Range Cell Migration Correction (RCMC)
- Azimuth compression
- SAR image generation

## 🧠 Pipeline Overview

1. Read RAW `.dat` file
2. Extract metadata & packets
3. Decode radar IQ data
4. Convert to frequency domain
5. Apply range compression
6. Apply Doppler correction
7. Perform azimuth compression
8. Generate SAR image

## 📁 Project Structure
