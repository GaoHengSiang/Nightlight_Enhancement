This repository contains scripts for "ENHANCEMENT OF IMAGES UNDER NIGHTLIGHT EFFECT BY PERCEPTUAL
ANCHORING", authored by Heng-Siang Gao, Nguyen Trung Kien Hoang, and Homer H. Chen. 

# Nightlight Enhance CLI Manual

## Description
A Python script to process an image file with adjustable nightlight enhancement.

---

## Syntax
```bash
python nightlight_enhance.py -i <input_file> -s <strength> [-m]
```

| Argument               | Required | Type     | Description                                    |
|------------------------|----------|----------|------------------------------------------------|
| `-i`, `--input`        | Yes      | `str`    | Path to the input image file.                 |
| `-s`, `--strength`     | Yes      | `float`  | Nightlight strength (0 to 100).               |
| `-m`, `--map`          | No       | `flag`   | Enable post-gamut mapping (default: disabled).|

