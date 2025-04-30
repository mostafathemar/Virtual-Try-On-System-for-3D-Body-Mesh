# Virtual-Try-On-System-for-3D-Body-Mesh
A robust virtual try-on system that aligns and fits 3D clothing onto SMPL-X body meshes with collision-aware positioning and multi-format visualization.

## Features

- **SMPL-X Landmark Alignment**: Precise shoulder and hip landmark matching
- **Collision-Aware Fitting**: KDTree-accelerated penetration resolution
- **Multi-Format Support**: Handles OBJ, GLTF, USDZ, and ABC formats
- **Cross-Platform Visualization**: PyRender, Matplotlib, and HTML outputs
- **Gender-Specific Processing**: Supports male and female body types

## Core Pipeline Architecture
``` graph TD
        A[Input Meshes] --> B[Landmark Extraction]
        B --> C[Geometric Alignment]
        C --> D[Collision Resolution]
        D --> E[Visualization]
        E --> F[Output Generation]
```    
## Installation

```bash
# Clone repository
git clone https://github.com/mostafathemar/Virtual-Try-On-System-for-3D-Body-Mesh.git
cd virtual-try-on
```
# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

## License:
Please see the LICENSE file for details.

## Contact:
If you have any questions, please feel free to contact us at mostafathemar@gmail.com.
