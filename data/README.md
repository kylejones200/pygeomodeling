# Sample Data

This directory contains sample GRDECL files for testing and tutorials.

## Files

### `sample_small.grdecl`
- **Size**: 5×5×3 grid (75 cells)
- **Purpose**: Quick testing and tutorials
- **Properties**: PERMX, PERMY, PERMZ, PORO, NTG
- **Description**: Simplified reservoir model with 3 layers of varying permeability

**Characteristics:**
- Layer 1: High permeability (~500 mD), good porosity (0.25)
- Layer 2: Medium permeability (~250 mD), moderate porosity (0.20)
- Layer 3: Low permeability (~100 mD), lower porosity (0.15)

### `SPE9.GRDECL` (if available)
- **Size**: 24×25×15 grid (9,000 cells)
- **Purpose**: Full-scale testing and benchmarking
- **Source**: SPE Comparative Solution Project
- **Note**: Download separately from SPE website if needed

## Usage

### Python

```python
from spe9_geomodeling import GRDECLParser

# Load sample data
parser = GRDECLParser('data/sample_small.grdecl')
data = parser.load_data()

print(f"Grid dimensions: {data['dimensions']}")
print(f"Properties: {list(data['properties'].keys())}")
```

### Command Line

```bash
# View sample data
python -c "from spe9_geomodeling import GRDECLParser; \
           p = GRDECLParser('data/sample_small.grdecl'); \
           d = p.load_data(); \
           print('Loaded:', list(d['properties'].keys()))"
```

## Creating Your Own Sample Data

To create custom sample GRDECL files:

1. Start with the SPECGRID keyword defining dimensions
2. Add property sections (PERMX, PORO, etc.)
3. Ensure each property has nx × ny × nz values
4. Use Fortran column-major ordering

Example structure:
```
SPECGRID
nx ny nz 1 F
/

PERMX
value1 value2 ... valueN
/

PORO
value1 value2 ... valueN
/
```

## Data Format

GRDECL (Grid Eclipse) format specifications:
- Keywords in UPPERCASE
- Data sections end with `/`
- Comments start with `--`
- Values can be in scientific notation (e.g., 1.5E+02)
- Fortran column-major order (varies fastest in X, then Y, then Z)

## License

Sample data files are provided for educational and testing purposes.
