# Proposal: Change ycen to Fractional Offsets Only

**Date**: 2025-12-08
**Status**: Proposed (not yet implemented)

## Summary

Change the `ycen` parameter in `slitdec()` to accept only fractional pixel offsets [0-1), and have the algorithm internally add `nrows/2` to center the extraction. This better matches the real-world preprocessing workflow and clarifies the interface.

## Current Situation

### Real-world Preprocessing Workflow

In production use, there is an outer loop around `slitdec()` that:

1. **Identifies each spectral order** in the full detector image
2. **Cuts out a rectangle** containing just that order (typically narrow, e.g., 176 rows)
3. **Column-shifts each column** by `floor(ycen[col])` to align the order horizontally
4. **Result**: A narrow strip where:
   - Rows are nearly aligned across columns
   - Only fractional pixel shifts (0-1) remain to be corrected
   - The strip is centered approximately at the middle rows

### Current Test Data

The existing test data files (CRIRES1, Hsim, Rsim, etc.) already have this preprocessing applied:

- **CRIRES1**: 176 rows × 2048 columns
  - `ycen` values: ~88.5 (middle row ≈ 88, plus fractional offset ≈ 0.5)
  - Already cut from full detector
  - Already column-shifted to align the order

### Current ycen Format

**Current behavior**: `ycen` contains absolute row positions

```python
# CRIRES1 example
ycen = [88.535, 88.524, 88.513, ...]  # Absolute row positions
```

The C code extracts:
```c
ycen_offset = floor(ycen)      // Integer row: 88
ycen_frac = ycen - floor(ycen)  // Fractional part: 0.535, 0.524, ...
```

### Problem with Current Format

When converting data from other sources (e.g., CR2RES trace tables):

- **Y1029 det2 order4**: 2048 rows × 2048 columns
  - `ycen` values: ~887 (absolute position in full detector)
  - **Not preprocessed** - still at original detector coordinates
  - Cannot be directly used with `slitdec()` without preprocessing

The current format conflates two pieces of information:
1. **Where** the order is located (integer row offset)
2. **Fractional alignment** within pixels

## Proposed Change

### New ycen Format

**Proposed behavior**: `ycen` contains only fractional offsets [0-1)

```python
# CRIRES1 example (after conversion)
ycen = [0.535, 0.524, 0.513, ...]  # Fractional offsets only
```

### Algorithm Change

`slitdec()` internally adds `nrows/2` to center the extraction:

```c
// In C code, before using ycen:
for (int x = 0; x < ncols; x++) {
    ycen[x] += nrows / 2.0;
}

// Then extract as before:
ycen_offset = floor(ycen)      // Integer row offset from center
ycen_frac = ycen - floor(ycen)  // Fractional alignment
```

## Rationale

### 1. Matches Preprocessing Workflow

The preprocessing already:
- Cuts to a narrow strip (removing the integer offset context)
- Column-shifts by `floor(ycen)` (removing absolute positioning)
- Leaves only fractional shifts to be handled

The ycen format should reflect this - it's describing **fractional alignment within the strip**, not **absolute position in the original detector**.

### 2. Clarifies the Interface

**Current (ambiguous)**:
- Is ycen absolute? Relative to what?
- Why is CRIRES1 ycen ≈ 88 for a 176-row image?

**Proposed (clear)**:
- ycen is always fractional offset from pixel boundaries [0-1)
- The algorithm centers the extraction at `nrows/2`
- Preprocessing has already handled absolute positioning

### 3. Simplifies Data Preparation

**Current**: Different sources need different ycen handling
- Test data: Keep absolute positions (88.5)
- CR2RES: Need to figure out how to shift to middle

**Proposed**: All sources use the same format
- Extract fractional part: `ycen_frac = ycen - floor(ycen)`
- Save only the fractional part
- Consistent across all data sources

### 4. Makes Physical Sense

The fractional offset represents:
- **Sub-pixel alignment**: Where within each pixel the spectrum falls
- **Independent of detector size**: Same meaning for 176-row or 2048-row strips
- **Invariant to centering**: Describes the pattern, not the absolute position

## Implementation Plan

### 1. Update C Code

In `slitdec/slitdec.c`, add centering before using ycen:

```c
// After input validation, before main algorithm
double center_row = nrows / 2.0;
for (int x = 0; x < ncols; x++) {
    ycen[x] += center_row;
}
```

**Note**: Since the wrapper creates a copy of ycen, this doesn't modify the Python input.

### 2. Update Wrapper Documentation

In `slitdec/slitdec_wrapper.cpp`, update docstring:

```cpp
"ycen : ndarray (ncols,)\n"
"    Fractional y-offset from pixel boundaries for each column [0-1).\n"
"    The algorithm will center the extraction at nrows/2.\n"
"    Should contain only fractional parts after preprocessing.\n"
```

### 3. Update Test Data Preparation

**make_curvedelta.py**: Save fractional ycen

```python
# After fitting, extract fractional part
ycen_frac = ycen - np.floor(ycen)

np.savez(
    output_file,
    ycen=ycen_frac,  # Save fractional part only
    ...
)
```

### 4. Update Conversion Scripts

**cr2res_to_curvedelta.py**: Save fractional ycen

```python
# Evaluate center line position
ycen = np.polyval(center_poly[::-1], x_cols)

# Extract fractional part
ycen_frac = ycen - np.floor(ycen)

np.savez(
    output_path,
    ycen=ycen_frac,  # Save fractional part only
    ...
)
```

### 5. Update Existing Test Data

Regenerate all curvedelta NPZ files:
- CRIRES1: ycen 88.5 → 0.5
- Hsim, Rsim, etc.: Extract fractional parts
- Verify tests still pass

### 6. Update Documentation

**CLAUDE.md**: Document the ycen format convention
**interface.md**: Update parameter description

## Migration Path

1. ✅ **Document proposal** (this file)
2. Review and discuss
3. Update C code
4. Update wrapper docs
5. Update data preparation scripts
6. Regenerate test data
7. Run tests to verify
8. Update all documentation
9. Commit changes

## Backward Compatibility

**Breaking change**: This changes the expected input format for ycen.

**Impact**:
- Internal test data: Can be regenerated
- External users: Need to update their ycen preparation
  - Simple change: `ycen_new = ycen_old - np.floor(ycen_old)`

**Mitigation**:
- Clear documentation of the change
- Version bump to indicate API change
- Migration guide in release notes

## Example: Before and After

### Before (Current)

```python
# CRIRES1: 176 rows
ycen = np.array([88.535, 88.524, 88.513, ...])  # Absolute positions

result = slitdec(
    im,           # 176 × 2048
    pix_unc,
    mask,
    ycen,         # Absolute row positions
    slitcurve,
    slitdeltas,
)
```

### After (Proposed)

```python
# CRIRES1: 176 rows
ycen = np.array([0.535, 0.524, 0.513, ...])  # Fractional offsets only

result = slitdec(
    im,           # 176 × 2048
    pix_unc,
    mask,
    ycen,         # Fractional offsets [0-1)
    slitcurve,    # Algorithm adds nrows/2 internally
    slitdeltas,
)
```

## Questions to Resolve

1. Should we validate that ycen is in [0-1)? Or allow any fractional values?
2. Should the centering be `nrows / 2` or `(nrows - 1) / 2` or `(nrows + 1) / 2`?
3. Do we need a flag to disable auto-centering for edge cases?

## References

- Current C code: `slitdec/slitdec.c` lines 956-961 (ycen_offset extraction)
- Wrapper: `slitdec/slitdec_wrapper.cpp` lines 180-210 (ycen parameter)
- Test fixture: `tests/conftest.py` lines 356-444 (ycen loading)
- Documentation: `CLAUDE.md` section on ycen values
