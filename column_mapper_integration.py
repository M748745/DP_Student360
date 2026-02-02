"""
Column Mapper Integration for Student 360 Analytics
====================================================

This module provides helper functions to integrate the Universal Columns Catalog
into the Student 360 application, making it easy to use semantic column mapping
across all tabs and analyses.

Usage:
    from column_mapper_integration import col, validate_dataframe, get_analysis_columns

    # Resolve column names with semantic mapping
    df[col('gpa')]  # Automatically resolves to 'cumulative_gpa'
    df[col('aid_amount')]  # Automatically resolves to 'financial_aid_monetary_amount'

    # Validate dataframe against catalog
    validate_dataframe(df)

    # Get columns for specific analysis
    columns = get_analysis_columns('performance_comparison')
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from universal_columns_catalog import (
    get_mapper,
    get_column_group,
    ColumnCategory,
    ColumnDefinition,
    UNIVERSAL_COLUMNS_CATALOG
)

# Initialize global mapper instance
_MAPPER = get_mapper()


# ═════════════════════════════════════════════════════════════════════════════
# CORE COLUMN RESOLUTION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def col(column_name: str) -> str:
    """
    Resolve a column name or alias to its canonical form.
    This is the primary function to use throughout the application.

    Args:
        column_name: Column name or alias

    Returns:
        Canonical column name

    Raises:
        ValueError: If column name cannot be resolved

    Example:
        df[col('gpa')]  # Returns cumulative_gpa
        df[col('aid')]  # Returns financial_aid_monetary_amount
        df.groupby(col('nationality'))
    """
    canonical = _MAPPER.resolve(column_name)
    if canonical is None:
        raise ValueError(
            f"Column '{column_name}' not found in catalog. "
            f"Check spelling or add to universal_columns_catalog.py"
        )
    return canonical


def col_safe(column_name: str, default: str = None) -> Optional[str]:
    """
    Safely resolve a column name, returning None or default if not found

    Args:
        column_name: Column name or alias
        default: Default value to return if not found

    Returns:
        Canonical column name, default, or None
    """
    canonical = _MAPPER.resolve(column_name)
    return canonical if canonical is not None else default


def cols(*column_names: str) -> List[str]:
    """
    Resolve multiple column names at once

    Args:
        *column_names: Variable number of column names or aliases

    Returns:
        List of canonical column names

    Example:
        df[cols('gpa', 'aid', 'nationality')]
    """
    return [col(name) for name in column_names]


def col_exists(column_name: str) -> bool:
    """
    Check if a column exists in the catalog

    Args:
        column_name: Column name or alias to check

    Returns:
        True if column exists, False otherwise
    """
    return _MAPPER.resolve(column_name) is not None


# ═════════════════════════════════════════════════════════════════════════════
# DATAFRAME VALIDATION & MAPPING
# ═════════════════════════════════════════════════════════════════════════════

def map_dataframe_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Map all DataFrame column names to their canonical forms

    Args:
        df: Input DataFrame
        inplace: If True, modify DataFrame in place

    Returns:
        DataFrame with canonical column names

    Example:
        df = map_dataframe_columns(df)
        # Now df columns use canonical names
    """
    column_mapping = {}
    for col_name in df.columns:
        canonical = _MAPPER.resolve(col_name)
        if canonical:
            column_mapping[col_name] = canonical

    if inplace:
        df.rename(columns=column_mapping, inplace=True)
        return df
    else:
        return df.rename(columns=column_mapping)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate a DataFrame against the column catalog

    Args:
        df: DataFrame to validate
        required_columns: List of required column names (canonical or aliases)

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'missing_columns': List[str],
            'unknown_columns': List[str],
            'mapped_columns': Dict[str, str]
        }
    """
    results = {
        'valid': True,
        'missing_columns': [],
        'unknown_columns': [],
        'mapped_columns': {}
    }

    # Check which columns can be mapped
    for col_name in df.columns:
        canonical = _MAPPER.resolve(col_name)
        if canonical:
            results['mapped_columns'][col_name] = canonical
        else:
            results['unknown_columns'].append(col_name)
            results['valid'] = False

    # Check required columns
    if required_columns:
        for req_col in required_columns:
            canonical = col(req_col)  # This will raise if not in catalog
            if canonical not in df.columns and canonical not in results['mapped_columns'].values():
                # Check if any alias is present
                found = False
                col_def = _MAPPER.get_definition(req_col)
                if col_def:
                    for alias in [canonical] + col_def.aliases:
                        if alias in df.columns:
                            found = True
                            break

                if not found:
                    results['missing_columns'].append(canonical)
                    results['valid'] = False

    return results


def get_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get schema information for DataFrame columns that are in the catalog

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with columns: column_name, canonical_name, category, data_type, description
    """
    schema_data = []

    for col_name in df.columns:
        col_def = _MAPPER.get_definition(col_name)
        if col_def:
            schema_data.append({
                'column_name': col_name,
                'canonical_name': col_def.name,
                'category': col_def.category.value,
                'data_type': col_def.data_type.value,
                'description': col_def.description,
                'nullable': col_def.nullable
            })

    return pd.DataFrame(schema_data)


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS-SPECIFIC HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_analysis_columns(analysis_type: str) -> List[str]:
    """
    Get recommended columns for a specific analysis type

    Args:
        analysis_type: Type of analysis (e.g., 'performance_comparison', 'aid_correlation')

    Returns:
        List of canonical column names

    Available analysis types:
        - performance_analysis
        - aid_correlation
        - uae_vs_international_comparison
        - housing_impact_analysis
        - risk_assessment
        - first_gen_comparison
        - etc.
    """
    return _MAPPER.get_columns_for_analysis(analysis_type)


def get_category_columns(category: str) -> List[str]:
    """
    Get all columns in a specific category

    Args:
        category: Category name (e.g., 'academic_performance', 'financial', 'housing')

    Returns:
        List of canonical column names

    Available categories:
        - identifier, personal_info, demographic
        - academic_performance, enrollment, financial
        - housing, grades, risk_success
        - engagement, international, derived
    """
    try:
        category_enum = ColumnCategory(category.lower())
        col_defs = _MAPPER.get_by_category(category_enum)
        return [col_def.name for col_def in col_defs]
    except ValueError:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Valid categories: {[c.value for c in ColumnCategory]}"
        )


def get_column_group(group_name: str) -> List[str]:
    """
    Get a predefined column group

    Args:
        group_name: Name of the group

    Returns:
        List of column names

    Available groups:
        - student_basic_info
        - academic_core
        - financial_core
        - housing_core
        - uae_national_analysis
        - performance_comparison
        - financial_aid_analysis
        - housing_impact_analysis
        - risk_assessment
    """
    from universal_columns_catalog import COLUMN_GROUPS
    group = COLUMN_GROUPS.get(group_name)
    if group is None:
        raise ValueError(
            f"Column group '{group_name}' not found. "
            f"Available groups: {list(COLUMN_GROUPS.keys())}"
        )
    return group


# ═════════════════════════════════════════════════════════════════════════════
# CONDITIONAL COLUMN ACCESS (for optional columns)
# ═════════════════════════════════════════════════════════════════════════════

def get_column_if_exists(df: pd.DataFrame, column_name: str, default_value=None) -> pd.Series:
    """
    Get a column from DataFrame if it exists (by canonical name or alias)

    Args:
        df: DataFrame
        column_name: Column name or alias
        default_value: Value to return as Series if column not found

    Returns:
        Series if column exists, Series with default_value otherwise
    """
    canonical = col_safe(column_name)
    if canonical and canonical in df.columns:
        return df[canonical]

    # Check if any alias is present
    col_def = _MAPPER.get_definition(column_name)
    if col_def:
        for alias in col_def.aliases:
            if alias in df.columns:
                return df[alias]

    # Return default
    return pd.Series([default_value] * len(df), index=df.index)


def has_column(df: pd.DataFrame, column_name: str) -> bool:
    """
    Check if DataFrame has a column (by canonical name or any alias)

    Args:
        df: DataFrame
        column_name: Column name or alias

    Returns:
        True if column exists in DataFrame
    """
    canonical = col_safe(column_name)
    if canonical and canonical in df.columns:
        return True

    # Check aliases
    col_def = _MAPPER.get_definition(column_name)
    if col_def:
        for alias in col_def.aliases:
            if alias in df.columns:
                return True

    return False


# ═════════════════════════════════════════════════════════════════════════════
# FILTERING & QUERYING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def filter_active_students(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for active students only"""
    status_col = col('enrollment_enrollment_status')
    return df[df[status_col] == 'Active']


def filter_graduated_students(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for graduated students only"""
    status_col = col('enrollment_enrollment_status')
    return df[df[status_col] == 'Graduated']


def filter_uae_nationals(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for UAE national students"""
    nationality_col = col('nationality')
    return df[df[nationality_col] == 'UAE']


def filter_international_students(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for international students"""
    nationality_col = col('nationality')
    return df[df[nationality_col] != 'UAE']


def filter_housed_students(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for students with campus housing"""
    room_col = col('room_number')
    return df[df[room_col].notna()]


def filter_non_housed_students(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for students without campus housing"""
    room_col = col('room_number')
    return df[df[room_col].isna()]


def filter_aid_recipients(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for students receiving financial aid"""
    aid_col = col('financial_aid_monetary_amount')
    return df[df[aid_col] > 0]


def filter_high_performers(df: pd.DataFrame, gpa_threshold: float = 3.5) -> pd.DataFrame:
    """Filter for high-performing students (GPA >= threshold)"""
    gpa_col = col('cumulative_gpa')
    return df[df[gpa_col] >= gpa_threshold]


def filter_at_risk_students(df: pd.DataFrame, gpa_threshold: float = 2.5) -> pd.DataFrame:
    """Filter for at-risk students (GPA < threshold)"""
    gpa_col = col('cumulative_gpa')
    return df[df[gpa_col] < gpa_threshold]


# ═════════════════════════════════════════════════════════════════════════════
# COLUMN INFO HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_column_info(column_name: str) -> str:
    """Get formatted information about a column"""
    return _MAPPER.get_column_info(column_name)


def print_column_info(column_name: str):
    """Print information about a column"""
    print(get_column_info(column_name))


def list_all_columns() -> List[str]:
    """Get list of all canonical column names in catalog"""
    return list(UNIVERSAL_COLUMNS_CATALOG.keys())


def search_columns(search_term: str) -> List[str]:
    """
    Search for columns by name or description

    Args:
        search_term: Term to search for

    Returns:
        List of matching canonical column names
    """
    results = []
    search_lower = search_term.lower()

    for name, col_def in UNIVERSAL_COLUMNS_CATALOG.items():
        # Check name
        if search_lower in name.lower():
            results.append(name)
            continue

        # Check aliases
        if any(search_lower in alias.lower() for alias in col_def.aliases):
            results.append(name)
            continue

        # Check description
        if search_lower in col_def.description.lower():
            results.append(name)
            continue

    return results


# ═════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def print_quick_reference():
    """Print quick reference guide for common column operations"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COLUMN MAPPER - QUICK REFERENCE                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
───────────
  from column_mapper_integration import col, cols

  # Single column resolution
  df[col('gpa')]                    # → cumulative_gpa
  df[col('aid')]                    # → financial_aid_monetary_amount
  df.groupby(col('nationality'))    # → nationality

  # Multiple columns
  df[cols('gpa', 'aid', 'room')]    # → ['cumulative_gpa', 'financial_aid_monetary_amount', 'room_number']

FILTERING:
──────────
  from column_mapper_integration import (
      filter_active_students,
      filter_uae_nationals,
      filter_high_performers
  )

  active_df = filter_active_students(df)
  uae_df = filter_uae_nationals(df)
  high_perf_df = filter_high_performers(df, gpa_threshold=3.5)

ANALYSIS COLUMNS:
─────────────────
  from column_mapper_integration import get_analysis_columns, get_column_group

  # Get columns for specific analysis
  cols = get_analysis_columns('aid_correlation')
  df[cols]

  # Get predefined column groups
  academic_cols = get_column_group('academic_core')
  df[academic_cols]

DATAFRAME VALIDATION:
────────────────────
  from column_mapper_integration import validate_dataframe, map_dataframe_columns

  # Validate DataFrame
  results = validate_dataframe(df, required_columns=['student_id', 'gpa'])
  if results['valid']:
      print("✓ DataFrame is valid")

  # Map column names to canonical form
  df = map_dataframe_columns(df)

COLUMN INFO:
────────────
  from column_mapper_integration import print_column_info, search_columns

  # Get info about a column
  print_column_info('gpa')

  # Search for columns
  financial_cols = search_columns('financial')
  print(financial_cols)

COMMON COLUMN ALIASES:
─────────────────────
  gpa                → cumulative_gpa
  aid                → financial_aid_monetary_amount
  tuition            → enrollment_tuition_amount
  room               → room_number
  nationality        → nationality
  status             → enrollment_enrollment_status
  cohort             → cohort_year

ANALYSIS TYPES:
──────────────
  - performance_analysis
  - aid_correlation
  - uae_vs_international_comparison
  - housing_impact_analysis
  - risk_assessment
  - first_gen_comparison

COLUMN GROUPS:
─────────────
  - student_basic_info
  - academic_core
  - financial_core
  - housing_core
  - uae_national_analysis
  - performance_comparison
  - financial_aid_analysis
  - housing_impact_analysis
  - risk_assessment

╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION (for testing)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_quick_reference()

    print("\n" + "=" * 80)
    print("TESTING COLUMN MAPPER INTEGRATION")
    print("=" * 80)

    # Test basic resolution
    print("\n1. Basic column resolution:")
    print(f"   col('gpa') → {col('gpa')}")
    print(f"   col('aid') → {col('aid')}")
    print(f"   col('room') → {col('room')}")

    # Test multiple columns
    print("\n2. Multiple columns:")
    print(f"   cols('gpa', 'aid', 'nationality') → {cols('gpa', 'aid', 'nationality')}")

    # Test analysis columns
    print("\n3. Analysis columns for 'aid_correlation':")
    aid_cols = get_analysis_columns('aid_correlation')
    for ac in aid_cols:
        print(f"   - {ac}")

    # Test column groups
    print("\n4. Academic core columns:")
    academic = get_column_group('academic_core')
    for ac in academic:
        print(f"   - {ac}")

    # Test search
    print("\n5. Search for 'financial':")
    results = search_columns('financial')
    for r in results[:5]:
        print(f"   - {r}")

    print("\n" + "=" * 80)
