"""
Universal Columns Catalog for Student 360 Analytics Platform
==============================================================

This module defines a comprehensive, semantic column catalog that serves as the
single source of truth for all column references across all tabs and analyses.

Features:
- Centralized column definitions with semantic categories
- Flexible column name resolution (handles variations and aliases)
- Data type specifications
- Business logic and validation rules
- Semantic grouping for analysis patterns
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


# ═════════════════════════════════════════════════════════════════════════════
# ENUMS FOR SEMANTIC CATEGORIES
# ═════════════════════════════════════════════════════════════════════════════

class ColumnCategory(Enum):
    """Semantic categories for columns"""
    # Student Analytics Categories
    IDENTIFIER = "identifier"
    PERSONAL_INFO = "personal_info"
    DEMOGRAPHIC = "demographic"
    ACADEMIC_PERFORMANCE = "academic_performance"
    ENROLLMENT = "enrollment"
    FINANCIAL = "financial"
    HOUSING = "housing"
    GRADES = "grades"
    RISK_SUCCESS = "risk_success"
    ENGAGEMENT = "engagement"
    INTERNATIONAL = "international"
    DERIVED = "derived"

    # Data Warehouse Categories
    RAW_TABLE = "raw_table"
    DIMENSION_TABLE = "dimension_table"
    FACT_TABLE = "fact_table"
    CURATED_TABLE = "curated_table"

    # Data Governance Categories
    DATA_CLASSIFICATION = "data_classification"
    TECHNICAL_METADATA = "technical_metadata"
    BUSINESS_METADATA = "business_metadata"
    PERFORMANCE_MONITORING = "performance_monitoring"


class DataType(Enum):
    """Data types for validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"


# ═════════════════════════════════════════════════════════════════════════════
# COLUMN DEFINITION CLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnDefinition:
    """Definition of a single column with metadata"""
    name: str                           # Primary column name
    category: ColumnCategory            # Semantic category
    data_type: DataType                 # Expected data type
    description: str                    # Business description
    aliases: List[str] = None          # Alternative names
    nullable: bool = True               # Can contain null values
    validation_rules: Dict[str, Any] = None  # Validation constraints
    business_logic: str = None         # Business rules and calculations
    used_for: List[str] = None         # Common use cases

    def __post_init__(self):
        """Initialize default values"""
        if self.aliases is None:
            self.aliases = []
        if self.validation_rules is None:
            self.validation_rules = {}
        if self.used_for is None:
            self.used_for = []


# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSAL COLUMNS CATALOG
# ═════════════════════════════════════════════════════════════════════════════

UNIVERSAL_COLUMNS_CATALOG = {

    # ═══════════════════════════════════════════════════════════════════
    # IDENTIFIERS
    # ═══════════════════════════════════════════════════════════════════

    "student_id": ColumnDefinition(
        name="student_id",
        category=ColumnCategory.IDENTIFIER,
        data_type=DataType.STRING,
        description="Unique student identifier",
        aliases=["studentid", "id", "student_number"],
        nullable=False,
        used_for=["primary_key", "unique_counting", "joins"]
    ),

    "emirates_id": ColumnDefinition(
        name="emirates_id",
        category=ColumnCategory.IDENTIFIER,
        data_type=DataType.STRING,
        description="UAE National ID number",
        aliases=["eid", "national_id"],
        nullable=True,
        used_for=["uae_national_identification", "compliance"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # PERSONAL INFORMATION
    # ═══════════════════════════════════════════════════════════════════

    "first_name_en": ColumnDefinition(
        name="first_name_en",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student first name in English",
        aliases=["firstname_en", "given_name"],
        used_for=["student_identification", "communications"]
    ),

    "last_name_en": ColumnDefinition(
        name="last_name_en",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student last name in English",
        aliases=["lastname_en", "family_name", "surname"],
        used_for=["student_identification", "communications"]
    ),

    "first_name_ar": ColumnDefinition(
        name="first_name_ar",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student first name in Arabic",
        aliases=["firstname_ar"],
        used_for=["localization", "official_documents"]
    ),

    "last_name_ar": ColumnDefinition(
        name="last_name_ar",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student last name in Arabic",
        aliases=["lastname_ar"],
        used_for=["localization", "official_documents"]
    ),

    "middle_name": ColumnDefinition(
        name="middle_name",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student middle name",
        aliases=["middlename"],
        used_for=["full_name_construction"]
    ),

    "email_address": ColumnDefinition(
        name="email_address",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student email address (unified field)",
        aliases=["email"],
        business_logic="Aggregated from university_email or personal_email",
        used_for=["communications", "contact"]
    ),

    "university_email": ColumnDefinition(
        name="university_email",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="University-issued email address",
        aliases=["institutional_email"],
        used_for=["official_communications", "contact"]
    ),

    "personal_email": ColumnDefinition(
        name="personal_email",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student's personal email address",
        used_for=["alternative_contact", "communications"]
    ),

    "phone_number": ColumnDefinition(
        name="phone_number",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.STRING,
        description="Student contact phone number",
        aliases=["phone", "mobile", "contact_number"],
        used_for=["communications", "contact"]
    ),

    "date_of_birth": ColumnDefinition(
        name="date_of_birth",
        category=ColumnCategory.PERSONAL_INFO,
        data_type=DataType.DATE,
        description="Student date of birth",
        aliases=["dob", "birthdate"],
        used_for=["age_calculation", "demographics"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DEMOGRAPHIC INFORMATION
    # ═══════════════════════════════════════════════════════════════════

    "gender": ColumnDefinition(
        name="gender",
        category=ColumnCategory.DEMOGRAPHIC,
        data_type=DataType.CATEGORICAL,
        description="Student gender",
        aliases=["sex"],
        validation_rules={"allowed_values": ["Male", "Female", "Other"]},
        used_for=["demographic_analysis", "diversity_metrics"]
    ),

    "nationality": ColumnDefinition(
        name="nationality",
        category=ColumnCategory.DEMOGRAPHIC,
        data_type=DataType.STRING,
        description="Student nationality (country code or name)",
        aliases=["country", "citizenship"],
        used_for=["diversity_analysis", "regional_grouping", "uae_vs_international"]
    ),

    "age_at_first_enrollment": ColumnDefinition(
        name="age_at_first_enrollment",
        category=ColumnCategory.DEMOGRAPHIC,
        data_type=DataType.INTEGER,
        description="Student age at first enrollment",
        aliases=["enrollment_age", "starting_age"],
        validation_rules={"min": 16, "max": 100},
        used_for=["cohort_analysis", "demographic_segmentation"]
    ),

    "total_in_region": ColumnDefinition(
        name="total_in_region",
        category=ColumnCategory.DEMOGRAPHIC,
        data_type=DataType.INTEGER,
        description="Total students from same region",
        business_logic="Aggregated count by region",
        used_for=["regional_analysis", "diversity_metrics"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # ACADEMIC PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════

    "cumulative_gpa": ColumnDefinition(
        name="cumulative_gpa",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.FLOAT,
        description="Overall cumulative GPA (0.0-4.0 scale)",
        aliases=["gpa", "cgpa", "overall_gpa"],
        nullable=True,
        validation_rules={"min": 0.0, "max": 4.0},
        business_logic="Performance tiers: High (>=3.5), Mid (2.5-3.5), At-risk (<2.5)",
        used_for=[
            "performance_analysis",
            "risk_scoring",
            "uae_vs_international_comparison",
            "housing_impact_analysis",
            "aid_correlation"
        ]
    ),

    "term_gpa": ColumnDefinition(
        name="term_gpa",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.FLOAT,
        description="Single term/semester GPA",
        aliases=["semester_gpa", "current_gpa"],
        validation_rules={"min": 0.0, "max": 4.0},
        used_for=["term_performance_tracking"]
    ),

    "major_gpa": ColumnDefinition(
        name="major_gpa",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.FLOAT,
        description="Major-specific GPA",
        aliases=["program_gpa"],
        validation_rules={"min": 0.0, "max": 4.0},
        used_for=["program_performance_analysis"]
    ),

    "grade_point": ColumnDefinition(
        name="grade_point",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.FLOAT,
        description="Average grade point value",
        aliases=["avg_grade_point"],
        used_for=["grade_distribution_analysis"]
    ),

    "academic_standing": ColumnDefinition(
        name="academic_standing",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.CATEGORICAL,
        description="Academic status (Good Standing, Probation, etc.)",
        aliases=["standing", "status"],
        validation_rules={"allowed_values": ["Good Standing", "Academic Probation", "Academic Warning", "Dismissed"]},
        used_for=["risk_identification", "intervention_targeting"]
    ),

    "academic_program": ColumnDefinition(
        name="academic_program",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.CATEGORICAL,
        description="Degree program or major",
        aliases=["program", "major", "degree_program"],
        used_for=["program_analysis", "cohort_comparison"]
    ),

    "academic_level": ColumnDefinition(
        name="academic_level",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.CATEGORICAL,
        description="Student academic level or year (e.g., Freshman, Sophomore, Junior, Senior, or 1, 2, 3, 4)",
        aliases=["level", "year_level", "student_level", "class_level", "grade_level"],
        validation_rules={"allowed_values": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate", "1", "2", "3", "4", "5"]},
        used_for=["cohort_analysis", "progression_tracking", "retention_analysis"]
    ),

    "gpa_trend": ColumnDefinition(
        name="gpa_trend",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.STRING,
        description="GPA trajectory over time (Improving, Declining, Stable)",
        validation_rules={"allowed_values": ["Improving", "Declining", "Stable"]},
        business_logic="Calculated from term-by-term GPA changes",
        used_for=["performance_tracking", "intervention_targeting"]
    ),

    "avg_gpa": ColumnDefinition(
        name="avg_gpa",
        category=ColumnCategory.DERIVED,
        data_type=DataType.FLOAT,
        description="Average GPA (aggregated metric)",
        business_logic="Mean of cumulative_gpa across a group",
        used_for=["group_comparisons", "cohort_analysis"]
    ),

    "std_gpa": ColumnDefinition(
        name="std_gpa",
        category=ColumnCategory.DERIVED,
        data_type=DataType.FLOAT,
        description="GPA standard deviation (aggregated metric)",
        business_logic="Standard deviation of cumulative_gpa across a group",
        used_for=["variability_analysis", "consistency_tracking"]
    ),

    "min_gpa": ColumnDefinition(
        name="min_gpa",
        category=ColumnCategory.DERIVED,
        data_type=DataType.FLOAT,
        description="Minimum GPA in group (aggregated metric)",
        business_logic="Minimum cumulative_gpa in a group",
        used_for=["range_analysis", "at_risk_identification"]
    ),

    "max_gpa": ColumnDefinition(
        name="max_gpa",
        category=ColumnCategory.DERIVED,
        data_type=DataType.FLOAT,
        description="Maximum GPA in group (aggregated metric)",
        business_logic="Maximum cumulative_gpa in a group",
        used_for=["range_analysis", "top_performer_identification"]
    ),

    "avg_credits": ColumnDefinition(
        name="avg_credits",
        category=ColumnCategory.DERIVED,
        data_type=DataType.FLOAT,
        description="Average credits attempted (aggregated metric)",
        business_logic="Mean of credits_attempted across a group",
        used_for=["enrollment_intensity_analysis"]
    ),

    "avg_progress": ColumnDefinition(
        name="avg_progress",
        category=ColumnCategory.DERIVED,
        data_type=DataType.FLOAT,
        description="Average degree progress (aggregated metric)",
        business_logic="Mean of degree_progress_pct across a group",
        used_for=["completion_tracking", "cohort_analysis"]
    ),

    "high_performer_rate": ColumnDefinition(
        name="high_performer_rate",
        category=ColumnCategory.DERIVED,
        data_type=DataType.PERCENTAGE,
        description="Percentage of high performers in group",
        business_logic="(Students with GPA >= 3.5 / Total Students) * 100",
        validation_rules={"min": 0, "max": 100},
        used_for=["performance_benchmarking", "program_comparison"]
    ),

    "credits_attempted": ColumnDefinition(
        name="credits_attempted",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.INTEGER,
        description="Total credits attempted/pursued",
        aliases=["credits", "total_credits"],
        validation_rules={"min": 0},
        business_logic="Used to calculate degree progress and enrollment intensity",
        used_for=["progress_tracking", "cohort_comparison"]
    ),

    "degree_progress_pct": ColumnDefinition(
        name="degree_progress_pct",
        category=ColumnCategory.ACADEMIC_PERFORMANCE,
        data_type=DataType.PERCENTAGE,
        description="Percentage of degree completed (0-100)",
        aliases=["progress_pct", "completion_pct"],
        validation_rules={"min": 0, "max": 100},
        used_for=["retention_analysis", "graduation_prediction"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # ENROLLMENT STATUS
    # ═══════════════════════════════════════════════════════════════════

    "enrollment_enrollment_status": ColumnDefinition(
        name="enrollment_enrollment_status",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.CATEGORICAL,
        description="Current enrollment status",
        aliases=["enrollment_status", "status", "student_status"],
        validation_rules={"allowed_values": ["Active", "Graduated", "Suspended", "Withdrawn", "Leave of Absence"]},
        business_logic="Filters: Active vs Graduated for current vs historical cohorts",
        used_for=["active_filtering", "retention_metrics", "graduation_tracking"]
    ),

    "enrollment_type": ColumnDefinition(
        name="enrollment_type",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.CATEGORICAL,
        description="Enrollment category (Full-time, Part-time, Level)",
        aliases=["type", "program_type"],
        used_for=["cohort_grouping", "program_comparison"]
    ),

    "enrollment_date": ColumnDefinition(
        name="enrollment_date",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.DATE,
        description="Initial admission/enrollment date",
        aliases=["admission_date", "start_date"],
        used_for=["cohort_assignment", "tenure_calculation"]
    ),

    "cohort_year": ColumnDefinition(
        name="cohort_year",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.INTEGER,
        description="Year of admission/cohort",
        aliases=["admission_year", "entry_year"],
        business_logic="Primary grouping variable for year-on-year comparisons",
        used_for=[
            "cohort_analysis",
            "retention_tracking",
            "yoy_comparison"
        ]
    ),

    "cohort_term": ColumnDefinition(
        name="cohort_term",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.STRING,
        description="Admission semester/term (Fall, Spring, Summer)",
        aliases=["admission_term", "entry_term"],
        used_for=["sub_cohort_classification"]
    ),

    "expected_graduation": ColumnDefinition(
        name="expected_graduation",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.DATE,
        description="Projected graduation date",
        aliases=["expected_grad_date", "projected_graduation"],
        used_for=["retention_tracking", "progress_monitoring"]
    ),

    "actual_graduation_date": ColumnDefinition(
        name="actual_graduation_date",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.DATE,
        description="Actual graduation completion date",
        aliases=["graduation_date", "completion_date"],
        used_for=["graduation_tracking", "completion_analysis"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # FINANCIAL INFORMATION
    # ═══════════════════════════════════════════════════════════════════

    "enrollment_tuition_amount": ColumnDefinition(
        name="enrollment_tuition_amount",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CURRENCY,
        description="Total tuition charged (AED)",
        aliases=["tuition", "tuition_amount", "tuition_charged"],
        validation_rules={"min": 0, "currency": "AED"},
        used_for=["financial_metrics", "revenue_tracking"]
    ),

    "financial_aid_monetary_amount": ColumnDefinition(
        name="financial_aid_monetary_amount",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CURRENCY,
        description="Financial aid awarded (AED)",
        aliases=["aid_amount", "financial_aid", "aid"],
        validation_rules={"min": 0, "currency": "AED"},
        business_logic="Primary financial metric for aid analysis and equity studies",
        used_for=[
            "aid_recipient_identification",
            "aid_vs_gpa_correlation",
            "aid_distribution_equity",
            "aided_vs_non_aided_comparison"
        ]
    ),

    "scholarship_type": ColumnDefinition(
        name="scholarship_type",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CATEGORICAL,
        description="Type of scholarship/aid received",
        aliases=["aid_type"],
        used_for=["aid_classification", "scholarship_analysis"]
    ),

    "scholarship_amount": ColumnDefinition(
        name="scholarship_amount",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CURRENCY,
        description="Scholarship value (AED)",
        aliases=["scholarship"],
        validation_rules={"min": 0, "currency": "AED"},
        business_logic="Component of total financial aid",
        used_for=["scholarship_tracking"]
    ),

    "sponsorship_type": ColumnDefinition(
        name="sponsorship_type",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CATEGORICAL,
        description="Type of sponsorship",
        used_for=["sponsorship_classification"]
    ),

    "account_balance": ColumnDefinition(
        name="account_balance",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CURRENCY,
        description="Outstanding balance (AED)",
        aliases=["balance_due", "outstanding_balance"],
        validation_rules={"currency": "AED"},
        used_for=["financial_obligations", "collection_tracking"]
    ),

    "has_aid": ColumnDefinition(
        name="has_aid",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.BOOLEAN,
        description="Binary indicator if student received aid",
        business_logic="Derived: True if financial_aid_monetary_amount > 0",
        used_for=["aided_vs_non_aided_comparison"]
    ),

    "balance_due": ColumnDefinition(
        name="balance_due",
        category=ColumnCategory.FINANCIAL,
        data_type=DataType.CURRENCY,
        description="Outstanding balance owed by student (AED)",
        aliases=["outstanding_balance"],
        validation_rules={"currency": "AED"},
        used_for=["collections", "financial_health"]
    ),

    "aid_rate": ColumnDefinition(
        name="aid_rate",
        category=ColumnCategory.DERIVED,
        data_type=DataType.PERCENTAGE,
        description="Percentage of students receiving aid in group",
        business_logic="(Students with aid / Total Students) * 100",
        validation_rules={"min": 0, "max": 100},
        used_for=["aid_distribution_analysis", "equity_metrics"]
    ),

    "collection_pct": ColumnDefinition(
        name="collection_pct",
        category=ColumnCategory.DERIVED,
        data_type=DataType.PERCENTAGE,
        description="Collection rate percentage",
        business_logic="(Amount Collected / Amount Due) * 100",
        validation_rules={"min": 0, "max": 100},
        used_for=["financial_health", "collection_tracking"]
    ),

    "collection_range": ColumnDefinition(
        name="collection_range",
        category=ColumnCategory.DERIVED,
        data_type=DataType.STRING,
        description="Collection range category",
        used_for=["collection_segmentation"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # HOUSING INFORMATION
    # ═══════════════════════════════════════════════════════════════════

    "room_number": ColumnDefinition(
        name="room_number",
        category=ColumnCategory.HOUSING,
        data_type=DataType.STRING,
        description="Campus housing room identifier",
        aliases=["room", "housing_room"],
        nullable=True,
        business_logic="not null = Housed students; null = Non-housed students",
        used_for=[
            "housing_status_identification",
            "housed_vs_non_housed_comparison",
            "housing_impact_analysis"
        ]
    ),

    "housing_status": ColumnDefinition(
        name="housing_status",
        category=ColumnCategory.HOUSING,
        data_type=DataType.CATEGORICAL,
        description="Housing occupancy status",
        aliases=["occupancy_status"],
        validation_rules={"allowed_values": ["Occupied", "Vacant", "Reserved"]},
        used_for=["housing_assignment_tracking"]
    ),

    "rent_amount": ColumnDefinition(
        name="rent_amount",
        category=ColumnCategory.HOUSING,
        data_type=DataType.CURRENCY,
        description="Expected monthly/total rent (AED)",
        aliases=["rent"],
        validation_rules={"min": 0, "currency": "AED"},
        used_for=["housing_revenue_projection"]
    ),

    "rent_paid": ColumnDefinition(
        name="rent_paid",
        category=ColumnCategory.HOUSING,
        data_type=DataType.CURRENCY,
        description="Rent payment collected (AED)",
        validation_rules={"min": 0, "currency": "AED"},
        used_for=["housing_revenue_tracking"]
    ),

    "fee_paid": ColumnDefinition(
        name="fee_paid",
        category=ColumnCategory.HOUSING,
        data_type=DataType.CURRENCY,
        description="Fees collected (AED)",
        validation_rules={"min": 0, "currency": "AED"},
        business_logic="Aggregated with rent_paid for total housing payments",
        used_for=["total_payment_calculation"]
    ),

    "occupancy_status": ColumnDefinition(
        name="occupancy_status",
        category=ColumnCategory.HOUSING,
        data_type=DataType.CATEGORICAL,
        description="Current occupancy status of housing",
        aliases=["housing_occupancy"],
        validation_rules={"allowed_values": ["Occupied", "Vacant", "Reserved", "Maintenance"]},
        used_for=["occupancy_tracking", "capacity_planning"]
    ),

    "has_meal_plan": ColumnDefinition(
        name="has_meal_plan",
        category=ColumnCategory.HOUSING,
        data_type=DataType.BOOLEAN,
        description="Indicator if student has meal plan",
        used_for=["dining_services", "housing_services"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # GRADES & ACADEMIC TRACKING
    # ═══════════════════════════════════════════════════════════════════

    "is_passing_grade": ColumnDefinition(
        name="is_passing_grade",
        category=ColumnCategory.GRADES,
        data_type=DataType.BOOLEAN,
        description="Indicator of passing grade",
        used_for=["passing_rate_calculation"]
    ),

    "is_dfw_grade": ColumnDefinition(
        name="is_dfw_grade",
        category=ColumnCategory.GRADES,
        data_type=DataType.BOOLEAN,
        description="Indicator of D, F, or W (withdrawal) grade",
        business_logic="Risk indicator for academic intervention",
        used_for=["dfw_rate_calculation", "at_risk_identification"]
    ),

    "terms_enrolled": ColumnDefinition(
        name="terms_enrolled",
        category=ColumnCategory.GRADES,
        data_type=DataType.INTEGER,
        description="Number of terms/semesters attended",
        aliases=["semesters_enrolled"],
        validation_rules={"min": 0},
        used_for=["tenure_tracking", "progression_analysis"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # STUDENT SUCCESS & RISK
    # ═══════════════════════════════════════════════════════════════════

    "is_first_generation": ColumnDefinition(
        name="is_first_generation",
        category=ColumnCategory.RISK_SUCCESS,
        data_type=DataType.BOOLEAN,
        description="Indicator if first-generation college student",
        aliases=["first_gen"],
        business_logic="Key demographic for success factor analysis",
        used_for=[
            "first_gen_comparison",
            "equity_analysis",
            "support_targeting"
        ]
    ),

    "risk_score": ColumnDefinition(
        name="risk_score",
        category=ColumnCategory.RISK_SUCCESS,
        data_type=DataType.FLOAT,
        description="Calculated risk metric (0-100)",
        business_logic="Derived from GPA, aid status, enrollment, and other factors",
        validation_rules={"min": 0, "max": 100},
        used_for=["risk_assessment", "intervention_prioritization"]
    ),

    "risk_level": ColumnDefinition(
        name="risk_level",
        category=ColumnCategory.RISK_SUCCESS,
        data_type=DataType.CATEGORICAL,
        description="Risk categorization tier",
        business_logic="Derived from risk_score: Critical (>75), High (50-75), Moderate (25-50), Low (<25)",
        validation_rules={"allowed_values": ["Critical", "High", "Moderate", "Low"]},
        used_for=["risk_tier_analysis", "intervention_planning"]
    ),

    "is_at_risk": ColumnDefinition(
        name="is_at_risk",
        category=ColumnCategory.RISK_SUCCESS,
        data_type=DataType.BOOLEAN,
        description="Binary indicator if student is at risk",
        business_logic="Derived: True if risk_score >= 50 OR cumulative_gpa < 2.5",
        used_for=["at_risk_identification", "intervention_targeting"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # ENGAGEMENT & SUPPORT
    # ═══════════════════════════════════════════════════════════════════

    "engagement_score": ColumnDefinition(
        name="engagement_score",
        category=ColumnCategory.ENGAGEMENT,
        data_type=DataType.FLOAT,
        description="Student engagement metric",
        used_for=["engagement_tracking"]
    ),

    "attendance_rate": ColumnDefinition(
        name="attendance_rate",
        category=ColumnCategory.ENGAGEMENT,
        data_type=DataType.PERCENTAGE,
        description="Class attendance percentage",
        validation_rules={"min": 0, "max": 100},
        used_for=["success_factor_analysis", "at_risk_support"]
    ),

    "advisor_meeting_count": ColumnDefinition(
        name="advisor_meeting_count",
        category=ColumnCategory.ENGAGEMENT,
        data_type=DataType.INTEGER,
        description="Number of advisor meetings attended",
        aliases=["advising_sessions"],
        validation_rules={"min": 0},
        used_for=["support_utilization_tracking"]
    ),

    "counseling_visits_count": ColumnDefinition(
        name="counseling_visits_count",
        category=ColumnCategory.ENGAGEMENT,
        data_type=DataType.INTEGER,
        description="Counseling session count",
        aliases=["counseling_sessions"],
        validation_rules={"min": 0},
        used_for=["support_services_tracking"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # INTERNATIONAL STUDENT INFORMATION
    # ═══════════════════════════════════════════════════════════════════

    "is_international": ColumnDefinition(
        name="is_international",
        category=ColumnCategory.INTERNATIONAL,
        data_type=DataType.BOOLEAN,
        description="Indicator of international student status",
        business_logic="Derived from nationality != 'UAE'",
        used_for=["uae_vs_international_comparison", "diversity_metrics"]
    ),

    "visa_status": ColumnDefinition(
        name="visa_status",
        category=ColumnCategory.INTERNATIONAL,
        data_type=DataType.CATEGORICAL,
        description="Visa type or status",
        used_for=["international_student_classification"]
    ),

    "visa_expiry_date": ColumnDefinition(
        name="visa_expiry_date",
        category=ColumnCategory.INTERNATIONAL,
        data_type=DataType.DATE,
        description="Visa expiration date",
        used_for=["compliance_tracking", "retention_risk"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DERIVED/CALCULATED FIELDS
    # ═══════════════════════════════════════════════════════════════════

    "region": ColumnDefinition(
        name="region",
        category=ColumnCategory.DERIVED,
        data_type=DataType.CATEGORICAL,
        description="Geographic region derived from nationality",
        business_logic="Mapped from nationality codes to regions (GCC, MENA, Asia, etc.)",
        used_for=["regional_analysis", "diversity_metrics"]
    ),

    "performance_tier": ColumnDefinition(
        name="performance_tier",
        category=ColumnCategory.DERIVED,
        data_type=DataType.CATEGORICAL,
        description="Performance tier from GPA",
        business_logic="High (>=3.5), Mid (2.5-3.5), Low (<2.5)",
        validation_rules={"allowed_values": ["High", "Mid", "Low"]},
        used_for=["performance_segmentation"]
    ),

    "registration_status": ColumnDefinition(
        name="registration_status",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.CATEGORICAL,
        description="Registration status for current term",
        used_for=["current_term_enrollment"]
    ),

    "application_status": ColumnDefinition(
        name="application_status",
        category=ColumnCategory.ENROLLMENT,
        data_type=DataType.CATEGORICAL,
        description="Application status (Accepted, Pending, Denied)",
        validation_rules={"allowed_values": ["Accepted", "Pending", "Denied", "Withdrawn"]},
        used_for=["admission_tracking", "enrollment_funnel"]
    ),

    "student_count": ColumnDefinition(
        name="student_count",
        category=ColumnCategory.DERIVED,
        data_type=DataType.INTEGER,
        description="Count of students in group (aggregated metric)",
        business_logic="COUNT(student_id) or COUNT(DISTINCT student_id)",
        used_for=["group_sizing", "cohort_analysis"]
    ),

    "last_activity_date": ColumnDefinition(
        name="last_activity_date",
        category=ColumnCategory.ENGAGEMENT,
        data_type=DataType.DATE,
        description="Last system activity date",
        used_for=["engagement_monitoring", "inactive_identification"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA WAREHOUSE - RAW TABLES
    # ═══════════════════════════════════════════════════════════════════

    "RAW_STUDENT": ColumnDefinition(
        name="RAW_STUDENT",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw student source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_ENROLLMENT": ColumnDefinition(
        name="RAW_ENROLLMENT",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw enrollment source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_STUDENT_DEMOGRAPHIC": ColumnDefinition(
        name="RAW_STUDENT_DEMOGRAPHIC",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw student demographics source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_STUDENT_GRADE": ColumnDefinition(
        name="RAW_STUDENT_GRADE",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw student grades source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_FINANCIAL_AID_COUNSELING": ColumnDefinition(
        name="RAW_FINANCIAL_AID_COUNSELING",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw financial aid and counseling source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_HOUSING_APPLICATION": ColumnDefinition(
        name="RAW_HOUSING_APPLICATION",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw housing application source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_HOUSING_ASSIGNMENT": ColumnDefinition(
        name="RAW_HOUSING_ASSIGNMENT",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw housing assignment source table",
        used_for=["etl", "data_lineage"]
    ),

    "RAW_RECREATION_ENROLLMENT": ColumnDefinition(
        name="RAW_RECREATION_ENROLLMENT",
        category=ColumnCategory.RAW_TABLE,
        data_type=DataType.STRING,
        description="Raw recreation enrollment source table",
        used_for=["etl", "data_lineage"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA WAREHOUSE - DIMENSION TABLES
    # ═══════════════════════════════════════════════════════════════════

    "DIM_STUDENT": ColumnDefinition(
        name="DIM_STUDENT",
        category=ColumnCategory.DIMENSION_TABLE,
        data_type=DataType.STRING,
        description="Student dimension table (uppercase)",
        used_for=["data_warehouse", "dimensional_modeling"]
    ),

    "dim_student": ColumnDefinition(
        name="dim_student",
        category=ColumnCategory.DIMENSION_TABLE,
        data_type=DataType.STRING,
        description="Student dimension table (lowercase)",
        aliases=["DIM_STUDENT"],
        used_for=["data_warehouse", "dimensional_modeling"]
    ),

    "dim_housing_room": ColumnDefinition(
        name="dim_housing_room",
        category=ColumnCategory.DIMENSION_TABLE,
        data_type=DataType.STRING,
        description="Housing room dimension table",
        used_for=["data_warehouse", "dimensional_modeling"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA WAREHOUSE - FACT TABLES
    # ═══════════════════════════════════════════════════════════════════

    "fact_enrollment": ColumnDefinition(
        name="fact_enrollment",
        category=ColumnCategory.FACT_TABLE,
        data_type=DataType.STRING,
        description="Enrollment fact table",
        used_for=["data_warehouse", "analytics"]
    ),

    "fact_grades": ColumnDefinition(
        name="fact_grades",
        category=ColumnCategory.FACT_TABLE,
        data_type=DataType.STRING,
        description="Grades fact table",
        used_for=["data_warehouse", "analytics"]
    ),

    "fact_financial_aid": ColumnDefinition(
        name="fact_financial_aid",
        category=ColumnCategory.FACT_TABLE,
        data_type=DataType.STRING,
        description="Financial aid fact table",
        used_for=["data_warehouse", "analytics"]
    ),

    "fact_housing_applications": ColumnDefinition(
        name="fact_housing_applications",
        category=ColumnCategory.FACT_TABLE,
        data_type=DataType.STRING,
        description="Housing applications fact table",
        used_for=["data_warehouse", "analytics"]
    ),

    "fact_housing_occupancy": ColumnDefinition(
        name="fact_housing_occupancy",
        category=ColumnCategory.FACT_TABLE,
        data_type=DataType.STRING,
        description="Housing occupancy fact table",
        used_for=["data_warehouse", "analytics"]
    ),

    "fact_recreation": ColumnDefinition(
        name="fact_recreation",
        category=ColumnCategory.FACT_TABLE,
        data_type=DataType.STRING,
        description="Recreation fact table",
        used_for=["data_warehouse", "analytics"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA WAREHOUSE - CURATED TABLES
    # ═══════════════════════════════════════════════════════════════════

    "student_360": ColumnDefinition(
        name="student_360",
        category=ColumnCategory.CURATED_TABLE,
        data_type=DataType.STRING,
        description="Student 360 view (curated analytics table)",
        used_for=["analytics", "reporting"]
    ),

    "raw_sources": ColumnDefinition(
        name="raw_sources",
        category=ColumnCategory.CURATED_TABLE,
        data_type=DataType.STRING,
        description="List of raw data sources",
        used_for=["data_lineage", "documentation"]
    ),

    "curated_sources": ColumnDefinition(
        name="curated_sources",
        category=ColumnCategory.CURATED_TABLE,
        data_type=DataType.STRING,
        description="List of curated data sources",
        used_for=["data_lineage", "documentation"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA GOVERNANCE - CLASSIFICATION & SECURITY
    # ═══════════════════════════════════════════════════════════════════

    "data_classification": ColumnDefinition(
        name="data_classification",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.CATEGORICAL,
        description="Data sensitivity classification level",
        validation_rules={"allowed_values": ["Public", "Internal", "Confidential", "Restricted"]},
        used_for=["data_governance", "security", "compliance"]
    ),

    "data_owner": ColumnDefinition(
        name="data_owner",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.STRING,
        description="Person/team responsible for data ownership",
        used_for=["data_governance", "accountability"]
    ),

    "data_steward": ColumnDefinition(
        name="data_steward",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.STRING,
        description="Person/team responsible for data stewardship",
        used_for=["data_governance", "data_quality"]
    ),

    "contains_pii": ColumnDefinition(
        name="contains_pii",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.BOOLEAN,
        description="Indicates if column contains personally identifiable information",
        used_for=["privacy", "compliance", "gdpr"]
    ),

    "gdpr_applicable": ColumnDefinition(
        name="gdpr_applicable",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.BOOLEAN,
        description="Indicates if GDPR regulations apply",
        used_for=["privacy", "compliance"]
    ),

    "access_level": ColumnDefinition(
        name="access_level",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.CATEGORICAL,
        description="Required access level for data",
        validation_rules={"allowed_values": ["Public", "Authenticated", "Authorized", "Admin"]},
        used_for=["security", "access_control"]
    ),

    "compliance_tags": ColumnDefinition(
        name="compliance_tags",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.STRING,
        description="Compliance framework tags (GDPR, FERPA, etc.)",
        used_for=["compliance", "data_governance"]
    ),

    "retention_period": ColumnDefinition(
        name="retention_period",
        category=ColumnCategory.DATA_CLASSIFICATION,
        data_type=DataType.STRING,
        description="Data retention period (e.g., '7 years', 'permanent')",
        used_for=["compliance", "data_lifecycle"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA GOVERNANCE - TECHNICAL METADATA
    # ═══════════════════════════════════════════════════════════════════

    "source_table": ColumnDefinition(
        name="source_table",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="Source table name",
        used_for=["data_lineage", "etl_documentation"]
    ),

    "source_tables": ColumnDefinition(
        name="source_tables",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="Multiple source table names",
        used_for=["data_lineage", "etl_documentation"]
    ),

    "target_table": ColumnDefinition(
        name="target_table",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="Target table name",
        used_for=["data_lineage", "etl_documentation"]
    ),

    "product_column": ColumnDefinition(
        name="product_column",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="Product/application column name",
        used_for=["data_mapping", "integration"]
    ),

    "standardized_name": ColumnDefinition(
        name="standardized_name",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="Standardized column name across systems",
        used_for=["data_standardization", "integration"]
    ),

    "sql_logic": ColumnDefinition(
        name="sql_logic",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="SQL transformation logic",
        used_for=["etl_documentation", "data_transformation"]
    ),

    "transformation_type": ColumnDefinition(
        name="transformation_type",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.CATEGORICAL,
        description="Type of data transformation applied",
        validation_rules={"allowed_values": ["Direct", "Calculated", "Aggregated", "Derived", "Lookup"]},
        used_for=["etl_documentation", "data_lineage"]
    ),

    "validation_rules": ColumnDefinition(
        name="validation_rules",
        category=ColumnCategory.TECHNICAL_METADATA,
        data_type=DataType.STRING,
        description="Data validation rules and constraints",
        used_for=["data_quality", "validation"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA GOVERNANCE - BUSINESS METADATA
    # ═══════════════════════════════════════════════════════════════════

    "business_rule": ColumnDefinition(
        name="business_rule",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="Business rule definition",
        used_for=["business_logic", "documentation"]
    ),

    "business_rules": ColumnDefinition(
        name="business_rules",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="Multiple business rules",
        used_for=["business_logic", "documentation"]
    ),

    "business_impact": ColumnDefinition(
        name="business_impact",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.CATEGORICAL,
        description="Business impact level",
        validation_rules={"allowed_values": ["Critical", "High", "Medium", "Low"]},
        used_for=["impact_analysis", "prioritization"]
    ),

    "affected_dashboards": ColumnDefinition(
        name="affected_dashboards",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="List of dashboards using this data",
        used_for=["impact_analysis", "dependencies"]
    ),

    "affected_reports": ColumnDefinition(
        name="affected_reports",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="List of reports using this data",
        used_for=["impact_analysis", "dependencies"]
    ),

    "direct_downstream": ColumnDefinition(
        name="direct_downstream",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="Direct downstream dependencies",
        used_for=["dependency_tracking", "impact_analysis"]
    ),

    "indirect_downstream": ColumnDefinition(
        name="indirect_downstream",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="Indirect downstream dependencies",
        used_for=["dependency_tracking", "impact_analysis"]
    ),

    "ROI": ColumnDefinition(
        name="ROI",
        category=ColumnCategory.BUSINESS_METADATA,
        data_type=DataType.STRING,
        description="Return on investment metric",
        used_for=["business_value", "prioritization"]
    ),

    # ═══════════════════════════════════════════════════════════════════
    # DATA GOVERNANCE - PERFORMANCE & MONITORING
    # ═══════════════════════════════════════════════════════════════════

    "refresh_schedule": ColumnDefinition(
        name="refresh_schedule",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.STRING,
        description="Data refresh schedule (e.g., 'Daily 2AM', 'Hourly')",
        used_for=["etl_scheduling", "data_freshness"]
    ),

    "last_run": ColumnDefinition(
        name="last_run",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.DATE,
        description="Last execution timestamp",
        used_for=["monitoring", "job_tracking"]
    ),

    "next_run": ColumnDefinition(
        name="next_run",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.DATE,
        description="Next scheduled execution timestamp",
        used_for=["scheduling", "planning"]
    ),

    "last_change": ColumnDefinition(
        name="last_change",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.DATE,
        description="Last modified timestamp",
        used_for=["change_tracking", "audit"]
    ),

    "last_audit": ColumnDefinition(
        name="last_audit",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.DATE,
        description="Last audit timestamp",
        used_for=["compliance", "audit_trail"]
    ),

    "last_check": ColumnDefinition(
        name="last_check",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.DATE,
        description="Last quality check timestamp",
        used_for=["data_quality", "monitoring"]
    ),

    "change_frequency": ColumnDefinition(
        name="change_frequency",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.STRING,
        description="How often data changes (e.g., 'Daily', 'Weekly')",
        used_for=["monitoring", "refresh_planning"]
    ),

    "job_name": ColumnDefinition(
        name="job_name",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.STRING,
        description="ETL job name",
        used_for=["job_tracking", "monitoring"]
    ),

    "sla_status": ColumnDefinition(
        name="sla_status",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.CATEGORICAL,
        description="Service level agreement status",
        validation_rules={"allowed_values": ["Met", "At Risk", "Breached", "N/A"]},
        used_for=["sla_monitoring", "performance_tracking"]
    ),

    "success_rate": ColumnDefinition(
        name="success_rate",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.PERCENTAGE,
        description="Job success rate percentage",
        validation_rules={"min": 0, "max": 100},
        used_for=["reliability_monitoring", "quality_metrics"]
    ),

    "avg_query_time": ColumnDefinition(
        name="avg_query_time",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.FLOAT,
        description="Average query execution time (seconds)",
        used_for=["performance_monitoring", "optimization"]
    ),

    "records_processed": ColumnDefinition(
        name="records_processed",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.INTEGER,
        description="Number of records processed in last run",
        used_for=["volume_tracking", "monitoring"]
    ),

    "source_records": ColumnDefinition(
        name="source_records",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.INTEGER,
        description="Number of records in source",
        used_for=["reconciliation", "data_quality"]
    ),

    "target_records": ColumnDefinition(
        name="target_records",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.INTEGER,
        description="Number of records in target",
        used_for=["reconciliation", "data_quality"]
    ),

    "variance_pct": ColumnDefinition(
        name="variance_pct",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.PERCENTAGE,
        description="Variance percentage between source and target",
        validation_rules={"min": 0},
        used_for=["data_quality", "reconciliation"]
    ),

    "index_count": ColumnDefinition(
        name="index_count",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.INTEGER,
        description="Number of indexes on table",
        used_for=["performance_optimization", "monitoring"]
    ),

    "storage_mb": ColumnDefinition(
        name="storage_mb",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.FLOAT,
        description="Storage size in megabytes",
        used_for=["capacity_planning", "cost_management"]
    ),

    "growth_rate": ColumnDefinition(
        name="growth_rate",
        category=ColumnCategory.PERFORMANCE_MONITORING,
        data_type=DataType.PERCENTAGE,
        description="Data growth rate percentage",
        used_for=["capacity_planning", "forecasting"]
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING & RESOLUTION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class ColumnMapper:
    """Semantic column mapping and resolution system"""

    def __init__(self, catalog: Dict[str, ColumnDefinition] = None):
        """Initialize with column catalog"""
        self.catalog = catalog or UNIVERSAL_COLUMNS_CATALOG
        self._build_alias_map()

    def _build_alias_map(self):
        """Build reverse mapping from aliases to canonical names"""
        self.alias_to_canonical = {}
        for canonical_name, col_def in self.catalog.items():
            # Map canonical name to itself
            self.alias_to_canonical[canonical_name] = canonical_name
            # Map all aliases to canonical name
            for alias in col_def.aliases:
                self.alias_to_canonical[alias.lower()] = canonical_name

    def resolve(self, column_name: str) -> Optional[str]:
        """
        Resolve a column name (or alias) to its canonical name

        Args:
            column_name: Column name or alias to resolve

        Returns:
            Canonical column name, or None if not found
        """
        return self.alias_to_canonical.get(column_name.lower())

    def get_definition(self, column_name: str) -> Optional[ColumnDefinition]:
        """
        Get the full column definition for a column name or alias

        Args:
            column_name: Column name or alias

        Returns:
            ColumnDefinition object, or None if not found
        """
        canonical = self.resolve(column_name)
        if canonical:
            return self.catalog[canonical]
        return None

    def get_by_category(self, category: ColumnCategory) -> List[ColumnDefinition]:
        """
        Get all columns in a specific category

        Args:
            category: ColumnCategory enum value

        Returns:
            List of ColumnDefinition objects
        """
        return [col_def for col_def in self.catalog.values()
                if col_def.category == category]

    def get_columns_for_analysis(self, analysis_type: str) -> List[str]:
        """
        Get recommended columns for a specific analysis type

        Args:
            analysis_type: Type of analysis (e.g., 'performance_analysis', 'aid_correlation')

        Returns:
            List of canonical column names
        """
        return [name for name, col_def in self.catalog.items()
                if analysis_type in col_def.used_for]

    def validate_value(self, column_name: str, value: Any) -> bool:
        """
        Validate a value against column validation rules

        Args:
            column_name: Column name or alias
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        col_def = self.get_definition(column_name)
        if not col_def:
            return False

        if value is None:
            return col_def.nullable

        rules = col_def.validation_rules

        # Check min/max for numeric types
        if 'min' in rules and value < rules['min']:
            return False
        if 'max' in rules and value > rules['max']:
            return False

        # Check allowed values for categorical
        if 'allowed_values' in rules and value not in rules['allowed_values']:
            return False

        return True

    def get_column_info(self, column_name: str) -> str:
        """
        Get formatted information about a column

        Args:
            column_name: Column name or alias

        Returns:
            Formatted string with column information
        """
        col_def = self.get_definition(column_name)
        if not col_def:
            return f"Column '{column_name}' not found in catalog"

        info = f"""
Column: {col_def.name}
Category: {col_def.category.value}
Data Type: {col_def.data_type.value}
Description: {col_def.description}
Aliases: {', '.join(col_def.aliases) if col_def.aliases else 'None'}
Nullable: {col_def.nullable}
"""
        if col_def.business_logic:
            info += f"Business Logic: {col_def.business_logic}\n"
        if col_def.used_for:
            info += f"Used For: {', '.join(col_def.used_for)}\n"

        return info.strip()


# ═════════════════════════════════════════════════════════════════════════════
# PREDEFINED COLUMN GROUPS FOR COMMON ANALYSES
# ═════════════════════════════════════════════════════════════════════════════

COLUMN_GROUPS = {
    "student_basic_info": [
        "student_id", "first_name_en", "last_name_en", "email_address",
        "gender", "nationality", "date_of_birth"
    ],

    "academic_core": [
        "cumulative_gpa", "credits_attempted", "degree_progress_pct",
        "academic_standing", "enrollment_enrollment_status"
    ],

    "financial_core": [
        "enrollment_tuition_amount", "financial_aid_monetary_amount",
        "account_balance", "has_aid"
    ],

    "housing_core": [
        "room_number", "housing_status", "rent_amount", "rent_paid"
    ],

    "uae_national_analysis": [
        "student_id", "nationality", "emirates_id", "cumulative_gpa",
        "enrollment_enrollment_status", "cohort_year", "financial_aid_monetary_amount"
    ],

    "performance_comparison": [
        "student_id", "nationality", "cumulative_gpa", "credits_attempted",
        "degree_progress_pct", "is_first_generation"
    ],

    "financial_aid_analysis": [
        "student_id", "financial_aid_monetary_amount", "cumulative_gpa",
        "nationality", "has_aid", "enrollment_tuition_amount"
    ],

    "housing_impact_analysis": [
        "student_id", "room_number", "cumulative_gpa", "enrollment_enrollment_status",
        "financial_aid_monetary_amount"
    ],

    "risk_assessment": [
        "student_id", "cumulative_gpa", "risk_score", "risk_level",
        "financial_aid_monetary_amount", "is_first_generation", "attendance_rate"
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def get_mapper() -> ColumnMapper:
    """Get a configured ColumnMapper instance"""
    return ColumnMapper(UNIVERSAL_COLUMNS_CATALOG)


def get_columns_by_category(category: ColumnCategory) -> List[str]:
    """Get list of column names in a category"""
    mapper = get_mapper()
    return [col_def.name for col_def in mapper.get_by_category(category)]


def resolve_column(column_name: str) -> Optional[str]:
    """Quick resolve a column name to canonical form"""
    mapper = get_mapper()
    return mapper.resolve(column_name)


def get_column_group(group_name: str) -> List[str]:
    """Get a predefined column group"""
    return COLUMN_GROUPS.get(group_name, [])


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION (for testing)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test the mapper
    mapper = get_mapper()

    print("=" * 80)
    print("UNIVERSAL COLUMNS CATALOG - TEST")
    print("=" * 80)

    # Test alias resolution
    print("\n1. Testing alias resolution:")
    test_aliases = ["gpa", "aid_amount", "room", "student_number"]
    for alias in test_aliases:
        canonical = mapper.resolve(alias)
        print(f"   '{alias}' → '{canonical}'")

    # Test category grouping
    print("\n2. Academic Performance columns:")
    academic_cols = get_columns_by_category(ColumnCategory.ACADEMIC_PERFORMANCE)
    for col in academic_cols:
        print(f"   - {col}")

    # Test analysis-based retrieval
    print("\n3. Columns for 'aid_correlation' analysis:")
    aid_cols = mapper.get_columns_for_analysis('aid_correlation')
    for col in aid_cols:
        print(f"   - {col}")

    # Test column info
    print("\n4. Column information for 'cumulative_gpa':")
    print(mapper.get_column_info('cumulative_gpa'))

    print("\n" + "=" * 80)
    print(f"Total columns in catalog: {len(UNIVERSAL_COLUMNS_CATALOG)}")
    print("=" * 80)
