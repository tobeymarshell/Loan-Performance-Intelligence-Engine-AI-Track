
# Data Dictionary

## Loan Static Attributes
- `loan_id`: unique loan identifier
- `original_balance`: loan amount at origination
- `interest_rate`: note rate
- `credit_score_band`: FICO band at origination
- `ltv_band`: loan‑to‑value band
- `dti_band`: debt‑to‑income band
- `state`: US state code
- `loan_purpose`: Purchase, Refinance, Cash-out Refinance
- `occupancy_type`: Primary, Secondary, Investment
- `property_type`: Single Family, Condo, Townhouse, Multi-family
- `servicer_name`: name of servicing company
- `origination_month`: month of origination

## Monthly Performance
- `reporting_month`: month of performance observation
- `loan_age_months`: age of loan in months
- `remaining_term_months`: months left until maturity
- `current_balance`: outstanding principal balance
- `current_status`: Current, 30/60/90+ Delinquent, Prepaid, Charged Off
- `days_past_due`: number of days past due
- `modification_flag`: 1 if loan modified this month
- `prepayment_flag`: 1 if loan prepaid this month
- `default_flag`: 1 if loan defaulted this month
- `document_status`: Complete, Pending, Missing

## Targets (Training only)
- `next_3m_delinquency_flag`: 1 if 30+ days delinquent in next 3 months
- `next_6m_delinquency_flag`: 1 if 30+ days delinquent in next 6 months
- `next_12m_default_flag`: 1 if default occurs within 12 months
- `next_12m_prepayment_flag`: 1 if prepayment occurs within 12 months
- `next_state`: Current / Delinquent / Default / Prepaid
- `exception_required`: flag for manual review
- `exception_type`: type of exception (e.g., High DPD)
