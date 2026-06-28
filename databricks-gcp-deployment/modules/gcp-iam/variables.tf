###############################################################################
# GCP IAM Module - Variables
###############################################################################

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "databricks_account_id" {
  description = "Databricks Account ID"
  type        = string
  sensitive   = true
}
