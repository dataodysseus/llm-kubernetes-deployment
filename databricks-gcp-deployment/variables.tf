###############################################################################
# Root Variables
###############################################################################

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "genai-model-dev-deploy"
}

variable "gcp_region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "databricks_account_id" {
  description = "Databricks Account ID (found in Databricks account console)"
  type        = string
  default     = "bbd2dc60-e52e-4335-a663-156574f5ed23"
  sensitive   = true
}

variable "workspace_name" {
  description = "Databricks workspace name"
  type        = string
  default     = "databricks-gcp-dev"
}

# Networking
variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "databricks-vpc"
}

variable "subnet_cidr" {
  description = "CIDR for the primary subnet (nodes)"
  type        = string
  default     = "10.0.0.0/16"
}

variable "pod_cidr" {
  description = "CIDR for GKE pod secondary range"
  type        = string
  default     = "10.1.0.0/16"
}

variable "svc_cidr" {
  description = "CIDR for GKE services secondary range"
  type        = string
  default     = "10.2.0.0/20"
}
