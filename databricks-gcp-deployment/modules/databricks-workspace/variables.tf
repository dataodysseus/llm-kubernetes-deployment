###############################################################################
# Databricks Workspace Module - Variables
###############################################################################

variable "gcp_project_id" {
  type = string
}

variable "gcp_region" {
  type = string
}

variable "environment" {
  type = string
}

variable "workspace_name" {
  type = string
}

variable "databricks_account_id" {
  type      = string
  sensitive = true
}

variable "network_id" {
  type        = string
  description = "VPC network self link"
}

variable "subnet_id" {
  type        = string
  description = "Subnet self link"
}

variable "pod_subnet_id" {
  type        = string
  description = "Pod secondary range name"
}

variable "svc_subnet_id" {
  type        = string
  description = "Services secondary range name"
}

variable "gke_node_sa_email" {
  type        = string
  description = "GKE node service account email"
}

variable "storage_sa_email" {
  type        = string
  description = "Storage service account email"
}
