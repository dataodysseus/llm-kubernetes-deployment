###############################################################################
# GCP Networking Module - Variables
###############################################################################

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "network_name" {
  description = "Base name for the VPC"
  type        = string
}

variable "subnet_cidr" {
  description = "Primary subnet CIDR"
  type        = string
}

variable "pod_cidr" {
  description = "GKE pod secondary range CIDR"
  type        = string
}

variable "svc_cidr" {
  description = "GKE services secondary range CIDR"
  type        = string
}
