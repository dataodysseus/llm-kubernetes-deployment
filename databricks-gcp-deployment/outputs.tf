###############################################################################
# Root Outputs
###############################################################################

output "workspace_url" {
  description = "Databricks workspace URL"
  value       = module.databricks_workspace.workspace_url
}

output "workspace_id" {
  description = "Databricks workspace ID"
  value       = module.databricks_workspace.workspace_id
}

output "vpc_id" {
  description = "VPC network ID"
  value       = module.gcp_networking.vpc_id
}

output "subnet_id" {
  description = "Primary subnet ID"
  value       = module.gcp_networking.subnet_id
}

output "gke_node_sa_email" {
  description = "GKE node service account email"
  value       = module.gcp_iam.gke_node_sa_email
}

output "storage_sa_email" {
  description = "Storage service account email"
  value       = module.gcp_iam.storage_sa_email
}
