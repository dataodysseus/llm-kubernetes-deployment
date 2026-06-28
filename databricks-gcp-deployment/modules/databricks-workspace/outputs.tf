output "workspace_url" {
  description = "Databricks workspace URL"
  value       = databricks_mws_workspaces.workspace.workspace_url
}

output "workspace_id" {
  description = "Databricks workspace ID"
  value       = databricks_mws_workspaces.workspace.workspace_id
}

output "workspace_status" {
  description = "Databricks workspace status"
  value       = databricks_mws_workspaces.workspace.workspace_status
}
