output "gke_node_sa_email" {
  description = "GKE node service account email"
  value       = google_service_account.gke_node.email
}

output "storage_sa_email" {
  description = "Storage service account email"
  value       = google_service_account.storage.email
}

output "dbfs_bucket_name" {
  description = "DBFS root GCS bucket name"
  value       = google_storage_bucket.dbfs_root.name
}

output "dbfs_bucket_url" {
  description = "DBFS root GCS bucket URL"
  value       = google_storage_bucket.dbfs_root.url
}
