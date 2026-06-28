output "vpc_id" {
  description = "VPC self link"
  value       = google_compute_network.databricks_vpc.id
}

output "vpc_name" {
  description = "VPC network name"
  value       = google_compute_network.databricks_vpc.name
}

output "subnet_id" {
  description = "Primary subnet self link"
  value       = google_compute_subnetwork.databricks_subnet.id
}

output "subnet_name" {
  description = "Primary subnet name"
  value       = google_compute_subnetwork.databricks_subnet.name
}

output "pod_subnet_id" {
  description = "Pod secondary range name"
  value       = "pods"
}

output "svc_subnet_id" {
  description = "Services secondary range name"
  value       = "services"
}
