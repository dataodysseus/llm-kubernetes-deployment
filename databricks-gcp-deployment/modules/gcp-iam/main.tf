###############################################################################
# GCP IAM Module
# Creates service accounts and grants roles required by Databricks on GCP
###############################################################################

locals {
  # Databricks-required roles for GKE node service account
  gke_node_roles = [
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
  ]

  # Databricks-required roles for storage service account
  storage_roles = [
    "roles/storage.objectAdmin",
    "roles/storage.objectViewer",         
  ]
}

###############################################################################
# GKE Node Service Account
###############################################################################

resource "google_service_account" "gke_node" {
  account_id   = "databricks-gke-node-${var.environment}"
  display_name = "Databricks GKE Node SA (${var.environment})"
  project      = var.project_id
  description  = "Service account for Databricks GKE cluster nodes"
}

resource "google_project_iam_member" "gke_node_roles" {
  for_each = toset(local.gke_node_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_node.email}"
}

###############################################################################
# Storage Service Account (for DBFS/Unity Catalog root storage)
###############################################################################

resource "google_service_account" "storage" {
  account_id   = "databricks-storage-${var.environment}"
  display_name = "Databricks Storage SA (${var.environment})"
  project      = var.project_id
  description  = "Service account for Databricks DBFS and Unity Catalog storage"
}

resource "google_project_iam_member" "storage_roles" {
  for_each = toset(local.storage_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.storage.email}"
}

###############################################################################
# Allow Databricks control plane to impersonate the storage SA via Workload Identity
###############################################################################

resource "google_service_account_iam_binding" "storage_workload_identity" {
  service_account_id = google_service_account.storage.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    # Databricks uses this principal to impersonate the storage SA
    "serviceAccount:${var.project_id}.svc.id.goog[databricks/databricks]",
  ]
}

###############################################################################
# GCS bucket for DBFS root storage
###############################################################################

resource "google_storage_bucket" "dbfs_root" {
  name          = "databricks-dbfs-${var.project_id}-${var.environment}"
  project       = var.project_id
  location      = "US"
  force_destroy = var.environment != "prod" # Safety guard for prod

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    purpose     = "databricks-dbfs"
  }
}

resource "google_storage_bucket_iam_binding" "dbfs_storage_admin" {
  bucket = google_storage_bucket.dbfs_root.name
  role   = "roles/storage.objectAdmin"
  members = [
    "serviceAccount:${google_service_account.storage.email}",
  ]
}

###############################################################################
# Enable required GCP APIs
###############################################################################

locals {
  required_apis = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "networkmanagement.googleapis.com",
    "servicenetworking.googleapis.com",
    "iamcredentials.googleapis.com",
  ]
}

resource "google_project_service" "required_apis" {
  for_each = toset(local.required_apis)

  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}
