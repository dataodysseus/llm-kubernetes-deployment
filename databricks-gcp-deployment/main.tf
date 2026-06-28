###############################################################################
# Databricks on GCP - Root Module
# Auth: Google Workload Identity Federation via GitHub Actions
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
  }
}

###############################################################################
# Providers
# Auth comes entirely from environment variables set by the GitHub Actions
# google-github-actions/auth step — no hardcoded credentials needed.
#
# GCP provider   : uses GOOGLE_CREDENTIALS (set by auth action)
# Databricks provider: uses DATABRICKS_HOST + DATABRICKS_ACCOUNT_ID +
#                       DATABRICKS_GOOGLE_SERVICE_ACCOUNT (set in workflow)
###############################################################################

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  # Credentials come from GOOGLE_CREDENTIALS env var (set by WIF auth step)
}

provider "databricks" {
  alias      = "accounts"
  host       = "https://accounts.gcp.databricks.com"
  account_id = var.databricks_account_id
  # Auth: DATABRICKS_GOOGLE_SERVICE_ACCOUNT env var triggers google-id auth
  # The service account must be an Account Admin in Databricks
}

provider "databricks" {
  alias = "workspace"
  host  = module.databricks_workspace.workspace_url
  # Auth: same google-id flow, workspace-scoped
}

###############################################################################
# Modules
###############################################################################

module "gcp_networking" {
  source = "./modules/gcp-networking"

  project_id   = var.gcp_project_id
  region       = var.gcp_region
  environment  = var.environment
  network_name = var.network_name
  subnet_cidr  = var.subnet_cidr
  pod_cidr     = var.pod_cidr
  svc_cidr     = var.svc_cidr
}

module "gcp_iam" {
  source = "./modules/gcp-iam"

  project_id            = var.gcp_project_id
  environment           = var.environment
  databricks_account_id = var.databricks_account_id
}

module "databricks_workspace" {
  source = "./modules/databricks-workspace"

  providers = {
    databricks = databricks.accounts
    google     = google
  }

  gcp_project_id        = var.gcp_project_id
  gcp_region            = var.gcp_region
  environment           = var.environment
  workspace_name        = var.workspace_name
  databricks_account_id = var.databricks_account_id
  network_id            = module.gcp_networking.vpc_id
  subnet_id             = module.gcp_networking.subnet_id
  pod_subnet_id         = module.gcp_networking.pod_subnet_id
  svc_subnet_id         = module.gcp_networking.svc_subnet_id
  gke_node_sa_email     = module.gcp_iam.gke_node_sa_email
  storage_sa_email      = module.gcp_iam.storage_sa_email

  depends_on = [
    module.gcp_networking,
    module.gcp_iam,
  ]
}
