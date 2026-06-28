###############################################################################
# Databricks Workspace Module
# Creates the Databricks workspace on GCP using the Databricks provider
###############################################################################

terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

###############################################################################
# Network Configuration for Databricks
###############################################################################

resource "databricks_mws_networks" "databricks_network" {
  provider     = databricks
  account_id   = var.databricks_account_id
  network_name = "databricks-network-${var.environment}"
  gcp_network_info {
    network_project_id = var.gcp_project_id
    vpc_id             = var.network_id
    subnet_id          = var.subnet_id
    subnet_region      = var.gcp_region
    # pod_ip_range_name and service_ip_range_name removed (deprecated in 1.119.0)
  }
}

###############################################################################
# Databricks Workspace
###############################################################################

resource "databricks_mws_workspaces" "workspace" {
  provider       = databricks
  account_id     = var.databricks_account_id
  workspace_name = var.workspace_name
  location       = var.gcp_region

  cloud_resource_container {
    gcp {
      project_id = var.gcp_project_id
    }
  }

  network_id = databricks_mws_networks.databricks_network.network_id

  timeouts {
    create = "30m"
    read   = "10m"
    update = "20m"
  }
}
