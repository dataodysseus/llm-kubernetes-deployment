###############################################################################
# GCP Networking Module
# Creates VPC, subnet with secondary ranges required by Databricks on GKE
###############################################################################

resource "google_compute_network" "databricks_vpc" {
  name                    = "${var.network_name}-${var.environment}"
  project                 = var.project_id
  auto_create_subnetworks = false
  description             = "VPC for Databricks on GCP (${var.environment})"
}

resource "google_compute_subnetwork" "databricks_subnet" {
  name          = "${var.network_name}-subnet-${var.environment}"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.databricks_vpc.id
  ip_cidr_range = var.subnet_cidr

  # GKE requires two secondary ranges: pods and services
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pod_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.svc_cidr
  }

  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# Router + NAT so private nodes can reach the internet (e.g., PyPI)
resource "google_compute_router" "databricks_router" {
  name    = "databricks-router-${var.environment}"
  project = var.project_id
  region  = var.region
  network = google_compute_network.databricks_vpc.id
}

resource "google_compute_router_nat" "databricks_nat" {
  name                               = "databricks-nat-${var.environment}"
  project                            = var.project_id
  router                             = google_compute_router.databricks_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Firewall: allow internal traffic within VPC
resource "google_compute_firewall" "databricks_internal" {
  name    = "databricks-allow-internal-${var.environment}"
  project = var.project_id
  network = google_compute_network.databricks_vpc.name

  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }

  source_ranges = [
    var.subnet_cidr,
    var.pod_cidr,
    var.svc_cidr,
  ]

  description = "Allow all internal traffic within Databricks VPC"
}

# Firewall: allow Databricks control plane to reach nodes (SSH/API)
resource "google_compute_firewall" "databricks_control_plane" {
  name    = "databricks-allow-cp-${var.environment}"
  project = var.project_id
  network = google_compute_network.databricks_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["443", "8443-8451", "3306"]
  }

  # Databricks GCP control plane CIDR ranges
  source_ranges = ["35.199.224.0/19"]

  description = "Allow Databricks control plane inbound access"
}
