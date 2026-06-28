###############################################################################
# Prod Environment - Variable Values
###############################################################################

gcp_project_id        = "genai-model-dev-deploy"   # Update with prod project if separate
gcp_region            = "us-central1"
environment           = "prod"
databricks_account_id = "bbd2dc60-e52e-4335-a663-156574f5ed23"
workspace_name        = "databricks-gcp-prod"

# Networking (non-overlapping CIDRs)
network_name = "databricks-vpc"
subnet_cidr  = "10.10.0.0/16"
pod_cidr     = "10.11.0.0/16"
svc_cidr     = "10.12.0.0/20"
