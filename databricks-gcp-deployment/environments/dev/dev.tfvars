###############################################################################
# Dev Environment - Variable Values
# Copy this file to the root as terraform.tfvars (or use -var-file flag)
# Usage: terraform apply -var-file=environments/dev/dev.tfvars
###############################################################################

gcp_project_id        = "genai-model-dev-deploy"
gcp_region            = "us-central1"
environment           = "dev"
databricks_account_id = "bbd2dc60-e52e-4335-a663-156574f5ed23"
workspace_name        = "databricks-gcp-dev"

# Networking
network_name = "databricks-vpc"
subnet_cidr  = "10.0.0.0/16"
pod_cidr     = "10.1.0.0/16"
svc_cidr     = "10.2.0.0/20"
