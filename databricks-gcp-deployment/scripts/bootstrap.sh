#!/usr/bin/env bash
###############################################################################
# bootstrap.sh - One-time setup script before running Terraform
#
# Run this ONCE before `terraform init` and `terraform apply`.
# It authenticates to GCP, enables required APIs, and validates your setup.
###############################################################################

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID="genai-model-dev-deploy"
GCP_REGION="us-central1"
DATABRICKS_ACCOUNT_ID="bbd2dc60-e52e-4335-a663-156574f5ed23"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

###############################################################################
# Step 1: Check prerequisites
###############################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Databricks on GCP - Bootstrap Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

command -v gcloud    >/dev/null 2>&1 || err "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
command -v terraform >/dev/null 2>&1 || err "terraform not found. Install: https://developer.hashicorp.com/terraform/install"
command -v jq        >/dev/null 2>&1 || warn "jq not found (optional but helpful). Install: brew install jq"

log "Prerequisites found"

###############################################################################
# Step 2: Authenticate to GCP
###############################################################################
echo ""
echo "── Step 1: GCP Authentication ───────────────────────────"
echo ""
echo "Logging in to GCP (browser will open)..."
gcloud auth login --update-adc

echo ""
echo "Setting application default credentials (used by Terraform)..."
gcloud auth application-default login

gcloud config set project "$GCP_PROJECT_ID"
gcloud config set compute/region "$GCP_REGION"
log "GCP project set to: $GCP_PROJECT_ID"

###############################################################################
# Step 3: Enable required APIs
###############################################################################
echo ""
echo "── Step 2: Enabling GCP APIs ────────────────────────────"
echo ""

APIS=(
  "compute.googleapis.com"
  "container.googleapis.com"
  "iam.googleapis.com"
  "cloudresourcemanager.googleapis.com"
  "storage.googleapis.com"
  "logging.googleapis.com"
  "monitoring.googleapis.com"
  "networkmanagement.googleapis.com"
  "servicenetworking.googleapis.com"
  "iamcredentials.googleapis.com"
)

for api in "${APIS[@]}"; do
  echo -n "  Enabling $api ... "
  gcloud services enable "$api" --project="$GCP_PROJECT_ID" --quiet
  echo "done"
done

log "All required APIs enabled"

###############################################################################
# Step 4: Verify Databricks account access
###############################################################################
echo ""
echo "── Step 3: Verify Setup ─────────────────────────────────"
echo ""

CURRENT_USER=$(gcloud config get-value account)
log "Logged in as: $CURRENT_USER"
log "GCP Project:  $GCP_PROJECT_ID"
log "Region:       $GCP_REGION"
log "DB Account:   $DATABRICKS_ACCOUNT_ID"

###############################################################################
# Step 5: Initialize Terraform
###############################################################################
echo ""
echo "── Step 4: Terraform Init ───────────────────────────────"
echo ""

# Navigate to the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

terraform init -upgrade
log "Terraform initialized"

###############################################################################
# Done
###############################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}  Bootstrap complete! Next steps:${NC}"
echo ""
echo "  1. Review variables:"
echo "     cat environments/dev/dev.tfvars"
echo ""
echo "  2. Plan the deployment:"
echo "     terraform plan -var-file=environments/dev/dev.tfvars"
echo ""
echo "  3. Apply:"
echo "     terraform apply -var-file=environments/dev/dev.tfvars"
echo ""
echo "  4. Get workspace URL:"
echo "     terraform output workspace_url"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
