
terraform {
  required_version = ">= 1.6"

  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }

  # Local state is fine for a learning project. For a team: S3 backend + DynamoDB/S3 native locking, e.g.
  # backend "s3" { bucket = "..." key = "mlops-fraud/terraform.tfstate" region = "..." use_lockfile = true }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform" }
  }
}
