variable "project" {
  type    = string
  default = "mlops-fraud"
}

variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "github_repo" {
  description = "owner/repo allowed to assume the CI/CD role via OIDC"
  type        = string
}

variable "create_github_oidc_provider" {
  description = "Only ONE GitHub OIDC provider can exist per account. Set false if it already exists."
  type        = bool
  default     = true
}

variable "mlflow_allowed_cidrs" {
  description = "Who may reach the MLflow tracking server (port 5000). CI runners need access; LOCK THIS DOWN."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "api_allowed_cidrs" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "mlflow_image_tag" {
  type    = string
  default = "latest"
}

variable "api_desired_count" {
  description = "Bootstrap with 0 (no champion model exists yet), run the CD pipeline once, then apply with 2."
  type        = number
  default     = 2
}
