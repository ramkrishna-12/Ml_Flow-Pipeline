data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

# Default VPC + public subnets keep the lab cheap (no NAT gateway). Production: private subnets + NAT/VPC endpoints.
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  name        = var.project
  bucket_name = "${var.project}-${data.aws_caller_identity.current.account_id}-${var.aws_region}"
  mlflow_host = "mlflow.${var.project}.local"
}

# ---------------------------------------------------------------- security groups
resource "aws_security_group" "alb" {
  name_prefix = "${local.name}-alb-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "API"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.api_allowed_cidrs
  }
  ingress {
    description = "MLflow UI/API (CI + humans)"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = var.mlflow_allowed_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
}

resource "aws_security_group" "ecs" {
  name_prefix = "${local.name}-ecs-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "MLflow from ALB"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    description     = "API from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    description = "API -> MLflow over Cloud Map (task to task)"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    self        = true
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.name}-rds-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "Postgres from ECS tasks only"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  lifecycle { create_before_destroy = true }
}

# ---------------------------------------------------------------- S3: MLflow artifacts + training data
resource "aws_s3_bucket" "mlops" {
  bucket        = local.bucket_name
  force_destroy = true # lab convenience
}

resource "aws_s3_bucket_versioning" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "mlops" {
  bucket                  = aws_s3_bucket.mlops.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------- ECR
resource "aws_ecr_repository" "api" {
  name         = "${local.name}/api"
  force_delete = true
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "mlflow" {
  name         = "${local.name}/mlflow-server"
  force_delete = true
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "keep_recent" {
  for_each   = { api = aws_ecr_repository.api.name, mlflow = aws_ecr_repository.mlflow.name }
  repository = each.value
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 15 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 15 }
      action       = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------- RDS Postgres = MLflow backend store (runs, params, metrics, registry)
resource "random_password" "db" {
  length  = 32
  special = false # goes into a URI; avoid characters that need escaping
}

resource "aws_secretsmanager_secret" "db" {
  name_prefix             = "${local.name}/mlflow-db-"
  recovery_window_in_days = 0 # lab convenience
}

resource "aws_secretsmanager_secret_version" "db" {
  secret_id     = aws_secretsmanager_secret.db.id
  secret_string = random_password.db.result
}

resource "aws_db_subnet_group" "mlflow" {
  name_prefix = "${local.name}-"
  subnet_ids  = data.aws_subnets.default.ids
}

resource "aws_db_instance" "mlflow" {
  identifier_prefix       = "${local.name}-"
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  storage_encrypted       = true
  db_name                 = "mlflow"
  username                = "mlflow"
  password                = random_password.db.result
  db_subnet_group_name    = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  publicly_accessible     = false
  backup_retention_period = 1
  skip_final_snapshot     = true # lab convenience; production: false + deletion_protection = true
}
