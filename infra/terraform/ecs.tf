resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${local.name}/mlflow"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 14
}

# ---------------------------------------------------------------- service discovery: API -> mlflow.<project>.local
resource "aws_service_discovery_private_dns_namespace" "main" {
  name = "${var.project}.local"
  vpc  = data.aws_vpc.default.id
}

resource "aws_service_discovery_service" "mlflow" {
  name = "mlflow"
  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.main.id
    routing_policy = "MULTIVALUE"
    dns_records {
      ttl  = 10
      type = "A"
    }
  }
}

# ---------------------------------------------------------------- ALB: :80 -> API, :5000 -> MLflow
resource "aws_lb" "main" {
  name               = substr(local.name, 0, 32)
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids
}

resource "aws_lb_target_group" "api" {
  name                 = "${substr(local.name, 0, 24)}-api"
  port                 = 8000
  protocol             = "HTTP"
  vpc_id               = data.aws_vpc.default.id
  target_type          = "ip"
  deregistration_delay = 30
  health_check {
    path                = "/ready" # readiness, not liveness: no model => no traffic
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_target_group" "mlflow" {
  name                 = "${substr(local.name, 0, 22)}-mlflow"
  port                 = 5000
  protocol             = "HTTP"
  vpc_id               = data.aws_vpc.default.id
  target_type          = "ip"
  deregistration_delay = 30
  health_check {
    path    = "/health"
    matcher = "200"
  }
}

resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP" # production: 443 + ACM certificate, and redirect 80 -> 443
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener" "mlflow" {
  load_balancer_arn = aws_lb.main.arn
  port              = 5000
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}

# ---------------------------------------------------------------- MLflow tracking server + registry
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${local.name}-mlflow"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.mlflow_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name         = "mlflow"
    image        = "${aws_ecr_repository.mlflow.repository_url}:${var.mlflow_image_tag}"
    essential    = true
    portMappings = [{ containerPort = 5000, protocol = "tcp" }]
    # $${VAR} renders as ${VAR} for the shell inside the container (password comes from Secrets Manager).
    command = [
      "sh", "-c",
      "exec mlflow server --host 0.0.0.0 --port 5000 --workers 2 --backend-store-uri postgresql://$${DB_USER}:$${DB_PASS}@$${DB_HOST}:5432/$${DB_NAME} --artifacts-destination s3://${aws_s3_bucket.mlops.bucket}/artifacts --serve-artifacts"
    ]
    environment = [
      { name = "DB_USER", value = aws_db_instance.mlflow.username },
      { name = "DB_NAME", value = aws_db_instance.mlflow.db_name },
      { name = "DB_HOST", value = aws_db_instance.mlflow.address },
      { name = "AWS_DEFAULT_REGION", value = var.aws_region },
      # MLflow >= 3.5 rejects unknown Host headers (DNS-rebinding protection): allow-list ours explicitly.
      {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = join(",", [aws_lb.main.dns_name, "${aws_lb.main.dns_name}:*", local.mlflow_host, "${local.mlflow_host}:*"])
      },
    ]
    secrets = [{ name = "DB_PASS", valueFrom = aws_secretsmanager_secret.db.arn }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.mlflow.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "mlflow"
      }
    }
  }])
}

resource "aws_ecs_service" "mlflow" {
  name                              = "mlflow"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.mlflow.arn
  desired_count                     = 1
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 120

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true # lab: avoids a NAT gateway. Production: private subnets + NAT / VPC endpoints.
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow.arn
    container_name   = "mlflow"
    container_port   = 5000
  }
  service_registries {
    registry_arn = aws_service_discovery_service.mlflow.arn
  }

  depends_on = [aws_lb_listener.mlflow]
}

# ---------------------------------------------------------------- Inference API
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  # No task role on purpose: the API talks to MLflow over HTTP only and needs zero AWS permissions.

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name         = "api"
    image        = "${aws_ecr_repository.api.repository_url}:latest"
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = "http://${local.mlflow_host}:5000" },
      { name = "MODEL_NAME", value = "fraud-detector" },
      { name = "CHAMPION_ALIAS", value = "champion" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name                               = "api"
  cluster                            = aws_ecs_cluster.main.id
  task_definition                    = aws_ecs_task_definition.api.arn
  desired_count                      = var.api_desired_count
  launch_type                        = "FARGATE"
  health_check_grace_period_seconds  = 90
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  # New tasks that never pass /ready (e.g. cannot load the champion) => automatic rollback to the last good revision.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  # CD registers new task-definition revisions (new image tag). Terraform must not fight it.
  lifecycle {
    ignore_changes = [task_definition]
  }

  depends_on = [aws_lb_listener.api, aws_ecs_service.mlflow]
}
