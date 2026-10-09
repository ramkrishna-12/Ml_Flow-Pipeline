output "mlflow_tracking_uri" {
  value = "http://${aws_lb.main.dns_name}:5000"
}

output "api_url" {
  value = "http://${aws_lb.main.dns_name}"
}

output "ecr_mlflow_repo_url" {
  value = aws_ecr_repository.mlflow.repository_url
}

# Paste these into: GitHub repo -> Settings -> Secrets and variables -> Actions -> Variables
output "github_actions_variables" {
  value = {
    AWS_REGION          = var.aws_region
    AWS_ROLE_ARN        = aws_iam_role.github_actions.arn
    MLFLOW_TRACKING_URI = "http://${aws_lb.main.dns_name}:5000"
    DATA_BUCKET         = aws_s3_bucket.mlops.bucket
    ECR_API_REPO        = aws_ecr_repository.api.name
    ECS_CLUSTER         = aws_ecs_cluster.main.name
    ECS_API_SERVICE     = aws_ecs_service.api.name
    ECS_API_TASK_FAMILY = aws_ecs_task_definition.api.family
    API_URL             = "http://${aws_lb.main.dns_name}"
  }
}
